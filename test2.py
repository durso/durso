def entropy_regularized_xgboost_loss_clean(alpha=0.1):
    """
    EDHD-FG-XGBoost with correct chain rule
    
    Returns derivatives w.r.t. raw margin f (not probabilities p)
    """
    def objective(preds, dtrain):
        labels = dtrain.get_label()
        f = preds  # Raw margin from XGBoost trees
        
        # Clip for numerical stability
        f = np.clip(f, -500, 500)
        
        # Transform to probabilities
        p = 1 / (1 + np.exp(-f))
        p = np.clip(p, 1e-15, 1 - 1e-15)
        
        # Loss components
        bce = -(labels * np.log(p) + (1 - labels) * np.log(1 - p))
        entropy = -(p * np.log(p) + (1 - p) * np.log(1 - p))
        
        # Sigmoid derivative (needed for chain rule)
        p_prime = p * (1 - p)
        
        # Gradients w.r.t. probabilities
        grad_bce_p = p - labels
        grad_entropy_p = alpha * (np.log(1 - p) - np.log(p))
        grad_entropy_p = np.nan_to_num(grad_entropy_p, nan=0.0)
        
        # CHAIN RULE: Convert to derivatives w.r.t. raw margin
        grad = (grad_bce_p + grad_entropy_p) * p_prime
        
        # Hessians (simplified)
        hess = (p_prime ** 2) * (1 / (p * (1 - p))) + \
               2 * p_prime * (1 - 2*p) * (grad_bce_p + grad_entropy_p)
        
        # Safety
        hess = np.maximum(hess, 1e-8)
        
        return grad, hess
    
    return objective
