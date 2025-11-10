def entropy_regularized_xgboost_loss(alpha=0.1):
    
    def objective(preds, dtrain):
        """
        Calculate gradient and Hessian for XGBoost tree construction
        XGBoost uses these to build optimal splits
        """
        labels = dtrain.get_label()
        
        # Clip predictions to avoid log(0)
        preds = np.clip(preds, 1e-15, 1 - 1e-15)
        
        # ===== PART 1: Binary Cross-Entropy (Primary Loss) =====
        # L_BCE = -[y*log(p) + (1-y)*log(1-p)]
        bce = -(labels * np.log(preds) + (1 - labels) * np.log(1 - preds))
        
        # Gradient: dL_BCE/dp = p - y
        grad_bce = preds - labels
        
        # Hessian: d²L_BCE/dp² = p(1-p)
        hess_bce = preds * (1 - preds)
        
        # ===== PART 2: Entropy Regularization Term =====
        # H(p) = -[p*log(p) + (1-p)*log(1-p)]
        # Higher entropy (p closer to 0.5) = less confident
        entropy = -(preds * np.log(preds) + (1 - preds) * np.log(1 - preds))
        
        # Gradient of entropy term: d(-α*H)/dp = α * [log(p) - log(1-p)]
        # (Simplified: log(1-p) - log(p))
        grad_entropy = alpha * (np.log(1 - preds) - np.log(preds))
        
        # Hessian of entropy term: d²(-α*H)/dp² = α / [p(1-p)]
        hess_entropy = alpha / (preds * (1 - preds))
        
        # ===== TOTAL LOSS =====
        # Combine both components
        grad = grad_bce + grad_entropy
        hess = hess_bce + hess_entropy
        
        return grad, hess
    
    return objective
