import numpy as np
import xgboost as xgb
import shap

# Setup: Train TWO models, ONLY difference is num_parallel_trees
base_params = {
    'max_depth': 6,
    'eta': 0.1,
    'lambda': 1.0,
    'subsample': 0.7,
    'colsample_bytree': 1.0,
    'seed': 42,  # FIXED seed
}

# Model 1: Sequential (no parallel)
params_seq = {**base_params, 'num_parallel_trees': 1}
model_seq = xgb.train(params_seq, dtrain, num_boost_round=100)

# Model 2: Parallel
params_par = {**base_params, 'num_parallel_trees': 5}
model_par = xgb.train(params_par, dtrain, num_boost_round=100)

# Get SHAP values
explainer_seq = shap.TreeExplainer(model_seq)
shap_seq = explainer_seq.shap_values(X_test)

explainer_par = shap.TreeExplainer(model_par)
shap_par = explainer_par.shap_values(X_test)

# === STABILITY METRICS ===

# 1. Absolute difference in individual feature SHAP values
individual_diff = np.abs(shap_seq - shap_par)

print("=" * 60)
print("INDIVIDUAL FEATURE SHAP STABILITY")
print("=" * 60)
print(f"Mean absolute difference: {individual_diff.mean():.6f}")
print(f"Median absolute difference: {np.median(individual_diff):.6f}")
print(f"Max absolute difference: {individual_diff.max():.6f}")
print(f"Std of differences: {individual_diff.std():.6f}")

# Per-feature stability
print("\nPer-feature stability:")
for i, feature in enumerate(feature_names):
    diff = individual_diff[:, i]
    print(f"  {feature:20s}: mean_diff={diff.mean():.6f}, max_diff={diff.max():.6f}")

# 2. Feature importance ranking stability
importance_seq = np.abs(shap_seq).mean(axis=0)
importance_par = np.abs(shap_par).mean(axis=0)

rank_seq = np.argsort(-importance_seq)  # Descending order
rank_par = np.argsort(-importance_par)

print("\n" + "=" * 60)
print("FEATURE IMPORTANCE RANKING STABILITY")
print("=" * 60)
print(f"Top 10 features (sequential):")
for rank in range(10):
    idx = rank_seq[rank]
    print(f"  {rank+1}. {feature_names[idx]:20s}: {importance_seq[idx]:.6f}")

print(f"\nTop 10 features (parallel):")
for rank in range(10):
    idx = rank_par[rank]
    print(f"  {rank+1}. {feature_names[idx]:20s}: {importance_par[idx]:.6f}")

# Calculate rank correlation (Spearman or Kendall)
from scipy.stats import spearmanr, kendalltau
spearman_corr, spearman_p = spearmanr(rank_seq, rank_par)
kendall_corr, kendall_p = kendalltau(rank_seq, rank_par)

print(f"\nRank correlation (Spearman): {spearman_corr:.4f} (p={spearman_p:.4e})")
print(f"Rank correlation (Kendall): {kendall_corr:.4f} (p={kendall_p:.4e})")

# 3. Prediction-level SHAP stability (per sample)
print("\n" + "=" * 60)
print("PREDICTION-LEVEL SHAP STABILITY")
print("=" * 60)

# For each sample, sum of SHAP values should equal prediction difference
pred_seq = model_seq.predict(xgb.DMatrix(X_test))
pred_par = model_par.predict(xgb.DMatrix(X_test))

shap_sum_seq = shap_seq.sum(axis=1)  # Sum across features
shap_sum_par = shap_par.sum(axis=1)

print(f"Mean SHAP sum (sequential): {shap_sum_seq.mean():.6f}")
print(f"Mean SHAP sum (parallel): {shap_sum_par.mean():.6f}")
print(f"Difference in mean sums: {abs(shap_sum_seq.mean() - shap_sum_par.mean()):.6f}")

# 4. Group-level aggregation stability
print("\n" + "=" * 60)
print("GROUP-LEVEL SHAP AGGREGATION STABILITY")
print("=" * 60)

# Define feature groups
feature_groups = {
    'group_A': feature_names[:5],
    'group_B': feature_names[5:10],
    'group_C': feature_names[10:],
}

for group_name, features in feature_groups.items():
    indices = [i for i, f in enumerate(feature_names) if f in features]
    
    group_sum_seq = np.abs(shap_seq[:, indices]).sum(axis=1).mean()
    group_sum_par = np.abs(shap_par[:, indices]).sum(axis=1).mean()
    group_diff = abs(group_sum_seq - group_sum_par)
    pct_diff = (group_diff / max(group_sum_seq, group_sum_par)) * 100
    
    print(f"{group_name}:")
    print(f"  Sequential: {group_sum_seq:.6f}")
    print(f"  Parallel:   {group_sum_par:.6f}")
    print(f"  Difference: {group_diff:.6f} ({pct_diff:.2f}%)")

# 5. Statistical summary
print("\n" + "=" * 60)
print("SUMMARY: IS SHAP STABLE WITH num_parallel_trees?")
print("=" * 60)

stability_score = 1 - (individual_diff.mean() / importance_seq.mean())
print(f"Stability score: {stability_score:.4f}")
print(f"  (1.0 = perfectly stable, 0.0 = completely unstable)")

if individual_diff.mean() < 0.001:
    print("\n✅ VERY STABLE: Parallel trees have minimal SHAP impact")
elif individual_diff.mean() < 0.01:
    print("\n✅ STABLE: Parallel trees have minor SHAP impact")
elif individual_diff.mean() < 0.05:
    print("\n⚠️  MODERATE: Parallel trees have noticeable SHAP impact")
else:
    print("\n❌ UNSTABLE: Parallel trees significantly affect SHAP values")
