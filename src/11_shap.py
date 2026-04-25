"""Phase 9 — SHAP interpretability.

Generates the interpretability artefacts the paper needs:

  1. Tree-SHAP on the RF base of the Phase 6 champion (meta coef 0.376,
     largest weight of all five bases per table13_stack_v5_weights.csv).
     This is the primary "which features matter" figure.

  2. Meta-weighted feature importance aggregation across RF / NR-Boost / ET
     (the three tree bases in the stack). Weighted by the Phase 6 L2-LR
     meta coefficients so rankings reflect stack-level importance rather
     than any single base's.

  3. Partial Dependence Plots for the top-5 features from the RF SHAP
     ranking.

Outputs
-------
src/figures/fig18a_shap_summary_rf.png
src/figures/fig18b_shap_bar_rf.png
src/figures/fig19_pdp_top5.png
src/tables/table18_shap_importance.csv
src/tables/table18_meta_weighted_importance.csv
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import importlib.util, sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.inspection import PartialDependenceDisplay

import shap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Chinese label rendering
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC",
                                    "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

_stack = _load("stack_v3", "05_stacking.py")  # for fit_nrboost / predict_nrboost (NRBoost uses XGBoost)


RS = 42
TARGET = "离职行为"
INTENT = "离职意向"

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)

np.random.seed(RS)


# ─── data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")
feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]

y_all = df[TARGET].values.astype(int)
X_raw = df[feat_cols].values.astype(np.float32)
X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
y_tr, y_te = y_all[train_idx], y_all[test_idx]

sw_v6 = np.load("data/processed/sample_weights_v6.npy").astype(np.float32)

print(f"train={len(y_tr)}  test={len(y_te)}")
print(f"features ({len(feat_cols)}):", feat_cols)


# ─── 1. Refit RF with Phase 6 champion config ──────────────────────────
print("\n── fitting RF (Phase 6 champion config) ──")
rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                             class_weight="balanced_subsample",
                             n_jobs=-1, random_state=RS)
rf.fit(X_tr, y_tr, sample_weight=sw_v6)


# ─── 2. Tree-SHAP on RF (training set) ─────────────────────────────────
print("\n── computing tree-SHAP on RF ──")
explainer = shap.TreeExplainer(rf)
shap_out = explainer.shap_values(X_tr, check_additivity=False)
# RF classifier: shap_values returns ndarray (N, F, 2) for newer versions, list of 2 arrays for older
if isinstance(shap_out, list):
    shap_vals = shap_out[1]  # positive class
elif shap_out.ndim == 3:
    shap_vals = shap_out[:, :, 1]
else:
    shap_vals = shap_out
print(f"  shap_vals shape = {shap_vals.shape}")
mean_abs = np.abs(shap_vals).mean(axis=0)
shap_df = pd.DataFrame({
    "feature": feat_cols,
    "mean_abs_shap_rf": mean_abs,
    "rf_builtin_importance": rf.feature_importances_,
}).sort_values("mean_abs_shap_rf", ascending=False).reset_index(drop=True)
print("\nSHAP ranking (top-10 RF):")
for i, row in shap_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:14s}  SHAP={row['mean_abs_shap_rf']:.4f}  "
          f"RF-builtin={row['rf_builtin_importance']:.4f}")


# ─── 3. SHAP summary beeswarm (fig18a) ─────────────────────────────────
plt.figure(figsize=(8.5, 5.5))
shap.summary_plot(shap_vals, X_tr, feature_names=feat_cols,
                  show=False, plot_size=None, max_display=12)
plt.title("SHAP summary — RF base of Phase 6 champion\n(meta coef 0.376, largest of 5 bases)",
          fontsize=11, pad=12)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig18a_shap_summary_rf.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nwrote {OUT_FIGS / 'fig18a_shap_summary_rf.png'}")

# Bar plot (fig18b)
plt.figure(figsize=(7.5, 4.8))
shap.summary_plot(shap_vals, X_tr, feature_names=feat_cols, plot_type="bar",
                  show=False, max_display=12)
plt.title("SHAP global feature importance — RF base\n(mean |SHAP| over training set)",
          fontsize=11, pad=10)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig18b_shap_bar_rf.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {OUT_FIGS / 'fig18b_shap_bar_rf.png'}")


# ─── 4. Meta-weighted feature importance across tree bases ─────────────
print("\n── computing meta-weighted importance (RF + NRBoost + ET) ──")
# Phase 6 champion meta coefs (from table13_stack_v5_weights.csv row A: Phase 6 CL)
META_COEF = {"RF": 0.37645965, "NRBoost": 0.01412834,
             "MTMLP": 0.31558276, "SVM": -0.04802189, "ET": 0.28435129}
# Only tree bases have feature_importances_; we aggregate those three and normalise weights
TREE_BASES = ["RF", "NRBoost", "ET"]
meta_sum = sum(max(META_COEF[k], 0.0) for k in TREE_BASES)
norm_weights = {k: max(META_COEF[k], 0.0) / meta_sum for k in TREE_BASES}
print("  tree-base normalised weights (within RF+NRB+ET):",
      {k: f"{v:.3f}" for k, v in norm_weights.items()})

# Fit NRBoost and ET with Phase 6 champion hyperparams
print("  fitting NRBoost (XGBoost-based) — uses pos_weight prior, not per-sample CL ...")
nrb = _stack.fit_nrboost(X_tr, y_tr, q=0.7, n_rounds=400, n_stages=2,
                          drop_frac=0.10, seed=RS)

print("  fitting ExtraTrees ...")
et = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                           class_weight="balanced_subsample",
                           n_jobs=-1, random_state=RS)
et.fit(X_tr, y_tr, sample_weight=sw_v6)

def _nrb_importance(booster, n_features):
    """Normalised feature-gain from the final NRBoost XGBoost booster."""
    imp = np.zeros(n_features)
    try:
        score = booster.get_score(importance_type="gain")
    except Exception:
        return imp
    for k, v in score.items():
        i = int(k.lstrip("f"))
        if 0 <= i < n_features:
            imp[i] += v
    s = imp.sum()
    return imp / s if s > 0 else imp

nrb_imp = _nrb_importance(nrb, len(feat_cols))
rf_imp = rf.feature_importances_
et_imp = et.feature_importances_

# weighted sum (normalised by meta weights)
agg_imp = (norm_weights["RF"] * rf_imp
           + norm_weights["NRBoost"] * nrb_imp
           + norm_weights["ET"] * et_imp)

mw_df = pd.DataFrame({
    "feature": feat_cols,
    "RF_imp": rf_imp,
    "NRBoost_imp": nrb_imp,
    "ET_imp": et_imp,
    "meta_weighted_imp": agg_imp,
    "mean_abs_shap_rf": mean_abs,
}).sort_values("meta_weighted_imp", ascending=False).reset_index(drop=True)
mw_df["rank_meta_weighted"] = mw_df.index + 1
mw_df["rank_shap_rf"] = mw_df["mean_abs_shap_rf"].rank(ascending=False).astype(int)

print("\nMeta-weighted importance (top-10):")
for i, row in mw_df.head(10).iterrows():
    print(f"  {i+1:2d}. {row['feature']:14s}  "
          f"RF={row['RF_imp']:.3f}  NRB={row['NRBoost_imp']:.3f}  "
          f"ET={row['ET_imp']:.3f}  agg={row['meta_weighted_imp']:.4f}")

mw_df.to_csv(OUT_TABLES / "table18_meta_weighted_importance.csv", index=False)
shap_df.to_csv(OUT_TABLES / "table18_shap_importance.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table18_shap_importance.csv'}")
print(f"wrote {OUT_TABLES / 'table18_meta_weighted_importance.csv'}")

# rank-correlation between SHAP and meta-weighted rankings
from scipy.stats import spearmanr
rho, p_rank = spearmanr(mw_df["rank_meta_weighted"], mw_df["rank_shap_rf"])
print(f"\nSpearman rank-correlation (SHAP-RF vs meta-weighted): ρ={rho:.3f}  p={p_rank:.4f}")


# ─── 5. PDP for top-5 features ─────────────────────────────────────────
print("\n── computing PDP for top-5 features (by meta-weighted importance) ──")
top5 = mw_df.head(5)["feature"].tolist()
top5_idx = [feat_cols.index(f) for f in top5]
print(f"  top-5: {top5}")

fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
disp = PartialDependenceDisplay.from_estimator(
    rf, X_tr, features=top5_idx, feature_names=feat_cols,
    kind="average", ax=axes, grid_resolution=30, response_method="predict_proba",
)
for ax in axes.flatten() if hasattr(axes, "flatten") else [axes]:
    ax.grid(alpha=0.3)
plt.suptitle("Partial Dependence — top-5 features (by meta-weighted tree importance) on RF",
             fontsize=12, y=1.05)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig19_pdp_top5.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {OUT_FIGS / 'fig19_pdp_top5.png'}")


# ─── final summary ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  FINAL SUMMARY (Phase 9 SHAP)")
print("=" * 72)
print(f"Top-5 features (meta-weighted): {top5}")
print(f"Top-5 features (SHAP-RF):       {shap_df.head(5)['feature'].tolist()}")
print(f"Rank-correlation ρ = {rho:.3f} (p={p_rank:.4f})")
print(f"\nArtifacts:")
print(f"  {OUT_FIGS / 'fig18a_shap_summary_rf.png'}")
print(f"  {OUT_FIGS / 'fig18b_shap_bar_rf.png'}")
print(f"  {OUT_FIGS / 'fig19_pdp_top5.png'}")
print(f"  {OUT_TABLES / 'table18_shap_importance.csv'}")
print(f"  {OUT_TABLES / 'table18_meta_weighted_importance.csv'}")
