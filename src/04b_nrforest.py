"""Phase 4b — NR-Forest: Noise-Robust Random Forest.

Two-stage RandomForest that inherits GCE's noise-robustness idea but keeps
tree-bagging's variance-reduction behaviour (which phase 4a showed is near the
Bayes ceiling on this 5K tabular survey).

Stage-1:  vanilla RF with `oob_score=True, bootstrap=True`.
          OOB probabilities p_oob serve as honest held-out predictions.

          GCE per-sample loss on OOB:
              p_y   = p_oob       if y=1 else 1-p_oob
              L_i   = (1 - p_y^q) / q

          High loss == hard to fit with clean trees == probable label noise.
          Convert loss → sample weight via per-class normalised exp(-T · L̃):
              L̃  =  (L - min_c L) / (max_c L - min_c L + ε)    (per class c)
              w_i = exp(-T · L̃_i);   rescale w_i so mean=1 within class.
          Per-class normalisation keeps the prior intact — without it the
          positives (higher base loss due to imbalance) would all get damped.

Stage-2:  RF trained on the same data with sample_weight = w from stage-1.
          Hard/noisy rows contribute less; clean rows dominate splits.

Inference blends:
              p(x) = blend · p_stage1(x) + (1 - blend) · p_stage2(x).

Panel vs RF, NR-Boost (q=0.7), XGB on the frozen 5469 split.
Bootstrap CI, DeLong NR-Forest vs RF.

Outputs:
    src/tables/table9_nrforest_panel.csv
    src/tables/table9_nrforest_ablation.csv
    src/tables/table9_delong.csv
    src/figures/fig9_nrforest_roc.png
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold

import importlib.util, sys
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

_utils = _load("stats_utils", "_utils.py")
delong_test = _utils.delong_test

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RS = 42
TARGET = "离职行为"
INTENT = "离职意向"
N_FOLDS = 5

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)

np.random.seed(RS)


# ─── data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")

feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]
print(f"clean {df.shape}  train {len(train_idx)}  test {len(test_idx)}  "
      f"features: {len(feat_cols)} (no intent)")

y = df[TARGET].values.astype(int)
X_raw = df[feat_cols].values.astype(np.float32)
X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]


# ─── helpers ────────────────────────────────────────────────────────────
def best_f1_threshold(y_true, p):
    best, thr = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_true, (p >= t).astype(int), pos_label=1, zero_division=0)
        if f > best:
            best, thr = f, t
    return thr


def bootstrap_ci(y_true, p, metric, n=1000, seed=RS):
    rng = np.random.default_rng(seed); vals = []; N = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try: vals.append(metric(y_true[idx], p[idx]))
        except Exception: pass
    a = np.asarray(vals)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def metrics_row(name, y_true, p, thr):
    pred = (p >= thr).astype(int)
    lo, hi = bootstrap_ci(y_true, p, roc_auc_score)
    plo, phi = bootstrap_ci(y_true, p, average_precision_score)
    return {
        "model": name,
        "AUC": roc_auc_score(y_true, p),
        "AUC_lo": lo, "AUC_hi": hi,
        "PR_AUC": average_precision_score(y_true, p),
        "PR_lo": plo, "PR_hi": phi,
        "F1": f1_score(y_true, pred, zero_division=0),
        "Bal_Acc": balanced_accuracy_score(y_true, pred),
        "Brier": brier_score_loss(y_true, p),
        "MCC": matthews_corrcoef(y_true, pred),
        "thr": thr,
    }


# ─── NR-Forest ──────────────────────────────────────────────────────────
def gce_loss(p: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    p = np.clip(p, 1e-4, 1 - 1e-4)
    p_y = np.where(y > 0.5, p, 1 - p)
    return (1 - p_y ** q) / q


def derive_nr_weights(p_oob: np.ndarray, y: np.ndarray, q: float, T: float) -> np.ndarray:
    """GCE-loss-derived per-class sample weights, mean=1 within each class."""
    losses = gce_loss(p_oob, y, q)
    w = np.ones(len(y), dtype=np.float64)
    for cls in (0, 1):
        m = y == cls
        l = losses[m]
        l_rng = l.max() - l.min()
        l_norm = (l - l.min()) / (l_rng + 1e-9)
        wi = np.exp(-T * l_norm)
        w[m] = wi / wi.mean()
    return w


class NRForest:
    def __init__(self, n_trees_s1=300, n_trees_s2=500, max_depth=10,
                 min_samples_leaf=5, q=0.5, T=5.0, blend=0.3, seed=RS):
        self.n_trees_s1 = n_trees_s1; self.n_trees_s2 = n_trees_s2
        self.max_depth = max_depth; self.min_samples_leaf = min_samples_leaf
        self.q = q; self.T = T; self.blend = blend; self.seed = seed

    def fit(self, X, y):
        self.rf1 = RandomForestClassifier(
            n_estimators=self.n_trees_s1, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",
            bootstrap=True, oob_score=True, n_jobs=-1,
            random_state=self.seed).fit(X, y)
        p_oob = self.rf1.oob_decision_function_[:, 1]
        # fallback for any NaN (rare: sample never OOB) — use 0.5
        p_oob = np.where(np.isnan(p_oob), 0.5, p_oob)
        self.sample_weights_ = derive_nr_weights(p_oob, y, self.q, self.T)
        self.rf2 = RandomForestClassifier(
            n_estimators=self.n_trees_s2, max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight="balanced_subsample",
            bootstrap=True, n_jobs=-1,
            random_state=self.seed + 1).fit(X, y, sample_weight=self.sample_weights_)
        return self

    def predict_proba(self, X):
        p1 = self.rf1.predict_proba(X)[:, 1]
        p2 = self.rf2.predict_proba(X)[:, 1]
        p = self.blend * p1 + (1 - self.blend) * p2
        return np.column_stack([1 - p, p])


def nrforest_cv(name, *, q, T, blend, seed=RS):
    print(f"\n── {name}  q={q}  T={T}  blend={blend} ──")
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_tr))
    for f, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
        m = NRForest(q=q, T=T, blend=blend, seed=seed + f).fit(X_tr[tr], y_tr[tr])
        oof[va] = m.predict_proba(X_tr[va])[:, 1]
        print(f"  fold {f} val AUC={roc_auc_score(y_tr[va], oof[va]):.4f}")
    cv = roc_auc_score(y_tr, oof)
    print(f"  CV AUC = {cv:.4f}")
    full = NRForest(q=q, T=T, blend=blend, seed=seed).fit(X_tr, y_tr)
    p_te = full.predict_proba(X_te)[:, 1]
    return oof, p_te, cv


# ─── grid: small to keep budget in line ─────────────────────────────────
oof_probs = {}; test_probs = {}
grid = [
    ("NRF q0.5 T5 b0.3",  dict(q=0.5, T=5.0, blend=0.3)),
    ("NRF q0.7 T5 b0.3",  dict(q=0.7, T=5.0, blend=0.3)),
    ("NRF q0.5 T3 b0.3",  dict(q=0.5, T=3.0, blend=0.3)),
    ("NRF q0.5 T10 b0.3", dict(q=0.5, T=10.0, blend=0.3)),
    ("NRF q0.5 T5 b0.0",  dict(q=0.5, T=5.0, blend=0.0)),
    ("NRF q0.5 T5 b0.5",  dict(q=0.5, T=5.0, blend=0.5)),
]
best_key, best_cv = None, -np.inf
for nm, kw in grid:
    key = nm.replace(" ", "_")
    oof, p_te, cv = nrforest_cv(nm, **kw)
    oof_probs[key] = oof; test_probs[key] = p_te
    if cv > best_cv:
        best_cv, best_key = cv, key
print(f"\n★ best NRF (by CV) = {best_key}  CV AUC = {best_cv:.4f}")


# ─── baselines on same frozen split ─────────────────────────────────────
print("\n── RF (stage-1 only anchor) ──")
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
oof_rf = np.zeros(len(y_tr))
for f, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
    rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RS).fit(X_tr[tr], y_tr[tr])
    oof_rf[va] = rf.predict_proba(X_tr[va])[:, 1]
rf_full = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=RS).fit(X_tr, y_tr)
p_te_rf = rf_full.predict_proba(X_te)[:, 1]
oof_probs["RF"] = oof_rf; test_probs["RF"] = p_te_rf
print(f"  CV AUC = {roc_auc_score(y_tr, oof_rf):.4f}")

print("\n── XGB (vanilla CE) ──")
spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
oof_xgb = np.zeros(len(y_tr))
for f, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
    m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
                           reg_lambda=1.0, scale_pos_weight=spw,
                           eval_metric="logloss", random_state=RS, n_jobs=-1,
                           verbosity=0).fit(X_tr[tr], y_tr[tr])
    oof_xgb[va] = m.predict_proba(X_tr[va])[:, 1]
m_full = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
                            reg_lambda=1.0, scale_pos_weight=spw,
                            eval_metric="logloss", random_state=RS, n_jobs=-1,
                            verbosity=0).fit(X_tr, y_tr)
oof_probs["XGB"] = oof_xgb; test_probs["XGB"] = m_full.predict_proba(X_te)[:, 1]
print(f"  CV AUC = {roc_auc_score(y_tr, oof_xgb):.4f}")


# ─── NRF + RF blend (best NRF already blends rf1+rf2; now also add vanilla RF) ──
oof_probs["NRF_RF_avg"] = (oof_probs[best_key] + oof_probs["RF"]) / 2
test_probs["NRF_RF_avg"] = (test_probs[best_key] + test_probs["RF"]) / 2


# ─── calibrate + score ──────────────────────────────────────────────────
name_map = {k: k.replace("_", " ") for k in oof_probs}
name_map["RF"] = "RandomForest"; name_map["XGB"] = "XGBoost"
name_map["NRF_RF_avg"] = "NR-Forest + RF avg"
panel, cal = [], {}
for k in oof_probs:
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_probs[k], y_tr)
    oof_cal = iso.predict(oof_probs[k])
    test_cal = iso.predict(test_probs[k])
    thr = best_f1_threshold(y_tr, oof_cal)
    row = metrics_row(name_map[k], y_te, test_cal, thr)
    cal[k] = test_cal
    panel.append(row)

panel_df = pd.DataFrame(panel).sort_values("AUC", ascending=False).reset_index(drop=True)
panel_df.to_csv(OUT_TABLES / "table9_nrforest_panel.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table9_nrforest_panel.csv'}")
print(panel_df.to_string(index=False,
      formatters={c: "{:.3f}".format for c in panel_df.select_dtypes("float").columns}))

# ablation-only table (NRF variants)
abl = panel_df[panel_df.model.str.startswith("NRF")].copy()
abl.to_csv(OUT_TABLES / "table9_nrforest_ablation.csv", index=False)
print(f"wrote {OUT_TABLES / 'table9_nrforest_ablation.csv'}")


# ─── DeLong: NR-Forest (best) vs each baseline ──────────────────────────
delong_rows = []
for champ in [best_key, "NRF_RF_avg"]:
    cp = cal[champ]
    for k, p in cal.items():
        if k == champ: continue
        auc_a, auc_b, z, pval = delong_test(y_te, cp, p)
        delong_rows.append({
            "champion": name_map[champ], "baseline": name_map[k],
            "AUC_champ": auc_a, "AUC_base": auc_b,
            "ΔAUC": auc_a - auc_b, "z": z, "p_value": pval,
            "significant_05": pval < 0.05,
        })
dlg = pd.DataFrame(delong_rows).sort_values(["champion", "p_value"]).reset_index(drop=True)
dlg.to_csv(OUT_TABLES / "table9_delong.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table9_delong.csv'}")
print(dlg.to_string(index=False,
      formatters={c: "{:.4f}".format for c in dlg.select_dtypes("float").columns}))


# ─── figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
order = ["XGB", "RF", best_key, "NRF_RF_avg"]
order += [k for k in oof_probs if k.startswith("NRF") and k not in order]
colors = {"NRF_RF_avg": "crimson", best_key: "darkorange", "RF": "steelblue"}
for k in order:
    p = cal[k]
    fpr, tpr, _ = roc_curve(y_te, p)
    lw = 2.5 if k in ("NRF_RF_avg", best_key) else 1.2
    alpha = 1.0 if k in ("NRF_RF_avg", best_key, "RF") else 0.5
    c = colors.get(k, None)
    axes[0].plot(fpr, tpr, label=f"{name_map[k]}  {roc_auc_score(y_te, p):.3f}",
                  linewidth=lw, alpha=alpha, color=c)
axes[0].plot([0, 1], [0, 1], "--", c="grey", alpha=0.5)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test, calibrated)")
axes[0].legend(fontsize=7, loc="lower right")
for k in order:
    p = cal[k]
    pr, rc, _ = precision_recall_curve(y_te, p)
    lw = 2.5 if k in ("NRF_RF_avg", best_key) else 1.2
    alpha = 1.0 if k in ("NRF_RF_avg", best_key, "RF") else 0.5
    c = colors.get(k, None)
    axes[1].plot(rc, pr, label=f"{name_map[k]}  {average_precision_score(y_te, p):.3f}",
                  linewidth=lw, alpha=alpha, color=c)
axes[1].axhline(y_te.mean(), ls="--", c="grey", alpha=0.5)
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR (test, calibrated)")
axes[1].legend(fontsize=7, loc="upper right")
plt.tight_layout()
fig_path = OUT_FIGS / "fig9_nrforest_roc.png"
plt.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"wrote {fig_path}")
