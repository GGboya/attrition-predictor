"""Phase 4 — NR-Boost: Noise-Robust Gradient Boosting.

Novel classifier combining three noise-handling techniques on top of XGBoost:

1) Generalized Cross-Entropy (GCE) custom objective  (Zhang & Sabuncu 2018)
       L_GCE(p, y) = (1 - p_y^q) / q,    p_y = p if y=1 else 1-p
   q → 0 recovers cross-entropy; q → 1 recovers MAE.  Intermediate q gives a
   provably noise-robust loss: gradient shrinks on hard-to-fit (noisy) samples.

   Derivatives wrt the raw margin (logit):
       grad = -(2y-1) · p_y^q · (1 - p_y)
       hess = p_y^q · (1 - p_y) · (p_y - q (1 - p_y))
   Hessian can turn negative when q is large and p_y is small → clip to ε.

2) Self-paced sample reweighting (Kumar 2010 / Jiang 2015 adapted):
       Stage-1: train GCE-boosted ensemble.
       Stage-2: downweight top `drop_frac` of samples by training GCE loss.
       Stage-3: refit with updated weights.
   Samples the model can't fit after stage-1 are probable label noise; we
   don't zero them (conservative), we damp them ×0.3.

3) Class-imbalance-aware weighting baked into the objective rather than
   `scale_pos_weight` (which is ignored with custom obj).

Full-panel comparison on frozen 5469 split:
    RF (the Phase-3 champion), XGB (vanilla CE), ICM-Net, + NR-Boost variants.
Bootstrap CI on test AUC, DeLong NR-Boost vs RF.

Outputs:
    src/tables/table8_nrboost_panel.csv
    src/tables/table8_nrboost_ablation.csv
    src/tables/table8_delong.csv
    src/figures/fig8_nrboost_roc.png
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


# ─── GCE custom objective for XGB ───────────────────────────────────────
def gce_objective(q: float = 0.5, pos_weight: float = 1.0):
    """Return an XGB-compatible objective implementing GCE loss.

    q: GCE parameter.  q→0 is CE; q→1 is MAE.  Typical noise-robust q=0.5.
    pos_weight: multiplier on (grad, hess) for positive class; folds
                class-imbalance handling into the custom objective, since
                `scale_pos_weight` is ignored when obj is custom.
    """
    def _obj(y_pred, dtrain):
        y = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        p_y = np.where(y > 0.5, p, 1.0 - p)
        y_sign = 2.0 * y - 1.0
        grad = -y_sign * (p_y ** q) * (1.0 - p_y)
        hess = (p_y ** q) * (1.0 - p_y) * (p_y - q * (1.0 - p_y))
        hess = np.maximum(hess, 1e-6)
        w = np.where(y > 0.5, pos_weight, 1.0)
        return grad * w, hess * w
    return _obj


def gce_per_sample_loss(p: np.ndarray, y: np.ndarray, q: float) -> np.ndarray:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    p_y = np.where(y > 0.5, p, 1 - p)
    return (1 - p_y ** q) / q


# ─── NR-Boost ───────────────────────────────────────────────────────────
def fit_nrboost(X_tr: np.ndarray, y_tr: np.ndarray,
                *, q: float = 0.5, n_rounds: int = 400,
                n_stages: int = 2, drop_frac: float = 0.10,
                damp: float = 0.3, lr: float = 0.05, max_depth: int = 5,
                subsample: float = 0.9, colsample: float = 0.9,
                pos_weight: float | None = None, seed: int = RS,
                use_self_paced: bool = True):
    """Train NR-Boost with staged self-paced weighting.

    Returns (booster, train_weights) — use booster.predict on DMatrix for
    margin logits at inference time.
    """
    if pos_weight is None:
        pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    N = len(y_tr)
    weights = np.ones(N, dtype=np.float64)
    params = dict(
        max_depth=max_depth, eta=lr, subsample=subsample,
        colsample_bytree=colsample, min_child_weight=3,
        reg_lambda=1.0, tree_method="hist", seed=seed, verbosity=0,
    )
    booster = None
    rounds_per_stage = n_rounds // max(n_stages, 1)
    total_rounds = 0

    for stage in range(n_stages):
        dtrain = xgb.DMatrix(X_tr, label=y_tr, weight=weights)
        booster = xgb.train(
            params, dtrain, num_boost_round=rounds_per_stage,
            obj=gce_objective(q=q, pos_weight=pos_weight),
            xgb_model=booster,      # warm-start from previous stage
            verbose_eval=False,
        )
        total_rounds += rounds_per_stage

        if stage == n_stages - 1 or not use_self_paced:
            break
        # compute per-sample GCE loss on current model; damp the hardest
        p = 1.0 / (1.0 + np.exp(-booster.predict(dtrain, output_margin=True)))
        losses = gce_per_sample_loss(p, y_tr, q)
        thresh = np.percentile(losses, 100 * (1 - drop_frac))
        weights = np.where(losses > thresh, weights * damp, weights)

    return booster


def predict_nrboost(booster, X) -> np.ndarray:
    d = xgb.DMatrix(X)
    margin = booster.predict(d, output_margin=True)
    return 1.0 / (1.0 + np.exp(-margin))


def nrboost_cv(name, *, q, n_stages, drop_frac, use_self_paced=True, seed=RS):
    print(f"\n── {name}  q={q}  stages={n_stages}  drop={drop_frac}  "
          f"self_paced={use_self_paced} ──")
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_tr))
    for f, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
        b = fit_nrboost(X_tr[tr], y_tr[tr], q=q, n_rounds=400,
                         n_stages=n_stages, drop_frac=drop_frac,
                         use_self_paced=use_self_paced, seed=seed + f)
        oof[va] = predict_nrboost(b, X_tr[va])
        print(f"  fold {f} val AUC={roc_auc_score(y_tr[va], oof[va]):.4f}")
    cv = roc_auc_score(y_tr, oof)
    print(f"  CV AUC = {cv:.4f}")
    # retrain full
    booster = fit_nrboost(X_tr, y_tr, q=q, n_rounds=400,
                           n_stages=n_stages, drop_frac=drop_frac,
                           use_self_paced=use_self_paced, seed=seed)
    p_te = predict_nrboost(booster, X_te)
    return oof, p_te, cv, booster


# ─── run: NR-Boost grid + ablations + anchor baselines ──────────────────
oof_probs = {}; test_probs = {}
q_grid = [0.3, 0.5, 0.7]
best_q, best_cv_auc = None, -np.inf
for q in q_grid:
    oof, p_te, cv, _ = nrboost_cv(f"NR-Boost q={q}", q=q, n_stages=2, drop_frac=0.10)
    oof_probs[f"NR_q{q}"] = oof; test_probs[f"NR_q{q}"] = p_te
    if cv > best_cv_auc:
        best_cv_auc, best_q = cv, q
print(f"\n★ best q (by CV) = {best_q}  CV AUC = {best_cv_auc:.4f}")

# ablation at best q
oof, p_te, _, _ = nrboost_cv("NR-Boost −self-paced", q=best_q, n_stages=1,
                              drop_frac=0.0, use_self_paced=False)
oof_probs["NR_NSP"] = oof; test_probs["NR_NSP"] = p_te

oof, p_te, _, _ = nrboost_cv("NR-Boost −GCE (=plain XGB with pos_w)",
                              q=1e-3, n_stages=2, drop_frac=0.10)
# q→0 recovers CE, so q=1e-3 is effectively CE baseline but with our self-paced
oof_probs["NR_CE"] = oof; test_probs["NR_CE"] = p_te

# anchor baselines
print("\n── RandomForest anchor ──")
spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
oof_rf = np.zeros(len(y_tr))
for f, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
    rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RS)
    rf.fit(X_tr[tr], y_tr[tr])
    oof_rf[va] = rf.predict_proba(X_tr[va])[:, 1]
rf_full = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=RS).fit(X_tr, y_tr)
p_te_rf = rf_full.predict_proba(X_te)[:, 1]
oof_probs["RF"] = oof_rf; test_probs["RF"] = p_te_rf
print(f"  CV AUC = {roc_auc_score(y_tr, oof_rf):.4f}")

print("\n── XGB (vanilla CE, scale_pos_weight) ──")
oof_xgb = np.zeros(len(y_tr))
for f, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
    m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                           subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
                           reg_lambda=1.0, scale_pos_weight=spw,
                           eval_metric="logloss", random_state=RS, n_jobs=-1,
                           verbosity=0)
    m.fit(X_tr[tr], y_tr[tr])
    oof_xgb[va] = m.predict_proba(X_tr[va])[:, 1]
m_full = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
                            reg_lambda=1.0, scale_pos_weight=spw,
                            eval_metric="logloss", random_state=RS, n_jobs=-1,
                            verbosity=0).fit(X_tr, y_tr)
p_te_xgb = m_full.predict_proba(X_te)[:, 1]
oof_probs["XGB"] = oof_xgb; test_probs["XGB"] = p_te_xgb
print(f"  CV AUC = {roc_auc_score(y_tr, oof_xgb):.4f}")

# Best NR-Boost + RF ensemble
best_key = f"NR_q{best_q}"
oof_probs["NR_RF"] = (oof_probs[best_key] + oof_probs["RF"]) / 2
test_probs["NR_RF"] = (test_probs[best_key] + test_probs["RF"]) / 2


# ─── calibrate + score ──────────────────────────────────────────────────
name_map = {
    "NR_q0.3": "NR-Boost (q=0.3)", "NR_q0.5": "NR-Boost (q=0.5)",
    "NR_q0.7": "NR-Boost (q=0.7)",
    "NR_NSP":  "NR-Boost −self-paced",
    "NR_CE":   "NR-Boost −GCE (q≈0)",
    "RF":      "RandomForest", "XGB": "XGBoost",
    "NR_RF":   "NR-Boost + RF ensemble",
}
panel, ablation, cal = [], [], {}
for k in oof_probs:
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_probs[k], y_tr)
    oof_cal = iso.predict(oof_probs[k])
    test_cal = iso.predict(test_probs[k])
    thr = best_f1_threshold(y_tr, oof_cal)
    row = metrics_row(name_map[k], y_te, test_cal, thr)
    cal[k] = test_cal
    panel.append(row)
    if k.startswith("NR"):
        ablation.append(row)

panel_df = pd.DataFrame(panel).sort_values("AUC", ascending=False).reset_index(drop=True)
abl_df = pd.DataFrame(ablation).sort_values("AUC", ascending=False).reset_index(drop=True)
panel_df.to_csv(OUT_TABLES / "table8_nrboost_panel.csv", index=False)
abl_df.to_csv(OUT_TABLES / "table8_nrboost_ablation.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table8_nrboost_panel.csv'}")
print(panel_df.to_string(index=False,
      formatters={c: "{:.3f}".format for c in panel_df.select_dtypes("float").columns}))

# ─── DeLong: NR-Boost (best-q) and NR_RF ensemble vs each baseline ──────
delong_rows = []
for champ_key in [best_key, "NR_RF"]:
    champ_p = cal[champ_key]
    for k, p in cal.items():
        if k == champ_key:
            continue
        auc_a, auc_b, z, pval = delong_test(y_te, champ_p, p)
        delong_rows.append({
            "champion": name_map[champ_key],
            "baseline": name_map[k],
            "AUC_champ": auc_a, "AUC_base": auc_b,
            "ΔAUC": auc_a - auc_b, "z": z, "p_value": pval,
            "significant_05": pval < 0.05,
        })
dlg = pd.DataFrame(delong_rows).sort_values(["champion", "p_value"]).reset_index(drop=True)
dlg.to_csv(OUT_TABLES / "table8_delong.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table8_delong.csv'}")
print(dlg.to_string(index=False,
      formatters={c: "{:.4f}".format for c in dlg.select_dtypes("float").columns}))

# ─── figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
order = ["XGB", "RF", "NR_CE", "NR_NSP", "NR_q0.3", "NR_q0.5", "NR_q0.7", "NR_RF"]
colors = {"NR_RF": "crimson", best_key: "darkorange", "RF": "steelblue"}
for k in order:
    p = cal[k]
    fpr, tpr, _ = roc_curve(y_te, p)
    lw = 2.5 if k in ("NR_RF", best_key) else 1.2
    alpha = 1.0 if k in ("NR_RF", best_key, "RF") else 0.6
    c = colors.get(k, None)
    axes[0].plot(fpr, tpr, label=f"{name_map[k]}  {roc_auc_score(y_te, p):.3f}",
                  linewidth=lw, alpha=alpha, color=c)
axes[0].plot([0, 1], [0, 1], "--", c="grey", alpha=0.5)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test, calibrated)")
axes[0].legend(fontsize=8, loc="lower right")
for k in order:
    p = cal[k]
    pr, rc, _ = precision_recall_curve(y_te, p)
    lw = 2.5 if k in ("NR_RF", best_key) else 1.2
    alpha = 1.0 if k in ("NR_RF", best_key, "RF") else 0.6
    c = colors.get(k, None)
    axes[1].plot(rc, pr, label=f"{name_map[k]}  {average_precision_score(y_te, p):.3f}",
                  linewidth=lw, alpha=alpha, color=c)
axes[1].axhline(y_te.mean(), ls="--", c="grey", alpha=0.5)
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR (test, calibrated)")
axes[1].legend(fontsize=8, loc="upper right")
plt.tight_layout()
fig_path = OUT_FIGS / "fig8_nrboost_roc.png"
plt.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"wrote {fig_path}")
