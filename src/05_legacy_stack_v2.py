"""Phase 5 — OOF-tuned stacking, legacy v2.  SUPERSEDED by `05_stacking.py`.

Kept for reproducibility of earlier iterations.  v3 (current) adds seed-averaged
MT-MLP and a fifth base (ExtraTrees) on top of this v2 setup.

Legacy v1 (`05_legacy_stack_v1.py`) used 4 tree-based bases (RF, NR-Boost, XGB, LGB).
OOF correlations were all ≥0.86 → no orthogonal signal for the meta to exploit,
and Stack-LR collapsed to "just use RF" (w_RF=0.97).

v2 mixes four genuinely different inductive biases:
    - RF         (bagged trees — variance reduction, Phase 4b champion)
    - NR-Boost   (boosted trees with GCE noise-robust loss, Phase 4a)
    - MT-MLP     (neural net with ordinal intent head as auxiliary supervision;
                  at inference only the binary head is used, so no leakage.
                  This injects signal from 离职意向 that RF cannot access.)
    - SVM-RBF    (kernel method, non-tree, non-linear boundary)

Meta-learners: L2-logistic on logit-OOF, convex combo on prob-OOF.

Outputs:
    src/tables/table9b_stack_v2_panel.csv
    src/tables/table9b_stack_v2_weights.csv
    src/tables/table9b_stack_v2_delong.csv
    src/tables/table9b_stack_v2_corr.csv
    src/figures/fig9b_stack_v2_roc.png
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
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F

import importlib.util, sys
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

_utils = _load("stats_utils", "_utils.py")
delong_test = _utils.delong_test
_mtmod = _load("mt_model", "01a_mt_model.py")
MTMlp = _mtmod.MTMlp
joint_loss = _mtmod.joint_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RS = 42
TARGET = "离职行为"
INTENT = "离职意向"
N_FOLDS = 5
DEVICE = torch.device("cpu")

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)

np.random.seed(RS)
torch.manual_seed(RS)


# ─── data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")
feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]

y = df[TARGET].values.astype(int)
# bucketise fractional intent (1.0-5.0 in 0.25 steps) → integer 0..4 for CORN
y_ord_all = (df[INTENT].round().clip(1, 5).astype(int) - 1).values
X_raw = df[feat_cols].values.astype(np.float32)
X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
y_ord_tr = y_ord_all[train_idx]

print(f"clean {df.shape}  train {len(train_idx)}  test {len(test_idx)}  "
      f"features: {len(feat_cols)} (no intent)")


# ─── helpers ────────────────────────────────────────────────────────────
def gce_objective(q=0.5, pos_weight=1.0):
    def _obj(y_pred, dtrain):
        yv = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)
        p_y = np.where(yv > 0.5, p, 1.0 - p)
        y_sign = 2.0 * yv - 1.0
        grad = -y_sign * (p_y ** q) * (1.0 - p_y)
        hess = (p_y ** q) * (1.0 - p_y) * (p_y - q * (1.0 - p_y))
        hess = np.maximum(hess, 1e-6)
        w = np.where(yv > 0.5, pos_weight, 1.0)
        return grad * w, hess * w
    return _obj


def gce_loss_sample(p, y, q):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    p_y = np.where(y > 0.5, p, 1 - p)
    return (1 - p_y ** q) / q


def fit_nrboost(X_tr, y_tr, *, q=0.7, n_rounds=400, n_stages=2, drop_frac=0.10,
                damp=0.3, lr=0.05, max_depth=5, subsample=0.9, colsample=0.9,
                pos_weight=None, seed=RS):
    if pos_weight is None:
        pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    N = len(y_tr)
    weights = np.ones(N, dtype=np.float64)
    params = dict(max_depth=max_depth, eta=lr, subsample=subsample,
                  colsample_bytree=colsample, min_child_weight=3,
                  reg_lambda=1.0, tree_method="hist", seed=seed, verbosity=0)
    booster = None
    rounds_per_stage = n_rounds // max(n_stages, 1)
    for stage in range(n_stages):
        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=weights)
        booster = xgb.train(params, dtr, num_boost_round=rounds_per_stage,
                            obj=gce_objective(q=q, pos_weight=pos_weight),
                            xgb_model=booster, verbose_eval=False)
        if stage == n_stages - 1:
            break
        p = 1.0 / (1.0 + np.exp(-booster.predict(dtr, output_margin=True)))
        losses = gce_loss_sample(p, y_tr, q)
        thresh = np.percentile(losses, 100 * (1 - drop_frac))
        weights = np.where(losses > thresh, weights * damp, weights)
    return booster


def predict_nrboost(booster, X):
    d = xgb.DMatrix(X)
    return 1.0 / (1.0 + np.exp(-booster.predict(d, output_margin=True)))


def fit_mtmlp(Xs_tr, y_bin_tr, y_ord_tr, *, in_dim, epochs=200, bs=128,
              lam=0.7, lr=1e-3, wd=1e-4, dropout=0.2, pos_weight=None, seed=RS,
              patience=25, val_frac=0.15):
    """Train an MT-MLP with early stopping on an inner val split."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    N = len(y_bin_tr)
    perm = rng.permutation(N)
    n_val = int(N * val_frac)
    va_idx = perm[:n_val]; tr_idx = perm[n_val:]

    xt = torch.from_numpy(Xs_tr).float()
    yb = torch.from_numpy(y_bin_tr.astype(np.float32))
    yo = torch.from_numpy(y_ord_tr.astype(np.int64))
    Xt_tr, Xt_va = xt[tr_idx], xt[va_idx]
    yb_tr, yb_va = yb[tr_idx], yb[va_idx]
    yo_tr, yo_va = yo[tr_idx], yo[va_idx]

    model = MTMlp(in_dim=in_dim, dropout=dropout).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    best_val = float("inf"); best_state = None; bad = 0
    n_tr = len(Xt_tr)
    for ep in range(epochs):
        model.train()
        order = torch.randperm(n_tr)
        for i in range(0, n_tr, bs):
            b = order[i:i + bs]
            logit_bin, logits_ord = model(Xt_tr[b])
            loss = joint_loss(logit_bin, logits_ord, yb_tr[b], yo_tr[b],
                              lam=lam, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lbv, lov = model(Xt_va)
            vl = joint_loss(lbv, lov, yb_va, yo_va, lam=lam,
                            pos_weight=pos_weight).item()
        if vl < best_val - 1e-4:
            best_val = vl; best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience: break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_mtmlp(model, Xs):
    model.eval()
    with torch.no_grad():
        logit_bin, _ = model(torch.from_numpy(Xs).float())
        return torch.sigmoid(logit_bin).cpu().numpy()


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
        if len(np.unique(y_true[idx])) < 2: continue
        try: vals.append(metric(y_true[idx], p[idx]))
        except Exception: pass
    a = np.asarray(vals)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def metrics_row(name, y_true, p, thr):
    pred = (p >= thr).astype(int)
    lo, hi = bootstrap_ci(y_true, p, roc_auc_score)
    plo, phi = bootstrap_ci(y_true, p, average_precision_score)
    return {"model": name, "AUC": roc_auc_score(y_true, p),
            "AUC_lo": lo, "AUC_hi": hi,
            "PR_AUC": average_precision_score(y_true, p),
            "PR_lo": plo, "PR_hi": phi,
            "F1": f1_score(y_true, pred, zero_division=0),
            "Bal_Acc": balanced_accuracy_score(y_true, pred),
            "Brier": brier_score_loss(y_true, p),
            "MCC": matthews_corrcoef(y_true, pred), "thr": thr}


# ─── OOF matrix ─────────────────────────────────────────────────────────
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
fold_idx = list(skf.split(X_tr, y_tr))
N_tr = len(y_tr); in_dim = X_tr.shape[1]
spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

BASES = ["RF", "NRBoost", "MTMLP", "SVM"]
oof = {k: np.zeros(N_tr) for k in BASES}
test_preds = {k: np.zeros(len(y_te)) for k in BASES}

print("\n── building OOF matrix on shared 5-fold split ──")
for f, (tr, va) in enumerate(fold_idx):
    print(f"  fold {f}: ", end="", flush=True)
    Xtr_f, Xva_f = X_tr[tr], X_tr[va]
    ytr_f = y_tr[tr]
    yo_tr_f = y_ord_tr[tr]

    # RF
    rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RS).fit(Xtr_f, ytr_f)
    oof["RF"][va] = rf.predict_proba(Xva_f)[:, 1]
    print("RF ", end="", flush=True)

    # NR-Boost q=0.7
    b = fit_nrboost(Xtr_f, ytr_f, q=0.7, n_rounds=400, n_stages=2,
                    drop_frac=0.10, seed=RS + f)
    oof["NRBoost"][va] = predict_nrboost(b, Xva_f)
    print("NRB ", end="", flush=True)

    # MT-MLP — standardise inputs within fold
    scaler = StandardScaler().fit(Xtr_f)
    Xtr_s = scaler.transform(Xtr_f).astype(np.float32)
    Xva_s = scaler.transform(Xva_f).astype(np.float32)
    mt = fit_mtmlp(Xtr_s, ytr_f, yo_tr_f, in_dim=in_dim, lam=0.7,
                    pos_weight=spw, seed=RS + f)
    oof["MTMLP"][va] = predict_mtmlp(mt, Xva_s)
    print("MT ", end="", flush=True)

    # SVM-RBF — probability=True is slow but we need probs
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              probability=True, random_state=RS).fit(Xtr_s, ytr_f)
    oof["SVM"][va] = svm.predict_proba(Xva_s)[:, 1]
    print("SVM")

# full-train base models → test preds
print("\n── full-train base models → test predictions ──")
rf_full = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=RS).fit(X_tr, y_tr)
test_preds["RF"] = rf_full.predict_proba(X_te)[:, 1]

b_full = fit_nrboost(X_tr, y_tr, q=0.7, n_rounds=400, n_stages=2, drop_frac=0.10, seed=RS)
test_preds["NRBoost"] = predict_nrboost(b_full, X_te)

scaler_full = StandardScaler().fit(X_tr)
Xtr_s_full = scaler_full.transform(X_tr).astype(np.float32)
Xte_s_full = scaler_full.transform(X_te).astype(np.float32)
mt_full = fit_mtmlp(Xtr_s_full, y_tr, y_ord_tr, in_dim=in_dim, lam=0.7,
                     pos_weight=spw, seed=RS)
test_preds["MTMLP"] = predict_mtmlp(mt_full, Xte_s_full)

svm_full = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                probability=True, random_state=RS).fit(Xtr_s_full, y_tr)
test_preds["SVM"] = svm_full.predict_proba(Xte_s_full)[:, 1]

for k in BASES:
    print(f"  {k:8s}  CV AUC={roc_auc_score(y_tr, oof[k]):.4f}  "
          f"test AUC={roc_auc_score(y_te, test_preds[k]):.4f}")


# ─── OOF correlation diagnostic ─────────────────────────────────────────
print("\n── base OOF correlation matrix (Pearson) ──")
oof_mat = np.column_stack([oof[k] for k in BASES])
corr = np.corrcoef(oof_mat, rowvar=False)
print("       " + "  ".join(f"{k:8s}" for k in BASES))
for i, k in enumerate(BASES):
    print(f"{k:6s} " + "  ".join(f"{corr[i, j]:8.3f}" for j in range(len(BASES))))
corr_df = pd.DataFrame(corr, index=BASES, columns=BASES)
corr_df.to_csv(OUT_TABLES / "table9b_stack_v2_corr.csv")


# ─── Meta 1: L2 logistic on logit(OOF) ──────────────────────────────────
def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))

Z_tr = _logit(oof_mat)
Z_te = _logit(np.column_stack([test_preds[k] for k in BASES]))

meta_lr = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                              class_weight="balanced", max_iter=2000,
                              random_state=RS).fit(Z_tr, y_tr)
oof_meta_lr = np.zeros(N_tr)
for tr, va in fold_idx:
    m = LogisticRegression(C=1.0, penalty="l2", solver="lbfgs",
                            class_weight="balanced", max_iter=2000,
                            random_state=RS).fit(Z_tr[tr], y_tr[tr])
    oof_meta_lr[va] = m.predict_proba(Z_tr[va])[:, 1]
test_meta_lr = meta_lr.predict_proba(Z_te)[:, 1]
print(f"\n  stacked-LR    CV AUC={roc_auc_score(y_tr, oof_meta_lr):.4f}  "
      f"test AUC={roc_auc_score(y_te, test_meta_lr):.4f}  "
      f"weights={dict(zip(BASES, np.round(meta_lr.coef_.ravel(), 3)))}")


# ─── Meta 2: convex combo ───────────────────────────────────────────────
def _neg_logloss(w, P, y):
    w = np.clip(w, 0, None)
    if w.sum() < 1e-9: return 1e9
    w = w / w.sum()
    p = (P * w).sum(axis=1)
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return log_loss(y, p)

P_tr = oof_mat
P_te = np.column_stack([test_preds[k] for k in BASES])
w0 = np.ones(len(BASES)) / len(BASES)
res = minimize(_neg_logloss, w0, args=(P_tr, y_tr), method="SLSQP",
                bounds=[(0, 1)] * len(BASES),
                constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
w_cvx = res.x / res.x.sum()
oof_cvx_full = (P_tr * w_cvx).sum(axis=1)
test_cvx = (P_te * w_cvx).sum(axis=1)
print(f"  convex-combo  weights={dict(zip(BASES, np.round(w_cvx, 3)))}")
print(f"  convex-combo  CV AUC={roc_auc_score(y_tr, oof_cvx_full):.4f}  "
      f"test AUC={roc_auc_score(y_te, test_cvx):.4f}")

oof_meta_cvx = np.zeros(N_tr); fold_weights = []
for tr, va in fold_idx:
    res_f = minimize(_neg_logloss, w0, args=(P_tr[tr], y_tr[tr]), method="SLSQP",
                     bounds=[(0, 1)] * len(BASES),
                     constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    wf = res_f.x / res_f.x.sum()
    fold_weights.append(wf)
    oof_meta_cvx[va] = (P_tr[va] * wf).sum(axis=1)
print(f"  convex-combo  nested-CV AUC={roc_auc_score(y_tr, oof_meta_cvx):.4f}")


# ─── mean baseline ──────────────────────────────────────────────────────
oof_mean = P_tr.mean(axis=1)
test_mean = P_te.mean(axis=1)

oof_probs = {**oof, "Stack_LR": oof_meta_lr, "Stack_CVX": oof_meta_cvx, "MeanAvg": oof_mean}
test_probs = {**test_preds, "Stack_LR": test_meta_lr, "Stack_CVX": test_cvx, "MeanAvg": test_mean}

name_map = {"RF": "RandomForest", "NRBoost": "NR-Boost (q=0.7)",
            "MTMLP": "MT-MLP (intent-aux)", "SVM": "SVM-RBF",
            "Stack_LR": "Stacking — L2 logistic",
            "Stack_CVX": "Stacking — convex combo",
            "MeanAvg": "Mean-avg of 4 bases"}

# ─── calibrate + score ──────────────────────────────────────────────────
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
panel_df.to_csv(OUT_TABLES / "table9b_stack_v2_panel.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table9b_stack_v2_panel.csv'}")
print(panel_df.to_string(index=False,
      formatters={c: "{:.3f}".format for c in panel_df.select_dtypes("float").columns}))

w_df = pd.DataFrame({
    "base": BASES,
    "stack_LR_coef": meta_lr.coef_.ravel(),
    "convex_weight": w_cvx,
    "convex_weight_fold_mean": np.mean(fold_weights, axis=0),
    "convex_weight_fold_std":  np.std(fold_weights, axis=0),
})
w_df.to_csv(OUT_TABLES / "table9b_stack_v2_weights.csv", index=False)
print(f"wrote {OUT_TABLES / 'table9b_stack_v2_weights.csv'}")
print(w_df.to_string(index=False,
      formatters={c: "{:.4f}".format for c in w_df.select_dtypes("float").columns}))

delong_rows = []
for champ in ["Stack_LR", "Stack_CVX", "MeanAvg"]:
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
dlg.to_csv(OUT_TABLES / "table9b_stack_v2_delong.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table9b_stack_v2_delong.csv'}")
print(dlg.to_string(index=False,
      formatters={c: "{:.4f}".format for c in dlg.select_dtypes("float").columns}))

# ─── figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
order = ["SVM", "MTMLP", "NRBoost", "RF", "MeanAvg", "Stack_CVX", "Stack_LR"]
colors = {"Stack_LR": "crimson", "Stack_CVX": "darkorange", "RF": "steelblue"}
for k in order:
    p = cal[k]
    fpr, tpr, _ = roc_curve(y_te, p)
    lw = 2.5 if k in ("Stack_LR", "Stack_CVX") else 1.2
    alpha = 1.0 if k in ("Stack_LR", "Stack_CVX", "RF") else 0.55
    c = colors.get(k, None)
    axes[0].plot(fpr, tpr, label=f"{name_map[k]}  {roc_auc_score(y_te, p):.3f}",
                  linewidth=lw, alpha=alpha, color=c)
axes[0].plot([0, 1], [0, 1], "--", c="grey", alpha=0.5)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test, calibrated)")
axes[0].legend(fontsize=8, loc="lower right")
for k in order:
    p = cal[k]
    pr, rc, _ = precision_recall_curve(y_te, p)
    lw = 2.5 if k in ("Stack_LR", "Stack_CVX") else 1.2
    alpha = 1.0 if k in ("Stack_LR", "Stack_CVX", "RF") else 0.55
    c = colors.get(k, None)
    axes[1].plot(rc, pr, label=f"{name_map[k]}  {average_precision_score(y_te, p):.3f}",
                  linewidth=lw, alpha=alpha, color=c)
axes[1].axhline(y_te.mean(), ls="--", c="grey", alpha=0.5)
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR (test, calibrated)")
axes[1].legend(fontsize=8, loc="upper right")
plt.tight_layout()
fig_path = OUT_FIGS / "fig9b_stack_v2_roc.png"
plt.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"wrote {fig_path}")
