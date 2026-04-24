"""Phase 6c — Stacking v4: FE + asymmetric label-noise weights + ablations.

Mirrors 05_stacking.py but adds two orthogonal levers on top of the same
5-base architecture (RF, NR-Boost, MT-MLP seed-avg, SVM-RBF, ExtraTrees):

  (FE) 41 engineered features from 06a_features.py, replacing the 12 raw cols
  (CL) per-sample weights from 06b_cleanlab.py (flagged 0-labels w=0.3)

Four ablations are trained on the SAME folds, SAME seeds, SAME test set:
  A. base-v3    — Phase 5 stack (re-run here; used as DeLong anchor)
  B. +FE only
  C. +CL only
  D. +FE +CL   — Phase 6 full

Sample-weight plumbing:
  - RF, ExtraTrees, SVM-RBF: native `sample_weight` in `.fit()`
  - NR-Boost (xgb): initial `weights` array multiplied by CL weights, then
    the self-paced routine re-weights on top
  - MT-MLP: no per-sample-weight support (would require editing 01a_mt_model.py);
    per plan fallback we train MT-MLP on the same X but WITHOUT CL weights.
    Ablations C and D therefore pass weights to 4 of 5 bases only — this is
    documented in the panel CSV note column.

Outputs
-------
src/tables/table10_stack_v4_panel.csv       rows for all 4 ablations × meta-learners
src/tables/table10_stack_v4_ablation.csv    compact 4-row summary (meta = L2-LR C=10)
src/tables/table10_stack_v4_delong.csv      Phase 6 full vs Phase 5 stack + RF + +FE + +CL
src/tables/table10_stack_v4_weights.csv     meta weights for D (Phase 6 full)
src/figures/fig10_stack_v4_roc.png          ROC + PR, 4 ablations overlaid
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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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
_feats = _load("features_v6", "06a_features.py")
build_features = _feats.build_features

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RS = 42
TARGET = "离职行为"
INTENT = "离职意向"
N_FOLDS = 5
N_MT_SEEDS = 5
DEVICE = torch.device("cpu")

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)

np.random.seed(RS)
torch.manual_seed(RS)


# ─── helpers copied from 05_stacking.py (same semantics, verbatim) ──────
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
                pos_weight=None, seed=RS, init_weight=None):
    if pos_weight is None:
        pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    N = len(y_tr)
    weights = np.ones(N, dtype=np.float64) if init_weight is None else init_weight.astype(np.float64).copy()
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
        if stage == n_stages - 1: break
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
    if best_state is not None: model.load_state_dict(best_state)
    return model


def predict_mtmlp(model, Xs):
    model.eval()
    with torch.no_grad():
        logit_bin, _ = model(torch.from_numpy(Xs).float())
        return torch.sigmoid(logit_bin).cpu().numpy()


def fit_predict_mtmlp_seedavg(Xs_tr, y_bin_tr, y_ord_tr, Xs_pred, *, in_dim,
                               pos_weight, base_seed, n_seeds=N_MT_SEEDS):
    ps = np.zeros(len(Xs_pred))
    for s in range(n_seeds):
        m = fit_mtmlp(Xs_tr, y_bin_tr, y_ord_tr, in_dim=in_dim, lam=0.7,
                       pos_weight=pos_weight, seed=base_seed * 100 + s)
        ps += predict_mtmlp(m, Xs_pred)
    return ps / n_seeds


def best_f1_threshold(y_true, p):
    best, thr = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_true, (p >= t).astype(int), pos_label=1, zero_division=0)
        if f > best: best, thr = f, t
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


# ─── data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")
feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]

y = df[TARGET].values.astype(int)
y_ord_all = (df[INTENT].round().clip(1, 5).astype(int) - 1).values
X_raw = df[feat_cols].values.astype(np.float32)
X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
y_ord_tr = y_ord_all[train_idx]

# cleanlab weights (pre-computed by 06b)
SW_PATH = Path("data/processed/sample_weights_v6.npy")
sw_v6 = np.load(SW_PATH) if SW_PATH.exists() else np.ones(len(y_tr), dtype=np.float32)
print(f"clean {df.shape}  train {len(train_idx)}  test {len(test_idx)}")
print(f"raw features: {len(feat_cols)}  MT seeds: {N_MT_SEEDS}")
print(f"CL weights loaded from {SW_PATH}  flagged={(sw_v6 < 1.0).sum()}  w_flag={sw_v6[sw_v6<1.0].min() if (sw_v6<1.0).any() else 'n/a'}")

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
fold_idx = list(skf.split(X_tr_raw, y_tr))

# engineered features (depends on fold_idx for leak-safe target encoding)
X_tr_fe, X_te_fe, fe_names = build_features(df, feat_cols, train_idx, test_idx, fold_idx)
print(f"FE matrix: {X_tr_fe.shape[1]} features (from {len(feat_cols)})")

BASES = ["RF", "NRBoost", "MTMLP", "SVM", "ET"]
spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))


def run_stack(X_tr, X_te, *, sample_weight=None, tag=""):
    """Train 5-base stack on given features; return dict of OOF + test probs + meta.

    `sample_weight` is applied to RF, NRBoost init, SVM, ET. MT-MLP is trained
    without per-sample weights (see module docstring).
    """
    in_dim = X_tr.shape[1]
    use_w = sample_weight is not None
    w = sample_weight if use_w else None

    oof = {k: np.zeros(len(y_tr)) for k in BASES}
    test_preds = {k: np.zeros(len(y_te)) for k in BASES}

    print(f"\n── [{tag}] OOF matrix (in_dim={in_dim}, sample_weight={'on' if use_w else 'off'}) ──")
    for f, (tr, va) in enumerate(fold_idx):
        print(f"  fold {f}: ", end="", flush=True)
        Xtr_f, Xva_f = X_tr[tr], X_tr[va]
        ytr_f = y_tr[tr]; yo_tr_f = y_ord_tr[tr]
        w_f = None if w is None else w[tr]

        rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                     class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=RS)
        rf.fit(Xtr_f, ytr_f, sample_weight=w_f)
        oof["RF"][va] = rf.predict_proba(Xva_f)[:, 1]
        print("RF ", end="", flush=True)

        b = fit_nrboost(Xtr_f, ytr_f, q=0.7, n_rounds=400, n_stages=2,
                        drop_frac=0.10, seed=RS + f, init_weight=w_f)
        oof["NRBoost"][va] = predict_nrboost(b, Xva_f)
        print("NRB ", end="", flush=True)

        scaler = StandardScaler().fit(Xtr_f)
        Xtr_s = scaler.transform(Xtr_f).astype(np.float32)
        Xva_s = scaler.transform(Xva_f).astype(np.float32)
        oof["MTMLP"][va] = fit_predict_mtmlp_seedavg(
            Xtr_s, ytr_f, yo_tr_f, Xva_s, in_dim=in_dim,
            pos_weight=spw, base_seed=RS + f)
        print(f"MT×{N_MT_SEEDS} ", end="", flush=True)

        svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                   probability=True, random_state=RS)
        svm.fit(Xtr_s, ytr_f, sample_weight=w_f)
        oof["SVM"][va] = svm.predict_proba(Xva_s)[:, 1]
        print("SVM ", end="", flush=True)

        et = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=RS)
        et.fit(Xtr_f, ytr_f, sample_weight=w_f)
        oof["ET"][va] = et.predict_proba(Xva_f)[:, 1]
        print("ET")

    print(f"── [{tag}] full-train → test ──")
    rf_full = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                      class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=RS)
    rf_full.fit(X_tr, y_tr, sample_weight=w)
    test_preds["RF"] = rf_full.predict_proba(X_te)[:, 1]

    b_full = fit_nrboost(X_tr, y_tr, q=0.7, n_rounds=400, n_stages=2,
                         drop_frac=0.10, seed=RS, init_weight=w)
    test_preds["NRBoost"] = predict_nrboost(b_full, X_te)

    scaler_full = StandardScaler().fit(X_tr)
    Xtr_s_full = scaler_full.transform(X_tr).astype(np.float32)
    Xte_s_full = scaler_full.transform(X_te).astype(np.float32)
    test_preds["MTMLP"] = fit_predict_mtmlp_seedavg(
        Xtr_s_full, y_tr, y_ord_tr, Xte_s_full, in_dim=in_dim,
        pos_weight=spw, base_seed=RS)

    svm_full = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                    probability=True, random_state=RS)
    svm_full.fit(Xtr_s_full, y_tr, sample_weight=w)
    test_preds["SVM"] = svm_full.predict_proba(Xte_s_full)[:, 1]

    et_full = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                    class_weight="balanced_subsample",
                                    n_jobs=-1, random_state=RS)
    et_full.fit(X_tr, y_tr, sample_weight=w)
    test_preds["ET"] = et_full.predict_proba(X_te)[:, 1]

    for k in BASES:
        print(f"  {k:8s}  CV AUC={roc_auc_score(y_tr, oof[k]):.4f}  "
              f"test AUC={roc_auc_score(y_te, test_preds[k]):.4f}")

    # meta-learners
    def _logit(p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    oof_mat = np.column_stack([oof[k] for k in BASES])
    Z_tr = _logit(oof_mat)
    Z_te = _logit(np.column_stack([test_preds[k] for k in BASES]))

    def fit_meta_lr(C, class_weight):
        m = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                class_weight=class_weight, max_iter=2000,
                                random_state=RS).fit(Z_tr, y_tr)
        oof_m = np.zeros(len(y_tr))
        for tr, va in fold_idx:
            mi = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                     class_weight=class_weight, max_iter=2000,
                                     random_state=RS).fit(Z_tr[tr], y_tr[tr])
            oof_m[va] = mi.predict_proba(Z_tr[va])[:, 1]
        return m, oof_m, m.predict_proba(Z_te)[:, 1]

    meta_lr_bal, oof_lr_bal, test_lr_bal = fit_meta_lr(C=1.0, class_weight="balanced")
    meta_lr_hC, oof_lr_hC, test_lr_hC = fit_meta_lr(C=10.0, class_weight=None)

    def _neg_logloss(w_vec, P, y):
        w_vec = np.clip(w_vec, 0, None)
        if w_vec.sum() < 1e-9: return 1e9
        w_vec = w_vec / w_vec.sum()
        p = (P * w_vec).sum(axis=1)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return log_loss(y, p)

    P_tr = oof_mat; P_te = np.column_stack([test_preds[k] for k in BASES])
    w0 = np.ones(len(BASES)) / len(BASES)
    res = minimize(_neg_logloss, w0, args=(P_tr, y_tr), method="SLSQP",
                    bounds=[(0, 1)] * len(BASES),
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w_cvx = res.x / res.x.sum()
    oof_cvx = (P_tr * w_cvx).sum(axis=1); test_cvx = (P_te * w_cvx).sum(axis=1)
    oof_mean = P_tr.mean(axis=1); test_mean = P_te.mean(axis=1)

    return {
        "oof": oof, "test_preds": test_preds,
        "oof_lr_bal": oof_lr_bal, "test_lr_bal": test_lr_bal,
        "oof_lr_hC": oof_lr_hC, "test_lr_hC": test_lr_hC,
        "oof_cvx": oof_cvx, "test_cvx": test_cvx,
        "oof_mean": oof_mean, "test_mean": test_mean,
        "meta_lr_bal": meta_lr_bal, "meta_lr_hC": meta_lr_hC,
        "w_cvx": w_cvx,
    }


# ─── run four ablations ────────────────────────────────────────────────
ablations = [
    ("A_base",  X_tr_raw, X_te_raw, None,   "base-v3 (rerun)"),
    ("B_FE",    X_tr_fe,  X_te_fe,  None,   "+FE"),
    ("C_CL",    X_tr_raw, X_te_raw, sw_v6,  "+CL"),
    ("D_FEcCL", X_tr_fe,  X_te_fe,  sw_v6,  "+FE +CL (Phase 6 full)"),
]

results = {}
for key, Xtr, Xte, w, label in ablations:
    print(f"\n{'=' * 72}\n  ABLATION {key} :: {label}\n{'=' * 72}")
    results[key] = run_stack(Xtr, Xte, sample_weight=w, tag=key)


# ─── calibrate + metrics per ablation + meta variant ──────────────────
PANEL_KEYS = [
    ("lr_bal", "Stacking L2 LR (bal)"),
    ("lr_hC",  "Stacking L2 LR (C=10)"),
    ("cvx",    "Stacking convex combo"),
    ("mean",   "Mean-avg 5 bases"),
]

panel_rows = []
cal_probs = {}   # (ablation_key, meta_key) -> calibrated test prob
for ab_key, _, _, _, ab_label in ablations:
    r = results[ab_key]
    for mk, mk_label in PANEL_KEYS:
        oof_k = r[f"oof_{mk}"]; test_k = r[f"test_{mk}"]
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof_k, y_tr)
        oof_cal = iso.predict(oof_k); test_cal = iso.predict(test_k)
        thr = best_f1_threshold(y_tr, oof_cal)
        row = metrics_row(f"[{ab_key}] {mk_label}", y_te, test_cal, thr)
        row["ablation"] = ab_label
        row["meta"] = mk_label
        cal_probs[(ab_key, mk)] = test_cal
        panel_rows.append(row)
    # also include the 5 raw bases for this ablation (calibrated)
    for b in BASES:
        iso = IsotonicRegression(out_of_bounds="clip").fit(r["oof"][b], y_tr)
        oof_cal = iso.predict(r["oof"][b]); test_cal = iso.predict(r["test_preds"][b])
        thr = best_f1_threshold(y_tr, oof_cal)
        row = metrics_row(f"[{ab_key}] {b}", y_te, test_cal, thr)
        row["ablation"] = ab_label
        row["meta"] = b
        cal_probs[(ab_key, f"base_{b}")] = test_cal
        panel_rows.append(row)

panel_df = pd.DataFrame(panel_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
panel_path = OUT_TABLES / "table10_stack_v4_panel.csv"
panel_df.to_csv(panel_path, index=False)
print(f"\nwrote {panel_path}")
print(panel_df[["model", "AUC", "AUC_lo", "AUC_hi", "PR_AUC", "F1", "Bal_Acc"]].head(20).to_string(index=False,
      formatters={c: "{:.3f}".format for c in panel_df.select_dtypes("float").columns}))


# ─── 4-row ablation summary (meta = L2-LR C=10) ───────────────────────
summary_rows = []
for ab_key, _, _, _, ab_label in ablations:
    base = panel_df[(panel_df.ablation == ab_label) & (panel_df.meta == "Stacking L2 LR (C=10)")].iloc[0]
    summary_rows.append({
        "ablation": ab_label,
        "AUC": base["AUC"], "AUC_lo": base["AUC_lo"], "AUC_hi": base["AUC_hi"],
        "PR_AUC": base["PR_AUC"], "F1": base["F1"], "Bal_Acc": base["Bal_Acc"],
        "Brier": base["Brier"], "MCC": base["MCC"],
    })
summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_TABLES / "table10_stack_v4_ablation.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nwrote {summary_path}")
print(summary_df.to_string(index=False,
      formatters={c: "{:.4f}".format for c in summary_df.select_dtypes("float").columns}))


# ─── DeLong: D vs {Phase 5 stack, base-v3 rerun, +FE, +CL, RF(D)} ─────
def _get(key): return cal_probs[key]
champ = _get(("D_FEcCL", "lr_hC"))
delong_rows = []
comparisons = [
    (("A_base",  "lr_hC"),    "base-v3 (rerun, L2-LR C=10)"),
    (("B_FE",    "lr_hC"),    "+FE only (L2-LR C=10)"),
    (("C_CL",    "lr_hC"),    "+CL only (L2-LR C=10)"),
    (("D_FEcCL", "base_RF"),  "RF in D ablation"),
    (("A_base",  "base_RF"),  "RF in base-v3 rerun"),
]
for key, label in comparisons:
    auc_a, auc_b, z, p = delong_test(y_te, champ, _get(key))
    delong_rows.append({
        "champion": "D_FEcCL L2-LR C=10",
        "baseline": label,
        "AUC_champ": auc_a, "AUC_base": auc_b, "ΔAUC": auc_a - auc_b,
        "z": z, "p_value": p, "significant_05": p < 0.05,
    })

# also compare D against persisted Phase 5 panel (stack_LR_C10)
phase5_panel = pd.read_csv(OUT_TABLES / "table9c_stack_v3_panel.csv")
phase5_auc = float(phase5_panel.query("model == 'Stacking — L2 LR (C=10)'")["AUC"].iloc[0])
delong_rows.append({
    "champion": "D_FEcCL L2-LR C=10",
    "baseline": "Phase 5 stack L2-LR C=10 (persisted)",
    "AUC_champ": roc_auc_score(y_te, champ),
    "AUC_base": phase5_auc,
    "ΔAUC": roc_auc_score(y_te, champ) - phase5_auc,
    "z": np.nan, "p_value": np.nan,  # cross-script DeLong needs both score vectors
    "significant_05": False,
})
dlg = pd.DataFrame(delong_rows)
dlg_path = OUT_TABLES / "table10_stack_v4_delong.csv"
dlg.to_csv(dlg_path, index=False)
print(f"\nwrote {dlg_path}")
print(dlg.to_string(index=False,
      formatters={c: "{:.4f}".format for c in dlg.select_dtypes("float").columns}))


# ─── meta weights for D ─────────────────────────────────────────────
rD = results["D_FEcCL"]
w_df = pd.DataFrame({
    "base": BASES,
    "LR_bal_coef": rD["meta_lr_bal"].coef_.ravel(),
    "LR_C10_coef": rD["meta_lr_hC"].coef_.ravel(),
    "convex_weight": rD["w_cvx"],
})
wp = OUT_TABLES / "table10_stack_v4_weights.csv"
w_df.to_csv(wp, index=False)
print(f"wrote {wp}")
print(w_df.to_string(index=False,
      formatters={c: "{:.4f}".format for c in w_df.select_dtypes("float").columns}))


# ─── figure: ROC + PR overlay (one meta per ablation) ──────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
colors = {"A_base": "steelblue", "B_FE": "seagreen",
          "C_CL": "darkorange", "D_FEcCL": "crimson"}
labels = {"A_base": "base-v3 rerun", "B_FE": "+FE",
          "C_CL": "+CL", "D_FEcCL": "+FE +CL (Phase 6)"}
for ab in ["A_base", "B_FE", "C_CL", "D_FEcCL"]:
    p = cal_probs[(ab, "lr_hC")]
    fpr, tpr, _ = roc_curve(y_te, p)
    axes[0].plot(fpr, tpr, label=f"{labels[ab]}  AUC={roc_auc_score(y_te, p):.4f}",
                  linewidth=2.3 if ab == "D_FEcCL" else 1.5,
                  color=colors[ab])
axes[0].plot([0, 1], [0, 1], "--", c="grey", alpha=0.5)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test, calibrated, meta=L2-LR C=10)")
axes[0].legend(fontsize=8, loc="lower right")

for ab in ["A_base", "B_FE", "C_CL", "D_FEcCL"]:
    p = cal_probs[(ab, "lr_hC")]
    pr, rc, _ = precision_recall_curve(y_te, p)
    axes[1].plot(rc, pr, label=f"{labels[ab]}  PR-AUC={average_precision_score(y_te, p):.4f}",
                  linewidth=2.3 if ab == "D_FEcCL" else 1.5,
                  color=colors[ab])
axes[1].axhline(y_te.mean(), ls="--", c="grey", alpha=0.5)
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR (test, calibrated)")
axes[1].legend(fontsize=8, loc="upper right")

plt.tight_layout()
fig_path = OUT_FIGS / "fig10_stack_v4_roc.png"
plt.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"wrote {fig_path}")


# ─── summary for user ───────────────────────────────────────────────
print("\n" + "=" * 72)
print("SUMMARY (meta = L2-LR C=10)")
print("=" * 72)
for _, row in summary_df.iterrows():
    print(f"  {row['ablation']:30s}  AUC={row['AUC']:.4f}  "
          f"[{row['AUC_lo']:.3f}, {row['AUC_hi']:.3f}]  "
          f"PR_AUC={row['PR_AUC']:.3f}  F1={row['F1']:.3f}")
