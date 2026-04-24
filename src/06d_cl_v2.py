"""Phase 6d — Continuous + bidirectional Cleanlab weighting; final benchmark panel.

Supersedes 06b's binary (flagged→0.3, else 1.0) weighting with:

  (a) **Continuous**: weight is a monotone function of the cleanlab
      quality_score, so rows with mild suspicion get mild downweighting.
      Keeps more information than a hard flag.

  (b) **Bidirectional**: asymmetric noise on both sides.
      Axis-A (missed leavers):  y=0 ∧ 离职意向 ≥ 3.5 ∧ low quality
      Axis-B (reluctant leavers): y=1 ∧ 离职意向 ≤ 2.5 ∧ low quality
      Phase-0 F12 already drops y=1∧intent≤2; Axis-B catches the 2<intent≤2.5
      grey band where F12 didn't fire but the model still disagrees.

Weight definition
-----------------
For a flagged row i:
    w_i = max(w_min, (quality_i / tau) ** alpha)
For an unflagged row: w_i = 1.
Global variants apply the same formula to ALL rows (no flag).

Pipeline
--------
1. RF 5-fold OOF probabilities → cleanlab quality scores.
2. Grid-search 6 weight schemes on RF OOF AUC (fast, ~7 min).
3. Pick winner by RF OOF AUC. Save weights to sample_weights_v7.npy.
4. Train full 5-base stack THREE times for DeLong comparison:
     - baseline (no weights)            ≡ Phase 5 stack
     - v6 binary weights                ≡ Phase 6 +CL
     - v7 continuous + bidirectional    ≡ ours
5. Pull 13 baseline rows from existing Phase 1–5 tables, add our 3 stacks,
   emit 16-row benchmark panel sorted by AUC.
6. DeLong pairs: v7 vs baseline, v7 vs v6, v6 vs baseline.

Outputs
-------
data/processed/sample_weights_v7.npy
src/tables/table11_cl_v2_search.csv      grid-search OOF AUC per scheme
src/tables/table11_cl_v2_panel.csv       16-row benchmark panel
src/tables/table11_cl_v2_delong.csv      DeLong on the 3 stack variants
src/figures/fig11_cl_v2_roc.png          ROC+PR overlay
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
from cleanlab.rank import get_label_quality_scores
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
SW_V7_PATH = Path("data/processed/sample_weights_v7.npy")
SW_V6_PATH = Path("data/processed/sample_weights_v6.npy")

np.random.seed(RS); torch.manual_seed(RS)


# ─── helpers (verbatim from 05_stacking.py / 06c_stack_v4.py) ──────────
def gce_objective(q=0.5, pos_weight=1.0):
    def _obj(y_pred, dtrain):
        yv = dtrain.get_label()
        p = 1.0 / (1.0 + np.exp(-y_pred)); p = np.clip(p, 1e-7, 1 - 1e-7)
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
                  colsample_bytree=colsample, min_child_weight=3, reg_lambda=1.0,
                  tree_method="hist", seed=seed, verbosity=0)
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
    perm = rng.permutation(N); n_val = int(N * val_frac)
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
            lb, lo = model(Xt_tr[b])
            loss = joint_loss(lb, lo, yb_tr[b], yo_tr[b], lam=lam, pos_weight=pos_weight)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lbv, lov = model(Xt_va)
            vl = joint_loss(lbv, lov, yb_va, yo_va, lam=lam, pos_weight=pos_weight).item()
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


# ─── data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")
feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]
y = df[TARGET].values.astype(int)
y_ord_all = (df[INTENT].round().clip(1, 5).astype(int) - 1).values
X_raw = df[feat_cols].values.astype(np.float32)
X_tr = X_raw[train_idx]; X_te = X_raw[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
y_ord_tr = y_ord_all[train_idx]
intent_tr = df.iloc[train_idx][INTENT].values.astype(float)

spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
fold_idx = list(skf.split(X_tr, y_tr))
BASES = ["RF", "NRBoost", "MTMLP", "SVM", "ET"]
print(f"clean {df.shape}  train {len(train_idx)}  test {len(test_idx)}  n_feat {X_tr.shape[1]}")


# ─── Step 1: RF OOF → quality scores ───────────────────────────────────
def rf_oof(sw=None):
    oof = np.zeros(len(y_tr))
    for tr, va in fold_idx:
        rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                     class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=RS)
        rf.fit(X_tr[tr], y_tr[tr], sample_weight=None if sw is None else sw[tr])
        oof[va] = rf.predict_proba(X_tr[va])[:, 1]
    return oof


print("\n── Step 1: RF OOF for quality scores ──")
oof_base = rf_oof(None)
auc_rf_base = roc_auc_score(y_tr, oof_base)
print(f"  baseline RF OOF AUC = {auc_rf_base:.4f}")

pred_probs = np.column_stack([1 - oof_base, oof_base])
quality = get_label_quality_scores(labels=y_tr, pred_probs=pred_probs,
                                    method="self_confidence")
print(f"  quality quantiles: 5%={np.percentile(quality, 5):.3f}  "
      f"25%={np.percentile(quality, 25):.3f}  50%={np.percentile(quality, 50):.3f}  "
      f"75%={np.percentile(quality, 75):.3f}  95%={np.percentile(quality, 95):.3f}")


# ─── Step 2: weight-scheme grid search on RF OOF ────────────────────
def make_weights(tau_A: float, tau_B: float, alpha: float, w_min: float,
                  global_cont: bool = False) -> tuple[np.ndarray, int, int]:
    """Continuous + bidirectional CL weights.

    tau_A: y=0 ∧ intent≥3.5 flagged when quality<tau_A (set 0 to disable axis-A)
    tau_B: y=1 ∧ intent≤2.5 flagged when quality<tau_B (set 0 to disable axis-B)
    alpha, w_min: shape parameters of continuous downweight.
    global_cont: if True, ignore flags and apply (quality^alpha) shrinkage to
                 ALL rows (clipped to [w_min, 1]).
    """
    n = len(y_tr)
    w = np.ones(n, dtype=np.float32)
    if global_cont:
        cont = np.clip(quality ** alpha, w_min, 1.0).astype(np.float32)
        return cont, int((cont < 1.0).sum()), 0
    flagA = (y_tr == 0) & (intent_tr >= 3.5) & (quality < tau_A) if tau_A > 0 else np.zeros(n, dtype=bool)
    flagB = (y_tr == 1) & (intent_tr <= 2.5) & (quality < tau_B) if tau_B > 0 else np.zeros(n, dtype=bool)
    flagged = flagA | flagB
    # per-row tau: use axis A's tau for y=0 rows, B's for y=1
    tau_eff = np.where(y_tr == 0, max(tau_A, 1e-6), max(tau_B, 1e-6))
    cont = (quality / tau_eff) ** alpha
    cont = np.maximum(cont, w_min).astype(np.float32)
    w[flagged] = cont[flagged]
    return w, int(flagA.sum()), int(flagB.sum())


grid = [
    # (tau_A, tau_B, alpha, w_min, global_cont, label)
    (0.5, 0.0, 1.0, 0.3, False, "axisA_linear"),
    (0.5, 0.5, 1.0, 0.3, False, "bidir_linear"),
    (0.5, 0.5, 2.0, 0.1, False, "bidir_steep"),
    (0.7, 0.7, 1.0, 0.3, False, "bidir_wide"),
    (0.7, 0.7, 2.0, 0.1, False, "bidir_wide_steep"),
    (0.0, 0.0, 0.5, 0.3, True,  "global_sqrt"),
    (0.0, 0.0, 1.0, 0.3, True,  "global_linear"),
]

print("\n── Step 2: weight-scheme grid search (RF OOF AUC) ──")
search_rows = [{"scheme": "baseline", "tau_A": 0, "tau_B": 0, "alpha": 0,
                 "w_min": 1, "global": False, "n_flag_A": 0, "n_flag_B": 0,
                 "rf_oof_auc": auc_rf_base, "delta": 0.0}]
best = {"label": "baseline", "auc": auc_rf_base, "w": None}
for tau_A, tau_B, alpha, w_min, glob, label in grid:
    w, nA, nB = make_weights(tau_A, tau_B, alpha, w_min, glob)
    oof = rf_oof(w)
    auc = roc_auc_score(y_tr, oof)
    delta = auc - auc_rf_base
    search_rows.append({"scheme": label, "tau_A": tau_A, "tau_B": tau_B,
                         "alpha": alpha, "w_min": w_min, "global": glob,
                         "n_flag_A": nA, "n_flag_B": nB,
                         "rf_oof_auc": auc, "delta": delta})
    flag = "**" if auc > best["auc"] else "  "
    print(f"  {flag}{label:22s}  flagA={nA:4d}  flagB={nB:4d}  "
          f"RF OOF AUC={auc:.4f}  Δ={delta:+.4f}")
    if auc > best["auc"]:
        best = {"label": label, "auc": auc, "w": w.copy()}

search_df = pd.DataFrame(search_rows)
search_path = OUT_TABLES / "table11_cl_v2_search.csv"
search_df.to_csv(search_path, index=False)
print(f"\nwrote {search_path}")
print(f"winner: {best['label']}  RF OOF AUC={best['auc']:.4f}  "
      f"Δ vs baseline={best['auc'] - auc_rf_base:+.4f}")

np.save(SW_V7_PATH, best["w"] if best["w"] is not None else np.ones(len(y_tr), dtype=np.float32))
print(f"wrote {SW_V7_PATH}")


# ─── Step 3: full 5-base stack × 3 weight configs ──────────────────
def run_stack(sample_weight, tag):
    in_dim = X_tr.shape[1]
    use_w = sample_weight is not None
    w = sample_weight if use_w else None
    oof = {k: np.zeros(len(y_tr)) for k in BASES}
    test_preds = {k: np.zeros(len(y_te)) for k in BASES}

    print(f"\n── [{tag}] OOF matrix (weights={'on' if use_w else 'off'}) ──")
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

    # full-train test predictions
    rf_f = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=RS)
    rf_f.fit(X_tr, y_tr, sample_weight=w)
    test_preds["RF"] = rf_f.predict_proba(X_te)[:, 1]
    b_f = fit_nrboost(X_tr, y_tr, q=0.7, n_rounds=400, n_stages=2,
                      drop_frac=0.10, seed=RS, init_weight=w)
    test_preds["NRBoost"] = predict_nrboost(b_f, X_te)
    scaler_f = StandardScaler().fit(X_tr)
    Xtr_sf = scaler_f.transform(X_tr).astype(np.float32)
    Xte_sf = scaler_f.transform(X_te).astype(np.float32)
    test_preds["MTMLP"] = fit_predict_mtmlp_seedavg(
        Xtr_sf, y_tr, y_ord_tr, Xte_sf, in_dim=in_dim,
        pos_weight=spw, base_seed=RS)
    svm_f = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                 probability=True, random_state=RS)
    svm_f.fit(Xtr_sf, y_tr, sample_weight=w)
    test_preds["SVM"] = svm_f.predict_proba(Xte_sf)[:, 1]
    et_f = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RS)
    et_f.fit(X_tr, y_tr, sample_weight=w)
    test_preds["ET"] = et_f.predict_proba(X_te)[:, 1]
    for k in BASES:
        print(f"  {k:8s}  CV AUC={roc_auc_score(y_tr, oof[k]):.4f}  "
              f"test AUC={roc_auc_score(y_te, test_preds[k]):.4f}")

    # L2-LR meta (C=10) — strongest in Phase 5 DeLong
    def _logit(p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps); return np.log(p / (1 - p))
    oof_mat = np.column_stack([oof[k] for k in BASES])
    Z_tr = _logit(oof_mat)
    Z_te = _logit(np.column_stack([test_preds[k] for k in BASES]))
    meta = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs",
                               max_iter=2000, random_state=RS).fit(Z_tr, y_tr)
    oof_meta = np.zeros(len(y_tr))
    for tr, va in fold_idx:
        mi = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs",
                                 max_iter=2000, random_state=RS).fit(Z_tr[tr], y_tr[tr])
        oof_meta[va] = mi.predict_proba(Z_tr[va])[:, 1]
    test_meta = meta.predict_proba(Z_te)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_meta, y_tr)
    oof_cal = iso.predict(oof_meta); test_cal = iso.predict(test_meta)
    thr = best_f1_threshold(y_tr, oof_cal)
    return {"oof": oof, "test_preds": test_preds, "meta_coef": meta.coef_.ravel(),
             "oof_cal": oof_cal, "test_cal": test_cal, "thr": thr}


print("\n" + "=" * 72)
print("  STEP 3: full 5-base stack × 3 weight configs")
print("=" * 72)
sw_v6 = np.load(SW_V6_PATH) if SW_V6_PATH.exists() else None
sw_v7 = best["w"]

stacks = {
    "baseline":    run_stack(None, "baseline (no weights)"),
    "v6_binary":   run_stack(sw_v6, "v6 binary CL"),
    "v7_continuous": run_stack(sw_v7, f"v7 continuous ({best['label']})"),
}


# ─── Step 4: benchmark panel ──────────────────────────────────────────
def _pick(df_panel, model_name, new_name=None):
    row = df_panel[df_panel["model"] == model_name]
    if row.empty: return None
    r = row.iloc[0].to_dict()
    if new_name is not None: r["model"] = new_name
    return r


print("\n── Step 4: assembling benchmark panel from existing tables ──")
panel_rows = []
t4 = pd.read_csv(OUT_TABLES / "table4_base_classifier.csv")
t7 = pd.read_csv(OUT_TABLES / "table7_icmnet_panel.csv")
t8 = pd.read_csv(OUT_TABLES / "table8_nrboost_panel.csv")
t9 = pd.read_csv(OUT_TABLES / "table9_nrforest_panel.csv")
t9c = pd.read_csv(OUT_TABLES / "table9c_stack_v3_panel.csv")

# classic single models
panel_rows.append(_pick(t7, "LogReg"))
panel_rows.append(_pick(t4, "MLP (binary only, no intent)", "MLP (binary)"))
panel_rows.append(_pick(t9c, "SVM-RBF"))
# trees
panel_rows.append(_pick(t9c, "RandomForest"))
panel_rows.append(_pick(t9c, "ExtraTrees"))
panel_rows.append(_pick(t4, "XGBoost"))
panel_rows.append(_pick(t4, "LightGBM"))
panel_rows.append(_pick(t4, "CatBoost"))
# repo specialized
panel_rows.append(_pick(t9c, "MT-MLP (seed-avg)"))
panel_rows.append(_pick(t7, "ICM-Net (full)"))
# NR-Boost best row (top of table8)
panel_rows.append(t8.iloc[0].to_dict())   # "NR-Boost + RF ensemble"
panel_rows.append(t9.iloc[0].to_dict())   # best NR-Forest
# voting + Phase 5 stack
panel_rows.append(_pick(t4, "Voting (XGB+LGB+CAT)"))
panel_rows.append(_pick(t9c, "Stacking — L2 LR (C=10)", "Stack v3 (Phase 5 production)"))

# our 3 stacks
for tag, key in [("Stack (baseline rerun, no CL)", "baseline"),
                  ("Stack + CL-v6 binary (Phase 6)", "v6_binary"),
                  ("Stack + CL-v7 continuous (Phase 6d, OURS)", "v7_continuous")]:
    r = stacks[key]
    row = metrics_row(tag, y_te, r["test_cal"], r["thr"])
    panel_rows.append(row)

panel_rows = [r for r in panel_rows if r is not None]
panel_df = pd.DataFrame(panel_rows)
# normalize column order
keep_cols = ["model", "AUC", "AUC_lo", "AUC_hi", "PR_AUC", "PR_lo", "PR_hi",
              "F1", "Bal_Acc", "Brier", "MCC", "thr"]
panel_df = panel_df[[c for c in keep_cols if c in panel_df.columns]]
panel_df = panel_df.sort_values("AUC", ascending=False).reset_index(drop=True)
panel_path = OUT_TABLES / "table11_cl_v2_panel.csv"
panel_df.to_csv(panel_path, index=False)
print(f"wrote {panel_path}")
print(panel_df[["model", "AUC", "AUC_lo", "AUC_hi", "PR_AUC", "F1", "Bal_Acc"]].to_string(
    index=False, formatters={c: "{:.4f}".format for c in panel_df.select_dtypes("float").columns}))


# ─── Step 5: DeLong ────────────────────────────────────────────────
print("\n── Step 5: DeLong on the 3 stack variants ──")
p_base = stacks["baseline"]["test_cal"]
p_v6 = stacks["v6_binary"]["test_cal"]
p_v7 = stacks["v7_continuous"]["test_cal"]
pairs = [("v7_continuous", "baseline", p_v7, p_base),
          ("v7_continuous", "v6_binary", p_v7, p_v6),
          ("v6_binary", "baseline", p_v6, p_base)]
d_rows = []
for ch, bl, pc, pb in pairs:
    auc_c, auc_b, z, p = delong_test(y_te, pc, pb)
    d_rows.append({"champion": ch, "baseline": bl,
                    "AUC_champ": auc_c, "AUC_base": auc_b,
                    "ΔAUC": auc_c - auc_b, "z": z, "p_value": p,
                    "significant_05": p < 0.05})
    print(f"  {ch:16s} vs {bl:12s}  AUC {auc_c:.4f} vs {auc_b:.4f}  "
          f"Δ={auc_c - auc_b:+.4f}  z={z:+.3f}  p={p:.4f}")
d_df = pd.DataFrame(d_rows)
d_path = OUT_TABLES / "table11_cl_v2_delong.csv"
d_df.to_csv(d_path, index=False)
print(f"wrote {d_path}")


# ─── Step 6: figure ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
# include 4 curves: Phase 5 (baseline rerun), Phase 6 +CL (v6), v7, and RF (strongest single)
curves = [
    ("RF (strongest single)", stacks["baseline"]["test_preds"]["RF"], "steelblue", 1.3),
    ("Stack baseline rerun (Phase 5)", p_base, "grey", 1.6),
    ("Stack + CL-v6 binary (Phase 6)", p_v6, "darkorange", 1.8),
    ("Stack + CL-v7 continuous (OURS)", p_v7, "crimson", 2.6),
]
for label, p, color, lw in curves:
    fpr, tpr, _ = roc_curve(y_te, p)
    axes[0].plot(fpr, tpr, label=f"{label}  AUC={roc_auc_score(y_te, p):.4f}",
                  linewidth=lw, color=color)
axes[0].plot([0, 1], [0, 1], "--", c="grey", alpha=0.4)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test, calibrated)")
axes[0].legend(fontsize=8, loc="lower right")
for label, p, color, lw in curves:
    pr, rc, _ = precision_recall_curve(y_te, p)
    axes[1].plot(rc, pr, label=f"{label}  PR-AUC={average_precision_score(y_te, p):.4f}",
                  linewidth=lw, color=color)
axes[1].axhline(y_te.mean(), ls="--", c="grey", alpha=0.4)
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR (test, calibrated)")
axes[1].legend(fontsize=8, loc="upper right")
plt.tight_layout()
fig_path = OUT_FIGS / "fig11_cl_v2_roc.png"
plt.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"\nwrote {fig_path}")

# ─── summary ───────────────────────────────────────────────────────
print("\n" + "=" * 72); print("  FINAL RESULT"); print("=" * 72)
row_v7 = panel_df[panel_df["model"].str.contains("CL-v7")].iloc[0]
row_v6 = panel_df[panel_df["model"].str.contains("CL-v6")].iloc[0]
row_base = panel_df[panel_df["model"].str.contains("baseline rerun")].iloc[0]
row_p5 = panel_df[panel_df["model"].str.contains("Phase 5 production")].iloc[0]
for label, r in [("Phase 5 production (persisted)", row_p5),
                 ("Stack baseline (fresh rerun)", row_base),
                 ("Stack + CL-v6 binary (Phase 6)", row_v6),
                 ("Stack + CL-v7 continuous (OURS)", row_v7)]:
    print(f"  {label:38s}  AUC={r['AUC']:.4f}  [{r['AUC_lo']:.3f}, {r['AUC_hi']:.3f}]  "
          f"PR={r['PR_AUC']:.3f}  F1={r['F1']:.3f}  Bal_Acc={r['Bal_Acc']:.3f}")
