"""Phase 7 — Stacking v5: orthogonal base expansion.

Adds three mechanically-orthogonal base learners to the Phase 6 champion stack
and asks whether decorrelation alone can push past AUC=0.7844 / F1=0.4252 /
Bal-Acc=0.7107.

Eight bases total:
  RF            — Phase 5 (tree ensemble)
  NR-Boost      — Phase 4a (XGBoost + GCE + self-paced)
  MT-MLP        — Phase 1 (shared encoder + CORN intent head, seed-avg × 5)
  SVM-RBF       — Phase 5 (kernel margin)
  ExtraTrees    — Phase 5 (random splits)
  TabPFN        — transformer prior-fitted on synthetic tabular tasks [Hollmann+ 2024]
  gcForest      — cascade forest [Zhou & Feng 2017] via deep-forest package
  kNN           — distance-weighted local similarity (k=50)

CL weights (Phase 6 CL-v6 binary) are applied to the 5 weight-supporting bases:
  RF, NR-Boost (as init_weight), ExtraTrees, SVM-RBF, gcForest.
TabPFN / kNN / MT-MLP use uniform weights (API / stability constraints).

Ablations — subsets of the same 8-base OOF/test matrix:
  A_phase6  RF+NRB+MT+SVM+ET                (Phase 6 CL champion, reproducibility anchor)
  B_tabpfn  + TabPFN
  C_gcf     + gcForest
  D_knn     + kNN
  E_full    all 8

Outputs
-------
src/tables/table13_stack_v5_panel.csv      per-base + 5 ablations × meta variants (calibrated)
src/tables/table13_stack_v5_ablation.csv   5-row summary (meta = L2-LR C=10)
src/tables/table13_stack_v5_delong.csv     E_full vs {A, B, C, D, RF, Phase5, Phase6}
src/tables/table13_stack_v5_corr.csv       8×8 OOF Pearson correlation
src/tables/table13_stack_v5_weights.csv    meta coefficients per ablation
src/figures/fig13_stack_v5_roc.png         ROC + PR, 5 ablations
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Use HuggingFace mirror (direct huggingface.co is unreachable from CN).
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

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
from sklearn.neighbors import KNeighborsClassifier
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

# TabPFN license check hits HuggingFace for the canonical license version name;
# if HF API is unreachable it falls back to hf_repo_id ("tabpfn-v2-classifier"),
# which is NOT the version name used at acceptance time ("tabpfn-2.6-license-v1.0"),
# so the check returns False even though the user accepted the license.
# Pre-mark the repos as accepted (token is still validated against PriorLabs API).
from tabpfn import browser_auth as _tpf_auth
for _r in ("tabpfn_2_5", "tabpfn_2_6", "tabpfn-v2-classifier", "tabpfn-v2-regressor"):
    _tpf_auth._accepted_repos.add(_r)
from tabpfn import TabPFNClassifier
from deepforest import CascadeForestClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RS = 42
TARGET = "离职行为"
INTENT = "离职意向"
N_FOLDS = 5
N_MT_SEEDS = 5
DEVICE = torch.device("cpu")
USE_CL_WEIGHTS = True

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)

np.random.seed(RS)
torch.manual_seed(RS)


# ─── helpers copied from 06c_stack_v4.py (verbatim) ─────────────────────
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


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ─── data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")
feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]

y = df[TARGET].values.astype(int)
y_ord_all = (df[INTENT].round().clip(1, 5).astype(int) - 1).values
X_raw = df[feat_cols].values.astype(np.float32)
X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
y_ord_tr = y_ord_all[train_idx]

SW_PATH = Path("data/processed/sample_weights_v6.npy")
sw_v6 = np.load(SW_PATH) if (SW_PATH.exists() and USE_CL_WEIGHTS) else np.ones(len(y_tr), dtype=np.float32)
print(f"clean {df.shape}  train {len(train_idx)}  test {len(test_idx)}  "
      f"n_feat {len(feat_cols)}  MT seeds {N_MT_SEEDS}")
print(f"CL weights: use={USE_CL_WEIGHTS}  flagged={(sw_v6 < 1.0).sum()}  "
      f"w_flag={sw_v6[sw_v6<1.0].min() if (sw_v6<1.0).any() else 'n/a'}")

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
fold_idx = list(skf.split(X_tr, y_tr))
N_tr = len(y_tr); in_dim = X_tr.shape[1]
spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

BASES = ["RF", "NRBoost", "MTMLP", "SVM", "ET", "TabPFN", "GCF", "KNN"]
WEIGHT_SUPPORTED = {"RF", "NRBoost", "SVM", "ET", "GCF"}  # MT-MLP / TabPFN / KNN use uniform

oof = {k: np.zeros(N_tr) for k in BASES}
test_preds = {k: np.zeros(len(y_te)) for k in BASES}

w_train = sw_v6 if USE_CL_WEIGHTS else None


# ─── 5-fold OOF for 8 bases (compute once, select subsets at meta time) ─
print("\n" + "=" * 72)
print("  STEP 1: 5-fold OOF for 8 bases")
print("=" * 72)
for f, (tr, va) in enumerate(fold_idx):
    print(f"\nfold {f}:", flush=True)
    Xtr_f, Xva_f = X_tr[tr], X_tr[va]
    ytr_f = y_tr[tr]; yo_tr_f = y_ord_tr[tr]
    w_f = None if w_train is None else w_train[tr]

    # ── standardized copies for scale-sensitive models ──
    scaler = StandardScaler().fit(Xtr_f)
    Xtr_s = scaler.transform(Xtr_f).astype(np.float32)
    Xva_s = scaler.transform(Xva_f).astype(np.float32)

    # ── RF ──
    rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RS)
    rf.fit(Xtr_f, ytr_f, sample_weight=w_f)
    oof["RF"][va] = rf.predict_proba(Xva_f)[:, 1]
    print("  RF done", flush=True)

    # ── NR-Boost ──
    b = fit_nrboost(Xtr_f, ytr_f, q=0.7, n_rounds=400, n_stages=2,
                    drop_frac=0.10, seed=RS + f, init_weight=w_f)
    oof["NRBoost"][va] = predict_nrboost(b, Xva_f)
    print("  NRBoost done", flush=True)

    # ── MT-MLP seed-avg (uniform weights) ──
    oof["MTMLP"][va] = fit_predict_mtmlp_seedavg(
        Xtr_s, ytr_f, yo_tr_f, Xva_s, in_dim=in_dim,
        pos_weight=spw, base_seed=RS + f)
    print(f"  MT×{N_MT_SEEDS} done", flush=True)

    # ── SVM ──
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              probability=True, random_state=RS)
    svm.fit(Xtr_s, ytr_f, sample_weight=w_f)
    oof["SVM"][va] = svm.predict_proba(Xva_s)[:, 1]
    print("  SVM done", flush=True)

    # ── ExtraTrees ──
    et = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=RS)
    et.fit(Xtr_f, ytr_f, sample_weight=w_f)
    oof["ET"][va] = et.predict_proba(Xva_f)[:, 1]
    print("  ET done", flush=True)

    # ── TabPFN (no sample_weight) ──
    # ignore_pretraining_limits=True needed: train fold ~3500 > 1000 CPU default cap.
    tpf = TabPFNClassifier(device="cpu", n_estimators=4, random_state=RS,
                            ignore_pretraining_limits=True)
    tpf.fit(Xtr_s, ytr_f)
    oof["TabPFN"][va] = tpf.predict_proba(Xva_s)[:, 1]
    print("  TabPFN done", flush=True)

    # ── gcForest cascade ──
    gcf = CascadeForestClassifier(
        n_estimators=2, n_trees=100, max_layers=5,
        use_predictor=True, predictor="forest",
        n_jobs=-1, random_state=RS, verbose=0)
    try:
        gcf.fit(Xtr_f, ytr_f, sample_weight=w_f)
    except TypeError:
        gcf.fit(Xtr_f, ytr_f)
    oof["GCF"][va] = gcf.predict_proba(Xva_f)[:, 1]
    print("  GCF done", flush=True)

    # ── kNN distance-weighted ──
    knn = KNeighborsClassifier(n_neighbors=50, weights="distance",
                                metric="euclidean", n_jobs=-1)
    knn.fit(Xtr_s, ytr_f)
    oof["KNN"][va] = knn.predict_proba(Xva_s)[:, 1]
    print("  KNN done", flush=True)


# ─── full-train refits for test predictions ────────────────────────────
print("\n" + "=" * 72)
print("  STEP 2: full-train refits → test predictions")
print("=" * 72)

scaler_full = StandardScaler().fit(X_tr)
Xtr_s_full = scaler_full.transform(X_tr).astype(np.float32)
Xte_s_full = scaler_full.transform(X_te).astype(np.float32)

rf_full = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=RS)
rf_full.fit(X_tr, y_tr, sample_weight=w_train)
test_preds["RF"] = rf_full.predict_proba(X_te)[:, 1]
print("  RF full done", flush=True)

b_full = fit_nrboost(X_tr, y_tr, q=0.7, n_rounds=400, n_stages=2,
                     drop_frac=0.10, seed=RS, init_weight=w_train)
test_preds["NRBoost"] = predict_nrboost(b_full, X_te)
print("  NRBoost full done", flush=True)

test_preds["MTMLP"] = fit_predict_mtmlp_seedavg(
    Xtr_s_full, y_tr, y_ord_tr, Xte_s_full, in_dim=in_dim,
    pos_weight=spw, base_seed=RS)
print(f"  MT×{N_MT_SEEDS} full done", flush=True)

svm_full = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                probability=True, random_state=RS)
svm_full.fit(Xtr_s_full, y_tr, sample_weight=w_train)
test_preds["SVM"] = svm_full.predict_proba(Xte_s_full)[:, 1]
print("  SVM full done", flush=True)

et_full = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RS)
et_full.fit(X_tr, y_tr, sample_weight=w_train)
test_preds["ET"] = et_full.predict_proba(X_te)[:, 1]
print("  ET full done", flush=True)

tpf_full = TabPFNClassifier(device="cpu", n_estimators=4, random_state=RS,
                             ignore_pretraining_limits=True)
tpf_full.fit(Xtr_s_full, y_tr)
test_preds["TabPFN"] = tpf_full.predict_proba(Xte_s_full)[:, 1]
print("  TabPFN full done", flush=True)

gcf_full = CascadeForestClassifier(
    n_estimators=2, n_trees=100, max_layers=5,
    use_predictor=True, predictor="forest",
    n_jobs=-1, random_state=RS, verbose=0)
try:
    gcf_full.fit(X_tr, y_tr, sample_weight=w_train)
except TypeError:
    gcf_full.fit(X_tr, y_tr)
test_preds["GCF"] = gcf_full.predict_proba(X_te)[:, 1]
print("  GCF full done", flush=True)

knn_full = KNeighborsClassifier(n_neighbors=50, weights="distance",
                                 metric="euclidean", n_jobs=-1)
knn_full.fit(Xtr_s_full, y_tr)
test_preds["KNN"] = knn_full.predict_proba(Xte_s_full)[:, 1]
print("  KNN full done", flush=True)

print("\n── per-base AUCs ──")
for k in BASES:
    print(f"  {k:8s}  CV AUC={roc_auc_score(y_tr, oof[k]):.4f}  "
          f"test AUC={roc_auc_score(y_te, test_preds[k]):.4f}")


# ─── correlation matrix ─────────────────────────────────────────────────
print("\n── OOF Pearson correlation (8×8) ──")
oof_mat_full = np.column_stack([oof[k] for k in BASES])
corr = np.corrcoef(oof_mat_full, rowvar=False)
header = "        " + "  ".join(f"{k:8s}" for k in BASES)
print(header)
for i, k in enumerate(BASES):
    print(f"{k:6s}  " + "  ".join(f"{corr[i, j]:8.3f}" for j in range(len(BASES))))
corr_df = pd.DataFrame(corr, index=BASES, columns=BASES)
corr_df.to_csv(OUT_TABLES / "table13_stack_v5_corr.csv")


# ─── meta helpers (operate on a base subset) ───────────────────────────
def fit_meta(bases_subset, C=10.0, class_weight=None):
    """Fit L2-LR meta on logit-OOF of the given base subset. Return OOF + test
    meta probs (both produced via CV refit for OOF, full-train fit for test)."""
    Z_tr = _logit(np.column_stack([oof[k] for k in bases_subset]))
    Z_te = _logit(np.column_stack([test_preds[k] for k in bases_subset]))
    m_full = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                 class_weight=class_weight, max_iter=2000,
                                 random_state=RS).fit(Z_tr, y_tr)
    oof_m = np.zeros(N_tr)
    for tr, va in fold_idx:
        mi = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                 class_weight=class_weight, max_iter=2000,
                                 random_state=RS).fit(Z_tr[tr], y_tr[tr])
        oof_m[va] = mi.predict_proba(Z_tr[va])[:, 1]
    return m_full, oof_m, m_full.predict_proba(Z_te)[:, 1]


def fit_convex(bases_subset):
    """Convex-combination meta (log-loss) on raw probs of the given base subset."""
    P_tr = np.column_stack([oof[k] for k in bases_subset])
    P_te = np.column_stack([test_preds[k] for k in bases_subset])
    n_b = len(bases_subset)
    w0 = np.ones(n_b) / n_b

    def _neg_logloss(w_vec, P, y):
        w_vec = np.clip(w_vec, 0, None)
        if w_vec.sum() < 1e-9: return 1e9
        w_vec = w_vec / w_vec.sum()
        p = (P * w_vec).sum(axis=1)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return log_loss(y, p)

    res = minimize(_neg_logloss, w0, args=(P_tr, y_tr), method="SLSQP",
                    bounds=[(0, 1)] * n_b,
                    constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    w_cvx = res.x / res.x.sum()
    return w_cvx, (P_tr * w_cvx).sum(axis=1), (P_te * w_cvx).sum(axis=1)


# ─── 5 ablations ───────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  STEP 3: five ablations on shared 8-base OOF matrix")
print("=" * 72)
PHASE6_BASES = ["RF", "NRBoost", "MTMLP", "SVM", "ET"]
ablations = [
    ("A_phase6",  PHASE6_BASES,                           "A: Phase 6 CL (5 bases)"),
    ("B_tabpfn",  PHASE6_BASES + ["TabPFN"],              "B: + TabPFN"),
    ("C_gcf",     PHASE6_BASES + ["GCF"],                 "C: + gcForest"),
    ("D_knn",     PHASE6_BASES + ["KNN"],                 "D: + kNN"),
    ("E_full",    BASES,                                  "E: all 8 (Phase 7 full)"),
]

results = {}
for key, subset, label in ablations:
    print(f"\n  [{key}] {label}  bases={subset}")
    m_lr, oof_lr, test_lr = fit_meta(subset, C=10.0, class_weight=None)
    m_lr_bal, oof_lr_bal, test_lr_bal = fit_meta(subset, C=1.0, class_weight="balanced")
    w_cvx, oof_cvx, test_cvx = fit_convex(subset)
    P_tr = np.column_stack([oof[k] for k in subset])
    P_te = np.column_stack([test_preds[k] for k in subset])
    oof_mean = P_tr.mean(axis=1); test_mean = P_te.mean(axis=1)
    results[key] = {
        "subset": subset,
        "meta_lr_hC": m_lr, "oof_lr_hC": oof_lr, "test_lr_hC": test_lr,
        "meta_lr_bal": m_lr_bal, "oof_lr_bal": oof_lr_bal, "test_lr_bal": test_lr_bal,
        "w_cvx": w_cvx, "oof_cvx": oof_cvx, "test_cvx": test_cvx,
        "oof_mean": oof_mean, "test_mean": test_mean,
    }
    print(f"    L2-LR C=10   CV AUC={roc_auc_score(y_tr, oof_lr):.4f}  "
          f"test AUC={roc_auc_score(y_te, test_lr):.4f}")


# ─── calibrate + panel ─────────────────────────────────────────────────
META_KEYS = [
    ("lr_hC",  "Stacking L2 LR (C=10)"),
    ("lr_bal", "Stacking L2 LR (bal)"),
    ("cvx",    "Stacking convex combo"),
    ("mean",   "Mean-avg"),
]

panel_rows = []
cal_probs = {}
for ab_key, subset, ab_label in ablations:
    r = results[ab_key]
    for mk, mk_label in META_KEYS:
        oof_k = r[f"oof_{mk}"]; test_k = r[f"test_{mk}"]
        iso = IsotonicRegression(out_of_bounds="clip").fit(oof_k, y_tr)
        oof_cal = iso.predict(oof_k); test_cal = iso.predict(test_k)
        thr = best_f1_threshold(y_tr, oof_cal)
        row = metrics_row(f"[{ab_key}] {mk_label}", y_te, test_cal, thr)
        row["ablation"] = ab_label; row["meta"] = mk_label
        cal_probs[(ab_key, mk)] = test_cal
        panel_rows.append(row)

# also include individual (calibrated) base predictions once
for b in BASES:
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof[b], y_tr)
    oof_cal = iso.predict(oof[b]); test_cal = iso.predict(test_preds[b])
    thr = best_f1_threshold(y_tr, oof_cal)
    row = metrics_row(f"[base] {b}", y_te, test_cal, thr)
    row["ablation"] = "base learner"; row["meta"] = b
    cal_probs[("base", b)] = test_cal
    panel_rows.append(row)

panel_df = pd.DataFrame(panel_rows).sort_values("AUC", ascending=False).reset_index(drop=True)
panel_path = OUT_TABLES / "table13_stack_v5_panel.csv"
panel_df.to_csv(panel_path, index=False)
print(f"\nwrote {panel_path}")
print(panel_df[["model", "AUC", "AUC_lo", "AUC_hi", "PR_AUC", "F1", "Bal_Acc"]].head(20).to_string(
    index=False,
    formatters={c: "{:.3f}".format for c in panel_df.select_dtypes("float").columns}))


# ─── 5-row ablation summary (meta = L2-LR C=10) ───────────────────────
summary_rows = []
for ab_key, _, ab_label in ablations:
    base = panel_df[(panel_df.ablation == ab_label) & (panel_df.meta == "Stacking L2 LR (C=10)")].iloc[0]
    summary_rows.append({
        "ablation": ab_label,
        "AUC": base["AUC"], "AUC_lo": base["AUC_lo"], "AUC_hi": base["AUC_hi"],
        "PR_AUC": base["PR_AUC"], "F1": base["F1"], "Bal_Acc": base["Bal_Acc"],
        "Brier": base["Brier"], "MCC": base["MCC"], "thr": base["thr"],
    })
summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_TABLES / "table13_stack_v5_ablation.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nwrote {summary_path}")
print(summary_df.to_string(index=False,
      formatters={c: "{:.4f}".format for c in summary_df.select_dtypes("float").columns}))


# ─── DeLong: E_full vs {A, B, C, D, RF, Phase5, Phase6} ───────────────
print("\n── DeLong: E_full vs others (test AUC) ──")
champ = cal_probs[("E_full", "lr_hC")]
auc_champ = roc_auc_score(y_te, champ)
delong_rows = []
for ab in ["A_phase6", "B_tabpfn", "C_gcf", "D_knn"]:
    other = cal_probs[(ab, "lr_hC")]
    a, b, z, p = delong_test(y_te, champ, other)
    delong_rows.append({"champion": "E_full L2-LR C=10", "baseline": f"{ab} L2-LR C=10",
                        "AUC_champ": a, "AUC_base": b, "ΔAUC": a - b,
                        "z": z, "p_value": p, "significant_05": p < 0.05})
    print(f"  E_full vs {ab:10s}  Δ={a-b:+.4f}  z={z:+.3f}  p={p:.4f}")

# vs RF in E_full ablation
rf_cal = cal_probs[("base", "RF")]
a, b, z, p = delong_test(y_te, champ, rf_cal)
delong_rows.append({"champion": "E_full L2-LR C=10", "baseline": "RandomForest (calibrated)",
                    "AUC_champ": a, "AUC_base": b, "ΔAUC": a - b,
                    "z": z, "p_value": p, "significant_05": p < 0.05})
print(f"  E_full vs RF            Δ={a-b:+.4f}  z={z:+.3f}  p={p:.4f}")

# vs TabPFN solo, gcForest solo, kNN solo
for b_name in ["TabPFN", "GCF", "KNN"]:
    other = cal_probs[("base", b_name)]
    a, b_auc, z, p = delong_test(y_te, champ, other)
    delong_rows.append({"champion": "E_full L2-LR C=10", "baseline": f"{b_name} solo (calibrated)",
                        "AUC_champ": a, "AUC_base": b_auc, "ΔAUC": a - b_auc,
                        "z": z, "p_value": p, "significant_05": p < 0.05})
    print(f"  E_full vs {b_name:10s}  Δ={a-b_auc:+.4f}  z={z:+.3f}  p={p:.4f}")

# vs Phase 5 stack (persisted) — cross-script, AUC only
phase5_panel = pd.read_csv(OUT_TABLES / "table9c_stack_v3_panel.csv")
phase5_auc = float(phase5_panel.query("model == 'Stacking — L2 LR (C=10)'")["AUC"].iloc[0])
delong_rows.append({"champion": "E_full L2-LR C=10",
                    "baseline": "Phase 5 stack L2-LR C=10 (persisted)",
                    "AUC_champ": auc_champ, "AUC_base": phase5_auc,
                    "ΔAUC": auc_champ - phase5_auc, "z": np.nan, "p_value": np.nan,
                    "significant_05": False})

# vs Phase 6 champion (persisted)
phase6_panel = pd.read_csv(OUT_TABLES / "table10_stack_v4_panel.csv")
phase6_row = phase6_panel[phase6_panel.model == "[C_CL] Stacking L2 LR (C=10)"]
if len(phase6_row):
    phase6_auc = float(phase6_row["AUC"].iloc[0])
    delong_rows.append({"champion": "E_full L2-LR C=10",
                        "baseline": "Phase 6 +CL L2-LR C=10 (persisted)",
                        "AUC_champ": auc_champ, "AUC_base": phase6_auc,
                        "ΔAUC": auc_champ - phase6_auc, "z": np.nan, "p_value": np.nan,
                        "significant_05": False})

dlg = pd.DataFrame(delong_rows)
dlg_path = OUT_TABLES / "table13_stack_v5_delong.csv"
dlg.to_csv(dlg_path, index=False)
print(f"\nwrote {dlg_path}")


# ─── meta weights per ablation ─────────────────────────────────────────
w_rows = []
for ab_key, subset, ab_label in ablations:
    r = results[ab_key]
    for i, b in enumerate(subset):
        w_rows.append({
            "ablation": ab_label, "base": b,
            "LR_C10_coef": float(r["meta_lr_hC"].coef_.ravel()[i]),
            "LR_bal_coef": float(r["meta_lr_bal"].coef_.ravel()[i]),
            "convex_weight": float(r["w_cvx"][i]),
        })
w_df = pd.DataFrame(w_rows)
w_path = OUT_TABLES / "table13_stack_v5_weights.csv"
w_df.to_csv(w_path, index=False)
print(f"wrote {w_path}")


# ─── ROC + PR figure (5 ablations, meta=L2-LR C=10) ──────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
colors = {"A_phase6": "steelblue", "B_tabpfn": "crimson",
          "C_gcf": "seagreen", "D_knn": "darkorange", "E_full": "black"}
labels = {"A_phase6": "A: Phase 6 CL (5 bases)", "B_tabpfn": "B: + TabPFN",
          "C_gcf": "C: + gcForest", "D_knn": "D: + kNN",
          "E_full": "E: all 8 (Phase 7)"}
for ab in ["A_phase6", "B_tabpfn", "C_gcf", "D_knn", "E_full"]:
    p = cal_probs[(ab, "lr_hC")]
    fpr, tpr, _ = roc_curve(y_te, p)
    lw = 2.5 if ab == "E_full" else 1.5
    axes[0].plot(fpr, tpr,
                  label=f"{labels[ab]}  AUC={roc_auc_score(y_te, p):.4f}",
                  linewidth=lw, color=colors[ab])
axes[0].plot([0, 1], [0, 1], "--", c="grey", alpha=0.5)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test, calibrated)")
axes[0].legend(fontsize=8, loc="lower right")

for ab in ["A_phase6", "B_tabpfn", "C_gcf", "D_knn", "E_full"]:
    p = cal_probs[(ab, "lr_hC")]
    pr, rc, _ = precision_recall_curve(y_te, p)
    lw = 2.5 if ab == "E_full" else 1.5
    axes[1].plot(rc, pr,
                  label=f"{labels[ab]}  PR={average_precision_score(y_te, p):.4f}",
                  linewidth=lw, color=colors[ab])
axes[1].axhline(y_te.mean(), ls="--", c="grey", alpha=0.5)
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR (test, calibrated)")
axes[1].legend(fontsize=8, loc="upper right")

plt.tight_layout()
fig_path = OUT_FIGS / "fig13_stack_v5_roc.png"
plt.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"wrote {fig_path}")


# ─── final summary ────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  FINAL RESULT")
print("=" * 72)
for _, row in summary_df.iterrows():
    print(f"  {row['ablation']:35s}  AUC={row['AUC']:.4f}  "
          f"[{row['AUC_lo']:.3f}, {row['AUC_hi']:.3f}]  "
          f"PR={row['PR_AUC']:.3f}  F1={row['F1']:.3f}  "
          f"Bal_Acc={row['Bal_Acc']:.3f}  MCC={row['MCC']:.3f}")

best_row = summary_df.iloc[summary_df["AUC"].idxmax()]
print(f"\nBest by AUC: {best_row['ablation']} — AUC={best_row['AUC']:.4f}")
