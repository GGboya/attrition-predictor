"""Phase 3 — ICM-Net training + full baseline panel + ablation.

Train ICM-Net with 3-stage schedule (intent warmup → behavior cascade →
joint finetune) on the frozen 5469-row split. Compare against:

    Classical : LR, RF, SVM-RBF
    GBDT      : XGBoost, LightGBM, CatBoost
    NN        : vanilla MLP, MT-MLP (lam best from Phase 1)
    Ensemble  : Voting(3 trees), Voting(MT + 3 trees)

Ablation on ICM-Net (remove one component at a time):
    full / −cascade (intent_feed zeroed) / −Mixup / −SCE

Statistical testing:
    - Bootstrap 95% CI on test AUC and PR-AUC.
    - Two-sided DeLong test: ICM-Net vs each baseline on the test set.

Outputs:
    src/tables/table7_icmnet_panel.csv        all models, test metrics + CI
    src/tables/table7_icmnet_ablation.csv     ICM-Net variants
    src/tables/table7_delong.csv              DeLong p-values
    src/figures/fig7_icmnet_roc.png           ROC / PR overlay
    models/icmnet_calibrated.pkl              ICM-Net state + isotonic
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

import json
import pickle
from pathlib import Path

# Trees FIRST to avoid OpenMP clash with torch.
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import importlib.util, sys
def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

_m = _load("mt_model", "01a_mt_model.py")
MTMlp = _m.MTMlp; joint_loss = _m.joint_loss

_icm = _load("icmnet", "03a_icmnet_model.py")
ICMNet = _icm.ICMNet
forward_train_step = _icm.forward_train_step
set_requires_grad = _icm.set_requires_grad
corn_class_probs = _icm.corn_class_probs

# pull DeLong from _utils.py (reuse validated impl)
_utils = _load("stats_utils", "_utils.py")
delong_test = _utils.delong_test
_fast_delong = _utils._fast_delong

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scipy.stats as sp_stats

# ─── config ─────────────────────────────────────────────────────────────
RS = 42
TARGET = "离职行为"
INTENT = "离职意向"
N_FOLDS = 5
EPOCHS_S1 = 60     # intent warmup
EPOCHS_S2 = 120    # behavior cascade
EPOCHS_S3 = 120    # joint finetune
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 20

# ICM-Net hyperparams (tunable)
ALPHA_MIX = 0.2
MAX_INTENT_GAP = 1
SCE_ALPHA = 0.5
SCE_BETA = 0.5
LAM_BIN = 0.7
LAM_ORD = 0.3

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)
OUT_MODELS = Path("models"); OUT_MODELS.mkdir(exist_ok=True, parents=True)

torch.manual_seed(RS); np.random.seed(RS)

# ─── data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")

feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]
print(f"clean {df.shape}  train {len(train_idx)}  test {len(test_idx)}  "
      f"features: {len(feat_cols)} (no intent)")

y = df[TARGET].values.astype(int)
intent_bin = np.clip(df[INTENT].round().values.astype(int), 1, 5) - 1  # {0..4}
X_raw = df[feat_cols].values.astype(np.float32)
X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
i_tr, i_te = intent_bin[train_idx], intent_bin[test_idx]


# ─── helpers ────────────────────────────────────────────────────────────
def best_f1_threshold(y_true, p):
    best, thr = 0.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        f = f1_score(y_true, (p >= t).astype(int), pos_label=1, zero_division=0)
        if f > best:
            best, thr = f, t
    return thr


def bootstrap_ci(y_true, p, metric, n=1000, seed=RS):
    rng = np.random.default_rng(seed)
    vals = []
    n_samp = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_samp, n_samp)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            vals.append(metric(y_true[idx], p[idx]))
        except Exception:
            pass
    arr = np.asarray(vals)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


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
        "F1": f1_score(y_true, pred, pos_label=1, zero_division=0),
        "Bal_Acc": balanced_accuracy_score(y_true, pred),
        "Brier": brier_score_loss(y_true, p),
        "MCC": matthews_corrcoef(y_true, pred),
        "thr": thr,
    }


# ─── ICM-Net training (3 stages + optional ablation flags) ──────────────
class EarlyStop:
    def __init__(self, patience=PATIENCE):
        self.best = -np.inf; self.bad = 0; self.patience = patience
        self.best_state = None

    def step(self, score, state):
        if score > self.best + 1e-6:
            self.best = score; self.bad = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            return False
        self.bad += 1
        return self.bad > self.patience


def train_icmnet(X_tr, y_tr, i_tr, X_val, y_val, i_val,
                 *, in_dim: int, pos_weight: float,
                 use_cascade: bool = True, use_mixup: bool = True,
                 use_sce: bool = True,
                 seed: int = RS):
    """Train ICM-Net. Returns model (best state loaded) and best val AUC."""
    torch.manual_seed(seed)
    model = ICMNet(in_dim=in_dim, n_ord_classes=5).to(DEVICE)
    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, device=DEVICE)
    it = torch.tensor(i_tr, device=DEVICE, dtype=torch.long)
    Xv = torch.tensor(X_val, device=DEVICE)

    zero_intent = not use_cascade
    n = len(Xt)

    def run_epochs(stage: int, n_epochs: int, params: list,
                   eval_metric=True) -> float:
        opt = torch.optim.Adam(params, lr=LR, weight_decay=1e-4)
        stopper = EarlyStop(PATIENCE)
        for ep in range(n_epochs):
            model.train()
            perm = torch.randperm(n, device=DEVICE)
            for b in range(0, n, BATCH_SIZE):
                idx = perm[b:b + BATCH_SIZE]
                xb, yb, ib = Xt[idx], yt[idx], it[idx]
                loss = forward_train_step(model, xb, yb, ib,
                                           stage=stage,
                                           use_mixup=use_mixup,
                                           alpha_mix=ALPHA_MIX,
                                           use_sce=use_sce,
                                           zero_intent=zero_intent,
                                           pos_weight=pos_weight,
                                           lam_bin=LAM_BIN, lam_ord=LAM_ORD,
                                           sce_alpha=SCE_ALPHA, sce_beta=SCE_BETA)
                opt.zero_grad(); loss.backward(); opt.step()
            if not eval_metric:
                continue
            # eval: binary head AUC on val
            model.eval()
            with torch.no_grad():
                lb, _, _ = model(Xv, zero_intent=zero_intent)
                p_val = torch.sigmoid(lb).cpu().numpy()
            val_auc = roc_auc_score(y_val, p_val)
            if stopper.step(val_auc, model.state_dict()):
                break
        if stopper.best_state is not None:
            model.load_state_dict(stopper.best_state)
        return stopper.best

    # Stage 1 — intent warmup (encoder + intent_head only)
    set_requires_grad(model.encoder, True)
    set_requires_grad(model.intent_head, True)
    set_requires_grad(model.behavior_head, False)
    run_epochs(1, EPOCHS_S1,
               list(model.encoder.parameters()) + list(model.intent_head.parameters()),
               eval_metric=False)

    # Stage 2 — behavior cascade (freeze encoder + intent_head)
    set_requires_grad(model.encoder, False)
    set_requires_grad(model.intent_head, False)
    set_requires_grad(model.behavior_head, True)
    run_epochs(2, EPOCHS_S2,
               list(model.behavior_head.parameters()))

    # Stage 3 — joint finetune (all unfrozen)
    set_requires_grad(model.encoder, True)
    set_requires_grad(model.intent_head, True)
    set_requires_grad(model.behavior_head, True)
    best = run_epochs(3, EPOCHS_S3, list(model.parameters()))
    return model, best


def predict_icmnet(model, X, zero_intent=False):
    model.eval()
    Xt = torch.tensor(X, device=DEVICE)
    with torch.no_grad():
        lb, _, _ = model(Xt, zero_intent=zero_intent)
        p = torch.sigmoid(lb).cpu().numpy()
    return p


def icmnet_cv_and_fit(name: str, *, use_cascade=True, use_mixup=True, use_sce=True,
                      seed: int = RS):
    print(f"\n── {name} ──")
    scaler = StandardScaler().fit(X_tr_raw)
    Xs = scaler.transform(X_tr_raw).astype(np.float32)
    Xte_s = scaler.transform(X_te_raw).astype(np.float32)
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_tr))
    for f, (tr, va) in enumerate(skf.split(Xs, y_tr)):
        m, best = train_icmnet(Xs[tr], y_tr[tr], i_tr[tr], Xs[va], y_tr[va], i_tr[va],
                                in_dim=Xs.shape[1], pos_weight=pos_weight,
                                use_cascade=use_cascade, use_mixup=use_mixup,
                                use_sce=use_sce, seed=seed + f)
        oof[va] = predict_icmnet(m, Xs[va], zero_intent=not use_cascade)
        print(f"  fold {f} val AUC={best:.4f}")
    cv_auc = roc_auc_score(y_tr, oof)
    print(f"  CV AUC={cv_auc:.4f}")

    # retrain on full train (90/10 inner split for early stop)
    tr, va = train_test_split(np.arange(len(y_tr)), test_size=0.1,
                               stratify=y_tr, random_state=seed)
    model, _ = train_icmnet(Xs[tr], y_tr[tr], i_tr[tr], Xs[va], y_tr[va], i_tr[va],
                             in_dim=Xs.shape[1], pos_weight=pos_weight,
                             use_cascade=use_cascade, use_mixup=use_mixup,
                             use_sce=use_sce, seed=seed)
    p_test = predict_icmnet(model, Xte_s, zero_intent=not use_cascade)
    return oof, p_test, model, scaler, cv_auc


# ─── MT-MLP trainer (same as Phase 3, reimplemented here for isolation) ─
def train_nn_legacy(X_tr, y_tr, i_tr, X_val, y_val, i_val, lam: float | None,
                     pos_weight: float, in_dim: int, seed: int = RS):
    torch.manual_seed(seed)
    model = MTMlp(in_dim=in_dim, n_ord_classes=5).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    stopper = EarlyStop(PATIENCE)
    Xt = torch.tensor(X_tr, device=DEVICE); yt = torch.tensor(y_tr, device=DEVICE)
    it = torch.tensor(i_tr, device=DEVICE, dtype=torch.long); Xv = torch.tensor(X_val, device=DEVICE)
    for ep in range(200):
        model.train()
        perm = torch.randperm(len(Xt), device=DEVICE)
        for b in range(0, len(Xt), BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            xb, yb, ib = Xt[idx], yt[idx], it[idx]
            lb, lo = model(xb)
            if lam is None or lam >= 1.0:
                pw = torch.tensor([pos_weight], device=DEVICE)
                loss = F.binary_cross_entropy_with_logits(lb, yb.float(), pos_weight=pw)
            else:
                loss = joint_loss(lb, lo, yb, ib, lam=lam,
                                  pos_weight=pos_weight, n_ord_classes=5)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            lb, _ = model(Xv)
            p_val = torch.sigmoid(lb).cpu().numpy()
        val_auc = roc_auc_score(y_val, p_val)
        if stopper.step(val_auc, model.state_dict()):
            break
    model.load_state_dict(stopper.best_state)
    return model, stopper.best


def nn_cv_and_fit(name, lam, seed=RS):
    print(f"\n── {name} ──")
    scaler = StandardScaler().fit(X_tr_raw)
    Xs = scaler.transform(X_tr_raw).astype(np.float32)
    Xte_s = scaler.transform(X_te_raw).astype(np.float32)
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_tr))
    for f, (tr, va) in enumerate(skf.split(Xs, y_tr)):
        m, best = train_nn_legacy(Xs[tr], y_tr[tr], i_tr[tr], Xs[va], y_tr[va], i_tr[va],
                                   lam, pos_weight, in_dim=Xs.shape[1], seed=seed + f)
        m.eval()
        with torch.no_grad():
            lb, _ = m(torch.tensor(Xs[va], device=DEVICE))
            oof[va] = torch.sigmoid(lb).cpu().numpy()
        print(f"  fold {f} val AUC={best:.4f}")
    cv_auc = roc_auc_score(y_tr, oof)
    print(f"  CV AUC={cv_auc:.4f}")
    tr, va = train_test_split(np.arange(len(y_tr)), test_size=0.1, stratify=y_tr, random_state=seed)
    m, _ = train_nn_legacy(Xs[tr], y_tr[tr], i_tr[tr], Xs[va], y_tr[va], i_tr[va],
                            lam, pos_weight, in_dim=Xs.shape[1], seed=seed)
    m.eval()
    with torch.no_grad():
        lb, _ = m(torch.tensor(Xte_s, device=DEVICE))
        p_te = torch.sigmoid(lb).cpu().numpy()
    return oof, p_te, cv_auc


# ─── classical + tree baselines ─────────────────────────────────────────
def sklearn_cv_and_fit(name, make_model, needs_scale=False, seed=RS):
    if needs_scale:
        scaler = StandardScaler().fit(X_tr_raw)
        X_tr = scaler.transform(X_tr_raw); X_te = scaler.transform(X_te_raw)
    else:
        X_tr = X_tr_raw; X_te = X_te_raw
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_tr))
    for f, (tr, va) in enumerate(skf.split(X_tr, y_tr)):
        m = make_model(); m.fit(X_tr[tr], y_tr[tr])
        oof[va] = m.predict_proba(X_tr[va])[:, 1]
    cv = roc_auc_score(y_tr, oof)
    print(f"── {name} CV AUC={cv:.4f}")
    m = make_model().fit(X_tr, y_tr)
    p_te = m.predict_proba(X_te)[:, 1]
    return oof, p_te, cv


# ─── run all models ─────────────────────────────────────────────────────
oof_probs = {}; test_probs = {}; rows = []
spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

# classical
oof, p_te, _ = sklearn_cv_and_fit("LR", lambda: LogisticRegression(
    max_iter=5000, C=1.0, class_weight="balanced", random_state=RS),
    needs_scale=True)
oof_probs["LR"] = oof; test_probs["LR"] = p_te

oof, p_te, _ = sklearn_cv_and_fit("RF", lambda: RandomForestClassifier(
    n_estimators=400, max_depth=10, min_samples_leaf=5,
    class_weight="balanced_subsample", n_jobs=-1, random_state=RS))
oof_probs["RF"] = oof; test_probs["RF"] = p_te

oof, p_te, _ = sklearn_cv_and_fit("SVM-RBF", lambda: SVC(
    C=1.0, gamma="scale", probability=True, class_weight="balanced",
    random_state=RS), needs_scale=True)
oof_probs["SVM"] = oof; test_probs["SVM"] = p_te

# GBDT
oof, p_te, _ = sklearn_cv_and_fit("XGB", lambda: xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
    reg_lambda=1.0, scale_pos_weight=spw, eval_metric="logloss",
    random_state=RS, n_jobs=-1, verbosity=0))
oof_probs["XGB"] = oof; test_probs["XGB"] = p_te

oof, p_te, _ = sklearn_cv_and_fit("LGB", lambda: lgb.LGBMClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, min_child_samples=20,
    is_unbalance=True, random_state=RS, n_jobs=-1, verbosity=-1))
oof_probs["LGB"] = oof; test_probs["LGB"] = p_te

oof, p_te, _ = sklearn_cv_and_fit("CAT", lambda: cat.CatBoostClassifier(
    iterations=400, depth=5, learning_rate=0.05,
    auto_class_weights="Balanced", random_seed=RS, verbose=0))
oof_probs["CAT"] = oof; test_probs["CAT"] = p_te

# NN
oof, p_te, _ = nn_cv_and_fit("MLP (binary)", lam=1.0)
oof_probs["MLP"] = oof; test_probs["MLP"] = p_te

oof, p_te, _ = nn_cv_and_fit("MT-MLP (lam=0.7)", lam=0.7)
oof_probs["MT"] = oof; test_probs["MT"] = p_te

# Voting
oof_probs["VOTE3"] = (oof_probs["XGB"] + oof_probs["LGB"] + oof_probs["CAT"]) / 3
test_probs["VOTE3"] = (test_probs["XGB"] + test_probs["LGB"] + test_probs["CAT"]) / 3

oof_probs["VOTE4"] = (oof_probs["MT"] + oof_probs["XGB"] + oof_probs["LGB"] + oof_probs["CAT"]) / 4
test_probs["VOTE4"] = (test_probs["MT"] + test_probs["XGB"] + test_probs["LGB"] + test_probs["CAT"]) / 4

# ICM-Net (full)
oof, p_te, icm_model, icm_scaler, icm_cv = icmnet_cv_and_fit(
    "ICM-Net (full)", use_cascade=True, use_mixup=True, use_sce=True)
oof_probs["ICM"] = oof; test_probs["ICM"] = p_te

# ICM-Net ablations
oof_nc, p_te_nc, _, _, _ = icmnet_cv_and_fit(
    "ICM-Net −cascade", use_cascade=False, use_mixup=True, use_sce=True)
oof_probs["ICM_NC"] = oof_nc; test_probs["ICM_NC"] = p_te_nc

oof_nm, p_te_nm, _, _, _ = icmnet_cv_and_fit(
    "ICM-Net −Mixup", use_cascade=True, use_mixup=False, use_sce=True)
oof_probs["ICM_NM"] = oof_nm; test_probs["ICM_NM"] = p_te_nm

oof_ns, p_te_ns, _, _, _ = icmnet_cv_and_fit(
    "ICM-Net −SCE", use_cascade=True, use_mixup=True, use_sce=False)
oof_probs["ICM_NS"] = oof_ns; test_probs["ICM_NS"] = p_te_ns


# ─── isotonic calibration on OOF + metrics at F1-optimal CV threshold ───
panel = []
ablation = []
name_map = {
    "LR": "LogReg", "RF": "RandomForest", "SVM": "SVM-RBF",
    "XGB": "XGBoost", "LGB": "LightGBM", "CAT": "CatBoost",
    "MLP": "Vanilla MLP", "MT": "MT-MLP (lam=0.7)",
    "VOTE3": "Voting (3 trees)", "VOTE4": "Voting (MT+3 trees)",
    "ICM": "ICM-Net (full)", "ICM_NC": "ICM-Net −cascade",
    "ICM_NM": "ICM-Net −Mixup", "ICM_NS": "ICM-Net −SCE",
}
cal = {}
for k in oof_probs:
    iso = IsotonicRegression(out_of_bounds="clip").fit(oof_probs[k], y_tr)
    oof_cal = iso.predict(oof_probs[k])
    test_cal = iso.predict(test_probs[k])
    thr = best_f1_threshold(y_tr, oof_cal)
    row = metrics_row(name_map[k], y_te, test_cal, thr)
    cal[k] = test_cal
    if k.startswith("ICM"):
        ablation.append(row)
    panel.append(row)

panel_df = pd.DataFrame(panel).sort_values("AUC", ascending=False).reset_index(drop=True)
abl_df = pd.DataFrame(ablation).sort_values("AUC", ascending=False).reset_index(drop=True)
panel_df.to_csv(OUT_TABLES / "table7_icmnet_panel.csv", index=False)
abl_df.to_csv(OUT_TABLES / "table7_icmnet_ablation.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table7_icmnet_panel.csv'}")
print(panel_df.to_string(index=False,
      formatters={c: "{:.3f}".format for c in panel_df.select_dtypes("float").columns}))

# ─── DeLong: ICM-Net (full) vs every baseline ───────────────────────────
rows_d = []
icm_p = cal["ICM"]
for k, p in cal.items():
    if k == "ICM":
        continue
    auc_a, auc_b, z, p_val = delong_test(y_te, icm_p, p)
    rows_d.append({
        "baseline": name_map[k],
        "AUC_ICM": auc_a, "AUC_baseline": auc_b,
        "ΔAUC": auc_a - auc_b,
        "z": z, "p_value": p_val,
        "significant_05": p_val < 0.05,
    })
dlg = pd.DataFrame(rows_d).sort_values("p_value").reset_index(drop=True)
dlg.to_csv(OUT_TABLES / "table7_delong.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table7_delong.csv'}")
print(dlg.to_string(index=False,
      formatters={c: "{:.4f}".format for c in dlg.select_dtypes("float").columns}))

# ─── figure: ROC + PR overlays, ICM-Net highlighted ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
order = ["LR", "SVM", "RF", "XGB", "LGB", "CAT", "MLP", "MT", "VOTE3", "VOTE4", "ICM"]
colors = {"ICM": "crimson"}
for k in order:
    p = cal[k]
    fpr, tpr, _ = roc_curve(y_te, p)
    lw = 2.6 if k == "ICM" else 1.2
    alpha = 1.0 if k == "ICM" else 0.55
    c = colors.get(k, None)
    axes[0].plot(fpr, tpr, label=f"{name_map[k]}  {roc_auc_score(y_te, p):.3f}",
                  linewidth=lw, alpha=alpha, color=c)
axes[0].plot([0, 1], [0, 1], "--", c="grey", alpha=0.5)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test, calibrated)")
axes[0].legend(fontsize=7, loc="lower right")
for k in order:
    p = cal[k]
    pr, rc, _ = precision_recall_curve(y_te, p)
    lw = 2.6 if k == "ICM" else 1.2
    alpha = 1.0 if k == "ICM" else 0.55
    c = colors.get(k, None)
    axes[1].plot(rc, pr, label=f"{name_map[k]}  {average_precision_score(y_te, p):.3f}",
                  linewidth=lw, alpha=alpha, color=c)
axes[1].axhline(y_te.mean(), ls="--", c="grey", alpha=0.5)
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR (test, calibrated)")
axes[1].legend(fontsize=7, loc="upper right")
plt.tight_layout()
fig_path = OUT_FIGS / "fig7_icmnet_roc.png"
plt.savefig(fig_path, dpi=140, bbox_inches="tight")
print(f"wrote {fig_path}")

# ─── persist ICM-Net (calibrated) for downstream CF work ────────────────
iso_icm = IsotonicRegression(out_of_bounds="clip").fit(oof_probs["ICM"], y_tr)
model_path = OUT_MODELS / "icmnet_calibrated.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "state_dict": {k: v.cpu() for k, v in icm_model.state_dict().items()},
        "scaler_mean": icm_scaler.mean_,
        "scaler_scale": icm_scaler.scale_,
        "feat_cols": feat_cols,
        "isotonic": iso_icm,
        "in_dim": len(feat_cols),
        "thr_f1": best_f1_threshold(y_tr, iso_icm.predict(oof_probs["ICM"])),
    }, f)
print(f"wrote {model_path}")

with open(OUT_MODELS / "icmnet_meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "cv_auc": icm_cv,
        "test_auc_raw": roc_auc_score(y_te, test_probs["ICM"]),
        "test_auc_calibrated": roc_auc_score(y_te, cal["ICM"]),
        "delong_vs_top": dlg.iloc[0].to_dict(),
        "feat_cols": feat_cols,
        "hyperparams": {
            "alpha_mix": ALPHA_MIX, "sce_alpha": SCE_ALPHA, "sce_beta": SCE_BETA,
            "lam_bin": LAM_BIN, "lam_ord": LAM_ORD,
            "epochs_s1": EPOCHS_S1, "epochs_s2": EPOCHS_S2, "epochs_s3": EPOCHS_S3,
        },
    }, f, ensure_ascii=False, indent=2, default=float)
