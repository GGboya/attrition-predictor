"""Phase 7.5 — Threshold re-selection for Bal-Acc.

Phase 6 champion (5-base stack + CL-v6) was tuned for argmax(F1) giving
thr=0.19, F1=0.4252, Bal_Acc=0.7107. This script reproduces those calibrated
probabilities, then sweeps alternative threshold criteria on CV OOF and
reports the trade-off on held-out test:

  F1       — Phase 6 default, argmax F1
  Bal_Acc  — argmax balanced accuracy   (the metric we want to push)
  MCC      — argmax Matthews corrcoef   (penalises false positives more)
  Youden J — argmax (Sens + Spec - 1)   (equivalent to Bal_Acc up to affine)

The MODEL is unchanged; only the decision threshold moves. Hence this is
a free lunch for whichever metric we optimise — at the cost of the others.

Outputs
-------
src/tables/table14_threshold_sweep.csv
data/processed/phase6_meta_oof_probs.npy   calibrated CV-OOF probs (4375,)
data/processed/phase6_meta_test_probs.npy  calibrated test probs (1094,)
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import importlib.util, sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import torch


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

_mtmod = _load("mt_model", "01a_mt_model.py")
MTMlp = _mtmod.MTMlp
joint_loss = _mtmod.joint_loss

RS = 42
TARGET = "离职行为"
INTENT = "离职意向"
N_FOLDS = 5
N_MT_SEEDS = 5
DEVICE = torch.device("cpu")

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_PROC = Path("data/processed")

np.random.seed(RS)
torch.manual_seed(RS)


# ─── helpers (verbatim from 07) ─────────────────────────────────────────
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


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


# ─── threshold sweep ────────────────────────────────────────────────────
def _sweep_best(y, p, metric_fn, lo=0.02, hi=0.98, step=0.005):
    best_val, best_thr = -np.inf, 0.5
    for t in np.arange(lo, hi, step):
        pred = (p >= t).astype(int)
        if pred.sum() == 0 or pred.sum() == len(pred): continue
        v = metric_fn(y, pred)
        if v > best_val: best_val, best_thr = v, t
    return float(best_thr), float(best_val)


def _youden_j(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens + spec - 1.0


CRITERIA = [
    ("F1", lambda y, p: f1_score(y, p, pos_label=1, zero_division=0)),
    ("Bal_Acc", balanced_accuracy_score),
    ("MCC", matthews_corrcoef),
    ("Youden_J", _youden_j),
]


def _eval_at_threshold(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {
        "thr": thr,
        "F1": f1_score(y, pred, zero_division=0),
        "Bal_Acc": balanced_accuracy_score(y, pred),
        "MCC": matthews_corrcoef(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall (Sens)": sens,
        "Specificity": spec,
        "Pred_Pos_Rate": pred.mean(),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


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

sw_v6 = np.load(OUT_PROC / "sample_weights_v6.npy")
print(f"clean {df.shape}  train {len(train_idx)}  test {len(test_idx)}  "
      f"n_feat {len(feat_cols)}  MT seeds {N_MT_SEEDS}")
print(f"CL weights flagged={(sw_v6 < 1.0).sum()}  w_flag={sw_v6[sw_v6<1.0].min():.3f}")

skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
fold_idx = list(skf.split(X_tr, y_tr))
N_tr = len(y_tr); in_dim = X_tr.shape[1]
spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

BASES = ["RF", "NRBoost", "MTMLP", "SVM", "ET"]
WEIGHT_SUPPORTED = {"RF", "NRBoost", "SVM", "ET"}
oof = {k: np.zeros(N_tr) for k in BASES}
test_preds = {k: np.zeros(len(y_te)) for k in BASES}
w_train = sw_v6


# ─── 5-fold OOF for 5 bases ─────────────────────────────────────────────
print("\n" + "=" * 72)
print("  STEP 1: 5-fold OOF (5 Phase-6 bases)")
print("=" * 72)
for f, (tr, va) in enumerate(fold_idx):
    print(f"\nfold {f}:", flush=True)
    Xtr_f, Xva_f = X_tr[tr], X_tr[va]
    ytr_f = y_tr[tr]; yo_tr_f = y_ord_tr[tr]
    w_f = w_train[tr]

    scaler = StandardScaler().fit(Xtr_f)
    Xtr_s = scaler.transform(Xtr_f).astype(np.float32)
    Xva_s = scaler.transform(Xva_f).astype(np.float32)

    rf = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=RS)
    rf.fit(Xtr_f, ytr_f, sample_weight=w_f)
    oof["RF"][va] = rf.predict_proba(Xva_f)[:, 1]
    print("  RF done", flush=True)

    b = fit_nrboost(Xtr_f, ytr_f, q=0.7, n_rounds=400, n_stages=2,
                    drop_frac=0.10, seed=RS + f, init_weight=w_f)
    oof["NRBoost"][va] = predict_nrboost(b, Xva_f)
    print("  NRBoost done", flush=True)

    oof["MTMLP"][va] = fit_predict_mtmlp_seedavg(
        Xtr_s, ytr_f, yo_tr_f, Xva_s, in_dim=in_dim,
        pos_weight=spw, base_seed=RS + f)
    print(f"  MT×{N_MT_SEEDS} done", flush=True)

    svm = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
              probability=True, random_state=RS)
    svm.fit(Xtr_s, ytr_f, sample_weight=w_f)
    oof["SVM"][va] = svm.predict_proba(Xva_s)[:, 1]
    print("  SVM done", flush=True)

    et = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=RS)
    et.fit(Xtr_f, ytr_f, sample_weight=w_f)
    oof["ET"][va] = et.predict_proba(Xva_f)[:, 1]
    print("  ET done", flush=True)


# ─── full-train refits → test predictions ──────────────────────────────
print("\n" + "=" * 72)
print("  STEP 2: full-train refits → test")
print("=" * 72)

scaler_full = StandardScaler().fit(X_tr)
Xtr_s_full = scaler_full.transform(X_tr).astype(np.float32)
Xte_s_full = scaler_full.transform(X_te).astype(np.float32)

rf_full = RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                  class_weight="balanced_subsample",
                                  n_jobs=-1, random_state=RS)
rf_full.fit(X_tr, y_tr, sample_weight=w_train)
test_preds["RF"] = rf_full.predict_proba(X_te)[:, 1]

b_full = fit_nrboost(X_tr, y_tr, q=0.7, n_rounds=400, n_stages=2,
                     drop_frac=0.10, seed=RS, init_weight=w_train)
test_preds["NRBoost"] = predict_nrboost(b_full, X_te)

test_preds["MTMLP"] = fit_predict_mtmlp_seedavg(
    Xtr_s_full, y_tr, y_ord_tr, Xte_s_full, in_dim=in_dim,
    pos_weight=spw, base_seed=RS)

svm_full = SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
               probability=True, random_state=RS)
svm_full.fit(Xtr_s_full, y_tr, sample_weight=w_train)
test_preds["SVM"] = svm_full.predict_proba(Xte_s_full)[:, 1]

et_full = ExtraTreesClassifier(n_estimators=400, max_depth=12, min_samples_leaf=3,
                                class_weight="balanced_subsample",
                                n_jobs=-1, random_state=RS)
et_full.fit(X_tr, y_tr, sample_weight=w_train)
test_preds["ET"] = et_full.predict_proba(X_te)[:, 1]

print("── per-base AUCs ──")
for k in BASES:
    print(f"  {k:8s}  CV AUC={roc_auc_score(y_tr, oof[k]):.4f}  "
          f"test AUC={roc_auc_score(y_te, test_preds[k]):.4f}")


# ─── meta L2-LR C=10 → isotonic calibration ────────────────────────────
print("\n" + "=" * 72)
print("  STEP 3: meta L2-LR C=10, isotonic calibrate, threshold sweep")
print("=" * 72)

Z_tr = _logit(np.column_stack([oof[k] for k in BASES]))
Z_te = _logit(np.column_stack([test_preds[k] for k in BASES]))

m_full = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs",
                             max_iter=2000, random_state=RS).fit(Z_tr, y_tr)
oof_meta = np.zeros(N_tr)
for tr, va in fold_idx:
    mi = LogisticRegression(C=10.0, penalty="l2", solver="lbfgs",
                             max_iter=2000, random_state=RS).fit(Z_tr[tr], y_tr[tr])
    oof_meta[va] = mi.predict_proba(Z_tr[va])[:, 1]
test_meta = m_full.predict_proba(Z_te)[:, 1]

iso = IsotonicRegression(out_of_bounds="clip").fit(oof_meta, y_tr)
oof_cal = iso.predict(oof_meta)
test_cal = iso.predict(test_meta)

auc_oof = roc_auc_score(y_tr, oof_cal)
auc_test = roc_auc_score(y_te, test_cal)
print(f"\n  meta CV AUC = {auc_oof:.4f}   test AUC = {auc_test:.4f}")

# Persist calibrated probs for reuse
np.save(OUT_PROC / "phase6_meta_oof_probs.npy", oof_cal)
np.save(OUT_PROC / "phase6_meta_test_probs.npy", test_cal)
# Also cache pre-calibration meta probs so calibrator experiments can
# swap calibrators without rerunning the full stack pipeline.
np.save(OUT_PROC / "phase6_meta_oof_raw.npy", oof_meta)
np.save(OUT_PROC / "phase6_meta_test_raw.npy", test_meta)
print(f"  saved  data/processed/phase6_meta_{{oof,test}}_probs.npy (isotonic)")
print(f"  saved  data/processed/phase6_meta_{{oof,test}}_raw.npy (pre-calibration)")


# ─── threshold sweep: argmax on CV OOF, evaluate on test ────────────────
print("\n── threshold sweep on CV OOF ──")
rows = []
for name, fn in CRITERIA:
    thr, val_oof = _sweep_best(y_tr, oof_cal, fn)
    oof_metrics = _eval_at_threshold(y_tr, oof_cal, thr)
    test_metrics = _eval_at_threshold(y_te, test_cal, thr)
    row = {"criterion": name, "thr": thr}
    for k, v in oof_metrics.items():
        if k == "thr": continue
        row[f"OOF_{k}"] = v
    for k, v in test_metrics.items():
        if k == "thr": continue
        row[f"TEST_{k}"] = v
    rows.append(row)
    print(f"\n  argmax {name} on CV OOF → thr={thr:.3f}")
    print(f"    OOF  : F1={oof_metrics['F1']:.4f}  "
          f"Bal_Acc={oof_metrics['Bal_Acc']:.4f}  MCC={oof_metrics['MCC']:.4f}  "
          f"Sens={oof_metrics['Recall (Sens)']:.3f}  Spec={oof_metrics['Specificity']:.3f}  "
          f"PPR={oof_metrics['Pred_Pos_Rate']:.3f}")
    print(f"    TEST : F1={test_metrics['F1']:.4f}  "
          f"Bal_Acc={test_metrics['Bal_Acc']:.4f}  MCC={test_metrics['MCC']:.4f}  "
          f"Sens={test_metrics['Recall (Sens)']:.3f}  Spec={test_metrics['Specificity']:.3f}  "
          f"PPR={test_metrics['Pred_Pos_Rate']:.3f}")

df_out = pd.DataFrame(rows)
out_path = OUT_TABLES / "table14_threshold_sweep.csv"
df_out.to_csv(out_path, index=False)
print(f"\nwrote {out_path}")


# ─── final summary ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  SUMMARY — trade-off across threshold criteria (TEST)")
print("=" * 72)
print(f"  {'criterion':<10s}  {'thr':>6s}  {'F1':>6s}  {'Bal_Acc':>8s}  "
      f"{'MCC':>6s}  {'Sens':>6s}  {'Spec':>6s}  {'PPR':>6s}")
for r in rows:
    print(f"  {r['criterion']:<10s}  {r['thr']:>6.3f}  "
          f"{r['TEST_F1']:>6.4f}  {r['TEST_Bal_Acc']:>8.4f}  {r['TEST_MCC']:>6.4f}  "
          f"{r['TEST_Recall (Sens)']:>6.3f}  {r['TEST_Specificity']:>6.3f}  "
          f"{r['TEST_Pred_Pos_Rate']:>6.3f}")
print(f"\n  Phase 6 reference (argmax F1): F1=0.4252, Bal_Acc=0.7107, MCC=0.3187")
