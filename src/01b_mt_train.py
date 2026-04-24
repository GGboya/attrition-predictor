"""Phase 1 trainer/evaluator.

Compares on the frozen 80/20 split (leak-free: 离职意向 NEVER enters X):

  A. Single-task binary MLP (same encoder, binary head only)
  B. MT-MLP (shared encoder + binary + ordinal intent head, CORN)
      — tune lambda in {0.3, 0.5, 0.7} via 5-fold CV
  C. XGBoost  (scale_pos_weight)
  D. LightGBM (is_unbalance=True)
  E. CatBoost (auto_class_weights="Balanced")
  F. Voting ensemble of C/D/E

All probabilities calibrated with isotonic regression fit on CV OOF probs.
Metrics reported at F1-optimal threshold picked on CV OOF (not test).

Outputs:
  src/tables/table4_base_classifier.csv
  src/figures/fig2_base_roc_pr.png
  models/mt_mlp_calibrated.pkl   (state_dict + scaler + isotonic calibrator)
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

# Import tree libs FIRST so their OpenMP runtimes load before torch's.
import xgboost as xgb
import lightgbm as lgb
import catboost as cat

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import VotingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    f1_score, matthews_corrcoef, precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import importlib.util, sys
_spec = importlib.util.spec_from_file_location("mt_model", Path(__file__).with_name("01a_mt_model.py"))
_m = importlib.util.module_from_spec(_spec); sys.modules["mt_model"] = _m; _spec.loader.exec_module(_m)
MTMlp = _m.MTMlp
joint_loss = _m.joint_loss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RS = 42
TARGET = "离职行为"
INTENT = "离职意向"
N_FOLDS = 5
EPOCHS = 200
BATCH_SIZE = 256
LR = 1e-3
PATIENCE = 20
LAMBDAS = [0.3, 0.5, 0.7]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)
OUT_MODELS = Path("models"); OUT_MODELS.mkdir(exist_ok=True, parents=True)

torch.manual_seed(RS)
np.random.seed(RS)


# ─── data ──────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")

feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]
print(f"clean {df.shape}  train {len(train_idx)}  test {len(test_idx)}  "
      f"features: {len(feat_cols)} (no intent)")

y = df[TARGET].values.astype(int)
intent_bin = np.clip(df[INTENT].round().values.astype(int), 1, 5) - 1  # → {0..4}

X_raw = df[feat_cols].values.astype(np.float32)
X_tr_raw, X_te_raw = X_raw[train_idx], X_raw[test_idx]
y_tr, y_te = y[train_idx], y[test_idx]
i_tr, i_te = intent_bin[train_idx], intent_bin[test_idx]


# ─── helpers ───────────────────────────────────────────────────────────
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


# ─── MT-MLP trainer ────────────────────────────────────────────────────
class EarlyStop:
    def __init__(self, patience=PATIENCE):
        self.best = -np.inf
        self.bad = 0
        self.patience = patience
        self.best_state = None

    def step(self, score, state):
        if score > self.best + 1e-6:
            self.best = score
            self.bad = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in state.items()}
            return False
        self.bad += 1
        return self.bad > self.patience


def train_nn(X_tr, y_tr, i_tr, X_val, y_val, i_val, lam: float | None,
             pos_weight: float, in_dim: int, max_epochs=EPOCHS):
    """Train one MT-MLP (if lam<1) or single-task MLP (if lam=None).
    Returns best state dict (by val AUC) and best val AUC."""
    use_mt = lam is not None
    model = MTMlp(in_dim=in_dim, n_ord_classes=5).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    stopper = EarlyStop(PATIENCE)

    Xt = torch.tensor(X_tr, device=DEVICE)
    yt = torch.tensor(y_tr, device=DEVICE)
    it = torch.tensor(i_tr, device=DEVICE, dtype=torch.long)
    Xv = torch.tensor(X_val, device=DEVICE)

    n = len(Xt)
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for b in range(0, n, BATCH_SIZE):
            idx = perm[b:b + BATCH_SIZE]
            xb, yb, ib = Xt[idx], yt[idx], it[idx]
            lb, lo = model(xb)
            if use_mt:
                loss = joint_loss(lb, lo, yb, ib, lam=lam,
                                  pos_weight=pos_weight, n_ord_classes=5)
            else:
                pw = torch.tensor([pos_weight], device=DEVICE)
                loss = nn.functional.binary_cross_entropy_with_logits(
                    lb, yb.float(), pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
        # val AUC on binary head
        model.eval()
        with torch.no_grad():
            lb, _ = model(Xv)
            p_val = torch.sigmoid(lb).cpu().numpy()
        val_auc = roc_auc_score(y_val, p_val)
        if stopper.step(val_auc, model.state_dict()):
            break
    model.load_state_dict(stopper.best_state)
    return model, stopper.best


def predict_nn(model, X):
    model.eval()
    Xt = torch.tensor(X, device=DEVICE)
    with torch.no_grad():
        lb, _ = model(Xt)
        p = torch.sigmoid(lb).cpu().numpy()
    return p


# ─── CV pipeline for NN (tune lambda for MT, or single-task) ──────────
def nn_cv_and_fit(name: str, lam: float | None, seed: int = RS):
    """Return: oof probs on train, test probs (from one model retrained on full train)."""
    print(f"\n── {name}  lam={lam} ──")
    # standardise
    scaler = StandardScaler().fit(X_tr_raw)
    Xs = scaler.transform(X_tr_raw).astype(np.float32)
    Xte_s = scaler.transform(X_te_raw).astype(np.float32)
    pos_weight = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_tr))
    for f, (tr, va) in enumerate(skf.split(Xs, y_tr)):
        m, best = train_nn(Xs[tr], y_tr[tr], i_tr[tr], Xs[va], y_tr[va], i_tr[va],
                           lam, pos_weight, in_dim=Xs.shape[1])
        oof[va] = predict_nn(m, Xs[va])
        print(f"  fold {f} val AUC={best:.4f}")
    cv_auc = roc_auc_score(y_tr, oof)
    print(f"  CV AUC={cv_auc:.4f}")

    # retrain on full train (80/20 inner split for early-stop only)
    from sklearn.model_selection import train_test_split
    tr, va = train_test_split(np.arange(len(y_tr)), test_size=0.1,
                               stratify=y_tr, random_state=seed)
    model, _ = train_nn(Xs[tr], y_tr[tr], i_tr[tr], Xs[va], y_tr[va], i_tr[va],
                        lam, pos_weight, in_dim=Xs.shape[1])
    p_test = predict_nn(model, Xte_s)
    return oof, p_test, model, scaler, cv_auc


def tree_cv_and_fit(name: str, make_model, seed: int = RS):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    oof = np.zeros(len(y_tr))
    for f, (tr, va) in enumerate(skf.split(X_tr_raw, y_tr)):
        m = make_model()
        m.fit(X_tr_raw[tr], y_tr[tr])
        oof[va] = m.predict_proba(X_tr_raw[va])[:, 1]
    print(f"── {name} CV AUC={roc_auc_score(y_tr, oof):.4f}")
    m = make_model().fit(X_tr_raw, y_tr)
    p_test = m.predict_proba(X_te_raw)[:, 1]
    return oof, p_test


# ─── run everything ────────────────────────────────────────────────────
results = []
test_probs = {}
oof_probs = {}

# single-task MLP (same encoder, binary head only — lam=1.0 in our formulation
# reduces MT to binary-only because ordinal-term weight is 0)
oof, p_te, _, _, cv_auc = nn_cv_and_fit("Single-task MLP (binary only)", lam=1.0)
thr = best_f1_threshold(y_tr, oof)
results.append(metrics_row("MLP (binary only, no intent)", y_te, p_te, thr))
oof_probs["MLP"] = oof; test_probs["MLP"] = p_te

# MT-MLP — scan lambda
best_lam, best_cv = None, -np.inf
best_oof, best_p_te, best_model, best_scaler = None, None, None, None
for lam in LAMBDAS:
    oof, p_te, model, scaler, cv_auc = nn_cv_and_fit(f"MT-MLP (lam={lam})", lam=lam)
    if cv_auc > best_cv:
        best_cv, best_lam = cv_auc, lam
        best_oof, best_p_te = oof, p_te
        best_model, best_scaler = model, scaler
print(f"\n  best lambda = {best_lam}  CV AUC={best_cv:.4f}")
thr = best_f1_threshold(y_tr, best_oof)
results.append(metrics_row(f"MT-MLP (lam={best_lam}, no intent)", y_te, best_p_te, thr))
oof_probs["MT"] = best_oof; test_probs["MT"] = best_p_te

# trees
spw = (y_tr == 0).sum() / max(1, (y_tr == 1).sum())
oof, p_te = tree_cv_and_fit("XGBoost", lambda: xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, min_child_weight=3,
    reg_lambda=1.0, scale_pos_weight=spw, eval_metric="logloss",
    random_state=RS, n_jobs=-1, verbosity=0))
thr = best_f1_threshold(y_tr, oof)
results.append(metrics_row("XGBoost", y_te, p_te, thr))
oof_probs["XGB"] = oof; test_probs["XGB"] = p_te

oof, p_te = tree_cv_and_fit("LightGBM", lambda: lgb.LGBMClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.9, min_child_samples=20,
    is_unbalance=True, random_state=RS, n_jobs=-1, verbosity=-1))
thr = best_f1_threshold(y_tr, oof)
results.append(metrics_row("LightGBM", y_te, p_te, thr))
oof_probs["LGB"] = oof; test_probs["LGB"] = p_te

oof, p_te = tree_cv_and_fit("CatBoost", lambda: cat.CatBoostClassifier(
    iterations=400, depth=5, learning_rate=0.05,
    auto_class_weights="Balanced", random_seed=RS, verbose=0))
thr = best_f1_threshold(y_tr, oof)
results.append(metrics_row("CatBoost", y_te, p_te, thr))
oof_probs["CAT"] = oof; test_probs["CAT"] = p_te

# voting (avg of XGB/LGB/CAT)
oof_vote = (oof_probs["XGB"] + oof_probs["LGB"] + oof_probs["CAT"]) / 3
p_vote = (test_probs["XGB"] + test_probs["LGB"] + test_probs["CAT"]) / 3
thr = best_f1_threshold(y_tr, oof_vote)
results.append(metrics_row("Voting (XGB+LGB+CAT)", y_te, p_vote, thr))
oof_probs["VOTE"] = oof_vote; test_probs["VOTE"] = p_vote

# MT + trees voting
oof_mega = (best_oof + oof_probs["XGB"] + oof_probs["LGB"] + oof_probs["CAT"]) / 4
p_mega = (best_p_te + test_probs["XGB"] + test_probs["LGB"] + test_probs["CAT"]) / 4
thr = best_f1_threshold(y_tr, oof_mega)
results.append(metrics_row("Voting (MT+XGB+LGB+CAT)", y_te, p_mega, thr))


# ─── table ─────────────────────────────────────────────────────────────
out = pd.DataFrame(results)
out_path = OUT_TABLES / "table4_base_classifier.csv"
out.to_csv(out_path, index=False)
print(f"\nwrote {out_path}")
print(out.to_string(index=False,
                    formatters={c: "{:.3f}".format for c in out.select_dtypes("float").columns}))


# ─── figure ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for name, p in test_probs.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    axes[0].plot(fpr, tpr, label=f"{name}  AUC={roc_auc_score(y_te, p):.3f}")
axes[0].plot([0, 1], [0, 1], "--", c="grey", alpha=0.5)
axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC (test)")
axes[0].legend(fontsize=8)
for name, p in test_probs.items():
    pr, rc, _ = precision_recall_curve(y_te, p)
    axes[1].plot(rc, pr, label=f"{name}  PR-AUC={average_precision_score(y_te, p):.3f}")
axes[1].axhline(y_te.mean(), ls="--", c="grey", alpha=0.5, label=f"base={y_te.mean():.3f}")
axes[1].set(xlabel="Recall", ylabel="Precision", title="PR (test)")
axes[1].legend(fontsize=8)
plt.tight_layout()
fig_path = OUT_FIGS / "fig2_base_roc_pr.png"
plt.savefig(fig_path, dpi=140)
print(f"wrote {fig_path}")


# ─── calibrate MT-MLP on its own OOF, save for Phase 4 ─────────────────
iso = IsotonicRegression(out_of_bounds="clip").fit(best_oof, y_tr)
p_test_cal = iso.predict(best_p_te)
print(f"\n[calibration] MT test AUC raw={roc_auc_score(y_te, best_p_te):.4f}  "
      f"cal={roc_auc_score(y_te, p_test_cal):.4f}  "
      f"Brier raw={brier_score_loss(y_te, best_p_te):.4f}  "
      f"cal={brier_score_loss(y_te, p_test_cal):.4f}")

model_path = OUT_MODELS / "mt_mlp_calibrated.pkl"
with open(model_path, "wb") as f:
    pickle.dump({
        "state_dict": {k: v.cpu() for k, v in best_model.state_dict().items()},
        "scaler_mean": best_scaler.mean_,
        "scaler_scale": best_scaler.scale_,
        "feat_cols": feat_cols,
        "isotonic": iso,
        "best_lam": best_lam,
        "in_dim": len(feat_cols),
        "thr_f1": best_f1_threshold(y_tr, iso.predict(best_oof)),
    }, f)
print(f"wrote {model_path}")

with open(OUT_MODELS / "mt_mlp_meta.json", "w", encoding="utf-8") as f:
    json.dump({
        "best_lambda": best_lam,
        "cv_auc": best_cv,
        "test_auc_raw": roc_auc_score(y_te, best_p_te),
        "test_auc_calibrated": roc_auc_score(y_te, p_test_cal),
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "feat_cols": feat_cols,
    }, f, ensure_ascii=False, indent=2)
