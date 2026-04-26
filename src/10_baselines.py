"""Phase 8 — Standalone baseline family comparison.

Runs 7 standalone baselines on the SAME train/test split as the Phase 6
champion, each through:
    CV 5-fold OOF  →  isotonic calibration  →  full-train refit
    →  test prediction  →  F1 + Bal-Acc optimal thresholds
    →  1000-bootstrap 95% CI  →  DeLong vs champion

Baselines:
    b1 LR (L2, C=1.0)            12d scaled,  CL off, class_weight='balanced'
    b2 RF (independent)          12d raw,     CL on
    b3 XGBoost                   12d raw,     CL on
    b4 LightGBM                  12d raw,     CL on
    b5 CatBoost                  12d raw,     CL on
    b6 kNN (k=50, dist-weighted) 12d scaled,  CL off
    b7 SVM-RBF                   12d scaled,  CL on

Outputs
-------
src/tables/table17_baselines_panel.csv
src/tables/table17_baselines_delong.csv
src/tables/table17_baselines_calibration.csv
src/figures/fig17_baselines_roc.png
src/figures/fig17_baselines_forest.png
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, balanced_accuracy_score, brier_score_loss,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_recall_curve, roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _load(name, fn):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(fn))
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod; spec.loader.exec_module(mod)
    return mod

_utils = _load("stats_utils", "_utils.py")
delong_test = _utils.delong_test


RS = 42
N_FOLDS = 5
N_BOOT = 1000
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
N_tr, N_te = len(y_tr), len(y_te)

sw_v6 = np.load("data/processed/sample_weights_v6.npy").astype(np.float32)
assert len(sw_v6) == N_tr, f"sample_weights_v6 shape mismatch: {sw_v6.shape} vs N_tr={N_tr}"

champion_test = np.load("data/processed/phase6_meta_test_probs.npy")
champion_oof = np.load("data/processed/phase6_meta_oof_probs.npy")
assert champion_test.shape == (N_te,)
assert champion_oof.shape == (N_tr,)

print(f"train={N_tr}  test={N_te}  pos_rate_test={y_te.mean():.3f}")
print(f"champion   oof AUC={roc_auc_score(y_tr, champion_oof):.4f}  "
      f"test AUC={roc_auc_score(y_te, champion_test):.4f}")


# ─── helpers ────────────────────────────────────────────────────────────
def best_threshold(y, p, metric_fn):
    """Grid search τ ∈ [0.05, 0.95) by 0.01 maximising metric_fn(y, pred)."""
    best, thr = -1.0, 0.5
    for t in np.arange(0.05, 0.95, 0.01):
        pred = (p >= t).astype(int)
        try:
            v = metric_fn(y, pred)
        except Exception:
            continue
        if v > best:
            best, thr = v, float(t)
    return thr


def bootstrap_ci(y, p, metric, n=N_BOOT, seed=RS):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            vals.append(metric(y[idx], p[idx]))
        except Exception:
            pass
    a = np.asarray(vals)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


def sens_spec(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec


def ece_mce(y, p, n_bins=10):
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0] -= 1e-9; q[-1] += 1e-9
    q = np.unique(q)
    ids = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    N = len(y); ece = 0.0; mce = 0.0
    for b in range(len(q) - 1):
        m = ids == b
        if m.sum() == 0: continue
        gap = abs(p[m].mean() - y[m].mean())
        ece += (m.sum() / N) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


# ─── baseline registry ──────────────────────────────────────────────────
def make_lr():
    return LogisticRegression(penalty="l2", C=1.0, class_weight="balanced",
                              solver="lbfgs", max_iter=2000, random_state=RS)


def make_rf():
    return RandomForestClassifier(n_estimators=500, max_depth=12,
                                   min_samples_leaf=3, class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=RS)


def make_xgb():
    spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    return xgb.XGBClassifier(
        max_depth=6, learning_rate=0.05, n_estimators=500,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, eval_metric="logloss",
        tree_method="hist", n_jobs=-1, random_state=RS, verbosity=0)


def make_lgb():
    spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    return lgb.LGBMClassifier(
        num_leaves=31, learning_rate=0.05, n_estimators=500,
        subsample=0.9, colsample_bytree=0.9,
        scale_pos_weight=spw, n_jobs=-1,
        random_state=RS, verbosity=-1)


def make_cat():
    spw = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    return CatBoostClassifier(
        depth=6, learning_rate=0.05, iterations=500,
        loss_function="Logloss", scale_pos_weight=spw,
        random_seed=RS, verbose=False,
        allow_writing_files=False, thread_count=-1)


def make_knn():
    return KNeighborsClassifier(n_neighbors=50, weights="distance",
                                 metric="euclidean", n_jobs=-1)


def make_svm():
    return SVC(kernel="rbf", C=1.0, gamma="scale",
               class_weight="balanced", probability=True, random_state=RS)


# (name, factory, feature_space ["raw12" | "scaled12"], use_cl_weights)
BASELINES = [
    ("LR",       make_lr,  "scaled12", False),
    ("RF",       make_rf,  "raw12",    True),
    ("XGB",      make_xgb, "raw12",    True),
    ("LGBM",     make_lgb, "raw12",    True),
    ("CatBoost", make_cat, "raw12",    True),
    ("kNN",      make_knn, "scaled12", False),
    ("SVM",      make_svm, "scaled12", True),
]


# ─── fit+predict loop ───────────────────────────────────────────────────
def _fit_fold(name, factory, Xtr_f, ytr_f, w_f):
    m = factory()
    try:
        if w_f is not None:
            m.fit(Xtr_f, ytr_f, sample_weight=w_f)
        else:
            m.fit(Xtr_f, ytr_f)
    except TypeError:
        m.fit(Xtr_f, ytr_f)
    return m


def _predict_proba(m, X):
    return m.predict_proba(X)[:, 1]


skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
fold_idx = list(skf.split(X_tr, y_tr))

results = {}  # name -> dict(oof_raw, test_raw, oof_cal, test_cal, iso)

for name, factory, featspace, use_cl in BASELINES:
    print(f"\n── {name} ({featspace}, CL={'on' if use_cl else 'off'}) ──")
    oof = np.zeros(N_tr)
    for f, (tr, va) in enumerate(fold_idx):
        if featspace == "scaled12":
            scaler = StandardScaler().fit(X_tr[tr])
            Xtr_f = scaler.transform(X_tr[tr]).astype(np.float32)
            Xva_f = scaler.transform(X_tr[va]).astype(np.float32)
        else:
            Xtr_f = X_tr[tr]; Xva_f = X_tr[va]
        ytr_f = y_tr[tr]
        w_f = sw_v6[tr] if use_cl else None
        m = _fit_fold(name, factory, Xtr_f, ytr_f, w_f)
        oof[va] = _predict_proba(m, Xva_f)
        print(f"  fold {f}: AUC={roc_auc_score(y_tr[va], oof[va]):.4f}")

    if featspace == "scaled12":
        scaler_full = StandardScaler().fit(X_tr)
        Xtr_full = scaler_full.transform(X_tr).astype(np.float32)
        Xte_full = scaler_full.transform(X_te).astype(np.float32)
    else:
        Xtr_full = X_tr; Xte_full = X_te
    w_full = sw_v6 if use_cl else None
    m_full = _fit_fold(name, factory, Xtr_full, y_tr, w_full)
    test_raw = _predict_proba(m_full, Xte_full)

    iso = IsotonicRegression(out_of_bounds="clip").fit(oof, y_tr)
    oof_cal = iso.predict(oof)
    test_cal = iso.predict(test_raw)

    print(f"  CV AUC={roc_auc_score(y_tr, oof):.4f}  "
          f"test AUC (raw)={roc_auc_score(y_te, test_raw):.4f}  "
          f"test AUC (cal)={roc_auc_score(y_te, test_cal):.4f}")

    results[name] = dict(oof_raw=oof, test_raw=test_raw,
                         oof_cal=oof_cal, test_cal=test_cal)


# ─── panel & calibration tables ─────────────────────────────────────────
print("\n" + "=" * 72)
print("  Panel: per-baseline metrics (test, calibrated)")
print("=" * 72)

panel_rows = []
calib_rows = []
for name, *_ in [(n, *rest) for n, *rest in BASELINES]:
    r = results[name]
    oof_cal = r["oof_cal"]; test_cal = r["test_cal"]

    thr_f1 = best_threshold(y_tr, oof_cal, lambda y, p: f1_score(y, p, zero_division=0))
    thr_ba = best_threshold(y_tr, oof_cal, balanced_accuracy_score)

    for crit, thr in [("F1", thr_f1), ("Bal-Acc", thr_ba)]:
        pred = (test_cal >= thr).astype(int)
        sens, spec = sens_spec(y_te, pred)
        auc_lo, auc_hi = bootstrap_ci(y_te, test_cal, roc_auc_score)
        pr_lo, pr_hi = bootstrap_ci(y_te, test_cal, average_precision_score)
        panel_rows.append({
            "baseline": name,
            "threshold_criterion": crit,
            "threshold": thr,
            "AUC": roc_auc_score(y_te, test_cal),
            "AUC_lo": auc_lo, "AUC_hi": auc_hi,
            "PR_AUC": average_precision_score(y_te, test_cal),
            "PR_lo": pr_lo, "PR_hi": pr_hi,
            "F1": f1_score(y_te, pred, zero_division=0),
            "Bal_Acc": balanced_accuracy_score(y_te, pred),
            "MCC": matthews_corrcoef(y_te, pred),
            "Sens": sens, "Spec": spec,
            "Brier": brier_score_loss(y_te, test_cal),
        })

    # calibration once per baseline
    ece, mce = ece_mce(y_te, test_cal, n_bins=10)
    calib_rows.append({
        "baseline": name,
        "AUC": roc_auc_score(y_te, test_cal),
        "ECE": ece, "MCE": mce,
        "Brier": brier_score_loss(y_te, test_cal),
    })

# add champion row for reference
thr_f1_ch = best_threshold(y_tr, champion_oof, lambda y, p: f1_score(y, p, zero_division=0))
thr_ba_ch = best_threshold(y_tr, champion_oof, balanced_accuracy_score)
for crit, thr in [("F1", thr_f1_ch), ("Bal-Acc", thr_ba_ch)]:
    pred = (champion_test >= thr).astype(int)
    sens, spec = sens_spec(y_te, pred)
    auc_lo, auc_hi = bootstrap_ci(y_te, champion_test, roc_auc_score)
    pr_lo, pr_hi = bootstrap_ci(y_te, champion_test, average_precision_score)
    panel_rows.append({
        "baseline": "Phase6-Champion",
        "threshold_criterion": crit,
        "threshold": thr,
        "AUC": roc_auc_score(y_te, champion_test),
        "AUC_lo": auc_lo, "AUC_hi": auc_hi,
        "PR_AUC": average_precision_score(y_te, champion_test),
        "PR_lo": pr_lo, "PR_hi": pr_hi,
        "F1": f1_score(y_te, pred, zero_division=0),
        "Bal_Acc": balanced_accuracy_score(y_te, pred),
        "MCC": matthews_corrcoef(y_te, pred),
        "Sens": sens, "Spec": spec,
        "Brier": brier_score_loss(y_te, champion_test),
    })

ece_ch, mce_ch = ece_mce(y_te, champion_test, n_bins=10)
calib_rows.append({
    "baseline": "Phase6-Champion",
    "AUC": roc_auc_score(y_te, champion_test),
    "ECE": ece_ch, "MCE": mce_ch,
    "Brier": brier_score_loss(y_te, champion_test),
})

pd.DataFrame(panel_rows).to_csv(OUT_TABLES / "table17_baselines_panel.csv", index=False)
pd.DataFrame(calib_rows).to_csv(OUT_TABLES / "table17_baselines_calibration.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table17_baselines_panel.csv'}")
print(f"wrote {OUT_TABLES / 'table17_baselines_calibration.csv'}")


# ─── DeLong vs champion ────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  DeLong: each baseline vs Phase 6 champion (test)")
print("=" * 72)
delong_rows = []
for name, *_ in BASELINES:
    test_cal = results[name]["test_cal"]
    auc_a, auc_b, z, p = delong_test(y_te, test_cal, champion_test)
    print(f"  {name:10s}  AUC={auc_a:.4f}  champion={auc_b:.4f}  "
          f"Δ={auc_a-auc_b:+.4f}  z={z:+.3f}  p={p:.4f}")
    delong_rows.append({
        "baseline": name,
        "AUC_baseline": auc_a, "AUC_champion": auc_b,
        "delta": auc_a - auc_b, "z": z, "p_value": p,
        "champion_better_p05": bool(auc_b > auc_a and p < 0.05),
    })
pd.DataFrame(delong_rows).to_csv(OUT_TABLES / "table17_baselines_delong.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table17_baselines_delong.csv'}")


# ─── figures ────────────────────────────────────────────────────────────
print("\ndrawing figures...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
colors = plt.cm.tab10(np.linspace(0, 1, len(BASELINES) + 1))
for i, (name, *_) in enumerate(BASELINES):
    test_cal = results[name]["test_cal"]
    fpr, tpr, _ = roc_curve(y_te, test_cal)
    ax1.plot(fpr, tpr, lw=1.2, color=colors[i],
             label=f"{name} (AUC={roc_auc_score(y_te, test_cal):.3f})")
    prec, rec, _ = precision_recall_curve(y_te, test_cal)
    ax2.plot(rec, prec, lw=1.2, color=colors[i],
             label=f"{name} (AP={average_precision_score(y_te, test_cal):.3f})")
fpr_ch, tpr_ch, _ = roc_curve(y_te, champion_test)
ax1.plot(fpr_ch, tpr_ch, lw=2.5, color="black", ls="--",
         label=f"Champion (AUC={roc_auc_score(y_te, champion_test):.3f})")
prec_ch, rec_ch, _ = precision_recall_curve(y_te, champion_test)
ax2.plot(rec_ch, prec_ch, lw=2.5, color="black", ls="--",
         label=f"Champion (AP={average_precision_score(y_te, champion_test):.3f})")
ax1.plot([0, 1], [0, 1], ":", color="gray", lw=1)
ax1.set_xlabel("False positive rate"); ax1.set_ylabel("True positive rate")
ax1.set_title("ROC — test set"); ax1.legend(loc="lower right", fontsize=8)
ax1.grid(alpha=0.3)
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("PR — test set"); ax2.legend(loc="upper right", fontsize=8)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig17_baselines_roc.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {OUT_FIGS / 'fig17_baselines_roc.png'}")

fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), sharey=True)
labels = [name for name, *_ in BASELINES] + ["Phase 6 Champion"]

# gather predictions + Bal-Acc thresholds (from panel_df rows where criterion=="Bal-Acc")
panel_df_ba = pd.DataFrame(panel_rows)
panel_df_ba = panel_df_ba[panel_df_ba["threshold_criterion"] == "Bal-Acc"].set_index("baseline")

aucs, auc_los, auc_his = [], [], []
bas,  ba_los,  ba_his  = [], [], []
for name, *_ in BASELINES:
    test_cal = results[name]["test_cal"]
    aucs.append(roc_auc_score(y_te, test_cal))
    lo, hi = bootstrap_ci(y_te, test_cal, roc_auc_score)
    auc_los.append(lo); auc_his.append(hi)
    thr_ba = float(panel_df_ba.loc[name, "threshold"])
    ba_metric = lambda y, p, t=thr_ba: balanced_accuracy_score(y, (p >= t).astype(int))
    bas.append(balanced_accuracy_score(y_te, (test_cal >= thr_ba).astype(int)))
    blo, bhi = bootstrap_ci(y_te, test_cal, ba_metric)
    ba_los.append(blo); ba_his.append(bhi)

aucs.append(roc_auc_score(y_te, champion_test))
lo_ch, hi_ch = bootstrap_ci(y_te, champion_test, roc_auc_score)
auc_los.append(lo_ch); auc_his.append(hi_ch)
thr_ba_ch_val = float(panel_df_ba.loc["Phase6-Champion", "threshold"])
ba_metric_ch = lambda y, p, t=thr_ba_ch_val: balanced_accuracy_score(y, (p >= t).astype(int))
bas.append(balanced_accuracy_score(y_te, (champion_test >= thr_ba_ch_val).astype(int)))
blo_ch, bhi_ch = bootstrap_ci(y_te, champion_test, ba_metric_ch)
ba_los.append(blo_ch); ba_his.append(bhi_ch)

y_pos = np.arange(len(labels))[::-1]

# ---- left panel: AUC ----
ax = axes[0]
for i, (lab, a, lo, hi) in enumerate(zip(labels, aucs, auc_los, auc_his)):
    is_champ = (lab == "Phase 6 Champion")
    color = "crimson" if is_champ else "tab:blue"
    ax.errorbar(a, y_pos[i], xerr=[[a - lo], [hi - a]], fmt="o",
                capsize=4, color=color, markersize=9 if is_champ else 7, lw=2)
    ax.text(a, y_pos[i] + 0.22, f"{a:.4f} [{lo:.3f},{hi:.3f}]",
            ha="center", fontsize=8.5, color=color)
ax.axvline(aucs[-1], color="crimson", ls=":", alpha=0.4)
ax.set_yticks(y_pos); ax.set_yticklabels(labels, fontsize=10)
ax.set_xlim(0.60, 0.86)
ax.set_xlabel("Test AUC (calibrated)")
ax.set_title("(a) AUC", fontsize=11, loc="left")
ax.grid(axis="x", alpha=0.3)

# ---- right panel: Bal-Acc @ BalAcc-optimal threshold ----
ax2 = axes[1]
for i, (lab, v, lo, hi) in enumerate(zip(labels, bas, ba_los, ba_his)):
    is_champ = (lab == "Phase 6 Champion")
    color = "crimson" if is_champ else "tab:green"
    ax2.errorbar(v, y_pos[i], xerr=[[v - lo], [hi - v]], fmt="s",
                 capsize=4, color=color, markersize=9 if is_champ else 7, lw=2)
    ax2.text(v, y_pos[i] + 0.22, f"{v:.4f} [{lo:.3f},{hi:.3f}]",
             ha="center", fontsize=8.5, color=color)
ax2.axvline(bas[-1], color="crimson", ls=":", alpha=0.4)
ax2.set_xlim(0.58, 0.78)
ax2.set_xlabel(r"Test Bal-Acc @ $\tau^\star_{\mathrm{BalAcc}}$ (train-OOF searched)")
ax2.set_title("(b) Balanced Accuracy", fontsize=11, loc="left")
ax2.grid(axis="x", alpha=0.3)

fig.suptitle("Baseline family vs Phase 6 champion — 1000-bootstrap 95% CI",
             fontsize=12.5, y=1.02)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig17_baselines_forest.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {OUT_FIGS / 'fig17_baselines_forest.png'}")


# ─── final summary ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  FINAL SUMMARY")
print("=" * 72)
panel_df = pd.DataFrame(panel_rows)
f1_view = panel_df[panel_df["threshold_criterion"] == "F1"].sort_values("AUC", ascending=False)
print("\n[F1 threshold] ordered by AUC:")
for _, r in f1_view.iterrows():
    flag = "  ★" if r["baseline"] == "Phase6-Champion" else ""
    print(f"  {r['baseline']:18s}  AUC={r['AUC']:.4f} [{r['AUC_lo']:.3f},{r['AUC_hi']:.3f}]  "
          f"F1={r['F1']:.4f}  Bal_Acc={r['Bal_Acc']:.4f}  "
          f"Sens={r['Sens']:.4f}  Spec={r['Spec']:.4f}{flag}")
