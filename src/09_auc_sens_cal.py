"""Phase 7.6 — Bootstrap CI for AUC / Sensitivity + calibration diagnostics.

Reuses Phase 6 calibrated meta probabilities (saved by 08_bal_acc_tune.py) to
produce three artefacts for the paper's internal-validation section:

  1. Bootstrap 95% CI for AUC on held-out test (1000 resamples).
  2. Bootstrap 95% CI for Sensitivity at the Bal-Acc-optimal threshold
     (thr=0.135) and the F1-optimal threshold (thr=0.185).
  3. Reliability diagram (10 quantile bins) + Expected Calibration Error
     (ECE), Maximum Calibration Error (MCE), and Brier score; plus a
     histogram of predicted probabilities.

Outputs
-------
src/tables/table21_champion_bootstrap_ci.csv
src/tables/table16_calibration.csv
src/tables/table16_reliability_bins_test.csv
src/figures/fig21_champion_ci.png
src/figures/fig16_calibration_reliability.png
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    brier_score_loss, confusion_matrix, roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RS = 42
N_BOOT = 1000

TARGET = "离职行为"
INTENT = "离职意向"

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)
OUT_PROC = Path("data/processed")


# ─── load ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
test_idx = np.load(OUT_PROC / "test_idx.npy")
y_te = df[TARGET].values.astype(int)[test_idx]

oof_cal = np.load(OUT_PROC / "phase6_meta_oof_probs.npy")
test_cal = np.load(OUT_PROC / "phase6_meta_test_probs.npy")
print(f"loaded calibrated probs: oof={oof_cal.shape}, test={test_cal.shape}")
print(f"test label support: pos={int(y_te.sum())}/{len(y_te)} ({y_te.mean():.3f})")


# ─── helpers ────────────────────────────────────────────────────────────
def _sens(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tp / max(tp + fn, 1)


def _spec(y, pred):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return tn / max(tn + fp, 1)


def bootstrap_metric(y, p, metric_fn, n=N_BOOT, seed=RS):
    rng = np.random.default_rng(seed)
    N = len(y)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, N, N)
        if len(np.unique(y[idx])) < 2: continue
        vals.append(metric_fn(y[idx], p[idx]))
    return np.asarray(vals)


# ─── 1. AUC bootstrap ───────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  1) AUC on test — bootstrap 95% CI")
print("=" * 72)
auc_point = roc_auc_score(y_te, test_cal)
auc_boot = bootstrap_metric(y_te, test_cal, roc_auc_score)
auc_ci = (np.percentile(auc_boot, 2.5), np.percentile(auc_boot, 97.5))
print(f"  point AUC           = {auc_point:.4f}")
print(f"  95% CI              = [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")


# ─── 2. Sens / Spec bootstrap at both thresholds ───────────────────────
print("\n" + "=" * 72)
print("  2) Sensitivity / Specificity on test — bootstrap 95% CI")
print("=" * 72)

THR_BALACC = 0.135
THR_F1 = 0.185

sens_rows = []
for thr_name, thr in [("Bal-Acc-optimal", THR_BALACC), ("F1-optimal", THR_F1)]:
    pred = (test_cal >= thr).astype(int)
    s_point = _sens(y_te, pred)
    sp_point = _spec(y_te, pred)
    sens_boot = bootstrap_metric(y_te, test_cal,
                                 lambda y, p: _sens(y, (p >= thr).astype(int)))
    spec_boot = bootstrap_metric(y_te, test_cal,
                                 lambda y, p: _spec(y, (p >= thr).astype(int)))
    s_ci = (np.percentile(sens_boot, 2.5), np.percentile(sens_boot, 97.5))
    sp_ci = (np.percentile(spec_boot, 2.5), np.percentile(spec_boot, 97.5))
    print(f"\n  threshold = {thr:.3f}  ({thr_name})")
    print(f"    Sens  point = {s_point:.4f}  95% CI [{s_ci[0]:.4f}, {s_ci[1]:.4f}]")
    print(f"    Spec  point = {sp_point:.4f}  95% CI [{sp_ci[0]:.4f}, {sp_ci[1]:.4f}]")
    sens_rows.append({
        "threshold_name": thr_name, "threshold": thr,
        "Sens_point": s_point, "Sens_lo": s_ci[0], "Sens_hi": s_ci[1],
        "Spec_point": sp_point, "Spec_lo": sp_ci[0], "Spec_hi": sp_ci[1],
    })


# ─── write table21 — champion bootstrap CI ─────────────────────────────
rows21 = [{
    "metric": "AUC",
    "point": auc_point, "lo": auc_ci[0], "hi": auc_ci[1],
    "threshold": np.nan, "threshold_name": "",
}]
for r in sens_rows:
    rows21.append({
        "metric": "Sensitivity",
        "point": r["Sens_point"], "lo": r["Sens_lo"], "hi": r["Sens_hi"],
        "threshold": r["threshold"], "threshold_name": r["threshold_name"],
    })
    rows21.append({
        "metric": "Specificity",
        "point": r["Spec_point"], "lo": r["Spec_lo"], "hi": r["Spec_hi"],
        "threshold": r["threshold"], "threshold_name": r["threshold_name"],
    })
df21 = pd.DataFrame(rows21)
df21.to_csv(OUT_TABLES / "table21_champion_bootstrap_ci.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table21_champion_bootstrap_ci.csv'}")


# ─── 3. Calibration ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  3) Calibration — reliability diagram + ECE/MCE/Brier")
print("=" * 72)


def quantile_bins(p, n_bins=10):
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0] -= 1e-9; q[-1] += 1e-9
    q = np.unique(q)
    return q


def reliability_stats(y, p, n_bins=10):
    q = quantile_bins(p, n_bins)
    ids = np.clip(np.searchsorted(q, p, side="right") - 1, 0, len(q) - 2)
    rows = []
    ece = 0.0; mce = 0.0; N = len(y)
    for b in range(len(q) - 1):
        m = ids == b
        if m.sum() == 0: continue
        p_mean = p[m].mean()
        y_mean = y[m].mean()
        gap = abs(p_mean - y_mean)
        weight = m.sum() / N
        ece += weight * gap
        mce = max(mce, gap)
        rows.append({"bin": b, "n": int(m.sum()), "mean_pred": p_mean,
                     "mean_true": y_mean, "gap": gap})
    return pd.DataFrame(rows), ece, mce


rel_df, ece, mce = reliability_stats(y_te, test_cal, n_bins=10)
brier = brier_score_loss(y_te, test_cal)
print(f"  ECE (quantile-binned, 10 bins) = {ece:.4f}")
print(f"  MCE                            = {mce:.4f}")
print(f"  Brier                          = {brier:.4f}")

y_tr = df[TARGET].values.astype(int)[np.load(OUT_PROC / "train_idx.npy")]
rel_df_oof, ece_oof, mce_oof = reliability_stats(y_tr, oof_cal, n_bins=10)
brier_oof = brier_score_loss(y_tr, oof_cal)
print(f"  ECE (CV OOF)                   = {ece_oof:.4f}")
print(f"  Brier (CV OOF)                 = {brier_oof:.4f}")

cal_table = pd.DataFrame([
    {"set": "TEST", "n": len(y_te), "AUC": auc_point,
     "ECE": ece, "MCE": mce, "Brier": brier},
    {"set": "CV OOF", "n": len(y_tr), "AUC": roc_auc_score(y_tr, oof_cal),
     "ECE": ece_oof, "MCE": mce_oof, "Brier": brier_oof},
])
cal_table.to_csv(OUT_TABLES / "table16_calibration.csv", index=False)
rel_df.to_csv(OUT_TABLES / "table16_reliability_bins_test.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table16_calibration.csv'}")
print(f"wrote {OUT_TABLES / 'table16_reliability_bins_test.csv'}")


# ─── 4. Forest plot: champion AUC + Sens + Spec with CI ────────────────
fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.6))

items = [
    ("AUC", auc_point, auc_ci[0], auc_ci[1]),
    (f"Sensitivity\n(τ={THR_BALACC}, Bal-Acc-opt)",
     sens_rows[0]["Sens_point"], sens_rows[0]["Sens_lo"], sens_rows[0]["Sens_hi"]),
    (f"Specificity\n(τ={THR_BALACC}, Bal-Acc-opt)",
     sens_rows[0]["Spec_point"], sens_rows[0]["Spec_lo"], sens_rows[0]["Spec_hi"]),
    (f"Sensitivity\n(τ={THR_F1}, F1-opt)",
     sens_rows[1]["Sens_point"], sens_rows[1]["Sens_lo"], sens_rows[1]["Sens_hi"]),
    (f"Specificity\n(τ={THR_F1}, F1-opt)",
     sens_rows[1]["Spec_point"], sens_rows[1]["Spec_lo"], sens_rows[1]["Spec_hi"]),
]

y_pos = np.arange(len(items))[::-1]
for i, (label, pt, lo, hi) in enumerate(items):
    yy = y_pos[i]
    ax.errorbar(pt, yy, xerr=[[pt - lo], [hi - pt]], fmt="o",
                capsize=4, color="tab:blue", markersize=8, lw=2)
    ax.text(pt, yy + 0.18, f"{pt:.4f} [{lo:.3f}, {hi:.3f}]",
            ha="center", fontsize=9, color="tab:blue")

ax.set_yticks(y_pos)
ax.set_yticklabels([it[0] for it in items], fontsize=10)
ax.set_xlim(0.45, 1.0)
ax.set_xlabel("Metric value (test set)")
ax.set_title("Phase 6 champion — point estimates with 1000-bootstrap 95% CI",
             fontsize=11)
ax.axvline(0.5, color="gray", ls=":", alpha=0.4, label="chance (0.5)")
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig21_champion_ci.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nwrote {OUT_FIGS / 'fig21_champion_ci.png'}")


# ─── 5. Calibration reliability diagram ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                gridspec_kw={"width_ratios": [1.2, 1]})

ax1.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="Perfect calibration")
ax1.plot(rel_df_oof["mean_pred"], rel_df_oof["mean_true"],
         "o-", color="tab:orange", lw=1.5, ms=6, alpha=0.55,
         label=f"CV OOF (ECE={ece_oof:.3f}, Brier={brier_oof:.3f})")
ax1.plot(rel_df["mean_pred"], rel_df["mean_true"],
         "s-", color="tab:blue", lw=2, ms=8,
         label=f"TEST (ECE={ece:.3f}, Brier={brier:.3f})")
ax1.set_xlim(0, max(rel_df["mean_pred"].max(), rel_df_oof["mean_pred"].max()) * 1.1)
ax1.set_ylim(0, max(rel_df["mean_true"].max(), rel_df_oof["mean_true"].max()) * 1.1)
ax1.set_xlabel("Mean predicted probability (bin)")
ax1.set_ylabel("Observed fraction of positives (bin)")
ax1.set_title("Reliability diagram — 10 quantile bins", fontsize=11)
ax1.legend(loc="upper left", fontsize=9)
ax1.grid(alpha=0.3)
ax1.set_aspect("equal", adjustable="datalim")

ax2.hist(test_cal[y_te == 0], bins=30, alpha=0.55, label="y=0 (留任)",
         color="tab:gray", density=False)
ax2.hist(test_cal[y_te == 1], bins=30, alpha=0.75, label="y=1 (离职)",
         color="crimson", density=False)
ax2.axvline(THR_BALACC, color="tab:blue", ls="--",
            label=f"Bal-Acc τ={THR_BALACC}")
ax2.axvline(THR_F1, color="tab:green", ls=":",
            label=f"F1 τ={THR_F1}")
ax2.set_xlabel("Predicted probability (calibrated)")
ax2.set_ylabel("Count")
ax2.set_title("Predicted-probability distribution (test)", fontsize=11)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(alpha=0.3)

plt.suptitle("Phase 6 champion — calibration", fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig16_calibration_reliability.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {OUT_FIGS / 'fig16_calibration_reliability.png'}")


# ─── final summary ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  FINAL SUMMARY — Phase 6 champion bootstrap CI + calibration")
print("=" * 72)
print(f"""
AUC  : {auc_point:.4f}  [CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f}]
Sens (τ={THR_BALACC}, Bal-Acc)
     : {sens_rows[0]['Sens_point']:.4f}  [CI {sens_rows[0]['Sens_lo']:.3f}-{sens_rows[0]['Sens_hi']:.3f}]
Spec (τ={THR_BALACC}, Bal-Acc)
     : {sens_rows[0]['Spec_point']:.4f}  [CI {sens_rows[0]['Spec_lo']:.3f}-{sens_rows[0]['Spec_hi']:.3f}]
Sens (τ={THR_F1}, F1)
     : {sens_rows[1]['Sens_point']:.4f}  [CI {sens_rows[1]['Sens_lo']:.3f}-{sens_rows[1]['Sens_hi']:.3f}]
Spec (τ={THR_F1}, F1)
     : {sens_rows[1]['Spec_point']:.4f}  [CI {sens_rows[1]['Spec_lo']:.3f}-{sens_rows[1]['Spec_hi']:.3f}]
Calibration (TEST): ECE={ece:.4f}  MCE={mce:.4f}  Brier={brier:.4f}
Calibration (OOF) : ECE={ece_oof:.4f}                Brier={brier_oof:.4f}
""")
