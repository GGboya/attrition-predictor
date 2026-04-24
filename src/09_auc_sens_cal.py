"""Phase 7.6 — AUC / Sensitivity significance vs BORF, + calibration.

Reuses Phase 6 calibrated meta probabilities (saved by 08_bal_acc_tune.py) to
produce the three artefacts the paper needs:

  1. Bootstrap 95% CI for AUC on held-out test, plus one-sided p-value for
     H0: our AUC ≤ BORF AUC (0.69). 1000 resamples.
  2. Bootstrap 95% CI for Sensitivity at the Bal-Acc-optimal threshold
     (thr=0.135), plus one-sided p-value for H0: our Sens ≤ BORF Sens (0.681).
  3. Reliability diagram (10 quantile bins) + Expected Calibration Error (ECE),
     Maximum Calibration Error (MCE), and Brier score. Also a histogram of
     predicted probabilities.

External reference (from docs/research-paper-2024-borf-turnover-data.md):
  BORF (Liu+ 2024):  AUC = 0.69,  Sens = 0.681,  Acc = 0.786
  NOTE: BORF's Acc=0.786 and AUC=0.69 are questionable together (see
  memory: project_borf_benchmark_inconsistency.md). We compare against
  their *reported* numbers regardless, with the caveat documented.

Outputs
-------
src/tables/table15_bootstrap_vs_borf.csv
src/tables/table16_calibration.csv
src/figures/fig15_forest_vs_borf.png
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
BORF_AUC = 0.69
BORF_SENS = 0.681

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


def one_sided_p_leq(vals, ref):
    """P-value for H0: metric ≤ ref (i.e. fraction of bootstrap replicates below ref)."""
    return float((vals <= ref).mean())


# ─── 1. AUC bootstrap + vs BORF ─────────────────────────────────────────
print("\n" + "=" * 72)
print("  1) AUC on test — bootstrap 95% CI + vs BORF 0.69")
print("=" * 72)
auc_point = roc_auc_score(y_te, test_cal)
auc_boot = bootstrap_metric(y_te, test_cal, roc_auc_score)
auc_ci = (np.percentile(auc_boot, 2.5), np.percentile(auc_boot, 97.5))
auc_lower_1s = np.percentile(auc_boot, 5)
p_auc = one_sided_p_leq(auc_boot, BORF_AUC)
print(f"  point AUC           = {auc_point:.4f}")
print(f"  95% CI (2-sided)    = [{auc_ci[0]:.4f}, {auc_ci[1]:.4f}]")
print(f"  5% lower (1-sided)  = {auc_lower_1s:.4f}")
print(f"  BORF AUC            = {BORF_AUC:.4f}")
print(f"  p(our AUC ≤ BORF)   = {p_auc:.4f}  ({N_BOOT} boot replicates)")


# ─── 2. Sens bootstrap + vs BORF, at Bal-Acc threshold ─────────────────
print("\n" + "=" * 72)
print("  2) Sensitivity on test — bootstrap 95% CI + vs BORF 0.681")
print("=" * 72)

# Use Bal-Acc-optimal threshold from phase 7.5 (found on CV OOF)
THR_BALACC = 0.135
THR_F1 = 0.185

sens_rows = []
for thr_name, thr in [("Bal-Acc-optimal", THR_BALACC), ("F1-optimal", THR_F1)]:
    pred = (test_cal >= thr).astype(int)
    s_point = _sens(y_te, pred)
    sp_point = _spec(y_te, pred)
    sens_boot = bootstrap_metric(y_te, test_cal, lambda y, p: _sens(y, (p >= thr).astype(int)))
    spec_boot = bootstrap_metric(y_te, test_cal, lambda y, p: _spec(y, (p >= thr).astype(int)))
    ci = (np.percentile(sens_boot, 2.5), np.percentile(sens_boot, 97.5))
    lower_1s = np.percentile(sens_boot, 5)
    p_val = one_sided_p_leq(sens_boot, BORF_SENS)
    print(f"\n  threshold = {thr:.3f}  ({thr_name})")
    print(f"    Sens  point = {s_point:.4f}  95% CI [{ci[0]:.4f}, {ci[1]:.4f}]  "
          f"1s-lower={lower_1s:.4f}")
    print(f"    Spec  point = {sp_point:.4f}")
    print(f"    BORF Sens   = {BORF_SENS:.4f}  →  p(ours ≤ BORF) = {p_val:.4f}")
    sens_rows.append({
        "threshold_name": thr_name, "threshold": thr,
        "Sens_point": s_point, "Sens_lo": ci[0], "Sens_hi": ci[1],
        "Sens_1s_lower": lower_1s,
        "Spec_point": sp_point,
        "BORF_Sens": BORF_SENS, "p_vs_BORF": p_val,
    })


# ─── write table15 ──────────────────────────────────────────────────────
rows15 = [{
    "metric": "AUC", "benchmark": "BORF 2024",
    "our_point": auc_point, "our_lo": auc_ci[0], "our_hi": auc_ci[1],
    "our_1s_lower": auc_lower_1s, "BORF": BORF_AUC,
    "delta": auc_point - BORF_AUC, "p_one_sided": p_auc,
}]
for r in sens_rows:
    rows15.append({
        "metric": f"Sensitivity @ thr={r['threshold']:.3f} ({r['threshold_name']})",
        "benchmark": "BORF 2024",
        "our_point": r["Sens_point"], "our_lo": r["Sens_lo"], "our_hi": r["Sens_hi"],
        "our_1s_lower": r["Sens_1s_lower"], "BORF": BORF_SENS,
        "delta": r["Sens_point"] - BORF_SENS, "p_one_sided": r["p_vs_BORF"],
    })
df15 = pd.DataFrame(rows15)
df15.to_csv(OUT_TABLES / "table15_bootstrap_vs_borf.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table15_bootstrap_vs_borf.csv'}")


# ─── 3. Calibration ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("  3) Calibration — reliability diagram + ECE/MCE/Brier")
print("=" * 72)

def quantile_bins(p, n_bins=10):
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0] -= 1e-9; q[-1] += 1e-9
    q = np.unique(q)  # collapse duplicates
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

# Also compute for OOF (on training set) for reference — should look similar
rel_df_oof, ece_oof, mce_oof = reliability_stats(
    np.load(OUT_PROC / "train_idx.npy").astype(int) * 0
    + df[TARGET].values.astype(int)[np.load(OUT_PROC / "train_idx.npy")],
    oof_cal, n_bins=10)  # ugly but works
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


# ─── 4. Forest plot: AUC + Sens vs BORF ─────────────────────────────────
fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.2))

items = [
    ("AUC",
     auc_point, auc_ci[0], auc_ci[1], BORF_AUC, "BORF AUC = 0.69"),
    ("Sensitivity\n(thr=0.135, Bal-Acc optimal)",
     sens_rows[0]["Sens_point"], sens_rows[0]["Sens_lo"], sens_rows[0]["Sens_hi"],
     BORF_SENS, "BORF Sens = 0.681"),
    ("Sensitivity\n(thr=0.185, F1 optimal)",
     sens_rows[1]["Sens_point"], sens_rows[1]["Sens_lo"], sens_rows[1]["Sens_hi"],
     BORF_SENS, "BORF Sens = 0.681"),
]

y_pos = np.arange(len(items))[::-1]
for i, (label, pt, lo, hi, borf, _) in enumerate(items):
    yy = y_pos[i]
    ax.errorbar(pt, yy, xerr=[[pt - lo], [hi - pt]], fmt="o",
                capsize=4, color="tab:blue", markersize=8, lw=2,
                label="Ours (point + 95% CI)" if i == 0 else None)
    ax.scatter([borf], [yy], marker="D", s=80, color="crimson", zorder=5,
               label="BORF (Liu+ 2024)" if i == 0 else None)
    ax.text(pt, yy + 0.18, f"{pt:.4f} [{lo:.3f},{hi:.3f}]",
            ha="center", fontsize=9, color="tab:blue")
    ax.text(borf, yy - 0.25, f"{borf:.3f}", ha="center", fontsize=9,
            color="crimson")

ax.set_yticks(y_pos)
ax.set_yticklabels([it[0] for it in items], fontsize=10)
ax.set_xlim(0.60, 0.90)
ax.set_xlabel("Metric value (test set)")
ax.set_title("Our Phase 6 champion vs BORF (Liu+ 2024)\nPoint estimates with 1000-bootstrap 95% CI",
             fontsize=11)
ax.axvline(0.5, color="gray", ls=":", alpha=0.3)
ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig15_forest_vs_borf.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"\nwrote {OUT_FIGS / 'fig15_forest_vs_borf.png'}")


# ─── 5. Calibration reliability diagram ─────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5),
                                gridspec_kw={"width_ratios": [1.2, 1]})

# reliability curve (left)
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

# histogram (right)
ax2.hist(test_cal[y_te == 0], bins=30, alpha=0.55, label="y=0 (留任)",
         color="tab:gray", density=False)
ax2.hist(test_cal[y_te == 1], bins=30, alpha=0.75, label="y=1 (离职)",
         color="crimson", density=False)
ax2.axvline(THR_BALACC, color="tab:blue", ls="--",
            label=f"Bal-Acc thr={THR_BALACC}")
ax2.axvline(THR_F1, color="tab:green", ls=":",
            label=f"F1 thr={THR_F1}")
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
print("  FINAL SUMMARY")
print("=" * 72)
print(f"""
AUC  : ours = {auc_point:.4f}  [CI {auc_ci[0]:.3f}-{auc_ci[1]:.3f}]
       BORF = {BORF_AUC:.4f}   Δ = +{auc_point-BORF_AUC:.4f}   p(ours ≤ BORF) = {p_auc:.4f}
Sens (Bal-Acc thr, preferred)
     : ours = {sens_rows[0]['Sens_point']:.4f}  [CI {sens_rows[0]['Sens_lo']:.3f}-{sens_rows[0]['Sens_hi']:.3f}]
       BORF = {BORF_SENS:.4f}  Δ = +{sens_rows[0]['Sens_point']-BORF_SENS:.4f}  p(ours ≤ BORF) = {sens_rows[0]['p_vs_BORF']:.4f}
Sens (F1 thr, legacy)
     : ours = {sens_rows[1]['Sens_point']:.4f}  [CI {sens_rows[1]['Sens_lo']:.3f}-{sens_rows[1]['Sens_hi']:.3f}]
       p(ours ≤ BORF) = {sens_rows[1]['p_vs_BORF']:.4f}
Calibration (TEST): ECE={ece:.4f}  MCE={mce:.4f}  Brier={brier:.4f}
""")
