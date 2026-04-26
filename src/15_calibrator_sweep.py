"""Phase 13b — Calibrator sweep for the Phase 6 Stack champion.

Background
----------
The Phase 6 champion uses isotonic calibration on top of the L2-LR meta
output. In the multi-dimensional scorecard (Phase 13) we found that this
choice scores worst among 8 models on test ECE (0.0321). This script
swaps the calibrator only, leaving everything else identical, and picks
the calibrator that minimizes ECE without sacrificing AUC or Bal-Acc.

Calibrators evaluated
---------------------
1. identity  — raw sigmoid from the L2-LR meta (no post-hoc fix)
2. isotonic  — current Phase 6 champion (PAV on OOF)
3. sigmoid   — Platt scaling (1-parameter logistic on OOF, C=1e12)
4. beta      — 3-parameter beta calibration (Kull et al., 2017), fit with
               a small logistic regression on {log p, log(1-p)} features
5. temperature — single-parameter temperature scaling on the meta logit

Inputs
------
data/processed/phase6_meta_oof_raw.npy   (4375,) pre-calibration meta probs
data/processed/phase6_meta_test_raw.npy  (1094,) pre-calibration meta probs
data/processed/{train,test}_idx.npy
data/processed/clean.csv                 for labels

If the two _raw.npy files do not exist, run `python src/08_bal_acc_tune.py`
first — a patched version now caches them.

Outputs
-------
src/tables/table23_calibrator_sweep.csv
src/figures/fig23_reliability_grid.png
"""
from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (brier_score_loss, balanced_accuracy_score,
                             roc_auc_score)
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parent.parent
PROC = ROOT / "data" / "processed"
TAB = ROOT / "src" / "tables"
FIG = ROOT / "src" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# Load cached pre-calibration meta probs
# ----------------------------------------------------------------------
raw_oof_path = PROC / "phase6_meta_oof_raw.npy"
raw_te_path = PROC / "phase6_meta_test_raw.npy"
if not raw_oof_path.exists() or not raw_te_path.exists():
    print("ERROR: pre-calibration meta probs missing.")
    print("  run:  python src/08_bal_acc_tune.py")
    sys.exit(2)

oof_raw = np.load(raw_oof_path)
te_raw = np.load(raw_te_path)
print(f"loaded  oof_raw {oof_raw.shape}  test_raw {te_raw.shape}")

# labels
train_idx = np.load(PROC / "train_idx.npy")
test_idx = np.load(PROC / "test_idx.npy")
df = pd.read_csv(PROC / "clean.csv")
y_col = "离职行为"
y_tr = df[y_col].to_numpy()[train_idx].astype(int)
y_te = df[y_col].to_numpy()[test_idx].astype(int)
assert len(y_tr) == len(oof_raw) and len(y_te) == len(te_raw)

EPS = 1e-6

def _clip(p):
    return np.clip(p, EPS, 1.0 - EPS)


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------
def ece_mce_quantile(y, p, n_bins=10):
    """Quantile-binned ECE/MCE — matches 09_auc_sens_cal.py and 10_baselines.py."""
    p = _clip(p)
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    q[0] = 0.0; q[-1] = 1.0 + 1e-9
    idx = np.clip(np.digitize(p, q) - 1, 0, n_bins - 1)
    e = 0.0; worst = 0.0
    n = len(p)
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        conf = p[m].mean(); acc = y[m].mean()
        e += (m.sum() / n) * abs(acc - conf)
        worst = max(worst, abs(acc - conf))
    return e, worst


def ece(y, p, n_bins=10):
    return ece_mce_quantile(y, p, n_bins)[0]


def mce(y, p, n_bins=10):
    return ece_mce_quantile(y, p, n_bins)[1]


def best_balacc_threshold(y, p, grid=None):
    if grid is None:
        grid = np.linspace(0.02, 0.80, 401)
    best_tau, best_ba = 0.5, -1.0
    for tau in grid:
        ba = balanced_accuracy_score(y, (p >= tau).astype(int))
        if ba > best_ba:
            best_ba, best_tau = ba, tau
    return best_tau, best_ba


# ----------------------------------------------------------------------
# Calibrators
# ----------------------------------------------------------------------
def fit_identity(p, y):
    return lambda q: q


def fit_isotonic(p, y):
    m = IsotonicRegression(out_of_bounds="clip").fit(p, y)
    return m.predict


def fit_platt(p, y):
    # logistic on the logit of p (standard Platt scaling)
    z = np.log(_clip(p) / (1.0 - _clip(p))).reshape(-1, 1)
    lr = LogisticRegression(C=1e12, solver="lbfgs").fit(z, y)
    def apply(q):
        zz = np.log(_clip(q) / (1.0 - _clip(q))).reshape(-1, 1)
        return lr.predict_proba(zz)[:, 1]
    return apply


def fit_beta(p, y):
    # Kull et al 2017 beta calibration:
    # logit(p_cal) = a * log p + b * log(1-p) + c
    z = np.column_stack([np.log(_clip(p)), np.log(1.0 - _clip(p))])
    lr = LogisticRegression(C=1e12, solver="lbfgs").fit(z, y)
    def apply(q):
        zz = np.column_stack([np.log(_clip(q)), np.log(1.0 - _clip(q))])
        return lr.predict_proba(zz)[:, 1]
    return apply


def fit_temperature(p, y):
    # single scalar T that rescales the meta logit; minimize NLL
    logits = np.log(_clip(p) / (1.0 - _clip(p)))
    def nll(T):
        T = max(T, 1e-3)
        pr = 1.0 / (1.0 + np.exp(-logits / T))
        pr = _clip(pr)
        return -(y * np.log(pr) + (1 - y) * np.log(1 - pr)).mean()
    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T = float(res.x)
    def apply(q):
        zz = np.log(_clip(q) / (1.0 - _clip(q)))
        return 1.0 / (1.0 + np.exp(-zz / T))
    return apply, T


# ----------------------------------------------------------------------
# Sweep
# ----------------------------------------------------------------------
CALIBRATORS = [
    ("identity",     fit_identity),
    ("isotonic",     fit_isotonic),
    ("platt",        fit_platt),
    ("beta",         fit_beta),
    ("temperature",  fit_temperature),
]

rows = []
calibrated_test = {}
calibrated_oof = {}
for name, fitter in CALIBRATORS:
    out = fitter(oof_raw, y_tr)
    T = None
    if name == "temperature":
        apply_fn, T = out
    else:
        apply_fn = out
    p_oof = _clip(apply_fn(oof_raw))
    p_te = _clip(apply_fn(te_raw))
    calibrated_test[name] = p_te
    calibrated_oof[name] = p_oof

    # find τ* on OOF, evaluate on test
    tau_star, ba_oof = best_balacc_threshold(y_tr, p_oof)

    rows.append({
        "calibrator": name,
        "T": T if T is not None else np.nan,
        "AUC_test": roc_auc_score(y_te, p_te),
        "ECE_test": ece(y_te, p_te),
        "MCE_test": mce(y_te, p_te),
        "Brier_test": brier_score_loss(y_te, p_te),
        "tau_star": tau_star,
        "BalAcc_oof": ba_oof,
        "BalAcc_test": balanced_accuracy_score(y_te, (p_te >= tau_star).astype(int)),
        "ECE_oof": ece(y_tr, p_oof),
        "Brier_oof": brier_score_loss(y_tr, p_oof),
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(TAB / "table23_calibrator_sweep.csv", index=False)
print(df_out.to_string(index=False))
print("\nwrote", TAB / "table23_calibrator_sweep.csv")


# ----------------------------------------------------------------------
# Reliability grid figure
# ----------------------------------------------------------------------
def reliability_bins(y, p, n=15):
    bins = np.linspace(0, 1, n + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n - 1)
    xs, ys, ws = [], [], []
    for b in range(n):
        m = idx == b
        if m.sum() == 0:
            continue
        xs.append(p[m].mean())
        ys.append(y[m].mean())
        ws.append(m.sum())
    return np.asarray(xs), np.asarray(ys), np.asarray(ws)


plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.5})
fig, axes = plt.subplots(1, 5, figsize=(18, 3.9), sharey=True)
for ax, (name, _) in zip(axes, CALIBRATORS):
    row = df_out.set_index("calibrator").loc[name]
    p_te = calibrated_test[name]
    xs, ys, ws = reliability_bins(y_te, p_te)
    ax.plot([0, 1], [0, 1], color="#999", linestyle="--", linewidth=0.8)
    ax.scatter(xs, ys, s=15 + 0.4 * ws, color="#1f77b4",
               edgecolor="white", linewidth=0.6, alpha=0.85, zorder=3)
    ax.plot(xs, ys, color="#1f77b4", linewidth=1.2, alpha=0.7)
    ax.set_xlim(0, 0.9); ax.set_ylim(0, 0.9)
    ax.set_xlabel("predicted prob")
    if ax is axes[0]:
        ax.set_ylabel("empirical freq (positive)")
    title = (f"{name}\n"
             f"ECE={row['ECE_test']:.4f}  "
             f"AUC={row['AUC_test']:.4f}  "
             f"BalAcc={row['BalAcc_test']:.4f}")
    ax.set_title(title, fontsize=9)
    ax.grid(alpha=0.25)

plt.suptitle("Calibrator sweep on Phase 6 Stack — reliability diagrams (test, N=1094, 15 bins)",
             fontsize=11, fontweight="bold")
plt.tight_layout()
out = FIG / "fig23_reliability_grid.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.savefig(FIG / "fig23_reliability_grid.pdf", bbox_inches="tight")
plt.close()
print("wrote", out)
