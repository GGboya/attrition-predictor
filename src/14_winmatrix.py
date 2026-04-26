"""Phase 13 — Multi-dimensional win/loss scorecard.

Consolidates all existing test-set evidence into a single scorecard:
    rows    = Phase 6 Stack champion + 7 standalone baselines
    columns = AUC, Bal-Acc @ tau*_BalAcc, ECE, Brier, HRCF compatibility

Output
------
src/tables/table22_multidim_comparison.csv   — raw numbers + ranks
src/figures/fig22_multidim_winmatrix.png     — paper-ready scorecard
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
TABDIR = ROOT / "src" / "tables"
FIGDIR = ROOT / "src" / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Load evidence
# ------------------------------------------------------------
panel = pd.read_csv(TABDIR / "table17_baselines_panel.csv")
calib = pd.read_csv(TABDIR / "table17_baselines_calibration.csv")

# Only keep the Bal-Acc-optimal threshold row per model
ba_rows = panel[panel["threshold_criterion"] == "Bal-Acc"].set_index("baseline")

ORDER = ["Phase6-Champion", "LR", "RF", "XGB", "LGBM", "CatBoost", "kNN", "SVM"]
DISPLAY = {
    "Phase6-Champion": "Phase 6 Stack (ours)",
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "XGB": "XGBoost",
    "LGBM": "LightGBM",
    "CatBoost": "CatBoost",
    "kNN": "kNN (k=50)",
    "SVM": "SVM-RBF",
}

# HRCF compatibility: Stack (via MT-MLP surrogate) ✓; LR ✓; SVM ✓;
# tree ensembles ✗; kNN ✗.
HRCF_COMPAT = {
    "Phase6-Champion": "yes",  # via MT-MLP surrogate (shown in Appendix)
    "LR":             "yes",
    "RF":             "no",
    "XGB":            "no",
    "LGBM":           "no",
    "CatBoost":       "no",
    "kNN":            "no",
    "SVM":            "partial",  # differentiable decision function but no probability gradient
}

rows = []
for name in ORDER:
    r = {
        "model":   name,
        "display": DISPLAY[name],
        "AUC":     float(ba_rows.loc[name, "AUC"]),
        "AUC_lo":  float(ba_rows.loc[name, "AUC_lo"]),
        "AUC_hi":  float(ba_rows.loc[name, "AUC_hi"]),
        "BalAcc":  float(ba_rows.loc[name, "Bal_Acc"]),
        "tau":     float(ba_rows.loc[name, "threshold"]),
        "Sens":    float(ba_rows.loc[name, "Sens"]),
        "Spec":    float(ba_rows.loc[name, "Spec"]),
        "ECE":     float(calib.set_index("baseline").loc[name, "ECE"]),
        "Brier":   float(ba_rows.loc[name, "Brier"]),
        "HRCF":    HRCF_COMPAT[name],
    }
    rows.append(r)

df = pd.DataFrame(rows)

# ------------------------------------------------------------
# 2. Compute ranks (lower = better for ECE/Brier, higher for others)
# ------------------------------------------------------------
df["rank_AUC"]    = df["AUC"].rank(ascending=False, method="min").astype(int)
df["rank_BalAcc"] = df["BalAcc"].rank(ascending=False, method="min").astype(int)
df["rank_ECE"]    = df["ECE"].rank(ascending=True, method="min").astype(int)
df["rank_Brier"]  = df["Brier"].rank(ascending=True, method="min").astype(int)

df.to_csv(TABDIR / "table22_multidim_comparison.csv", index=False)
print("wrote", TABDIR / "table22_multidim_comparison.csv")
print(df[["display", "AUC", "BalAcc", "ECE", "Brier",
          "rank_AUC", "rank_BalAcc", "rank_ECE", "rank_Brier", "HRCF"]].to_string(index=False))

# ------------------------------------------------------------
# 3. Scorecard figure
# ------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.2})

n_rows = len(df)
metric_cols = [
    ("AUC",     "AUC",     "rank_AUC",    "higher"),
    ("BalAcc",  r"Bal-Acc @ $\tau^\star$", "rank_BalAcc", "higher"),
    ("ECE",     "ECE",     "rank_ECE",    "lower"),
    ("Brier",   "Brier",   "rank_Brier",  "lower"),
]

fig, ax = plt.subplots(figsize=(12.0, 0.55 * n_rows + 2.4))
ax.set_xlim(0, 10)
ax.set_ylim(0, n_rows + 1.3)
ax.set_axis_off()

# --- column headers
col_x = {"model": 0.2, "AUC": 3.7, "BalAcc": 5.15, "ECE": 6.55, "Brier": 7.85, "HRCF": 9.25}
ax.text(col_x["model"], n_rows + 0.75, "Model", fontweight="bold", fontsize=10.5)
ax.text(col_x["AUC"],    n_rows + 0.75, "AUC",                ha="center", fontweight="bold", fontsize=10.5)
ax.text(col_x["BalAcc"], n_rows + 0.75, r"Bal-Acc @ $\tau^\star_{\mathrm{BalAcc}}$", ha="center", fontweight="bold", fontsize=10.5)
ax.text(col_x["ECE"],    n_rows + 0.75, "ECE (↓)",            ha="center", fontweight="bold", fontsize=10.5)
ax.text(col_x["Brier"],  n_rows + 0.75, "Brier (↓)",          ha="center", fontweight="bold", fontsize=10.5)
ax.text(col_x["HRCF"],   n_rows + 0.75, "HRCF-compat.",       ha="center", fontweight="bold", fontsize=10.5)

# --- separator line under header
ax.plot([0.1, 9.9], [n_rows + 0.45, n_rows + 0.45], color="#333", linewidth=1.1)

def rank_color(rank, n):
    # rank 1 = green, rank n = red, middle = yellow
    # use a 3-stop colormap
    cmap = LinearSegmentedColormap.from_list("wl", ["#2ca02c", "#F6C445", "#d62728"])
    t = (rank - 1) / max(1, n - 1)
    return cmap(t)

for i, row in df.iterrows():
    y = n_rows - i  # top row first
    is_champ = row["model"] == "Phase6-Champion"
    bg = "#FFF4D6" if is_champ else "white"
    # row background
    ax.add_patch(FancyBboxPatch((0.1, y - 0.30), 9.8, 0.60,
                                boxstyle="round,pad=0.0,rounding_size=0.08",
                                fc=bg, ec="#DDDDDD", lw=0.8))

    label = row["display"] + ("  ★" if is_champ else "")
    ax.text(col_x["model"], y, label, fontsize=9.6,
            fontweight="bold" if is_champ else "normal",
            va="center")

    for key, header, rank_key, direction in metric_cols:
        val = row[key]
        rank = row[rank_key]
        color = rank_color(rank, n_rows)
        ax.add_patch(FancyBboxPatch((col_x[header.split()[0] if " " not in header else key] - 0.55, y - 0.23),
                                    1.1, 0.46,
                                    boxstyle="round,pad=0.0,rounding_size=0.06",
                                    fc=color, ec="none", alpha=0.55))
        ax.text(col_x[header.split()[0] if " " not in header else key], y,
                f"{val:.4f}" if key in ("ECE", "Brier") else f"{val:.4f}",
                ha="center", va="center", fontsize=9.2, fontweight="bold",
                color="#111")
        ax.text(col_x[header.split()[0] if " " not in header else key], y - 0.33,
                f"rank {rank}", ha="center", va="center", fontsize=7.3, color="#555")

    # HRCF column — categorical
    hrcf = row["HRCF"]
    face = {"yes": "#2ca02c", "partial": "#F6C445", "no": "#d62728"}[hrcf]
    mark = {"yes": "✓", "partial": "~", "no": "✗"}[hrcf]
    ax.add_patch(FancyBboxPatch((col_x["HRCF"] - 0.35, y - 0.20), 0.70, 0.40,
                                boxstyle="round,pad=0.0,rounding_size=0.06",
                                fc=face, ec="none", alpha=0.75))
    ax.text(col_x["HRCF"], y, mark, ha="center", va="center",
            fontsize=13, fontweight="bold", color="white")

# --- legend / note
note_y = -0.1
ax.text(0.2, note_y,
        "Cells colored by rank (green=best, red=worst). "
        r"$\tau^\star_{\mathrm{BalAcc}}$ per model is searched on train-OOF. "
        "HRCF-compat. ✓ = differentiable, supports projection-based counterfactuals; "
        "~ = partial; ✗ = tree/non-differentiable.",
        fontsize=7.8, color="#555", style="italic")

ax.set_title("Multi-dimensional scorecard — Phase 6 Stack champion vs 7 standalone baselines (test set, N=1094)",
             fontsize=11.5, fontweight="bold", pad=10)

plt.tight_layout()
out = FIGDIR / "fig22_multidim_winmatrix.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
plt.savefig(FIGDIR / "fig22_multidim_winmatrix.pdf", bbox_inches="tight")
plt.close()
print("wrote", out)
