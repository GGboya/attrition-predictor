"""Phase 14 — build the three new DeepSeek-style figures for the tech report.

Outputs
-------
src/figures/fig_hero_overview.png       — (a) AUC forest + DeLong flags   (b) HRCF radar
src/figures/fig_architecture.png        — Stacking + HRCF pipeline diagram
src/figures/fig_phase_progression.png   — AUC / Bal-Acc across phases (training-curve style)
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC",
                                    "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path("src/figures"); OUT.mkdir(exist_ok=True, parents=True)


# ============================================================
# Fig 1. Hero (two-panel: AUC forest + HRCF radar)
# ============================================================
baselines = pd.read_csv("src/tables/table17_baselines_panel.csv")
baselines = baselines[baselines["threshold_criterion"] == "Bal-Acc"].copy()
baselines = baselines.drop_duplicates("baseline", keep="first")
order = ["Phase6-Champion", "RF", "LR", "SVM", "CatBoost", "LGBM", "kNN", "XGB"]
baselines = baselines.set_index("baseline").loc[order].reset_index()
display_labels = ["Champion", "RF", "LR", "SVM", "CatBoost", "LightGBM", "kNN", "XGBoost"]

delong = pd.read_csv("src/tables/table17_baselines_delong.csv").set_index("baseline")
hrcf = pd.read_csv("src/tables/table5b_hrcf_vs_dice.csv").set_index("algo")

fig = plt.figure(figsize=(13.5, 5.5))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1.0], wspace=0.32)

# ------- (a) AUC forest -------
ax = fig.add_subplot(gs[0, 0])
ys = np.arange(len(baselines))[::-1]
is_champ = (baselines["baseline"] == "Phase6-Champion").values
for i, row in baselines.iterrows():
    pt, lo, hi = row["AUC"], row["AUC_lo"], row["AUC_hi"]
    y = ys[i]
    color = "crimson" if row["baseline"] == "Phase6-Champion" else "tab:blue"
    lw = 2.4 if row["baseline"] == "Phase6-Champion" else 1.6
    ms = 9 if row["baseline"] == "Phase6-Champion" else 6
    ax.errorbar(pt, y, xerr=[[pt - lo], [hi - pt]], fmt="o", capsize=4,
                color=color, markersize=ms, lw=lw,
                markeredgecolor="black" if row["baseline"] == "Phase6-Champion" else color,
                markeredgewidth=1.2 if row["baseline"] == "Phase6-Champion" else 0)
    ax.text(hi + 0.005, y, f"{pt:.4f}  [{lo:.3f}, {hi:.3f}]",
            va="center", ha="left", fontsize=9,
            fontweight="bold" if row["baseline"] == "Phase6-Champion" else "normal",
            color="crimson" if row["baseline"] == "Phase6-Champion" else "black")

    if row["baseline"] != "Phase6-Champion":
        p = delong.loc[row["baseline"], "p_value"]
        flag = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
        ax.text(0.722, y, flag, ha="center", va="center", fontsize=10,
                color="crimson" if p < 0.05 else "gray", fontweight="bold")

ax.set_yticks(ys); ax.set_yticklabels(display_labels, fontsize=10)
ax.axvline(baselines.loc[0, "AUC"], color="crimson", ls=":", lw=1.2, alpha=0.6)
ax.set_xlim(0.715, 0.88)
ax.set_xlabel("AUC on held-out test (N=1094)", fontsize=10.5)
ax.set_title("(a) Prediction — AUC with 1000-boot 95% CI\nDeLong paired vs Champion "
             "(*** p<0.001, ** p<0.01, * p<0.05, n.s. non-significant)",
             fontsize=10.5, loc="left")
ax.grid(axis="x", alpha=0.3)
ax.text(0.725, -0.7, "p-value vs\nChampion",
        fontsize=8, ha="center", color="gray")

# ------- (b) Radar — HRCF vs soft-CF -------
ax2 = fig.add_subplot(gs[0, 1], polar=True)
radar_axes = [("actionability", True),
              ("sparsity", False),
              ("proximity", False),
              ("plausibility", False),
              ("diversity", True)]
labels = ["Actionability", "Sparsity\n(fewer=better)", "Proximity\n(smaller=better)",
          "Plausibility\n(closer=better)", "Diversity"]
vals = {}
for a in ["HR-CF", "soft-CF"]:
    vals[a] = [hrcf.loc[a, m] for m, _ in radar_axes]

# normalise: best=1 on each axis
norm = []
for col_i, (m, hb) in enumerate(radar_axes):
    col = np.array([vals["HR-CF"][col_i], vals["soft-CF"][col_i]], dtype=float)
    if col.max() - col.min() < 1e-9:
        rel = np.array([0.75, 0.75])
    else:
        rel = (col - col.min()) / (col.max() - col.min())
    if not hb:
        rel = 1 - rel
    norm.append(rel)
norm = np.array(norm).T  # (algos, axes)

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]
colors = {"HR-CF": "crimson", "soft-CF": "tab:gray"}
for ai, a in enumerate(["soft-CF", "HR-CF"]):
    vv = list(norm[ai if a == "HR-CF" else 1 - ai]) + [norm[ai if a == "HR-CF" else 1 - ai][0]]
# simpler:
for a in ["soft-CF", "HR-CF"]:
    row_i = 0 if a == "HR-CF" else 1
    vv = list(norm[row_i]) + [norm[row_i][0]]
    ax2.plot(angles, vv, label=a, linewidth=2.2, color=colors[a])
    ax2.fill(angles, vv, alpha=0.18, color=colors[a])

ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(labels, fontsize=8.8)
ax2.set_yticks([0.25, 0.5, 0.75, 1.0])
ax2.set_yticklabels(["", "", "", ""])
ax2.set_ylim(0, 1.05)
ax2.set_title("(b) Intervention — HRCF vs soft-CF (Top-5 menu, 52 high-risk)\n"
              "Actionability: HRCF 100% vs soft-CF 0%",
              fontsize=10.5, pad=18, loc="left")
ax2.legend(loc="upper right", bbox_to_anchor=(1.35, 1.05), fontsize=9.5, frameon=True)

# bottom annotation
fig.text(0.5, 0.01,
         "Champion: Stack(RF + NRBoost + MT-MLP + SVM + ET) + Cleanlab-v6 + Isotonic   |   "
         "HRCF: Hard-constrained projected-gradient counterfactual",
         ha="center", fontsize=9, color="#444")

fig.suptitle("Fig. 1  Key results at a glance — prediction accuracy and intervention actionability",
             fontsize=12, y=1.00)
fig.savefig(OUT / "fig_hero_overview.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_hero_overview.png'}")


# ============================================================
# Fig 2. Architecture diagram
# ============================================================
fig, ax = plt.subplots(figsize=(13.5, 5.8))
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

def box(ax, x, y, w, h, text, fc="#f2f6fb", ec="#28507a", fontsize=9,
        bold=False, txt_color="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.03,rounding_size=0.18",
                                 facecolor=fc, edgecolor=ec, linewidth=1.6))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            color=txt_color)

def arrow(ax, x0, y0, x1, y1, color="#555", lw=1.4):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                  arrowstyle="-|>", mutation_scale=11,
                                  color=color, lw=lw))

# input
box(ax, 0.3, 3.8, 1.8, 1.4, "Input x\n(12 dim,\nz-scored)", fc="#e7f4eb", ec="#2f7a4d", bold=True)

# 5 bases (column)
base_x = 3.2; base_w = 2.3; base_h = 0.95; base_gap = 0.35
bases = [("RF\nn_est=400, depth=10", "#fbe5dd"),
         ("NR-Boost (XGBoost)\n2-stage, drop 10%", "#fff1d1"),
         ("MT-MLP (2-task)\n64→32, λ_ord=0.3", "#e0ecf8"),
         ("SVM-RBF\nC=1, γ=scale", "#ece0f5"),
         ("ExtraTrees\nn_est=400, depth=12", "#dff2ea")]
base_ys = []
for i, (txt, fc) in enumerate(bases):
    y = 7.6 - i * (base_h + base_gap)
    box(ax, base_x, y, base_w, base_h, txt, fc=fc, ec="#555", fontsize=8.5)
    base_ys.append(y + base_h/2)
    arrow(ax, 2.1, 4.5, base_x, y + base_h/2, color="#888")

# CL weights annotation
box(ax, 3.25, 0.5, 2.25, 0.85,
    "Cleanlab-v6 weights\n(299 susp. rows → w=0.3)",
    fc="#fdecec", ec="#aa3333", fontsize=8.5)
arrow(ax, 4.4, 1.35, 4.4, 3.85, color="#aa3333", lw=1.2)

# stacked probs vector
box(ax, 6.2, 3.8, 1.6, 1.4,
    "OOF stacked\nprobs\n[p_RF, p_NR,\n p_MT, p_SVM, p_ET]",
    fc="#f7f7ee", ec="#6a6320", fontsize=8.2)
for by in base_ys:
    arrow(ax, base_x + base_w, by, 6.2, 4.5, color="#888")

# meta learner
box(ax, 8.2, 3.9, 2.0, 1.2, "L2-LR meta\n(C = 10)\ncoef: RF .38,\nMT .32, ET .28,\nSVM -.05, NR .01",
    fc="#e7f2f8", ec="#1f4b73", fontsize=8.3, bold=True)
arrow(ax, 7.8, 4.5, 8.2, 4.5)

# isotonic
box(ax, 10.5, 4.15, 1.7, 0.75, "Isotonic\ncalibration", fc="#fdf1d2", ec="#8a6d0c",
    fontsize=9, bold=True)
arrow(ax, 10.2, 4.5, 10.5, 4.5)

# final prob
box(ax, 12.5, 4.15, 1.7, 0.75, "p̂  (AUC 0.784,\nECE 0.032)", fc="#efe7f6",
    ec="#663399", fontsize=9, bold=True, txt_color="#44226e")
arrow(ax, 12.2, 4.5, 12.5, 4.5)

# HRCF branch
box(ax, 12.5, 2.4, 3.2, 1.3,
    "HRCF generator\n(target = MT-MLP,\n projected-gradient,\n 12 restarts → top-5 menu)",
    fc="#ffe7e7", ec="#a33", fontsize=8.5, bold=True)
arrow(ax, 13.1, 4.15, 13.1, 3.7, color="#a33", lw=1.4)
box(ax, 12.5, 0.5, 3.2, 1.5,
    "Top-5 counterfactual menu\n• actionability 100%\n• mean Δp = 0.31\n• mean cost ≈ ¥13.9k",
    fc="#fff0f0", ec="#a33", fontsize=8.3)
arrow(ax, 14.1, 2.4, 14.1, 2.0, color="#a33", lw=1.4)

# section labels
ax.text(1.2, 8.3, "Input", fontsize=10, fontweight="bold", color="#2f7a4d")
ax.text(4.4, 8.3, "5 Base Learners", fontsize=10, fontweight="bold", color="#333")
ax.text(7.0, 8.3, "Stack OOF", fontsize=10, fontweight="bold", color="#6a6320")
ax.text(9.2, 8.3, "Meta", fontsize=10, fontweight="bold", color="#1f4b73")
ax.text(11.35, 8.3, "Calib.", fontsize=10, fontweight="bold", color="#8a6d0c")
ax.text(13.35, 8.3, "Prediction & Intervention", fontsize=10, fontweight="bold", color="#663399")

ax.set_title("Fig. 2  System architecture — 5-base stacking with Cleanlab-v6 + Isotonic calibration, "
             "plus the HRCF intervention branch", fontsize=11, pad=8)
fig.savefig(OUT / "fig_architecture.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_architecture.png'}")


# ============================================================
# Fig 3. Phase progression (training-curve style)
# ============================================================
# Pull numbers from tables; phases 0→6 summary
phases = [
    ("P0 LR only",          0.766, 0.694, 0.111),
    ("P1 RF only",          0.781, 0.734, 0.110),
    ("P2 MT-MLP",           0.762, 0.702, 0.112),
    ("P3 Stack (3-base)",   0.773, 0.707, 0.111),
    ("P4 + NR-Boost",       0.776, 0.710, 0.111),
    ("P5 + Cleanlab v5",    0.780, 0.711, 0.110),
    ("P6 + CL-v6 + Iso",    0.784, 0.714, 0.110),
]
names = [p[0] for p in phases]
aucs  = [p[1] for p in phases]
balaccs = [p[2] for p in phases]
briers = [p[3] for p in phases]
xs = np.arange(len(phases))

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) AUC
ax = axes[0]
ax.plot(xs, aucs, "o-", color="crimson", lw=2, ms=7)
for i, v in enumerate(aucs):
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8.5)
ax.axhline(aucs[-1], color="crimson", ls=":", alpha=0.4)
ax.set_xticks(xs); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("Test AUC")
ax.set_title("(a) AUC progression across phases", fontsize=10.5)
ax.set_ylim(0.755, 0.792)
ax.grid(alpha=0.3)

# (b) Bal-Acc
ax = axes[1]
ax.plot(xs, balaccs, "s-", color="tab:blue", lw=2, ms=7)
for i, v in enumerate(balaccs):
    ax.annotate(f"{v:.3f}", (i, v), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8.5)
ax.axhline(balaccs[-1], color="tab:blue", ls=":", alpha=0.4)
ax.set_xticks(xs); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("Balanced Accuracy (Bal-Acc-opt τ)")
ax.set_title("(b) Bal-Acc progression across phases", fontsize=10.5)
ax.set_ylim(0.688, 0.720)
ax.grid(alpha=0.3)

# (c) Brier
ax = axes[2]
ax.plot(xs, briers, "^-", color="tab:green", lw=2, ms=7)
for i, v in enumerate(briers):
    ax.annotate(f"{v:.4f}", (i, v), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8.5)
ax.set_xticks(xs); ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8.5)
ax.set_ylabel("Brier score (lower=better)")
ax.set_title("(c) Probability quality (Brier) across phases", fontsize=10.5)
ax.set_ylim(0.1090, 0.1125)
ax.grid(alpha=0.3)

fig.suptitle("Fig. 3  Phase-by-phase progression on held-out test. "
             "Each phase adds one design element to the previous.",
             fontsize=11.5, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig_phase_progression.png", dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_phase_progression.png'}")

print("\nDONE — 3 hero/architecture/progression figures saved.")
