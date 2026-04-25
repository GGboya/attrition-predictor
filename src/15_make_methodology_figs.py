"""Phase 15 — journal-format methodology figures.

Outputs
-------
src/figures/fig_model_architecture.png
    Formal block diagram of the prediction + HRCF system (IEEE/Springer-style).
src/figures/fig_training_workflow.png
    Numbered step-by-step flowchart of training (5-fold OOF -> full-train refit
    -> meta -> calibration -> threshold) plus the inference path.
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Patch
from matplotlib.lines import Line2D

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUT = Path("src/figures"); OUT.mkdir(exist_ok=True, parents=True)

# ============================================================================
# Palette — neutral journal tones (grayscale-friendly for print)
# ============================================================================
C_INPUT   = "#E8EEF4"   # light blue-gray
C_FEAT    = "#DDE7D6"   # light sage
C_BASE    = "#F6E6CC"   # light tan
C_STACK   = "#EAE3F2"   # light lavender
C_META    = "#D6E4EA"   # steel-blue tint
C_CAL     = "#F2DFD6"   # light salmon
C_OUT     = "#DCE9DC"   # mint
C_HRCF    = "#F6D6D6"   # light rose

EDGE = "#2B2B2B"


def box(ax, x, y, w, h, text, fc=C_INPUT, ec=EDGE, lw=1.2,
        fontsize=8.8, bold=False, ital=False, align="center", txt_color="#1a1a1a"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        facecolor=fc, edgecolor=ec, linewidth=lw,
    ))
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.12
    style = "italic" if ital else "normal"
    ax.text(tx, y + h / 2, text, ha=ha, va="center",
            fontsize=fontsize, fontweight="bold" if bold else "normal",
            style=style, color=txt_color)


def arrow(ax, x0, y0, x1, y1, color=EDGE, lw=1.1, style="-|>", ms=10):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=style,
        mutation_scale=ms, color=color, lw=lw,
        shrinkA=0, shrinkB=0,
    ))


def group_frame(ax, x, y, w, h, label, fc="none", ec="#888", lw=0.7, ls="--"):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                            linewidth=lw, linestyle=ls))
    ax.text(x + 0.15, y + h - 0.22, label, fontsize=7.8, color="#555",
            style="italic")


# ============================================================================
# Figure 1. Model architecture — full pipeline
# ============================================================================
fig = plt.figure(figsize=(14.5, 7.5))
ax = fig.add_subplot(111)
ax.set_xlim(0, 18); ax.set_ylim(0, 10.5); ax.axis("off")

# ---- Column 1: Input & preprocessing -----------------------------------
group_frame(ax, 0.1, 0.4, 2.8, 9.8, "I.  Input & preprocessing")

box(ax, 0.3, 8.4, 2.4, 1.0,
    "Raw features\n$\\mathbf{x} \\in \\mathbb{R}^{12}$\n(demographic + work)",
    fc=C_INPUT, bold=True)
box(ax, 0.3, 6.8, 2.4, 1.2,
    "Feature engineering\n$\\phi(\\cdot): \\mathbb{R}^{12}\\!\\to\\!\\mathbb{R}^{41}$\n"
    "· Likert cum. dummies (16)\n· interactions & gaps (8)\n"
    "· log-income, quintile (2)\n· target encoding (2) — OOF",
    fc=C_FEAT, fontsize=7.8)
box(ax, 0.3, 5.3, 2.4, 1.0,
    "z-score standardisation\n(for LR, SVM, MT-MLP,\nk-NN)",
    fc=C_FEAT, fontsize=7.8)

box(ax, 0.3, 3.5, 2.4, 1.2,
    "Cleanlab v6 weight\nestimation\n$w_i = 0.3$ if flagged\n$w_i = 1.0$ otherwise",
    fc="#F5E6E6", fontsize=7.8, bold=True)
box(ax, 0.3, 1.9, 2.4, 1.0,
    "Labels\n$y_i\\in\\{0,1\\}$  (turnover)\n$y^{ord}_i\\in\\{1..5\\}$ (intent)",
    fc=C_INPUT, fontsize=7.8)

arrow(ax, 1.5, 8.4, 1.5, 8.0)
arrow(ax, 1.5, 6.8, 1.5, 6.3)
arrow(ax, 1.5, 5.3, 1.5, 4.7)
arrow(ax, 1.5, 3.5, 1.5, 2.9)

# ---- Column 2: 5 base learners ----------------------------------------
group_frame(ax, 3.1, 0.4, 4.8, 9.8, "II.  Base learners — 5 diverse families")

bx, bw, bh, gap = 3.3, 4.4, 1.10, 0.30
bases_info = [
    ("RF",          "Random Forest",
     "$T=400$, depth $=10$, min-leaf $=5$\nclass\\_weight = balanced\\_subsample",
     "#FBE6DD"),
    ("NR-Boost",    "Noise-robust XGBoost",
     "GCE loss ($q=0.7$), 2 self-paced stages\ndrop top-10% losers\nn\\_rounds $=400$, depth $=5$",
     "#FDF0CF"),
    ("MT-MLP",      "Multi-task MLP",
     "shared [64,32], ReLU, dropout $0.2$\nbinary + ordinal (CORN) heads\n$\\lambda=0.7$, seed-avg $\\times 5$",
     "#DDE5F3"),
    ("SVM-RBF",     "SVM, RBF kernel",
     "$C=1$, $\\gamma=$ scale, class\\_weight=balanced\nprobability via Platt scaling",
     "#E4D8EE"),
    ("ExtraTrees",  "Extra Randomised Trees",
     "$T=400$, depth $=12$, min-leaf $=3$\nrandom split thresholds",
     "#DDEFE0"),
]
base_centres = []
for i, (k, title, body, fc) in enumerate(bases_info):
    y = 8.9 - i * (bh + gap)
    ax.add_patch(FancyBboxPatch((bx, y), bw, bh,
                                 boxstyle="round,pad=0.02,rounding_size=0.10",
                                 facecolor=fc, edgecolor=EDGE, linewidth=1.1))
    ax.text(bx + 0.18, y + bh - 0.24, f"$f^{{({i+1})}}$   {title}",
            fontsize=9.2, fontweight="bold", color="#1a1a1a")
    ax.text(bx + 0.18, y + 0.30, body, fontsize=7.3, color="#222",
            linespacing=1.25)
    base_centres.append((bx + bw, y + bh / 2))
    # arrow from preprocessing into each base
    arrow(ax, 2.7, 7.35, bx, y + bh / 2, color="#777", lw=0.9)

# CL weights routed into weight-supporting bases
for i in [0, 1, 3, 4]:
    ax.add_patch(FancyArrowPatch(
        (2.7, 4.1), (bx - 0.05, base_centres[i][1] - 0.08),
        arrowstyle="->", mutation_scale=8, color="#a33",
        lw=0.8, connectionstyle="arc3,rad=0.15"))
ax.text(2.9, 4.35, "$w_i$", color="#a33", fontsize=8.5, fontweight="bold")

# ---- Column 3: Stacking / OOF probabilities ----------------------------
group_frame(ax, 8.1, 0.4, 2.4, 9.8, "III.  Stacking layer")

box(ax, 8.3, 5.4, 2.0, 2.0,
    "Stacked probabilities\n$\\mathbf{z}_i = [\\,p^{(1)},p^{(2)},\\ldots,p^{(5)}\\,]$\n"
    "$\\in [0,1]^{5}$\n\nTrain: OOF (5-fold CV)\nTest: full-train refit",
    fc=C_STACK, fontsize=7.8, bold=True)

# connect all 5 bases to stack box
for _, cy in base_centres:
    arrow(ax, bx + bw + 0.02, cy, 8.3, 6.4, color="#666", lw=0.9)

# logit transform note
box(ax, 8.3, 4.2, 2.0, 0.7,
    "logit $(\\cdot)$  transform",
    fc="#FFFFFF", ec="#888", fontsize=7.6, ital=True)
arrow(ax, 9.3, 5.4, 9.3, 4.9, lw=0.9)

# ---- Column 4: Meta-learner -------------------------------------------
group_frame(ax, 10.7, 0.4, 2.6, 9.8, "IV.  Meta-learner")

box(ax, 10.9, 5.0, 2.2, 2.4,
    "L2 Logistic Regression\n(meta)\n\n"
    "$\\hat p_{\\text{stack}} = \\sigma(\\beta^{\\top} \\text{logit}(\\mathbf{z})+b)$\n\n"
    "$C=10$ (weak $\\ell_2$)\nlbfgs, class\\_weight = none\n\n"
    "$\\beta$: RF .38, MT .32, ET .28\nSVM -.05, NR .01",
    fc=C_META, fontsize=7.6, bold=False)
arrow(ax, 10.3, 4.55, 10.9, 5.5, lw=1.0)

# ---- Column 5: Calibration + threshold --------------------------------
group_frame(ax, 13.5, 0.4, 2.0, 9.8, "V.  Calibration & decision")

box(ax, 13.7, 6.8, 1.6, 1.4,
    "Isotonic\ncalibration\n(PAV algorithm,\nfit on OOF)",
    fc=C_CAL, fontsize=7.8, bold=True)
arrow(ax, 13.1, 6.2, 13.7, 7.4, lw=1.0)

box(ax, 13.7, 5.0, 1.6, 1.2,
    "Calibrated prob.\n$\\hat p_i = g(\\hat p_{\\text{stack}})$\nECE $=0.032$",
    fc="#F7EED7", fontsize=7.6)
arrow(ax, 14.5, 6.8, 14.5, 6.2, lw=1.0)

box(ax, 13.7, 3.3, 1.6, 1.2,
    "Threshold\n$\\tau^{\\star}=0.135$\n(Bal-Acc optimal\n on OOF)",
    fc="#FFFFFF", fontsize=7.6, ital=True)
arrow(ax, 14.5, 5.0, 14.5, 4.5, lw=1.0)

box(ax, 13.7, 1.8, 1.6, 1.1,
    "$\\hat y_i = \\mathbb{1}[\\hat p_i \\geq \\tau^{\\star}]$",
    fc=C_OUT, fontsize=9.0, bold=True)
arrow(ax, 14.5, 3.3, 14.5, 2.9, lw=1.1)

# ---- Column 6: HRCF intervention branch ------------------------------
group_frame(ax, 15.7, 0.4, 2.2, 9.8, "VI.  Intervention (HRCF)")

box(ax, 15.9, 7.4, 1.8, 1.2,
    "If $\\hat p_i \\geq \\tau^{\\star}$:\nflag high-risk",
    fc="#FFF0F0", fontsize=7.8, bold=True)
arrow(ax, 15.3, 2.3, 15.9, 8.0, color="#a33", lw=1.0,
      style="-|>", ms=9)

box(ax, 15.9, 4.5, 1.8, 2.5,
    "HRCF generator\n\nProjected-gradient\non MT-MLP surrogate\n\n"
    "constraints:\n· immutable (6)\n· monotone (5)\n· bounded (1)\n· step-snap\n\n"
    "12 restarts + greedy\nL2 diversity",
    fc=C_HRCF, fontsize=7.4)
arrow(ax, 16.8, 7.4, 16.8, 7.0, color="#a33", lw=1.1)

box(ax, 15.9, 2.6, 1.8, 1.6,
    "Top-5 CF menu\n\n$\\Delta p \\approx 0.31$\ncost $\\approx$ ¥13.9k\n100% actionable",
    fc="#FFDFDF", fontsize=7.6, bold=True)
arrow(ax, 16.8, 4.5, 16.8, 4.2, color="#a33", lw=1.1)

# Title + legend
ax.text(9.0, 10.15, "Model Architecture", fontsize=14,
        fontweight="bold", ha="center", color="#111")
ax.text(9.0, 9.75,
        "Stacked ensemble with noise-robust sample weights and isotonic calibration "
        "(prediction) + hard-constrained projected-gradient counterfactuals (intervention)",
        fontsize=9.5, ha="center", color="#444", style="italic")

legend_patches = [
    Patch(facecolor=C_INPUT, edgecolor=EDGE, label="Input / labels"),
    Patch(facecolor=C_FEAT,  edgecolor=EDGE, label="Feature engineering"),
    Patch(facecolor="#F5E6E6", edgecolor=EDGE, label="Noise-aware weights"),
    Patch(facecolor=C_STACK, edgecolor=EDGE, label="Stacking layer"),
    Patch(facecolor=C_META,  edgecolor=EDGE, label="Meta-learner"),
    Patch(facecolor=C_CAL,   edgecolor=EDGE, label="Calibration"),
    Patch(facecolor=C_OUT,   edgecolor=EDGE, label="Decision"),
    Patch(facecolor=C_HRCF,  edgecolor=EDGE, label="HRCF intervention"),
]
ax.legend(handles=legend_patches, loc="lower center",
          bbox_to_anchor=(0.5, -0.01), ncol=8, fontsize=7.8,
          frameon=False, handletextpad=0.4, columnspacing=1.0)

fig.savefig(OUT / "fig_model_architecture.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT / "fig_model_architecture.pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_model_architecture.png'}")
print(f"wrote {OUT / 'fig_model_architecture.pdf'}")


# ============================================================================
# Figure 2. Training workflow — numbered flowchart
# ============================================================================
fig = plt.figure(figsize=(14.5, 8.8))
ax = fig.add_subplot(111)
ax.set_xlim(0, 18); ax.set_ylim(0, 12.5); ax.axis("off")

def step_box(ax, x, y, w, h, step_num, title, body,
             fc="#F3F3F3", ec=EDGE, title_fc="#3A3A3A", fontsize=8.0):
    # main rounded box
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                 boxstyle="round,pad=0.02,rounding_size=0.12",
                                 facecolor=fc, edgecolor=ec, linewidth=1.3))
    # numbered circle
    circ_r = 0.28
    cx, cy = x + 0.42, y + h - 0.42
    ax.add_patch(plt.Circle((cx, cy), circ_r, facecolor=title_fc,
                             edgecolor="none", zorder=3))
    ax.text(cx, cy, str(step_num), ha="center", va="center",
            fontsize=10, fontweight="bold", color="white", zorder=4)
    ax.text(x + 0.85, y + h - 0.42, title, fontsize=9.2,
            fontweight="bold", va="center", color="#1a1a1a")
    ax.text(x + 0.42, y + h - 0.95, body, fontsize=fontsize,
            va="top", ha="left", color="#222", linespacing=1.3)

# ---- Title ----
ax.text(9.0, 12.15, "Training Workflow",
        fontsize=14, fontweight="bold", ha="center", color="#111")
ax.text(9.0, 11.75,
        "From raw dataset (N = 5 469) to calibrated stacked classifier (Phase-6 champion)",
        fontsize=10, ha="center", color="#444", style="italic")

# ---- Step 1. Split ----
step_box(ax, 0.3, 9.4, 4.0, 1.9, 1,
         "Stratified train/test split",
         "• N = 5 469; 80 / 20 stratified on $y$\n"
         "• seed = 42\n"
         "• train: 4 375 (pos 14.5%)\n"
         "• test:   1 094 (pos 14.5%)  — frozen",
         fc=C_INPUT)

# ---- Step 2. Feature engineering ----
step_box(ax, 4.8, 9.4, 4.0, 1.9, 2,
         "Leak-safe feature build $\\phi$",
         "12 raw $\\rightarrow$ 41-d\n"
         "• 4 Likert × 4 cum. dummies (16)\n"
         "• 5 product + 3 gap interactions (8)\n"
         "• log / quintile of income (2)\n"
         "• composite index (1), target-enc. (2)",
         fc=C_FEAT)

# ---- Step 3. Cleanlab weights ----
step_box(ax, 9.3, 9.4, 4.2, 1.9, 3,
         "Cleanlab v6 noise weights",
         "• 5-fold CV RF confident-learning\n"
         "• flag rows with $\\hat p_{\\text{wrong}} > 0.5$ (299 rows)\n"
         "• $w_i = 0.3$ if flagged, else $1.0$\n"
         "• saved: sample\\_weights\\_v6.npy",
         fc="#F5E6E6")

# ---- Step 4. 5-fold OOF ----
step_box(ax, 0.3, 6.4, 9.2, 2.6, 4,
         "5-fold stratified CV — OOF base probabilities",
         "For each fold $f\\!\\in\\!\\{1..5\\}$:\n"
         "  • fit StandardScaler on $\\text{TR}_f$ only, transform $\\text{TR}_f,\\text{VA}_f$\n"
         "  • train each base $m\\!\\in\\!\\{\\text{RF}, \\text{NRB}, \\text{MT-MLP}, \\text{SVM}, \\text{ET}\\}$\n"
         "    on $\\text{TR}_f$ with $w_i$ (MT-MLP uses uniform; seed-avg ×5 with patience-25 early-stop)\n"
         "  • predict on $\\text{VA}_f$ $\\rightarrow$ fill $P^{\\text{oof}}_{m}[\\text{VA}_f]$\n"
         "Result: $P^{\\text{oof}} \\in [0,1]^{4375\\times 5}$, each row never-trained-on.",
         fc="#F1EBDE")

# 5 fold visualization inside the step
fx, fy, fw, fh = 5.2, 6.55, 4.0, 0.28
for i in range(5):
    yy = fy + (5 - 1 - i) * (fh + 0.05)
    for j in range(5):
        seg_w = fw / 5
        xx = fx + j * seg_w
        c = "#C57C7C" if i == j else "#8AA7C2"
        ax.add_patch(Rectangle((xx, yy), seg_w - 0.02, fh,
                                facecolor=c, edgecolor="white", linewidth=0.5))
    ax.text(fx - 0.15, yy + fh / 2, f"f{i+1}", fontsize=6.5,
            ha="right", va="center", color="#555")
ax.text(fx + fw / 2, fy - 0.22,
        "validation fold (predict)   training folds (fit)",
        fontsize=6.8, ha="center", color="#555")
# legend
ax.add_patch(Rectangle((fx, fy - 0.55), 0.18, 0.13, facecolor="#C57C7C"))
ax.text(fx + 0.22, fy - 0.49, "held-out", fontsize=6.8, color="#555")
ax.add_patch(Rectangle((fx + 1.1, fy - 0.55), 0.18, 0.13, facecolor="#8AA7C2"))
ax.text(fx + 1.32, fy - 0.49, "training", fontsize=6.8, color="#555")

# ---- Step 5. Full-train refit ----
step_box(ax, 9.9, 6.4, 3.8, 2.6, 5,
         "Full-train refit for test",
         "• fit StandardScaler on all 4 375\n"
         "• retrain each of the 5 bases\n  on the full training set\n  with $w_i$ applied\n"
         "• predict on test (1 094)\n"
         "  $\\rightarrow P^{\\text{te}} \\in [0,1]^{1094\\times 5}$",
         fc="#F1EBDE")

# ---- Step 6. Meta-learner ----
step_box(ax, 0.3, 3.4, 4.4, 2.6, 6,
         "Meta-learner (L2 logistic regression)",
         "• $Z^{\\text{oof}} = \\text{logit}(P^{\\text{oof}})$\n"
         "• fit $g_\\beta$: $C=10$, lbfgs, max\\_iter = 2 000\n"
         "• 5-fold CV refit on $Z^{\\text{oof}}$ to generate\n"
         "  meta-level OOF for downstream isotonic\n"
         "• full fit on all $Z^{\\text{oof}}$ for test scoring\n"
         "• learned $\\beta$: RF .38, MT .32, ET .28,\n  SVM $-$.05, NRB .01",
         fc=C_META)

# ---- Step 7. Isotonic ----
step_box(ax, 5.1, 3.4, 4.0, 2.6, 7,
         "Isotonic calibration",
         "• fit PAV on meta OOF probs vs $y$\n"
         "• monotone piecewise-constant\n"
         "  map $g: [0,1]\\to[0,1]$\n"
         "• apply $g$ to meta test probs\n"
         "  $\\Rightarrow \\hat p$ (calibrated)\n"
         "• result: ECE 0.032 (raw 0.071)",
         fc=C_CAL)

# ---- Step 8. Threshold ----
step_box(ax, 9.5, 3.4, 4.2, 2.6, 8,
         "Threshold selection (Bal-Acc optimal)",
         "Over meta-OOF calibrated probs:\n"
         "• sweep $\\tau \\in [0.05, 0.95]$ step 0.001\n"
         "• $\\tau^{\\star} = \\arg\\max_\\tau\\,"
         "\\frac{1}{2}[\\mathrm{TPR}(\\tau)+\\mathrm{TNR}(\\tau)]$\n"
         "• $\\tau^{\\star} = 0.135$\n"
         "• also report F1-optimal $\\tau = 0.185$",
         fc="#EAE0D3")

# ---- Step 9. Evaluation ----
step_box(ax, 0.3, 0.3, 6.4, 2.7, 9,
         "Held-out evaluation (test = 1 094, never touched)",
         "• apply full pipeline: $\\phi \\to$ scale $\\to$ 5 bases $\\to$ logit $\\to$ meta $\\to g \\to \\tau^{\\star}$\n"
         "• report:\n"
         "    AUC = 0.7844 [0.750, 0.816]    PR-AUC = 0.427    Brier = 0.110    ECE = 0.032\n"
         "    Sens@$\\tau^{\\star}$ = 0.730        Spec@$\\tau^{\\star}$ = 0.698    Bal-Acc = 0.714\n"
         "• 1 000-boot 95% CI; DeLong paired vs 7 single-model baselines",
         fc=C_OUT, fontsize=8.2)

# ---- Step 10. HRCF ----
step_box(ax, 7.0, 0.3, 6.7, 2.7, 10,
         "Intervention generation (HRCF)",
         "For every $x_i$ with $\\hat p_i \\geq \\tau^{\\star}$:\n"
         "• target = MT-MLP (differentiable surrogate for stack)\n"
         "• minimise $\\mathcal{L} = \\max(0, \\ell(x') - \\ell^\\star) + \\alpha\\,c(x,x')$\n"
         "• Adam + projection cascade (immutable / monotone / bound / snap)\n"
         "• 12 random restarts; greedy L2-diversity picks top-5\n"
         "• empirical yield: 52 high-risk $\\to$ 105 CFs; $\\Delta p\\!\\approx\\!0.31$; cost $\\approx$ ¥13.9k",
         fc=C_HRCF, fontsize=8.2)

# ---- Connectors ----
# 1 -> 2, 2 -> 3
arrow(ax, 4.3, 10.35, 4.8, 10.35, lw=1.4)
arrow(ax, 8.8, 10.35, 9.3, 10.35, lw=1.4)

# 3 -> 4 (down-left)
arrow(ax, 11.4, 9.4, 11.4, 9.1, lw=1.4)
arrow(ax, 11.4, 9.08, 4.9, 9.06, lw=1.4)  # horizontal back to step 4
# 2 -> 4 (down)
arrow(ax, 2.3, 9.4, 2.3, 9.05, lw=1.4)

# 4 -> 5
arrow(ax, 9.5, 7.7, 9.9, 7.7, lw=1.4)
# 4 -> 6
arrow(ax, 2.5, 6.4, 2.5, 6.0, lw=1.4)
# 5 -> 6 (across)
arrow(ax, 11.8, 6.4, 11.8, 6.12, lw=1.3)
arrow(ax, 11.8, 6.10, 2.6, 6.08, lw=1.3, style="-|>", ms=8)

# 6 -> 7 -> 8
arrow(ax, 4.7, 4.7, 5.1, 4.7, lw=1.4)
arrow(ax, 9.1, 4.7, 9.5, 4.7, lw=1.4)

# 8 -> 9 (down-left)
arrow(ax, 11.6, 3.4, 11.6, 3.15, lw=1.3)
arrow(ax, 11.6, 3.14, 3.5, 3.12, lw=1.3, style="-|>", ms=8)

# 9 -> 10
arrow(ax, 6.7, 1.65, 7.0, 1.65, lw=1.4)

# ---- sub-caption ----
fig.text(0.5, 0.02,
         "Solid arrows: data-flow dependencies.  Steps 4 and 5 share base-learner "
         "hyper-parameters and $w_i$.  Step 7 consumes the meta-OOF "
         "probabilities produced by step 6; step 8 thresholds are chosen on those "
         "same OOF probabilities so the held-out test remains untouched until step 9.",
         ha="center", fontsize=8.0, color="#555", style="italic")

fig.savefig(OUT / "fig_training_workflow.png", dpi=180, bbox_inches="tight")
fig.savefig(OUT / "fig_training_workflow.pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_training_workflow.png'}")
print(f"wrote {OUT / 'fig_training_workflow.pdf'}")

print("\nDONE — 2 methodology figures (architecture + training workflow).")
