"""Phase 16 — real model architecture figure (champion).

Produces an architecture diagram with actual visual iconography for each
component instead of text-in-a-box:
  - Random Forest / ExtraTrees: actual mini decision-tree icons
  - NR-Boost: sequential trees with residual arrows
  - MT-MLP: layered neurons with inter-layer connections
  - SVM:    2-class scatter + hyperplane + margin lines
  - Stack:  concrete 5-slot probability vector
  - Meta:   sigmoid node with β · z
  - Isotonic: PAV step function curve
  - Output: probability bar with τ★ threshold tick

Output:
  src/figures/fig_champion_architecture.png (+ .pdf)
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
from matplotlib.patches import (FancyArrowPatch, FancyBboxPatch, Rectangle,
                                 Circle)

plt.rcParams["font.sans-serif"] = ["Arial", "Helvetica", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42

OUT = Path("src/figures"); OUT.mkdir(exist_ok=True, parents=True)

# ─── palette (journal-friendly, print-safe) ─────────────────────────────
C_IN    = "#EEF2F7"
C_FEAT  = "#E3EDD9"
C_CL    = "#F5E3E3"
C_RF    = "#FBE3D3"
C_BOOST = "#FFF2CF"
C_MLP   = "#DFE6F6"
C_SVM   = "#EADAF0"
C_ET    = "#DAEBDA"
C_STACK = "#EAE0F4"
C_META  = "#CFE0E8"
C_CAL   = "#F5DBCB"
C_OUT   = "#DCE9DC"
DARK    = "#222"


# ─── drawing primitives ────────────────────────────────────────────────
def arrow(ax, x0, y0, x1, y1, color=DARK, lw=1.1, ms=10, style="-|>", **kw):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                  mutation_scale=ms, color=color, lw=lw,
                                  shrinkA=0, shrinkB=0, **kw))


def tree_icon(ax, cx, cy, w=0.7, h=0.65,
              root_c="#4a4a4a", node_c="#888",
              leaf_cs=("#4472a8", "#c04545", "#4472a8", "#c04545")):
    """A schematic binary decision tree: root → 2 internals → 4 leaves."""
    # root
    ax.add_patch(Circle((cx, cy + h/2), 0.055, facecolor=root_c,
                         edgecolor="black", linewidth=0.5, zorder=4))
    # 2 internal nodes
    mid_xs = [cx - w/4, cx + w/4]
    mid_y = cy + h/8
    for mx in mid_xs:
        ax.plot([cx, mx], [cy + h/2 - 0.04, mid_y + 0.035],
                 "-", color="#777", lw=0.8, zorder=2)
        ax.add_patch(Circle((mx, mid_y), 0.045, facecolor=node_c,
                             edgecolor="black", linewidth=0.4, zorder=4))
    # 4 leaves
    leaf_xs = np.linspace(cx - w/2 + 0.04, cx + w/2 - 0.04, 4)
    leaf_y = cy - h/2 + 0.03
    for i, lx in enumerate(leaf_xs):
        mx = mid_xs[0] if i < 2 else mid_xs[1]
        ax.plot([mx, lx], [mid_y - 0.03, leaf_y + 0.035],
                 "-", color="#777", lw=0.8, zorder=2)
        ax.add_patch(Rectangle((lx - 0.045, leaf_y - 0.03), 0.09, 0.065,
                                facecolor=leaf_cs[i], edgecolor="black",
                                linewidth=0.4, zorder=4))


def forest_icon(ax, cx, cy, w=2.3, h=0.95, n_show=3, ellipsis=True):
    """n small trees in a row, with '…' to hint at 400."""
    positions = np.linspace(cx - w/2 + 0.35, cx + w/2 - 0.35, n_show)
    for px in positions:
        tree_icon(ax, px, cy, w=0.55, h=h*0.88)
    if ellipsis:
        ax.text(positions[-1] + 0.32, cy, "...", fontsize=14,
                 va="center", ha="center", color="#555",
                 fontweight="bold")


def random_forest_icon(ax, cx, cy, w=2.3, h=0.95):
    forest_icon(ax, cx, cy, w, h, n_show=3)


def extratrees_icon(ax, cx, cy, w=2.3, h=0.95):
    """Like a forest but with zig-zag (random-split) branches."""
    positions = np.linspace(cx - w/2 + 0.35, cx + w/2 - 0.35, 3)
    for px in positions:
        # root
        ax.add_patch(Circle((px, cy + h*0.45), 0.055, facecolor="#4a4a4a",
                             edgecolor="black", linewidth=0.5, zorder=4))
        # crooked branches (random split illustrated)
        mids = [px - 0.14, px + 0.14]
        mid_y = cy + 0.10
        for mx in mids:
            # zig-zag path
            mid_pt = ((px + mx) / 2 + 0.03 * (1 if mx < px else -1),
                      (cy + h*0.45 + mid_y) / 2)
            ax.plot([px, mid_pt[0], mx],
                     [cy + h*0.45 - 0.04, mid_pt[1], mid_y + 0.035],
                     "-", color="#888", lw=0.7, zorder=2)
            ax.add_patch(Circle((mx, mid_y), 0.045, facecolor="#888",
                                 edgecolor="black", linewidth=0.4, zorder=4))
        # leaves
        leaf_xs = np.linspace(px - 0.22, px + 0.22, 4)
        leaf_y = cy - h/2 + 0.06
        leaf_cs = ["#4472a8", "#c04545", "#4472a8", "#c04545"]
        for i, lx in enumerate(leaf_xs):
            mx = mids[0] if i < 2 else mids[1]
            ax.plot([mx, lx + 0.02 * np.sin(i)], [mid_y - 0.03, leaf_y + 0.035],
                     "-", color="#888", lw=0.7, zorder=2)
            ax.add_patch(Rectangle((lx - 0.045, leaf_y - 0.03), 0.09, 0.06,
                                    facecolor=leaf_cs[i], edgecolor="black",
                                    linewidth=0.4, zorder=4))
    ax.text(positions[-1] + 0.32, cy, "...", fontsize=14,
             va="center", ha="center", color="#555", fontweight="bold")


def boosting_icon(ax, cx, cy, w=2.3, h=0.9):
    """T1 → T2 with residual arrow between. Shows stage-2 GCE drop."""
    xs = [cx - w*0.28, cx + w*0.04]
    for i, tx in enumerate(xs):
        tree_icon(ax, tx, cy, w=0.58, h=h*0.85)
        ax.text(tx, cy - h/2 - 0.02, f"$T_{{{i+1}}}$",
                 fontsize=8.5, ha="center", va="top",
                 style="italic", color="#333")
    # residual arrow (curved)
    ax.add_patch(FancyArrowPatch((xs[0] + 0.32, cy + 0.05),
                                  (xs[1] - 0.32, cy + 0.05),
                                  arrowstyle="-|>", mutation_scale=9,
                                  color="#c06060", lw=1.2,
                                  connectionstyle="arc3,rad=-0.25"))
    ax.text((xs[0] + xs[1]) / 2, cy + 0.30, "residual",
             fontsize=7, ha="center", color="#c06060", style="italic")
    # final + drop illustration
    drop_x = xs[1] + 0.55
    ax.text(drop_x, cy + 0.05, "drop\ntop-10%",
             fontsize=6.8, ha="center", va="center",
             color="#a24a3a", style="italic",
             bbox=dict(boxstyle="round,pad=0.14", facecolor="#ffeaea",
                       edgecolor="#c06060", linewidth=0.7))


def mlp_icon(ax, cx, cy, layers=(6, 8, 5, 2), w=2.15, h=0.9):
    """Neural-net columns of neurons with full-mesh light connections."""
    xs = np.linspace(cx - w/2 + 0.08, cx + w/2 - 0.08, len(layers))
    pos = []
    for i, (xp, n) in enumerate(zip(xs, layers)):
        ys = np.linspace(cy - h/2 + 0.07, cy + h/2 - 0.07, n)
        pos.append(list(zip([xp] * n, ys)))
    # connections
    for i in range(len(layers) - 1):
        for (x1, y1) in pos[i]:
            for (x2, y2) in pos[i+1]:
                ax.plot([x1, x2], [y1, y2], "-",
                         color="#999", lw=0.18, alpha=0.7, zorder=1)
    # neurons
    node_colors = ["#88a0c8"] * (len(layers) - 1) + ["#d4955c"]
    for i, col in enumerate(pos):
        for (x, y) in col:
            ax.add_patch(Circle((x, y), 0.048,
                                 facecolor=node_colors[i],
                                 edgecolor="black", linewidth=0.45,
                                 zorder=3))
    # head labels
    last = pos[-1]
    if len(last) >= 2:
        ax.annotate("bin", xy=(last[0][0] + 0.07, last[0][1]),
                     xytext=(last[0][0] + 0.32, last[0][1]),
                     fontsize=6.8, va="center", color="#7a4a1a",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#aaa"))
        ax.annotate("ord", xy=(last[1][0] + 0.07, last[1][1]),
                     xytext=(last[1][0] + 0.32, last[1][1]),
                     fontsize=6.8, va="center", color="#7a4a1a",
                     arrowprops=dict(arrowstyle="-", lw=0.5, color="#aaa"))


def svm_icon(ax, cx, cy, w=1.2, h=0.85):
    """Margin classifier: 2 classes, hyperplane, dashed margins, SVs circled."""
    ax.add_patch(Rectangle((cx - w/2, cy - h/2), w, h,
                            facecolor="#fbf8fc", edgecolor="#999",
                            linewidth=0.6, zorder=1))
    # hyperplane (diagonal)
    x0, y0 = cx - w/2 + 0.06, cy + h/2 - 0.06
    x1, y1 = cx + w/2 - 0.06, cy - h/2 + 0.06
    ax.plot([x0, x1], [y0, y1], "-", color="black", lw=1.2, zorder=3)
    # margin lines (parallel-offset)
    dx, dy = -(y1 - y0), (x1 - x0)
    L = np.hypot(dx, dy); dx, dy = dx / L, dy / L
    off = 0.10
    ax.plot([x0 + off*dx, x1 + off*dx], [y0 + off*dy, y1 + off*dy],
             "--", color="#777", lw=0.7, zorder=3)
    ax.plot([x0 - off*dx, x1 - off*dx], [y0 - off*dy, y1 - off*dy],
             "--", color="#777", lw=0.7, zorder=3)
    # points
    rng = np.random.default_rng(7)
    # positive (red circles) above-right
    for _ in range(7):
        ang = rng.uniform(0.15, 0.85)
        r = rng.uniform(0.10, 0.32)
        xx = cx + r * np.cos(ang); yy = cy + r * np.sin(ang)
        ax.plot(xx, yy, "o", color="#c04545", ms=3.3, zorder=4)
    # negative (blue squares) below-left
    for _ in range(7):
        ang = rng.uniform(np.pi + 0.15, np.pi + 0.85)
        r = rng.uniform(0.10, 0.32)
        xx = cx + r * np.cos(ang); yy = cy + r * np.sin(ang)
        ax.plot(xx, yy, "s", color="#4472a8", ms=3.3, zorder=4)
    # 2 support vectors circled (on the margins)
    for sv_off in [+off - 0.02, -off + 0.02]:
        svx = (x0 + x1) / 2 + sv_off * dx
        svy = (y0 + y1) / 2 + sv_off * dy
        ax.add_patch(Circle((svx, svy), 0.07, facecolor="none",
                             edgecolor="#222", linewidth=0.9, zorder=5))


def isotonic_icon(ax, cx, cy, w=1.15, h=0.95):
    """Monotone step function drawn inside a small panel."""
    ax.add_patch(FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                                 boxstyle="round,pad=0.02,rounding_size=0.05",
                                 facecolor=C_CAL, edgecolor=DARK, linewidth=1.0))
    # axes
    ax.plot([cx - w/2 + 0.12, cx + w/2 - 0.08],
             [cy - h/2 + 0.12, cy - h/2 + 0.12],
             "-", color="#555", lw=0.6)
    ax.plot([cx - w/2 + 0.12, cx - w/2 + 0.12],
             [cy - h/2 + 0.12, cy + h/2 - 0.14],
             "-", color="#555", lw=0.6)
    # step-function knots
    yk = np.array([0.04, 0.08, 0.12, 0.20, 0.35, 0.55, 0.75, 0.88, 0.94])
    xs = np.linspace(cx - w/2 + 0.14, cx + w/2 - 0.10, len(yk))
    ys = cy - h/2 + 0.14 + yk * (h - 0.30)
    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i+1]], [ys[i], ys[i]], "-",
                 color="#a24a3a", lw=1.6, zorder=3)
        ax.plot([xs[i+1], xs[i+1]], [ys[i], ys[i+1]], "-",
                 color="#a24a3a", lw=1.6, zorder=3)
    # diagonal reference (y = x) for context
    ax.plot([cx - w/2 + 0.14, cx + w/2 - 0.10],
             [cy - h/2 + 0.14, cy + h/2 - 0.16],
             ":", color="#888", lw=0.6, zorder=2)


# ─── figure ────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9.2))
ax = fig.add_subplot(111)
ax.set_xlim(0, 17); ax.set_ylim(0, 10.2); ax.axis("off")

# Title
ax.text(8.5, 9.85, "Phase-6 Champion — Model Architecture",
         fontsize=15.5, ha="center", fontweight="bold", color="#111")
ax.text(8.5, 9.50,
         "Stacking of 5 diverse base learners  →  L2-logistic meta  →  "
         "isotonic calibration  →  τ★-thresholded decision",
         fontsize=10.2, ha="center", style="italic", color="#444")

# ── Input x ── ----------------------------------------------------------
ax.add_patch(FancyBboxPatch((0.25, 4.4), 1.0, 1.2,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor=C_IN, edgecolor=DARK, linewidth=1.2))
ax.text(0.75, 5.38, "$\\mathbf{x}$", fontsize=17, ha="center",
         fontweight="bold")
ax.text(0.75, 5.05, r"$\in\mathbb{R}^{12}$", fontsize=9.5, ha="center")
# 12 mini dots illustrating the feature vector
for i in range(12):
    r, c = divmod(i, 6)
    ax.plot(0.48 + c * 0.11, 4.82 - r * 0.12, "o",
             color="#88a0c8", markersize=2.8)
ax.text(0.75, 4.28, "raw survey\nfeatures", fontsize=7, ha="center",
         color="#555", style="italic")

# ── Feature engineering φ ── ---------------------------------------------
ax.add_patch(FancyBboxPatch((1.75, 4.4), 1.2, 1.2,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor=C_FEAT, edgecolor=DARK, linewidth=1.2))
ax.text(2.35, 5.28, r"$\phi(\cdot)$", fontsize=16, ha="center",
         fontweight="bold")
ax.text(2.35, 4.95, "feature\nengineering", fontsize=8, ha="center",
         color="#333")
ax.text(2.35, 4.55, r"$\mathbb{R}^{12}\!\to\!\mathbb{R}^{41}$",
         fontsize=9, ha="center", color="#333")
# arrow x → φ with tensor shape label
arrow(ax, 1.28, 5.0, 1.75, 5.0, lw=1.3, ms=11)
ax.text(1.52, 5.20, "(12,)", fontsize=7, ha="center",
         color="#666", style="italic")

# ── standardisation ── --------------------------------------------------
ax.add_patch(Rectangle((3.2, 4.72), 0.85, 0.6,
                        facecolor="#F7F7F7", edgecolor="#888", linewidth=0.7))
ax.text(3.625, 5.10, "z-score", fontsize=8.5, ha="center", color="#333")
ax.text(3.625, 4.85, "(MLP, SVM)", fontsize=6.5, ha="center", color="#888",
         style="italic")
arrow(ax, 2.95, 5.0, 3.2, 5.02, lw=1.2)
ax.text(3.08, 5.22, "(41,)", fontsize=7, ha="center",
         color="#666", style="italic")

# ── 5 base learners (parallel) ──────────────────────────────────────────
base_data = [
    ("RF",       "Random Forest",          C_RF,    8.10, "$T=400,\\ d=10$"),
    ("NRB",      "NR-Boost (XGBoost)",     C_BOOST, 6.55, "GCE $q\\!=\\!0.7$, 2-stage"),
    ("MTMLP",    "MT-MLP",                 C_MLP,   5.00, "[64, 32] → bin + ord"),
    ("SVM",      "SVM (RBF)",              C_SVM,   3.45, "$C=1,\\ \\gamma=$scale"),
    ("ET",       "ExtraTrees",             C_ET,    1.90, "random splits, $T=400$"),
]
BOX_X, BOX_W, BOX_H = 4.55, 4.60, 1.35

for key, title, color, cy, hp_label in base_data:
    # container box
    ax.add_patch(FancyBboxPatch((BOX_X, cy - BOX_H/2), BOX_W, BOX_H,
                                 boxstyle="round,pad=0.02,rounding_size=0.10",
                                 facecolor=color, edgecolor=DARK, linewidth=1.1))
    # name header
    ax.text(BOX_X + 0.15, cy + BOX_H/2 - 0.17, title,
             fontsize=10.5, fontweight="bold", va="center", color="#111")
    # meta params
    ax.text(BOX_X + BOX_W - 0.15, cy + BOX_H/2 - 0.17, hp_label,
             fontsize=7.8, va="center", ha="right",
             color="#333", style="italic")

    # iconography centered
    ic_cx = BOX_X + BOX_W/2
    ic_cy = cy - 0.12
    if key == "RF":
        random_forest_icon(ax, ic_cx, ic_cy, w=2.9, h=0.95)
    elif key == "NRB":
        boosting_icon(ax, ic_cx, ic_cy, w=2.6, h=0.9)
    elif key == "MTMLP":
        mlp_icon(ax, ic_cx, ic_cy, layers=(6, 8, 5, 2), w=2.4, h=0.95)
    elif key == "SVM":
        svm_icon(ax, ic_cx, ic_cy, w=1.3, h=0.9)
    elif key == "ET":
        extratrees_icon(ax, ic_cx, ic_cy, w=2.9, h=0.95)

    # output probability node on right edge
    px_x = BOX_X + BOX_W + 0.30
    ax.add_patch(Circle((px_x, cy), 0.19, facecolor="white",
                         edgecolor="#333", linewidth=1.0, zorder=4))
    ax.text(px_x, cy, f"$p_{ {'RF':'1','NRB':'2','MTMLP':'3','SVM':'4','ET':'5'}[key] }$",
             fontsize=9, ha="center", va="center")

    # arrow from φ/standardiser into base
    src_x = 4.10 if key in ("MTMLP", "SVM") else 2.95
    src_y = 5.02
    arrow(ax, src_x, src_y, BOX_X - 0.03, cy, lw=0.9, color="#777", ms=9)

    # arrow from base-output circle into stack (done later)

# ── Stack vector ────────────────────────────────────────────────────────
STACK_X, STACK_CY = 10.55, 5.0
STACK_H = 2.0
ax.text(STACK_X + 0.22, STACK_CY + STACK_H/2 + 0.20,
         r"$\mathbf{z}\in[0,1]^{5}$",
         fontsize=10, ha="center", fontweight="bold", color="#222")
for i in range(5):
    y = STACK_CY + STACK_H/2 - 0.25 - i * 0.36
    ax.add_patch(Rectangle((STACK_X, y - 0.14), 0.45, 0.28,
                            facecolor=C_STACK, edgecolor=DARK, linewidth=0.9))
    ax.text(STACK_X + 0.22, y, f"$p_{i+1}$",
             fontsize=8.8, ha="center", va="center")
ax.text(STACK_X + 0.22, STACK_CY - STACK_H/2 - 0.20,
         "logit(·)", fontsize=8, ha="center", color="#666", style="italic")

# arrows from each base output circle to matching stack slot
base_ys = [8.10, 6.55, 5.00, 3.45, 1.90]
px_x = BOX_X + BOX_W + 0.30
for i, by in enumerate(base_ys):
    sy = STACK_CY + STACK_H/2 - 0.25 - i * 0.36
    arrow(ax, px_x + 0.19, by, STACK_X - 0.02, sy,
          color="#555", lw=0.9, ms=9)

# ── Meta learner (L2-LR) ────────────────────────────────────────────────
META_CX, META_CY = 12.45, 5.0
ax.add_patch(Circle((META_CX, META_CY), 0.50, facecolor=C_META,
                     edgecolor=DARK, linewidth=1.3))
ax.text(META_CX, META_CY + 0.13, r"$\sigma$", fontsize=16,
         ha="center", va="center")
ax.text(META_CX, META_CY - 0.18, r"$\beta^{\top}\mathbf{z}+b$",
         fontsize=8.5, ha="center", va="center", color="#222")
ax.text(META_CX, META_CY + 0.80, "L2-LR meta",
         fontsize=10, ha="center", fontweight="bold", color="#1f4b73")
ax.text(META_CX, META_CY - 0.80, r"$C=10$",
         fontsize=8.5, ha="center", color="#555", style="italic")
# coefficient annotation (small table)
coef_lines = [("RF",  "+0.38"), ("MT",  "+0.32"),
              ("ET",  "+0.28"), ("SVM", "-0.05"),
              ("NRB", "+0.01")]
for i, (k, v) in enumerate(coef_lines):
    ax.text(META_CX - 0.48, META_CY - 1.10 - i * 0.18,
             f"$\\beta_{{\\mathrm{{{k}}}}}$", fontsize=7, color="#555")
    ax.text(META_CX + 0.05, META_CY - 1.10 - i * 0.18,
             v, fontsize=7, color="#555", ha="left",
             family="monospace")
# arrow stack → meta
arrow(ax, STACK_X + 0.48, META_CY, META_CX - 0.50, META_CY,
       lw=1.3, ms=12)

# ── Isotonic calibration ────────────────────────────────────────────────
ISO_CX, ISO_CY = 14.10, 5.0
isotonic_icon(ax, ISO_CX, ISO_CY, w=1.25, h=1.05)
ax.text(ISO_CX, ISO_CY + 0.66, r"isotonic $\hat g$", fontsize=9.5,
         ha="center", fontweight="bold", color="#8a3b1e")
ax.text(ISO_CX, ISO_CY - 0.66, "PAV monotone", fontsize=7.5,
         ha="center", color="#555", style="italic")
# arrow meta → isotonic
arrow(ax, META_CX + 0.50, META_CY, ISO_CX - 0.625, META_CY,
       lw=1.3, ms=12)

# ── Output probability bar ─────────────────────────────────────────────
OUT_CX, OUT_CY = 15.75, 5.0
BAR_W, BAR_H = 1.05, 0.30
ax.add_patch(Rectangle((OUT_CX - BAR_W/2, OUT_CY - BAR_H/2), BAR_W, BAR_H,
                        facecolor="white", edgecolor=DARK, linewidth=1.2))
# shade proportional to p̂ ≈ 0.58 (illustrative)
ax.add_patch(Rectangle((OUT_CX - BAR_W/2, OUT_CY - BAR_H/2),
                        0.58 * BAR_W, BAR_H,
                        facecolor="#d47474", edgecolor="none"))
# τ★ mark
tau_x = OUT_CX - BAR_W/2 + 0.135 * BAR_W
ax.plot([tau_x, tau_x],
         [OUT_CY - BAR_H/2 - 0.07, OUT_CY + BAR_H/2 + 0.07],
         "-", color="#111", lw=2.0)
ax.text(tau_x, OUT_CY + BAR_H/2 + 0.20, r"$\tau^\star=0.135$",
         fontsize=8, ha="center", color="#111", fontweight="bold")
ax.text(OUT_CX, OUT_CY + 0.70, r"$\hat p\in[0,1]$",
         fontsize=10, ha="center", fontweight="bold", color="#333")
ax.text(OUT_CX - BAR_W/2, OUT_CY - 0.42, "0", fontsize=7.5, ha="center",
         color="#555")
ax.text(OUT_CX + BAR_W/2, OUT_CY - 0.42, "1", fontsize=7.5, ha="center",
         color="#555")
# arrow iso → output bar
arrow(ax, ISO_CX + 0.625, META_CY, OUT_CX - BAR_W/2 - 0.04, META_CY,
       lw=1.3, ms=12)

# Decision below the bar
ax.text(OUT_CX, OUT_CY - 1.05,
         r"$\hat y = \mathbb{1}[\hat p \geq \tau^\star]$",
         fontsize=11, ha="center", fontweight="bold", color="#245a24",
         bbox=dict(boxstyle="round,pad=0.22", facecolor=C_OUT,
                   edgecolor="#245a24", linewidth=1.0))

# ── Cleanlab weight flow (dashed, coming in from below) ───────────────
CL_X, CL_Y, CL_W, CL_H = 0.30, 2.3, 3.1, 1.0
ax.add_patch(FancyBboxPatch((CL_X, CL_Y - CL_H/2), CL_W, CL_H,
                             boxstyle="round,pad=0.03,rounding_size=0.08",
                             facecolor=C_CL, edgecolor="#a33", linewidth=1.2))
ax.text(CL_X + CL_W/2, CL_Y + 0.22, "Cleanlab v6",
         fontsize=10, ha="center", fontweight="bold", color="#8b2c2c")
ax.text(CL_X + CL_W/2, CL_Y - 0.12,
         r"$w_i\in\{0.3,\,1.0\}$  (299/4 375 flagged)",
         fontsize=8.2, ha="center", color="#5a2020")
ax.text(CL_X + CL_W/2, CL_Y - 0.35, "sample weights fed to RF, NRB, SVM, ET",
         fontsize=7, ha="center", color="#5a2020", style="italic")

# dashed arrows to the 4 weight-supporting bases
for cy in [8.10, 6.55, 3.45, 1.90]:
    ax.add_patch(FancyArrowPatch((CL_X + CL_W, CL_Y + 0.15),
                                  (BOX_X + 0.04, cy - 0.25),
                                  arrowstyle="->", mutation_scale=8,
                                  color="#a33", lw=0.8,
                                  linestyle=(0, (3, 2)),
                                  connectionstyle="arc3,rad=0.18"))

# ── Data-split note (top-left, small) ─────────────────────────────────
ax.text(0.25, 8.05,
         "Training:   $N_{\\mathrm{tr}}=4\\,375$  (14.5% positive)",
         fontsize=8, color="#333")
ax.text(0.25, 7.75,
         "Held-out:   $N_{\\mathrm{te}}=1\\,094$  (frozen, seed = 42)",
         fontsize=8, color="#333")

# ── Small legend on bottom ─────────────────────────────────────────────
leg_items = [
    ("#4472a8", "negative class leaf"),
    ("#c04545", "positive class leaf"),
    ("#88a0c8", "hidden neuron"),
    ("#d4955c", "output head"),
    ("#a33",    "weight flow (CL v6)"),
    ("#c06060", "boosting residual"),
]
lx, ly = 0.25, 0.55
for i, (col, txt) in enumerate(leg_items):
    ax.add_patch(Rectangle((lx + i * 2.75, ly), 0.18, 0.16,
                            facecolor=col, edgecolor="black", linewidth=0.3))
    ax.text(lx + i * 2.75 + 0.25, ly + 0.08, txt,
             fontsize=7.5, va="center", color="#333")

# ── Tensor-shape annotations on key arrows ────────────────────────────
ax.text(4.30, 5.25, "(41,)", fontsize=7, ha="center",
         color="#666", style="italic")
ax.text(STACK_X - 0.15, 5.92, "", fontsize=6.5)  # placeholder
ax.text(11.45, 5.22, "(5,)", fontsize=7, ha="center",
         color="#666", style="italic")
ax.text(13.25, 5.22, "(1,)", fontsize=7, ha="center",
         color="#666", style="italic")
ax.text(14.85, 5.22, "(1,)", fontsize=7, ha="center",
         color="#666", style="italic")

fig.savefig(OUT / "fig_champion_architecture.png", dpi=200, bbox_inches="tight")
fig.savefig(OUT / "fig_champion_architecture.pdf", bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig_champion_architecture.png'}")
print(f"wrote {OUT / 'fig_champion_architecture.pdf'}")
print("DONE")
