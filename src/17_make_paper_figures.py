#!/usr/bin/env python
"""Paper figures: Fig A (inference pipeline) + Fig B (training protocol).

Fig A shows the forward pass x -> y_hat with tensor-shape annotations on
every arrow and an example employee's numeric trace, so a reader can follow
one data point end-to-end. Fig B isolates the training-time machinery
(Cleanlab v6 weights, 5-fold OOF, meta refit, isotonic calibration).
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import (
    FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Polygon, Ellipse,
)
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,       # TrueType, not Type-3 (journal requirement)
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})
DPI = 300                      # journal raster requirement

# ---------- palette ----------
C_BG = "#F7F9FC"
C_BOX = "#FFFFFF"
C_EDGE = "#2C3E50"
C_PRIM = "#2980B9"
C_ACC = "#C0392B"
C_OK = "#27AE60"
C_WARN = "#E67E22"
C_MUT = "#95A5A6"
C_PUR = "#9B59B6"

# ---------- example trace ----------
# (feature, semantic_label, raw_code) — readers see meaning, not codes
INPUT_ROWS = [
    ("Gender",       "Male",           1),
    ("Education",    "Bachelor",       2),
    ("Marriage",     "Single",         1),
    ("Hometown",     "Tier-3 city",    3),
    ("Univ. type",   "Ord. bachelor",  2),
    ("Major",        "STEM",           1),
    ("Firm type",    "Private",        3),
    ("Income lv.",   "Low (3-5 k)",    2),
    ("Job pressure", "High (4 / 5)",   4),
    ("Atmosphere",   "Neutral (3/5)",  3),
    ("Job match",    "Neutral (3/5)",  3),
    ("Satisfaction", "Neutral (3/5)",  3),
]
EX_X12 = [r[2] for r in INPUT_ROWS]         # raw codes (only shown in bottom trace)
EX_PROFILE = "Male Bachelor · STEM · Private firm · low income · high pressure"

EX_P = [0.41, 0.38, 0.45, 0.33, 0.42]
EX_PTILDE = 0.402
EX_PHAT = 0.41
TAU = 0.135

# meta coefficients (learned from Phase 6)
BETA = {"RF": 0.376, "NRB": 0.014, "MT": 0.316, "SVM": -0.047, "ET": 0.280}
BIAS = -0.12


# ==========================================================================
# low-level primitives
# ==========================================================================

def rbox(ax, x, y, w, h, fc=C_BOX, ec=C_EDGE, lw=1.3, rad=0.06, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={rad}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha)
    ax.add_patch(p)


def arrow(ax, x1, y1, x2, y2, color=C_EDGE, lw=1.4, style="->",
          shape=None, ex=None, dashed=False):
    ls = "--" if dashed else "-"
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                        mutation_scale=15, color=color, lw=lw, linestyle=ls)
    ax.add_patch(a)
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    if shape:
        ax.text(mx, my + 0.16, shape, ha="center", va="bottom",
                fontsize=9, color=C_PRIM, weight="bold")
    if ex:
        ax.text(mx, my - 0.20, ex, ha="center", va="top",
                fontsize=7, color="#666", family="monospace")


# ==========================================================================
# icon primitives (trees, MLP, SVM, isotonic)
# ==========================================================================

def _tree(ax, cx, cy, scale=1.0, leaf_colors=("#E74C3C", "#3498DB", "#3498DB", "#E74C3C")):
    """small binary decision tree: root -> 2 internal -> 4 leaves."""
    s = scale
    r_root = 0.055 * s
    r_node = 0.045 * s
    # positions
    root = (cx, cy + 0.35 * s)
    l1 = (cx - 0.18 * s, cy + 0.08 * s)
    r1 = (cx + 0.18 * s, cy + 0.08 * s)
    leaves = [
        (cx - 0.27 * s, cy - 0.22 * s),
        (cx - 0.09 * s, cy - 0.22 * s),
        (cx + 0.09 * s, cy - 0.22 * s),
        (cx + 0.27 * s, cy - 0.22 * s),
    ]
    # edges
    for p in (l1, r1):
        ax.plot([root[0], p[0]], [root[1], p[1]], color=C_EDGE, lw=1.0, zorder=1)
    for p in leaves[:2]:
        ax.plot([l1[0], p[0]], [l1[1], p[1]], color=C_EDGE, lw=1.0, zorder=1)
    for p in leaves[2:]:
        ax.plot([r1[0], p[0]], [r1[1], p[1]], color=C_EDGE, lw=1.0, zorder=1)
    # nodes
    for pos, rr, fc in [(root, r_root, "#FDEBD0"), (l1, r_node, "#FDEBD0"), (r1, r_node, "#FDEBD0")]:
        ax.add_patch(Circle(pos, rr, fc=fc, ec=C_EDGE, lw=0.9, zorder=2))
    # leaves
    ls = 0.08 * s
    for (x, y), c in zip(leaves, leaf_colors):
        ax.add_patch(Rectangle((x - ls/2, y - ls/2), ls, ls,
                               fc=c, ec=C_EDGE, lw=0.7, zorder=2))


def _tree_random(ax, cx, cy, scale=1.0):
    """ExtraTrees-style: same tree but branches are zig-zag to hint at random splits."""
    s = scale
    root = (cx, cy + 0.35 * s)
    l1 = (cx - 0.18 * s, cy + 0.08 * s)
    r1 = (cx + 0.18 * s, cy + 0.08 * s)
    # draw slight zig-zag on edges
    def zig(a, b):
        midx = (a[0] + b[0]) / 2 + 0.04 * s * np.sign(b[0] - a[0])
        midy = (a[1] + b[1]) / 2
        ax.plot([a[0], midx, b[0]], [a[1], midy, b[1]], color=C_EDGE, lw=1.0, zorder=1)
    zig(root, l1); zig(root, r1)
    leaves = [
        (cx - 0.27 * s, cy - 0.22 * s), (cx - 0.09 * s, cy - 0.22 * s),
        (cx + 0.09 * s, cy - 0.22 * s), (cx + 0.27 * s, cy - 0.22 * s),
    ]
    zig(l1, leaves[0]); zig(l1, leaves[1])
    zig(r1, leaves[2]); zig(r1, leaves[3])
    for pos, rr in [(root, 0.055*s), (l1, 0.045*s), (r1, 0.045*s)]:
        ax.add_patch(Circle(pos, rr, fc="#D6EAF8", ec=C_EDGE, lw=0.9, zorder=2))
    colors = ["#E74C3C", "#3498DB", "#E74C3C", "#3498DB"]
    ls = 0.08 * s
    for (x, y), c in zip(leaves, colors):
        ax.add_patch(Rectangle((x - ls/2, y - ls/2), ls, ls,
                               fc=c, ec=C_EDGE, lw=0.7, zorder=2))


def icon_rf(ax, cx, cy, w, h):
    """3 trees + ..."""
    s = min(w, h) * 1.3
    offsets = [-w*0.33, 0, w*0.33]
    for dx in offsets:
        _tree(ax, cx + dx * 0.7, cy, scale=s)
    ax.text(cx + w*0.40, cy - h*0.05, "...", fontsize=7, color="#888", ha="left")


def icon_et(ax, cx, cy, w, h):
    s = min(w, h) * 1.3
    offsets = [-w*0.33, 0, w*0.33]
    for dx in offsets:
        _tree_random(ax, cx + dx * 0.7, cy, scale=s)
    ax.text(cx + w*0.40, cy - h*0.05, "...", fontsize=7, color="#888", ha="left")


def icon_nrb(ax, cx, cy, w, h):
    """2 trees in sequence + residual arrow + 'drop top 10%' note."""
    s = min(w, h) * 1.3
    _tree(ax, cx - w*0.28, cy + 0.05, scale=s,
          leaf_colors=("#E67E22", "#F4D03F", "#F4D03F", "#E67E22"))
    _tree(ax, cx + w*0.20, cy + 0.05, scale=s,
          leaf_colors=("#E67E22", "#F4D03F", "#F4D03F", "#E67E22"))
    # residual arrow between
    ax.add_patch(FancyArrowPatch((cx - w*0.10, cy - 0.18), (cx + w*0.04, cy - 0.18),
                                 arrowstyle="->", mutation_scale=10,
                                 color=C_ACC, lw=1.1))
    ax.text(cx - w*0.03, cy - 0.30, r"$r_{t}$", ha="center", va="top",
            fontsize=7, color=C_ACC, style="italic")
    ax.text(cx, cy - h*0.48, "GCE q=0.7 · drop top-10% loss",
            ha="center", va="top", fontsize=6.5, color="#666", style="italic")


def icon_mlp(ax, cx, cy, w, h):
    """4-column MLP: 41(reduced) -> 64(reduced) -> 32(reduced) -> (bin, ord)."""
    layers = [6, 8, 5, 2]
    col_x = np.linspace(cx - w*0.42, cx + w*0.35, len(layers))
    col_positions = []
    for n, x in zip(layers, col_x):
        ys = np.linspace(cy - h*0.35, cy + h*0.35, n)
        col_positions.append([(x, y) for y in ys])
    # connections (faded)
    for i in range(len(layers) - 1):
        for a in col_positions[i]:
            for b in col_positions[i + 1]:
                ax.plot([a[0], b[0]], [a[1], b[1]],
                        color=C_MUT, lw=0.3, alpha=0.35, zorder=1)
    # neurons
    for ci, col in enumerate(col_positions):
        for (x, y) in col:
            if ci == len(col_positions) - 1:
                # output heads: distinct colors
                fc = C_ACC if col.index((x, y)) == 0 else C_OK
            else:
                fc = "#AED6F1" if ci == 0 else "#85C1E9"
            ax.add_patch(Circle((x, y), 0.04, fc=fc, ec=C_EDGE, lw=0.7, zorder=2))
    # head labels
    ax.text(col_x[-1] + 0.10, col_positions[-1][0][1], "bin", ha="left", va="center",
            fontsize=6.5, color=C_ACC)
    ax.text(col_x[-1] + 0.10, col_positions[-1][1][1], "ord", ha="left", va="center",
            fontsize=6.5, color=C_OK)
    # layer shape labels
    ax.text(col_x[0], cy - h*0.48, "41", ha="center", fontsize=6, color="#666")
    ax.text(col_x[1], cy - h*0.48, "64", ha="center", fontsize=6, color="#666")
    ax.text(col_x[2], cy - h*0.48, "32", ha="center", fontsize=6, color="#666")


def icon_svm(ax, cx, cy, w, h):
    """scatter + diagonal hyperplane + dashed margin + support vectors."""
    rng = np.random.default_rng(3)
    # positives (top-left) and negatives (bottom-right)
    n_pts = 6
    pos_x = rng.uniform(cx - w*0.40, cx + w*0.05, n_pts)
    pos_y = rng.uniform(cy - h*0.05, cy + h*0.35, n_pts)
    neg_x = rng.uniform(cx - w*0.05, cx + w*0.40, n_pts)
    neg_y = rng.uniform(cy - h*0.35, cy + h*0.10, n_pts)
    ax.scatter(pos_x, pos_y, marker="o", c=C_ACC, s=14, ec="black", lw=0.5, zorder=3)
    ax.scatter(neg_x, neg_y, marker="s", c=C_PRIM, s=14, ec="black", lw=0.5, zorder=3)
    # hyperplane
    hp_x = np.array([cx - w*0.42, cx + w*0.42])
    hp_y = np.array([cy + h*0.35, cy - h*0.35])
    ax.plot(hp_x, hp_y, color=C_EDGE, lw=1.3, zorder=2)
    # margins
    dy = 0.09
    ax.plot(hp_x, hp_y + dy, color=C_EDGE, lw=0.8, linestyle="--", zorder=2)
    ax.plot(hp_x, hp_y - dy, color=C_EDGE, lw=0.8, linestyle="--", zorder=2)
    # circled support vectors
    ax.add_patch(Circle((pos_x[0], pos_y[0]), 0.06, fill=False, ec=C_WARN, lw=1.3, zorder=4))
    ax.add_patch(Circle((neg_x[0], neg_y[0]), 0.06, fill=False, ec=C_WARN, lw=1.3, zorder=4))
    ax.text(cx + w*0.35, cy + h*0.40, r"$w^{\top}\!z+b=0$", fontsize=6.5, ha="right", color="#444")


# ==========================================================================
# Fig A stage drawers
# ==========================================================================

def draw_input_box(ax, x, y, w, h):
    rbox(ax, x, y, w, h, fc="#F4F6F8")
    ax.text(x + w/2, y + h - 0.22,
            r"Raw input   $x \in \mathbb{R}^{12}$",
            ha="center", va="top", fontsize=10, weight="bold", color=C_EDGE)
    ax.text(x + w/2, y + h - 0.50,
            "example new-graduate (decoded)",
            ha="center", va="top", fontsize=7, style="italic", color="#777")
    # single-column table: feature : semantic label
    top_y = y + h - 0.75
    row_h = (top_y - (y + 0.20)) / len(INPUT_ROWS)
    for i, (feat, label, code) in enumerate(INPUT_ROWS):
        ty = top_y - (i + 0.5) * row_h
        ax.text(x + 0.12, ty, feat, ha="left", va="center",
                fontsize=7, color="#555")
        ax.text(x + w - 0.12, ty, label, ha="right", va="center",
                fontsize=7.2, color=C_PRIM, weight="bold")
    # thin divider above the list
    ax.plot([x + 0.12, x + w - 0.12], [top_y + 0.02, top_y + 0.02],
            color="#BBB", lw=0.6)


def draw_phi_box(ax, x, y, w, h):
    rbox(ax, x, y, w, h, fc="#ECF0F1")
    # title
    ax.text(x + w/2, y + h - 0.25,
            r"$\varphi:\;\mathbb{R}^{12}\to\mathbb{R}^{41}$",
            ha="center", va="top", fontsize=11, weight="bold", color=C_EDGE)
    ax.text(x + w/2, y + h - 0.60, "leak-safe feature expansion",
            ha="center", va="top", fontsize=7.5, style="italic", color="#666")

    # ---- stacked composition bar (upper) ----
    parts = [
        ("raw",         12, "#95A5A6", "original 12 coded inputs"),
        ("Likert cum.", 16, "#3498DB", "4 ordinal vars × 4 thresholds"),
        ("interact.",    5, "#E67E22", "satis×intent, match×opp, …"),
        ("gap",          3, "#27AE60", "match−pressure, …"),
        ("income",       2, "#9B59B6", "log(inc+1), inc²"),
        ("composite",    1, "#F39C12", "(satis+match+opp) / 3"),
        ("KFold TE",     2, "#E74C3C", "univ-type, major (leak-safe)"),
    ]
    total = sum(p[1] for p in parts)
    bx = x + 0.30
    bw = w - 0.60
    by = y + h - 1.60
    bh = 0.34
    cum = 0
    for label, n, col, _ in parts:
        frac = n / total
        sw = bw * frac
        ax.add_patch(Rectangle((bx + cum, by), sw, bh, fc=col, ec="black", lw=0.5))
        ax.text(bx + cum + sw/2, by + bh/2, str(n),
                ha="center", va="center", fontsize=7, color="white", weight="bold")
        cum += sw
    ax.text(bx, by - 0.18, "dim 0", fontsize=6, color="#888", ha="left", va="top")
    ax.text(bx + bw, by - 0.18, "dim 41", fontsize=6, color="#888", ha="right", va="top")
    ax.text(bx + bw/2, by + bh + 0.14, "41-dim feature vector (composition)",
            ha="center", va="bottom", fontsize=7.5, color="#333", weight="bold")

    # ---- breakdown rows (lower) — well-spaced so no overlap ----
    rowy = by - 0.55
    row_h = 0.38
    for i, (a, n, c, desc) in enumerate(parts):
        cy = rowy - i * row_h
        # color chip
        ax.add_patch(Rectangle((x + 0.28, cy - 0.09), 0.18, 0.18,
                               fc=c, ec="black", lw=0.5))
        # count badge
        ax.text(x + 0.37, cy, str(n), ha="center", va="center",
                fontsize=6.5, color="white", weight="bold")
        # label
        ax.text(x + 0.56, cy, a, ha="left", va="center",
                fontsize=7.5, weight="bold", color="#333")
        # description
        ax.text(x + 1.38, cy, desc, ha="left", va="center",
                fontsize=6.8, color="#555", style="italic")

    # bottom example shard — only leave a small gap
    ax.text(x + w/2, y + 0.18,
            r"$\varphi(x)\in\mathbb{R}^{41}$: first 12 dims = raw, then 29 engineered",
            ha="center", va="bottom", fontsize=7, color=C_PRIM)


def draw_zscore_box(ax, x, y, w, h):
    rbox(ax, x, y, w, h, fc="#EAF6FF")
    ax.text(x + w/2, y + h - 0.18, "z-score", ha="center", va="top",
            fontsize=9, weight="bold", color=C_EDGE)
    ax.text(x + w/2, y + h/2,
            r"$\dfrac{z-\mu_{\text{tr}}}{\sigma_{\text{tr}}}$",
            ha="center", va="center", fontsize=11)
    ax.text(x + w/2, y + 0.12, "(MLP, SVM)",
            ha="center", va="bottom", fontsize=6.5, style="italic", color="#666")


def draw_base_ensemble(ax, x, y, w, h, probs):
    """5 base learners stacked vertically.

    Within each panel: name + hyperparams centered along top,
    icon centered in the lower half. Probability circle sits
    OUTSIDE the panel on the right so there is no text/icon overlap.
    Returns list of (cx, cy) of the 5 output probability circles.
    """
    learners = [
        ("Random Forest", "n_est=400, max_depth=10  ·  CL-w",
         icon_rf,  probs[0], "p_{RF}"),
        ("NR-Boost",      "XGB + GCE q=0.7  ·  2-stage SP  ·  CL-w",
         icon_nrb, probs[1], "p_{NRB}"),
        ("MT-MLP",        "[41→64→32] + bin/ord heads  ·  λ=0.7  ·  seed×5",
         icon_mlp, probs[2], "p_{MT}"),
        ("SVM-RBF",       "C=1, γ=scale  ·  CL-w",
         icon_svm, probs[3], "p_{SVM}"),
        ("Extra Trees",   "n_est=400, max_depth=12  ·  CL-w",
         icon_et,  probs[4], "p_{ET}"),
    ]
    # section label
    ax.text(x + w/2 - 0.4, y + h + 0.15, "5 Base Learners (parallel)",
            ha="center", va="bottom", fontsize=10.5, weight="bold", color=C_EDGE)
    ax.text(x + w/2 - 0.4, y - 0.15, "CL-w = trained with Cleanlab-v6 sample weights",
            ha="center", va="top", fontsize=7, style="italic", color="#888")

    box_h = h / 5
    centers = []
    panel_right_pad = 0.85              # room reserved for p-circle outside panel
    bx_inner = x + 0.05
    bw_inner = w - panel_right_pad
    for i, (name, hp, icon_fn, p, plab) in enumerate(learners):
        by = y + h - (i + 1) * box_h
        rbox(ax, bx_inner, by + 0.08, bw_inner, box_h - 0.16, fc="#FDFEFF")
        # name — centered, top
        ax.text(bx_inner + bw_inner/2, by + box_h - 0.20, name,
                ha="center", va="top", fontsize=10, weight="bold", color=C_PRIM)
        # hp — centered, just below name, small font
        ax.text(bx_inner + bw_inner/2, by + box_h - 0.48, hp,
                ha="center", va="top", fontsize=6.5,
                style="italic", color="#666")
        # icon — centered in lower 55% of panel, no x-axis conflict with text
        icon_cx = bx_inner + bw_inner/2
        icon_cy = by + (box_h - 0.70) / 2 + 0.08   # bottom zone
        icon_fn(ax, icon_cx, icon_cy,
                w=bw_inner * 0.55,
                h=box_h - 0.95)
        # output probability circle — sits OUTSIDE right edge of panel
        cx = x + w - 0.30
        cy = by + box_h / 2
        ax.add_patch(Circle((cx, cy), 0.28,
                            fc=C_PRIM, ec=C_EDGE, lw=1.3, alpha=0.90))
        ax.text(cx, cy + 0.02, f"{p:.2f}",
                ha="center", va="center", fontsize=9, color="white", weight="bold")
        ax.text(cx, cy - 0.48, f"${plab}$",
                ha="center", va="top", fontsize=8, color=C_EDGE)
        centers.append((cx, cy))
    return centers


def draw_meta_sigma(ax, cx, cy, p_centers, p_tilde):
    """Minimal meta visualization: 5 weighted arrows from p-circles into σ.

    The 5 p-circles already drawn by draw_base_ensemble serve as the
    stacking vector; no duplicate sub-nodes or stack column.
    Returns (right_x, cy) of the σ node so caller can continue the spine.
    """
    keys = ["RF", "NRB", "MT", "SVM", "ET"]
    betas = [BETA[k] for k in keys]
    # draw weighted edges first (so σ sits on top)
    for (src_x, src_y), bk in zip(p_centers, betas):
        color = C_ACC if bk < 0 else C_PRIM
        lw = 0.7 + 4.0 * abs(bk)
        a = FancyArrowPatch((src_x + 0.30, src_y), (cx - 0.38, cy),
                            arrowstyle="->", mutation_scale=11,
                            color=color, lw=lw, alpha=0.85, zorder=2)
        ax.add_patch(a)
        # β label on edge — positioned 40% toward σ node
        lx = src_x + 0.30 + 0.42 * (cx - 0.38 - src_x - 0.30)
        ly = src_y + 0.42 * (cy - src_y)
        ax.text(lx, ly + 0.10, f"{bk:+.3f}",
                ha="center", va="bottom", fontsize=7.2,
                color=color, weight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.14",
                          fc="white", ec=color, lw=0.7, alpha=0.95))
    # σ node
    ax.add_patch(Circle((cx, cy), 0.38,
                        fc="#F9E79F", ec=C_EDGE, lw=1.6, zorder=5))
    ax.text(cx, cy + 0.03, r"$\sigma$",
            ha="center", va="center", fontsize=18, zorder=6)
    # labels around σ
    ax.text(cx, cy + 0.55, "L2-LR meta",
            ha="center", va="bottom", fontsize=9.5, weight="bold", color=C_EDGE)
    ax.text(cx, cy - 0.55,
            r"$\tilde p=\sigma\!\left(\sum_k \beta_k\,\mathrm{logit}(p_k)+b\right)$",
            ha="center", va="top", fontsize=8, color="#444")
    ax.text(cx, cy - 0.85,
            f"bias b = {BIAS:+.3f}",
            ha="center", va="top", fontsize=6.8, color="#666", style="italic")
    # emitted p̃ label
    ax.text(cx + 0.45, cy + 0.18, r"$\tilde p$",
            ha="left", va="center", fontsize=11, color=C_EDGE, weight="bold")
    ax.text(cx + 0.45, cy - 0.10, f"= {p_tilde:.3f}",
            ha="left", va="center", fontsize=7.5, color="#555", family="monospace")
    return (cx + 0.38, cy)


def _DEPRECATED_draw_meta_lr_box(ax, x, y, w, h):
    """kept for backward compat; not called."""
    rbox(ax, x, y, w, h, fc="#FEF9E7")
    # title bar
    ax.text(x + w/2, y + h - 0.25,
            "L2-LR meta  (weighted sum of logits)",
            ha="center", va="top", fontsize=10, weight="bold", color=C_EDGE)
    ax.text(x + w/2, y + h - 0.55,
            r"$\tilde p = \sigma\!\left(\sum_{k=1}^{5}\beta_k\,\mathrm{logit}(p_k)+b\right)$",
            ha="center", va="top", fontsize=9)

    # ---- 5 sub-nodes on left side of box ----
    keys = ["RF", "NRB", "MT", "SVM", "ET"]
    betas = [BETA[k] for k in keys]
    labels = [r"$p_{RF}$", r"$p_{NRB}$", r"$p_{MT}$", r"$p_{SVM}$", r"$p_{ET}$"]
    probs = EX_P
    node_x = x + 0.55
    node_ys = np.linspace(y + h - 1.30, y + 0.90, 5)
    # sigma node (center-right)
    sum_x = x + w * 0.62
    sum_y = y + h / 2 - 0.10
    ax.add_patch(Circle((sum_x, sum_y), 0.32,
                        fc="#F9E79F", ec=C_EDGE, lw=1.4, zorder=3))
    ax.text(sum_x, sum_y + 0.02, r"$\sigma$",
            ha="center", va="center", fontsize=16, zorder=4)
    ax.text(sum_x, sum_y - 0.50, "(weighted sum)",
            ha="center", va="top", fontsize=6.5, color="#666", style="italic")

    # weighted edges: color by sign, width by |β|
    for (ny, bk, lab, p) in zip(node_ys, betas, labels, probs):
        # sub-node
        ax.add_patch(Circle((node_x, ny), 0.19,
                            fc="#D6EAF8", ec=C_EDGE, lw=0.9, zorder=3))
        ax.text(node_x, ny + 0.01, lab, ha="center", va="center",
                fontsize=6.8, zorder=4)
        # small numeric value shown to left of sub-node
        ax.text(node_x - 0.30, ny, f"{p:.2f}", ha="right", va="center",
                fontsize=6.8, color=C_PRIM, weight="bold", family="monospace")
        # weighted arrow to sigma
        color = C_ACC if bk < 0 else C_PRIM
        lw = 0.6 + 4.2 * abs(bk)
        a = FancyArrowPatch((node_x + 0.19, ny), (sum_x - 0.32, sum_y),
                            arrowstyle="->", mutation_scale=10,
                            color=color, lw=lw, alpha=0.85, zorder=2)
        ax.add_patch(a)
        # beta label riding on the edge
        mid_x = (node_x + 0.19 + sum_x - 0.32) / 2
        mid_y = (ny + sum_y) / 2
        ax.text(mid_x, mid_y + 0.08, f"{bk:+.3f}",
                ha="center", va="bottom", fontsize=6.3,
                color=color, weight="bold", zorder=4,
                bbox=dict(boxstyle="round,pad=0.10", fc="white",
                          ec="none", alpha=0.7))

    # input-probe label on left
    ax.text(x + 0.15, y + h - 1.05, "input  p",
            ha="left", va="center", fontsize=7, color="#666", style="italic")
    # output label on right
    ax.text(sum_x + 0.45, sum_y,
            r"$\tilde p$", ha="left", va="center",
            fontsize=11, color=C_EDGE, weight="bold")
    ax.text(sum_x + 0.45, sum_y - 0.28,
            f"= {EX_PTILDE:.3f}", ha="left", va="center",
            fontsize=7.5, color="#555", family="monospace")
    # bias badge bottom-right
    ax.text(x + w - 0.30, y + 0.35,
            f"bias b = {BIAS:+.3f}",
            ha="right", va="bottom", fontsize=6.8, color="#666",
            style="italic")
    # legend for edge colors
    ax.text(x + 0.15, y + 0.35,
            "edge width ∝ |β|   blue: β > 0   red: β < 0",
            ha="left", va="bottom", fontsize=6.3, color="#888", style="italic")


def draw_isotonic_box(ax, x, y, w, h):
    rbox(ax, x, y, w, h, fc="#F0FFF4")
    ax.text(x + w/2, y + h - 0.18, "Isotonic", ha="center", va="top",
            fontsize=9, weight="bold", color=C_EDGE)
    ax.text(x + w/2, y + h - 0.45, r"$\hat g:[0,1]\to[0,1]$",
            ha="center", va="top", fontsize=7.5, color="#555")
    # mini axes
    ax0 = x + 0.30
    ay0 = y + 0.35
    aw = w - 0.55
    ah = h - 1.00
    ax.plot([ax0, ax0 + aw], [ay0, ay0], color="#333", lw=0.8)
    ax.plot([ax0, ax0], [ay0, ay0 + ah], color="#333", lw=0.8)
    # dashed diagonal reference (identity)
    ax.plot([ax0, ax0 + aw], [ay0, ay0 + ah], color="#AAA", lw=0.6, linestyle=":")
    # step function (piecewise monotone rising)
    knots_x = np.array([0.0, 0.10, 0.22, 0.38, 0.55, 0.72, 0.88, 1.00])
    knots_y = np.array([0.02, 0.06, 0.18, 0.32, 0.58, 0.78, 0.93, 0.99])
    xs = ax0 + knots_x * aw
    ys = ay0 + knots_y * ah
    # draw step (horizontal then vertical)
    for i in range(len(xs) - 1):
        ax.plot([xs[i], xs[i + 1]], [ys[i], ys[i]], color=C_OK, lw=1.6)
        ax.plot([xs[i + 1], xs[i + 1]], [ys[i], ys[i + 1]], color=C_OK, lw=1.6)
    # axis tick labels
    ax.text(ax0, ay0 - 0.10, "0", fontsize=6, ha="center", va="top", color="#777")
    ax.text(ax0 + aw, ay0 - 0.10, "1", fontsize=6, ha="center", va="top", color="#777")
    ax.text(ax0 - 0.05, ay0, "0", fontsize=6, ha="right", va="center", color="#777")
    ax.text(ax0 - 0.05, ay0 + ah, "1", fontsize=6, ha="right", va="center", color="#777")
    ax.text(ax0 + aw/2, ay0 - 0.22, r"$\tilde p$", fontsize=8, ha="center", va="top")
    ax.text(ax0 - 0.18, ay0 + ah/2, r"$\hat p$", fontsize=8, ha="right", va="center", rotation=90)
    # ECE annotation
    ax.text(x + w/2, y + 0.12, "ECE 0.071→0.032",
            ha="center", va="bottom", fontsize=6.5, style="italic", color=C_ACC)


def draw_threshold_box(ax, x, y, w, h, phat, tau):
    rbox(ax, x, y, w, h, fc="#FFF5F5")
    ax.text(x + w/2, y + h - 0.18, "Decision", ha="center", va="top",
            fontsize=9, weight="bold", color=C_EDGE)
    ax.text(x + w/2, y + h - 0.45, r"$\hat y = \mathbb{1}\{\hat p \geq \tau^\star\}$",
            ha="center", va="top", fontsize=7.5, color="#555")
    # horizontal probability bar
    bx = x + 0.20
    bw = w - 0.40
    by = y + h - 1.35
    bh = 0.26
    # background gradient (simple two-segment)
    for i in range(60):
        frac = i / 60
        col = plt.cm.RdYlGn_r(frac)
        ax.add_patch(Rectangle((bx + i*bw/60, by), bw/60 + 0.002, bh,
                               fc=col, ec="none"))
    ax.add_patch(Rectangle((bx, by), bw, bh, fc="none", ec=C_EDGE, lw=1.0))
    # ticks
    ax.text(bx, by - 0.12, "0", fontsize=7, ha="center", va="top", color="#555")
    ax.text(bx + bw, by - 0.12, "1", fontsize=7, ha="center", va="top", color="#555")
    # tau marker
    tx = bx + bw * tau
    ax.plot([tx, tx], [by - 0.05, by + bh + 0.05], color=C_ACC, lw=1.8)
    ax.text(tx, by + bh + 0.09, f"τ⋆={tau}", fontsize=6.5, ha="center",
            va="bottom", color=C_ACC, weight="bold")
    # p_hat marker
    px = bx + bw * phat
    ax.plot([px, px], [by - 0.03, by + bh + 0.03], color=C_EDGE, lw=1.3)
    ax.annotate(f"p̂={phat:.2f}", xy=(px, by + bh + 0.05),
                xytext=(px + 0.25, by + bh + 0.40),
                fontsize=7, color=C_EDGE, weight="bold",
                arrowprops=dict(arrowstyle="->", color=C_EDGE, lw=0.6))
    # decision label
    dec = 1 if phat >= tau else 0
    label = "resign" if dec else "stay"
    col = C_ACC if dec else C_OK
    ax.add_patch(Circle((x + w/2, y + 0.45), 0.28, fc=col, ec=C_EDGE, lw=1.4, alpha=0.9))
    ax.text(x + w/2, y + 0.45, str(dec), ha="center", va="center",
            fontsize=14, color="white", weight="bold")
    ax.text(x + w/2, y + 0.08, f"ŷ = {dec} ({label})",
            ha="center", va="bottom", fontsize=7.5, color=C_EDGE, weight="bold")


# ==========================================================================
# Fig A: inference pipeline
# ==========================================================================

def fig_a_inference():
    fig, ax = plt.subplots(figsize=(20, 11), facecolor=C_BG)
    ax.set_xlim(0, 20); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_facecolor(C_BG)

    # title
    ax.text(10, 10.55, "Figure A.  Inference Pipeline of the Champion Stacking Model",
            ha="center", fontsize=15, weight="bold", color=C_EDGE)
    ax.text(10, 10.15,
            r"Forward pass: $x \in \mathbb{R}^{12} \longrightarrow \hat y \in \{0,1\}$   |   "
            r"example employee traced in blue numerals below each arrow",
            ha="center", fontsize=10, style="italic", color="#555")

    y_mid = 5.5

    # Stage 1 · input
    draw_input_box(ax, 0.2, 4.1, 2.3, 2.8)
    arrow(ax, 2.5, y_mid, 3.1, y_mid,
          shape=r"$\mathbb{R}^{12}$", ex="(raw)")

    # Stage 2 · phi (feature engineering)
    draw_phi_box(ax, 3.1, 1.8, 3.1, 7.0)
    arrow(ax, 6.2, y_mid, 6.9, y_mid,
          shape=r"$\mathbb{R}^{41}$", ex="engineered")

    # Stage 3 · z-score
    draw_zscore_box(ax, 6.9, 4.85, 1.0, 1.3)
    arrow(ax, 7.9, y_mid, 8.5, y_mid,
          shape=r"$z\in\mathbb{R}^{41}$", ex="(std.)")

    # Stage 4 · base ensemble  (5 panels; each ends in a p_k circle)
    centers = draw_base_ensemble(ax, 8.5, 1.5, 4.1, 7.6, probs=EX_P)

    # Stage 5 · meta sigma node (fan-in of 5 weighted β edges, no stack column)
    sigma_x, sigma_y = 14.4, y_mid
    sigma_right = draw_meta_sigma(ax, sigma_x, sigma_y,
                                  p_centers=centers, p_tilde=EX_PTILDE)

    # arrow σ → isotonic
    arrow(ax, sigma_right[0] + 0.25, y_mid, 16.55, y_mid,
          shape=r"$\tilde p$", ex=f"{EX_PTILDE:.3f}")

    # Stage 6 · isotonic calibration
    draw_isotonic_box(ax, 16.55, 4.10, 1.80, 2.80)

    # arrow isotonic → decision
    arrow(ax, 18.35, y_mid, 18.85, y_mid,
          shape=r"$\hat p$", ex=f"{EX_PHAT:.2f}")

    # Stage 7 · threshold & decision
    draw_threshold_box(ax, 18.85, 3.90, 1.15, 3.20, phat=EX_PHAT, tau=TAU)

    # footer line 1: shape progression
    ax.text(10, 1.15,
            r"Tensor shape progression:  "
            r"$\mathbb{R}^{12}\;\to\;\mathbb{R}^{41}\;\to\;\mathbb{R}^{41}\;\to\;"
            r"[0,1]^{5}\;\to\;[0,1]\;\to\;[0,1]\;\to\;\{0,1\}$",
            ha="center", fontsize=10, color=C_EDGE,
            bbox=dict(boxstyle="round,pad=0.4", fc="#FFFFFF", ec="#BBB", lw=0.8))
    # footer line 2: semantic example trace (human-readable, not raw codes)
    ax.text(10, 0.55,
            f"Example:  {EX_PROFILE}   →   "
            f"p = [{', '.join(f'{p:.2f}' for p in EX_P)}]   →   "
            f"p̃ = {EX_PTILDE:.3f}   →   p̂ = {EX_PHAT:.2f}   "
            f"≥  τ⋆ = {TAU}   →   ŷ = 1  (predicted to resign)",
            ha="center", fontsize=8.5, color="#444")

    fig.savefig(OUT / "fig_A_inference.png", dpi=DPI, bbox_inches="tight", facecolor=C_BG)
    fig.savefig(OUT / "fig_A_inference.pdf",       bbox_inches="tight", facecolor=C_BG)
    fig.savefig(OUT / "fig_A_inference.svg",       bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_A_inference.png'}  (+ .pdf, .svg)")


# ==========================================================================
# Fig B: training protocol
# ==========================================================================

def draw_panel(ax, x, y, w, h, title, number, tint="#FFFFFF"):
    rbox(ax, x, y, w, h, fc=tint, ec=C_EDGE, lw=1.3)
    # corner badge
    ax.add_patch(Circle((x + 0.30, y + h - 0.30), 0.20, fc=C_PRIM, ec=C_EDGE, lw=1.0))
    ax.text(x + 0.30, y + h - 0.30, str(number),
            ha="center", va="center", fontsize=10, color="white", weight="bold")
    ax.text(x + 0.60, y + h - 0.30, title,
            ha="left", va="center", fontsize=10.5, weight="bold", color=C_EDGE)


def draw_cleanlab_panel(ax, x, y, w, h):
    draw_panel(ax, x, y, w, h, "Cleanlab v6 sample weighting", 1, tint="#FEF9E7")
    # flow: D_train -> 5-fold CV probs -> self-confidence -> threshold -> binary weights
    sy = y + h - 0.8
    # step 1: D_train
    rbox(ax, x + 0.4, sy - 0.70, 1.2, 0.55, fc="#FFFFFF")
    ax.text(x + 1.0, sy - 0.42, "D_train", ha="center", va="center",
            fontsize=8, weight="bold")
    ax.text(x + 1.0, sy - 0.62, "N=4375", ha="center", va="center",
            fontsize=7, color="#555")
    # arrow
    arrow(ax, x + 1.6, sy - 0.42, x + 2.0, sy - 0.42,
          shape="5-fold CV")
    # step 2: prob estimate matrix
    rbox(ax, x + 2.0, sy - 0.80, 1.5, 0.70, fc="#FFFFFF")
    ax.text(x + 2.75, sy - 0.42, "OOF p̂(y=1|x)",
            ha="center", va="center", fontsize=7.5, weight="bold")
    ax.text(x + 2.75, sy - 0.62, "RF surrogate",
            ha="center", va="center", fontsize=6.5, color="#666", style="italic")
    # arrow
    arrow(ax, x + 3.5, sy - 0.42, x + 3.95, sy - 0.42,
          shape="rank")
    # step 3: self-confidence histogram
    hx = x + 4.0
    hy = sy - 0.90
    rbox(ax, hx, hy, 1.3, 0.80, fc="#FFFFFF")
    ax.text(hx + 0.65, hy + 0.70, "self-conf.",
            ha="center", va="bottom", fontsize=7, weight="bold")
    # mini histogram
    heights = [0.04, 0.08, 0.14, 0.20, 0.32, 0.40, 0.28, 0.15]
    hhw = 1.05 / len(heights)
    for i, hv in enumerate(heights):
        col = "#E74C3C" if i < 1 else "#3498DB"
        ax.add_patch(Rectangle((hx + 0.12 + i*hhw, hy + 0.15), hhw*0.8, hv,
                               fc=col, ec="none"))
    # bottom 10% marker
    ax.plot([hx + 0.12 + 0.8*hhw, hx + 0.12 + 0.8*hhw],
            [hy + 0.12, hy + 0.62], color=C_ACC, lw=1.2, linestyle="--")
    ax.text(hx + 0.12 + 0.8*hhw, hy + 0.07, "bottom 10%",
            ha="center", va="top", fontsize=6, color=C_ACC)

    # step 4: binary weights output
    ay = y + 0.6
    rbox(ax, x + 0.7, ay - 0.5, w - 1.4, 0.85, fc="#FFFFFF")
    ax.text(x + w/2, ay - 0.05, "binary weight vector  w ∈ {0.3, 1.0}^N",
            ha="center", va="center", fontsize=9, weight="bold", color=C_EDGE)
    ax.text(x + w/2, ay - 0.28, "flagged (noisy) → w=0.3    clean → w=1.0",
            ha="center", va="center", fontsize=7.5, color="#555")
    # arrow down
    a = FancyArrowPatch((x + 2.75, sy - 0.95), (x + 2.75, ay + 0.40),
                        arrowstyle="->", mutation_scale=14, color=C_EDGE, lw=1.3)
    ax.add_patch(a)
    ax.text(x + 2.95, (sy - 0.95 + ay + 0.40)/2, "threshold @ 10%",
            ha="left", va="center", fontsize=7, color="#666")


def draw_oof_panel(ax, x, y, w, h):
    draw_panel(ax, x, y, w, h, "5-fold OOF generation (per base learner)", 2, tint="#EBF5FB")
    # 5 × 5 fold grid
    grid_x = x + 0.5
    grid_y = y + h - 3.0
    cell_w = 0.50
    cell_h = 0.30
    # col headers
    ax.text(grid_x - 0.35, grid_y + 5 * cell_h - 0.05, "fold",
            fontsize=7, ha="center", va="bottom", color="#555", rotation=90)
    for c in range(5):
        ax.text(grid_x + (c + 0.5) * cell_w, grid_y + 5 * cell_h + 0.10, f"G{c+1}",
                fontsize=6.5, ha="center", va="bottom", color="#555")
    for r in range(5):
        ax.text(grid_x - 0.10, grid_y + (4 - r + 0.5) * cell_h, f"f{r+1}",
                fontsize=7, ha="right", va="center", color="#555")
        for c in range(5):
            is_val = (c == r)
            fc = C_ACC if is_val else "#85C1E9"
            ax.add_patch(Rectangle((grid_x + c * cell_w, grid_y + (4 - r) * cell_h),
                                   cell_w, cell_h, fc=fc, ec="white", lw=1.3))
    # legend
    lx = grid_x + 5 * cell_w + 0.4
    ly = grid_y + 4 * cell_h
    ax.add_patch(Rectangle((lx, ly), 0.3, 0.2, fc="#85C1E9", ec="white", lw=1))
    ax.text(lx + 0.40, ly + 0.10, "train (w injected for CL-w ✓ learners)",
            fontsize=7, ha="left", va="center")
    ax.add_patch(Rectangle((lx, ly - 0.35), 0.3, 0.2, fc=C_ACC, ec="white", lw=1))
    ax.text(lx + 0.40, ly - 0.25, "validate → OOF pred",
            fontsize=7, ha="left", va="center")

    # lower: 5-learner stack (all produce one OOF column)
    learner_names = ["RF", "NR-Boost", "MT-MLP", "SVM", "ExtraTrees"]
    ly2 = y + 0.60
    for i, name in enumerate(learner_names):
        bx = x + 0.5 + i * (w - 1.0) / 5
        bw = (w - 1.0) / 5 - 0.10
        rbox(ax, bx, ly2, bw, 0.50, fc="#FFFFFF")
        ax.text(bx + bw/2, ly2 + 0.25, name,
                ha="center", va="center", fontsize=7.5, weight="bold", color=C_PRIM)
    # label "produces OOF column"
    ax.text(x + w/2, ly2 - 0.12, r"each yields OOF column $p_k^{\mathrm{OOF}}\in\mathbb{R}^N$",
            ha="center", va="top", fontsize=8, color="#555", style="italic")
    # arrow from grid to learner row
    a = FancyArrowPatch((x + w/2, grid_y - 0.10), (x + w/2, ly2 + 0.55),
                        arrowstyle="->", mutation_scale=12, color=C_EDGE, lw=1.2)
    ax.add_patch(a)


def draw_refit_panel(ax, x, y, w, h):
    draw_panel(ax, x, y, w, h, "Meta fit + Isotonic calibration", 3, tint="#F0FFF4")

    # step A: collect OOF matrix (N × 5)
    ax_x = x + 0.4
    ax_y = y + h - 1.4
    rbox(ax, ax_x, ax_y - 0.4, 1.8, 0.80, fc="#FFFFFF")
    ax.text(ax_x + 0.9, ax_y + 0.15, r"$\mathbf{P}^{\mathrm{OOF}}\in\mathbb{R}^{N\times 5}$",
            ha="center", va="center", fontsize=9, weight="bold")
    ax.text(ax_x + 0.9, ax_y - 0.18, r"logit transform",
            ha="center", va="center", fontsize=7, style="italic", color="#666")

    # arrow to meta fit
    arrow(ax, ax_x + 1.8, ax_y, ax_x + 2.4, ax_y, shape="fit")
    rbox(ax, ax_x + 2.4, ax_y - 0.4, 1.9, 0.80, fc="#FFFFFF")
    ax.text(ax_x + 3.35, ax_y + 0.15, "L2-LR (C=10)",
            ha="center", va="center", fontsize=9, weight="bold")
    ax.text(ax_x + 3.35, ax_y - 0.18, r"learn $\{\beta_k,b\}$",
            ha="center", va="center", fontsize=7, color="#666")

    # step B: p_tilde training predictions
    bx2 = x + 0.4
    by2 = ax_y - 1.15
    rbox(ax, bx2, by2 - 0.35, 1.8, 0.70, fc="#FFFFFF")
    ax.text(bx2 + 0.9, by2 - 0.00, r"$\tilde p_{\mathrm{train}}$",
            ha="center", va="center", fontsize=10, weight="bold")
    ax.text(bx2 + 0.9, by2 - 0.22, r"$\sigma(\beta^{\top}\mathrm{logit}(\mathbf{p})+b)$",
            ha="center", va="center", fontsize=7, color="#555")

    # arrow to isotonic fit
    arrow(ax, bx2 + 1.8, by2, bx2 + 2.4, by2, shape="PAV")

    # isotonic box
    rbox(ax, bx2 + 2.4, by2 - 0.48, 1.9, 0.95, fc="#FFFFFF")
    ax.text(bx2 + 3.35, by2 + 0.25, r"Isotonic regression",
            ha="center", va="center", fontsize=9, weight="bold")
    ax.text(bx2 + 3.35, by2 + 0.00, r"fit $\hat g:\tilde p\to \hat p$",
            ha="center", va="center", fontsize=7.5, color="#555")
    ax.text(bx2 + 3.35, by2 - 0.25, "ECE 0.071 → 0.032",
            ha="center", va="center", fontsize=7, color=C_ACC, style="italic")

    # bottom: pick tau* via balanced-acc search
    by3 = y + 0.55
    rbox(ax, x + 0.4, by3 - 0.40, w - 0.8, 0.75, fc="#FFF5F5")
    ax.text(x + w/2, by3 + 0.08, r"Threshold search:   "
            r"$\tau^\star=\arg\max_{\tau}\ \mathrm{BalAcc}(\hat p_{\mathrm{train}},y)=0.135$",
            ha="center", va="center", fontsize=9, color=C_ACC, weight="bold")
    ax.text(x + w/2, by3 - 0.20, "(note: τ⋆ is frozen from training set; never touches test set)",
            ha="center", va="center", fontsize=7, style="italic", color="#666")


def draw_artifacts_panel(ax, x, y, w, h):
    rbox(ax, x, y, w, h, fc="#F8F9FA", ec="#888", lw=1.0)
    ax.text(x + 0.30, y + h - 0.30, "Frozen artifacts → used by Fig A (inference)",
            ha="left", va="center", fontsize=10, weight="bold", color=C_EDGE)

    items = [
        ("5 × refit base models", "trained on full D_train with sample weights", C_PRIM),
        (r"$\{\beta_{RF},\beta_{NRB},\beta_{MT},\beta_{SVM},\beta_{ET},b\}$",
         "L2-LR meta coefficients (6 scalars)", C_OK),
        (r"$\hat g$ (isotonic PAV map)", "monotone piecewise-constant calibrator", C_WARN),
        (r"$\tau^\star = 0.135$", "decision threshold (bal-acc optimal)", C_ACC),
        (r"$\varphi$ parameters",
         "Likert thresholds, KFold TE means, z-score $(\\mu,\\sigma)$", C_PUR),
    ]
    for i, (a, b, c) in enumerate(items):
        ix = x + 0.5 + (i % 5) * (w - 1.0) / 5
        iy = y + h - 1.1
        ax.add_patch(Circle((ix + 0.18, iy), 0.13, fc=c, ec="black", lw=0.5))
        ax.text(ix + 0.42, iy + 0.05, a,
                fontsize=8, ha="left", va="center", weight="bold")
        ax.text(ix + 0.42, iy - 0.22, b,
                fontsize=6.5, ha="left", va="center", color="#666")


def fig_b_training():
    fig, ax = plt.subplots(figsize=(18, 11), facecolor=C_BG)
    ax.set_xlim(0, 18); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_facecolor(C_BG)

    ax.text(9, 10.55,
            "Figure B.  Training Protocol of the Champion Model",
            ha="center", fontsize=15, weight="bold", color=C_EDGE)
    ax.text(9, 10.15,
            "Three training-time machinery that does not appear at inference: "
            "Cleanlab v6 weights, 5-fold OOF, and meta+isotonic fitting",
            ha="center", fontsize=10, style="italic", color="#555")

    # Panel 1: Cleanlab
    draw_cleanlab_panel(ax, 0.4, 5.5, 5.8, 4.0)
    # Panel 2: OOF
    draw_oof_panel(ax, 6.4, 3.8, 5.2, 5.7)
    # Panel 3: Refit + isotonic
    draw_refit_panel(ax, 11.8, 5.5, 5.8, 4.0)
    # Panel 4: Artifacts (bottom bar)
    draw_artifacts_panel(ax, 0.4, 1.7, 17.2, 1.7)

    # inter-panel arrows
    a1 = FancyArrowPatch((6.2, 7.5), (6.4, 7.0), arrowstyle="->",
                         mutation_scale=16, color=C_EDGE, lw=1.3)
    ax.add_patch(a1)
    ax.text(6.3, 7.75, "w inject →", fontsize=7, ha="left", color="#555", style="italic")
    a2 = FancyArrowPatch((11.6, 6.7), (11.8, 7.5), arrowstyle="->",
                         mutation_scale=16, color=C_EDGE, lw=1.3)
    ax.add_patch(a2)
    ax.text(11.7, 7.05, r"$\mathbf{P}^{\mathrm{OOF}}$", fontsize=9, ha="left", color="#555")
    a3 = FancyArrowPatch((9, 3.6), (9, 3.5), arrowstyle="->",
                         mutation_scale=14, color=C_EDGE, lw=1.2)
    ax.add_patch(a3)
    # arrow from panel 3 down to artifacts
    a4 = FancyArrowPatch((14.5, 5.5), (14.5, 3.5), arrowstyle="->",
                         mutation_scale=16, color=C_EDGE, lw=1.5)
    ax.add_patch(a4)

    # footer: reproducibility note
    ax.text(9, 1.15,
            "Reproducibility: all seeds fixed (CV seed=42, base seeds=[0..4]); "
            "train/test split frozen at train_idx.npy / test_idx.npy; "
            "CL weights cached at sample_weights_v6.npy",
            ha="center", fontsize=8.5, color="#555", style="italic",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFFFFF", ec="#BBB", lw=0.6))

    fig.savefig(OUT / "fig_B_training.png", dpi=DPI, bbox_inches="tight", facecolor=C_BG)
    fig.savefig(OUT / "fig_B_training.pdf",       bbox_inches="tight", facecolor=C_BG)
    fig.savefig(OUT / "fig_B_training.svg",       bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"  wrote {OUT / 'fig_B_training.png'}  (+ .pdf, .svg)")


# ==========================================================================
# main
# ==========================================================================

if __name__ == "__main__":
    print("[17] making paper figures (Fig A + Fig B) ...")
    fig_a_inference()
    fig_b_training()
    print("[17] done.")
