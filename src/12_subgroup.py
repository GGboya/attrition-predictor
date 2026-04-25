"""Phase 10 — Subgroup robustness on the Phase 6 champion.

Slices the test set by four demographic dimensions and reports AUC, Sens,
Spec, sample size, and positive-class rate within each subgroup.

Uses the cached calibrated probabilities from Phase 6 directly — no
refitting.

Dimensions
----------
- 性别 (gender):                 1 男, 2 女
- 高校类型 (university tier):    1 重点, 2 一般, 3 民办
- 家庭所在地 (home region):       1 大中城市, 2 城镇/县, 3 农村
- 收入水平分位 (income tertile): low / mid / high — boundaries set on train

Outputs
-------
src/tables/table19_subgroup_performance.csv
src/figures/fig20_subgroup_auc_bars.png
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
    balanced_accuracy_score, confusion_matrix, roc_auc_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC",
                                    "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


RS = 42
N_BOOT = 1000
TARGET = "离职行为"
INTENT = "离职意向"
THR_BALACC = 0.135
THR_F1 = 0.185

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)


# ─── data ───────────────────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
train_idx = np.load("data/processed/train_idx.npy")
test_idx = np.load("data/processed/test_idx.npy")
test_probs = np.load("data/processed/phase6_meta_test_probs.npy")

y_all = df[TARGET].values.astype(int)
y_te = y_all[test_idx]

df_te = df.iloc[test_idx].reset_index(drop=True)
df_tr = df.iloc[train_idx].reset_index(drop=True)

print(f"test N={len(y_te)}  pos_rate={y_te.mean():.3f}  "
      f"overall AUC={roc_auc_score(y_te, test_probs):.4f}")


# ─── helpers ────────────────────────────────────────────────────────────
def subgroup_metrics(mask, name, group_label, thr=THR_BALACC):
    n = int(mask.sum())
    if n < 20:
        return {
            "dimension": name, "group": group_label, "n": n,
            "pos_rate": float("nan"),
            "AUC": float("nan"), "AUC_lo": float("nan"), "AUC_hi": float("nan"),
            "Sens": float("nan"), "Spec": float("nan"), "Bal_Acc": float("nan"),
            "note": "n<20, skipped",
        }
    y = y_te[mask]; p = test_probs[mask]
    if len(np.unique(y)) < 2:
        return {
            "dimension": name, "group": group_label, "n": n,
            "pos_rate": float(y.mean()),
            "AUC": float("nan"), "AUC_lo": float("nan"), "AUC_hi": float("nan"),
            "Sens": float("nan"), "Spec": float("nan"), "Bal_Acc": float("nan"),
            "note": "single-class",
        }
    auc = roc_auc_score(y, p)

    # bootstrap CI
    rng = np.random.default_rng(RS)
    aucs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        try: aucs.append(roc_auc_score(y[idx], p[idx]))
        except Exception: pass
    lo, hi = (float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))) \
        if aucs else (float("nan"), float("nan"))

    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1)
    bal = balanced_accuracy_score(y, pred)
    return {
        "dimension": name, "group": group_label, "n": n,
        "pos_rate": float(y.mean()),
        "AUC": float(auc), "AUC_lo": lo, "AUC_hi": hi,
        "Sens": float(sens), "Spec": float(spec), "Bal_Acc": float(bal),
        "note": "",
    }


# ─── subgroups ──────────────────────────────────────────────────────────
rows = []

# overall reference
rows.append(subgroup_metrics(np.ones(len(y_te), bool), "OVERALL", "all"))

# 性别
gender = df_te["性别"].values
for code, label in [(1, "男生"), (2, "女生")]:
    rows.append(subgroup_metrics(gender == code, "性别", label))

# 高校类型
uni = df_te["高校类型"].values
for code, label in [(1, "重点高校"), (2, "一般高校"), (3, "民办/独立")]:
    rows.append(subgroup_metrics(uni == code, "高校类型", label))

# 家庭所在地
region = df_te["家庭所在地"].values
for code, label in [(1, "大中城市"), (2, "城镇/县"), (3, "农村")]:
    rows.append(subgroup_metrics(region == code, "家庭所在地", label))

# 收入水平分位 (boundaries on train)
income_tr = df_tr["收入水平"].values.astype(float)
income_te = df_te["收入水平"].values.astype(float)
q33, q67 = np.quantile(income_tr, [1/3, 2/3])
print(f"\nincome tertile boundaries (train): q33={q33:.0f}  q67={q67:.0f}")

tercile_masks = [
    (income_te < q33, "low (<q33)"),
    ((income_te >= q33) & (income_te < q67), "mid"),
    (income_te >= q67, "high (>=q67)"),
]
for mask, label in tercile_masks:
    rows.append(subgroup_metrics(mask, "收入分位", label))

# 专业类型（bonus — cheap to add）
major = df_te["专业类型"].values
for code, label in [(1, "理工科类"), (2, "人文社科类")]:
    rows.append(subgroup_metrics(major == code, "专业类型", label))


# ─── write & print ──────────────────────────────────────────────────────
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT_TABLES / "table19_subgroup_performance.csv", index=False)
print(f"\nwrote {OUT_TABLES / 'table19_subgroup_performance.csv'}")

print("\n" + "=" * 80)
print(f"{'dimension':<14}{'group':<14}{'N':>5}{'pos%':>7}{'AUC':>8}"
      f"{'[lo,':>9}{'hi]':>8}{'Sens':>8}{'Spec':>8}{'BalAcc':>9}")
print("=" * 80)
for r in rows:
    if np.isnan(r["AUC"]):
        print(f"{r['dimension']:<14}{r['group']:<14}{r['n']:>5}{'—':>7}"
              f"{'—':>8}{r.get('note', ''):>26}")
        continue
    print(f"{r['dimension']:<14}{r['group']:<14}{r['n']:>5}"
          f"{r['pos_rate']*100:>6.1f}%"
          f"{r['AUC']:>8.4f}"
          f"{r['AUC_lo']:>9.3f}{r['AUC_hi']:>8.3f}"
          f"{r['Sens']:>8.3f}{r['Spec']:>8.3f}{r['Bal_Acc']:>9.4f}")


# ─── figure ─────────────────────────────────────────────────────────────
plot_df = out_df[out_df["dimension"] != "OVERALL"].copy()
overall = out_df[out_df["dimension"] == "OVERALL"].iloc[0]

dim_order = ["性别", "高校类型", "家庭所在地", "收入分位", "专业类型"]
plot_df["_dim_ord"] = plot_df["dimension"].map({d: i for i, d in enumerate(dim_order)})
plot_df = plot_df.sort_values(["_dim_ord"]).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(11, 5))
colors_map = {"性别": "tab:blue", "高校类型": "tab:orange",
              "家庭所在地": "tab:green", "收入分位": "tab:red",
              "专业类型": "tab:purple"}
xs = np.arange(len(plot_df))
bar_colors = [colors_map[d] for d in plot_df["dimension"]]
yerr_lo = plot_df["AUC"] - plot_df["AUC_lo"]
yerr_hi = plot_df["AUC_hi"] - plot_df["AUC"]
ax.bar(xs, plot_df["AUC"], yerr=[yerr_lo, yerr_hi], color=bar_colors,
       edgecolor="black", alpha=0.85, capsize=3)
ax.axhline(overall["AUC"], color="black", ls="--", lw=1.2,
           label=f"Overall AUC = {overall['AUC']:.4f}")
for i, (_, r) in enumerate(plot_df.iterrows()):
    ax.text(i, r["AUC"] + 0.01, f"{r['AUC']:.3f}", ha="center", fontsize=8)
    ax.text(i, 0.02, f"N={r['n']}", ha="center", fontsize=7, color="white")
ax.set_xticks(xs)
ax.set_xticklabels([f"{r['dimension']}\n{r['group']}"
                    for _, r in plot_df.iterrows()],
                    rotation=30, ha="right", fontsize=8.5)
ax.set_ylabel("AUC (Phase 6 champion, test set)")
ax.set_title("Subgroup AUC — robustness of Phase 6 champion across demographics",
             fontsize=11)
ax.set_ylim(0.5, 1.0)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="y", alpha=0.3)
# legend patches for dimensions
from matplotlib.patches import Patch
handles = [Patch(facecolor=c, edgecolor="black", label=d)
           for d, c in colors_map.items()]
handles.append(plt.Line2D([0], [0], color="black", ls="--",
                           label=f"Overall {overall['AUC']:.4f}"))
ax.legend(handles=handles, loc="lower right", fontsize=8.5)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig20_subgroup_auc_bars.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {OUT_FIGS / 'fig20_subgroup_auc_bars.png'}")


# ─── pass criteria ─────────────────────────────────────────────────────
failing = out_df[(out_df["dimension"] != "OVERALL")
                 & out_df["AUC"].notna()
                 & (out_df["AUC"] < 0.70)]
print("\n" + "=" * 72)
if len(failing) == 0:
    print("  PASS — all subgroup AUCs ≥ 0.70")
else:
    print(f"  FAIL — {len(failing)} subgroup(s) with AUC < 0.70:")
    for _, r in failing.iterrows():
        print(f"    {r['dimension']} / {r['group']}: AUC={r['AUC']:.4f}, N={r['n']}")
print("=" * 72)
