"""Phase 2c — cost-aware top-k intervention menu.

For each high-risk actual leaver with a feasible CF, we rank the top-k
candidates on HR cost and cost-effectiveness (Δp per ¥). The output is the
"intervention menu" an HR manager would choose from: cheapest feasible plan,
highest-effectiveness plan, and the Pareto frontier between them.

Outputs:
    src/tables/table5c_topk_costs.csv         aggregate summary
    src/tables/table5c_per_cf.csv             row-per-CF with all fields
    src/figures/fig3b_cf_cases.png            4 representative employee panels
    src/figures/fig3c_pareto.png              population-level cost/Δp Pareto
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import importlib.util, sys
_s1 = importlib.util.spec_from_file_location("mt_model", Path(__file__).with_name("01a_mt_model.py"))
_m = importlib.util.module_from_spec(_s1); sys.modules["mt_model"] = _m; _s1.loader.exec_module(_m)
MTMlp = _m.MTMlp

_s2 = importlib.util.spec_from_file_location("hrcf_algo", Path(__file__).with_name("02a_hrcf_algo.py"))
_a = importlib.util.module_from_spec(_s2); sys.modules["hrcf_algo"] = _a; _s2.loader.exec_module(_a)
HRCFGenerator = _a.HRCFGenerator
HRCFConfig = _a.HRCFConfig

RS = 42
np.random.seed(RS); torch.manual_seed(RS)
DEVICE = "cpu"

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)

# ─── load & pick cohort ───────────────────────────────────────────────
with open("models/mt_mlp_calibrated.pkl", "rb") as f:
    b = pickle.load(f)
feat_cols = b["feat_cols"]
model = MTMlp(in_dim=b["in_dim"], n_ord_classes=5).to(DEVICE)
model.load_state_dict(b["state_dict"]); model.eval()
scaler_mean, scaler_scale, iso = b["scaler_mean"], b["scaler_scale"], b["isotonic"]

df = pd.read_csv("data/processed/clean.csv")
test_idx = np.load("data/processed/test_idx.npy")
X_test = df.iloc[test_idx][feat_cols].values.astype(np.float32)
y_test = df.iloc[test_idx]["离职行为"].values.astype(int)

def predict_prob_batch(X):
    with torch.no_grad():
        xs = (X - scaler_mean) / scaler_scale
        logit, _ = model(torch.tensor(xs, dtype=torch.float32))
        p = torch.sigmoid(logit).cpu().numpy()
    return iso.predict(p)

p_test = predict_prob_batch(X_test)
HIGH_RISK_CUT = 0.30
TARGET_PROB = 0.20
eligible = np.where((p_test >= HIGH_RISK_CUT) & (y_test == 1))[0]
print(f"eligible high-risk actual leavers: {len(eligible)}")

# ─── generate CFs (same config as 4b best run) ────────────────────────
cfg = HRCFConfig(hard_project=True, alpha=1e-7, lr=0.15, n_restarts=12,
                 top_k=5, max_iters=1500, noise_sigma=0.3,
                 target_prob=TARGET_PROB, income_max_mult=2.0)
gen = HRCFGenerator(model, scaler_mean, scaler_scale, feat_cols,
                    isotonic=iso, config=cfg, device=DEVICE)

per_cf_rows = []
per_emp_rows = []
all_results = {}
for k, i in enumerate(eligible):
    x0 = X_test[i]
    r = gen.generate(x0)
    all_results[i] = r
    p0 = r["original_prob"]
    cands = r["candidates"]
    per_emp_rows.append({
        "emp_id": int(i),
        "p0": float(p0),
        "n_cf": len(cands),
        "success": bool(r["success"]),
        "min_cost": float(min(c["cost"] for c in cands)) if cands else np.nan,
        "min_cost_dp": float(max(c for c in [p0 - cd["prob"] for cd in cands]
                                  if True)) if cands else np.nan,
        "best_cpe_dp_per_kyuan": float(max((p0 - c["prob"]) / (c["cost"] / 1000.0 + 1e-6)
                                           for c in cands)) if cands else np.nan,
    })
    for ci, c in enumerate(cands):
        dp = p0 - c["prob"]
        cost_k = c["cost"] / 1000.0
        per_cf_rows.append({
            "emp_id": int(i),
            "cf_idx": ci,
            "p0": float(p0),
            "p_cf": float(c["prob"]),
            "dp": float(dp),
            "cost_yuan": float(c["cost"]),
            "dp_per_kyuan": float(dp / (cost_k + 1e-6)),
            "n_changes": len(c["changes"]),
            **{f"Δ_{fn}": float(dv) for fn, dv in c["changes"].items()},
        })
    if (k + 1) % 10 == 0:
        print(f"  [{k+1}/{len(eligible)}] done")

df_cf = pd.DataFrame(per_cf_rows).fillna(0.0)
df_emp = pd.DataFrame(per_emp_rows)

# ─── aggregate summary ────────────────────────────────────────────────
n_succ = int(df_emp["success"].sum())
print(f"\n── summary over {len(df_emp)} eligible employees "
      f"({n_succ} with ≥1 feasible CF) ──")

succ = df_emp[df_emp["success"]]
summary = {
    "n_eligible": len(df_emp),
    "n_feasible": n_succ,
    "feasibility_rate": n_succ / max(len(df_emp), 1),
    "mean_cfs_per_emp": float(succ["n_cf"].mean()),
    # Top-1 = cheapest plan per employee
    "top1_mean_cost_yuan": float(succ["min_cost"].mean()),
    "top1_median_cost_yuan": float(succ["min_cost"].median()),
    "top1_p10_cost_yuan": float(succ["min_cost"].quantile(0.1)),
    "top1_p90_cost_yuan": float(succ["min_cost"].quantile(0.9)),
    # Best cost-effectiveness per employee
    "bestCPE_dp_per_kyuan_mean": float(succ["best_cpe_dp_per_kyuan"].mean()),
    "bestCPE_dp_per_kyuan_median": float(succ["best_cpe_dp_per_kyuan"].median()),
}

# budget analysis: if HR has budget B ¥ per employee, what % can they flip?
budgets = [2000, 5000, 8000, 12000, 20000, 50000]
for B in budgets:
    n_within = int((succ["min_cost"] <= B).sum())
    summary[f"feasible_within_¥{B}"] = n_within
    summary[f"pct_within_¥{B}"] = n_within / max(len(df_emp), 1)

summary_df = pd.DataFrame([summary]).T.reset_index()
summary_df.columns = ["metric", "value"]
summary_df.to_csv(OUT_TABLES / "table5c_topk_costs.csv", index=False)
df_cf.to_csv(OUT_TABLES / "table5c_per_cf.csv", index=False)
print(f"wrote {OUT_TABLES / 'table5c_topk_costs.csv'}")
print(f"wrote {OUT_TABLES / 'table5c_per_cf.csv'}")
print(summary_df.to_string(index=False))

# ─── Pareto frontier across all CFs (population-level) ───────────────
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df_cf["cost_yuan"], df_cf["dp"], alpha=0.35, s=15, c="#4c72b0",
           label=f"all CFs  (n={len(df_cf)})")
# per-employee cheapest
cheapest_per_emp = df_cf.loc[df_cf.groupby("emp_id")["cost_yuan"].idxmin()]
ax.scatter(cheapest_per_emp["cost_yuan"], cheapest_per_emp["dp"], s=50,
           edgecolor="black", facecolor="#dd8452",
           label=f"cheapest per employee  (n={len(cheapest_per_emp)})")
# Pareto frontier: sort by cost, keep if dp strictly greater than running max
pts = df_cf.sort_values("cost_yuan").reset_index(drop=True)
keep = []
best_dp = -np.inf
for _, row in pts.iterrows():
    if row["dp"] > best_dp + 1e-6:
        keep.append(row); best_dp = row["dp"]
pareto = pd.DataFrame(keep)
ax.plot(pareto["cost_yuan"], pareto["dp"], "-", c="crimson", linewidth=2,
        label=f"Pareto frontier  (n={len(pareto)})")
ax.set_xlabel("intervention cost  ¥")
ax.set_ylabel("Δp  (original − counterfactual)")
ax.set_title("Cost vs prediction-drop across all HR-CF counterfactuals")
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig3c_pareto.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT_FIGS / 'fig3c_pareto.png'}")

# ─── 4 representative employee menu panels ────────────────────────────
# pick: cheapest-successful, most-expensive-successful, best-CPE, median-cost
succ_emps = df_emp[df_emp["success"]].copy()
picks: list[tuple[str, int]] = []
picks.append(("lowest cost", int(succ_emps.loc[succ_emps["min_cost"].idxmin(), "emp_id"])))
picks.append(("highest cost", int(succ_emps.loc[succ_emps["min_cost"].idxmax(), "emp_id"])))
picks.append(("median cost", int(succ_emps.loc[
    (succ_emps["min_cost"] - succ_emps["min_cost"].median()).abs().idxmin(), "emp_id"])))
picks.append(("best Δp/¥",
              int(succ_emps.loc[succ_emps["best_cpe_dp_per_kyuan"].idxmax(), "emp_id"])))

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()
for ax, (title, emp_id) in zip(axes, picks):
    r = all_results[emp_id]
    cands = r["candidates"]
    p0 = r["original_prob"]
    n = len(cands)
    # bar chart: cost stacked by feature-change cost contribution, annotated with Δp
    labels = [f"CF#{i+1}\nΔp={p0-c['prob']:.2f}" for i, c in enumerate(cands)]
    # reconstruct cost contribution per feature by applying cost weights
    cost_w = cfg.costs
    feat_contrib: dict[str, list[float]] = {fn: [0.0] * n for fn in cost_w}
    for i, c in enumerate(cands):
        for fn, dv in c["changes"].items():
            if fn in cost_w:
                feat_contrib.setdefault(fn, [0.0] * n)
                feat_contrib[fn][i] = abs(dv) * cost_w[fn]
    # filter out all-zero features
    feat_contrib = {k: v for k, v in feat_contrib.items() if any(abs(x) > 1e-3 for x in v)}
    # plot stacked bars
    bottom = np.zeros(n)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(feat_contrib), 1)))
    for (fn, vals), col in zip(feat_contrib.items(), colors):
        ax.bar(labels, vals, bottom=bottom, label=fn, color=col)
        bottom += np.array(vals)
    ax.set_title(f"{title}  (emp #{emp_id}, p₀={p0:.2f})", fontsize=11)
    ax.set_ylabel("intervention cost  ¥")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.3, axis="y")
plt.suptitle("HR-CF intervention menu — representative employees", fontsize=13)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig3b_cf_cases.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT_FIGS / 'fig3b_cf_cases.png'}")

# ─── print the 4 menus as text for sanity ─────────────────────────────
print("\n── representative intervention menus ──")
for title, emp_id in picks:
    r = all_results[emp_id]
    p0 = r["original_prob"]
    print(f"\n  {title}:  emp #{emp_id}  original p={p0:.3f}")
    for i, c in enumerate(r["candidates"]):
        dp = p0 - c["prob"]
        print(f"    CF#{i+1}  cost=¥{c['cost']:>7.0f}  Δp={dp:+.3f}  "
              f"Δp/¥k={dp/(c['cost']/1000+1e-6):+.3f}  "
              f"(n_changes={len(c['changes'])})")
        for fn, dv in c["changes"].items():
            print(f"      {fn:<10} Δ={dv:+.2f}")
