"""Phase 2b — run HR-CF vs soft-constraint baseline on high-risk test leavers.

Compares on five dimensions:
    1. Actionability (% top-k CFs that satisfy ALL hard constraints)
    2. Sparsity    (mean number of features changed)
    3. Proximity   (mean scaled-space L1 distance)
    4. Plausibility (mean k-NN distance to training leavers in scaled space,
                     lower = closer to observed data manifold)
    5. Diversity   (mean pairwise L2 in scaled space across the top-k menu)

Target population: test-set samples with calibrated p >= 0.5 (high-risk).
For each we request top-5 counterfactuals under two algorithms:
    * soft-CF  (gradient on prob + soft cost, no projection) — DiCE-style
    * HR-CF   (our algorithm, hard projection every step)

Outputs:
    src/tables/table5b_hrcf_vs_dice.csv
    src/figures/fig3a_hrcf_radar.png
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

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
_spec = importlib.util.spec_from_file_location(
    "mt_model", Path(__file__).with_name("01a_mt_model.py"))
_m = importlib.util.module_from_spec(_spec); sys.modules["mt_model"] = _m; _spec.loader.exec_module(_m)
MTMlp = _m.MTMlp

_spec2 = importlib.util.spec_from_file_location(
    "hrcf_algo", Path(__file__).with_name("02a_hrcf_algo.py"))
_a = importlib.util.module_from_spec(_spec2); sys.modules["hrcf_algo"] = _a; _spec2.loader.exec_module(_a)
HRCFGenerator = _a.HRCFGenerator
HRCFConfig = _a.HRCFConfig
IMMUTABLE_COLS = _a.IMMUTABLE_COLS
DOWN_COLS = _a.DOWN_COLS

RS = 42
np.random.seed(RS); torch.manual_seed(RS)
DEVICE = "cpu"  # CF is fast on CPU; avoid MPS/OMP dance

TARGET = "离职行为"
INTENT = "离职意向"

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)


# ─── load calibrated MT-MLP + data ────────────────────────────────────
with open("models/mt_mlp_calibrated.pkl", "rb") as f:
    bundle = pickle.load(f)

feat_cols = bundle["feat_cols"]
in_dim = bundle["in_dim"]
model = MTMlp(in_dim=in_dim, n_ord_classes=5).to(DEVICE)
model.load_state_dict(bundle["state_dict"])
model.eval()
scaler_mean = bundle["scaler_mean"]; scaler_scale = bundle["scaler_scale"]
iso = bundle["isotonic"]
print(f"loaded MT-MLP  in_dim={in_dim}  best_lam={bundle['best_lam']}  thr_f1={bundle['thr_f1']:.3f}")

df = pd.read_csv("data/processed/clean.csv")
test_idx = np.load("data/processed/test_idx.npy")
X_test = df.iloc[test_idx][feat_cols].values.astype(np.float32)
y_test = df.iloc[test_idx][TARGET].values.astype(int)

# training leavers for plausibility kNN manifold
train_idx = np.load("data/processed/train_idx.npy")
X_train = df.iloc[train_idx][feat_cols].values.astype(np.float32)
y_train = df.iloc[train_idx][TARGET].values.astype(int)
leavers_train = X_train[y_train == 1]
leavers_train_scl = (leavers_train - scaler_mean) / scaler_scale


# ─── pick high-risk subset ────────────────────────────────────────────
def predict_prob_batch(X_raw):
    with torch.no_grad():
        xs = (X_raw - scaler_mean) / scaler_scale
        xt = torch.tensor(xs, dtype=torch.float32, device=DEVICE)
        logit, _ = model(xt)
        p = torch.sigmoid(logit).cpu().numpy()
    return iso.predict(p)


probs_test = predict_prob_batch(X_test)
HIGH_RISK_CUT = 0.30       # calibrated prob (test max is ~0.57, base rate 0.15)
TARGET_PROB = 0.20         # slightly below F1-optimal threshold (0.21)
high_risk_mask = (probs_test >= HIGH_RISK_CUT) & (y_test == 1)
print(f"test: {len(y_test)}  pos={y_test.sum()}  "
      f"p>={HIGH_RISK_CUT} = {int((probs_test >= HIGH_RISK_CUT).sum())}  "
      f"high-risk actual leavers (eligible for CF): {int(high_risk_mask.sum())}  "
      f"target τ = {TARGET_PROB}")

# cap for runtime
MAX_SAMPLES = 60
eligible_idx = np.where(high_risk_mask)[0]
if len(eligible_idx) > MAX_SAMPLES:
    sel = np.random.default_rng(RS).choice(eligible_idx, MAX_SAMPLES, replace=False)
    sel.sort()
else:
    sel = eligible_idx
print(f"running CF on {len(sel)} samples\n")


# ─── build two generators ─────────────────────────────────────────────
cfg_hard = HRCFConfig(hard_project=True,  alpha=1e-7, lr=0.15, n_restarts=12,
                      top_k=5, max_iters=1500, noise_sigma=0.3,
                      target_prob=TARGET_PROB, income_max_mult=2.0)
cfg_soft = HRCFConfig(hard_project=False, alpha=1e-7, lr=0.15, n_restarts=12,
                      top_k=5, max_iters=1500, noise_sigma=0.3,
                      target_prob=TARGET_PROB, income_max_mult=2.0)

gen_hard = HRCFGenerator(model, scaler_mean, scaler_scale, feat_cols,
                         isotonic=iso, config=cfg_hard, device=DEVICE)
gen_soft = HRCFGenerator(model, scaler_mean, scaler_scale, feat_cols,
                         isotonic=iso, config=cfg_soft, device=DEVICE)


# ─── metrics ──────────────────────────────────────────────────────────
def per_sample_metrics(result: dict, gen_for_check: HRCFGenerator) -> dict:
    """Return per-sample metrics aggregated over that sample's top-k."""
    if not result["success"]:
        return dict(success=False, n_k=0, actionability=np.nan, sparsity=np.nan,
                    proximity=np.nan, plausibility=np.nan, diversity=np.nan,
                    mean_cost=np.nan, mean_dprob=np.nan)

    cands = result["candidates"]
    x0 = result["original"]
    p0 = result["original_prob"]
    # check ALL candidates for HARD-constraint satisfaction (use gen_hard rules)
    feas_flags = []
    sparsity_list = []; proximity_list = []; plaus_list = []
    dprob_list = []; cost_list = []
    scaled = lambda v: (v - scaler_mean) / scaler_scale
    for c in cands:
        feas, _ = gen_for_check._check_feasibility(c["x"], x0)
        feas_flags.append(bool(feas))
        sparsity_list.append(int(sum(abs(c["x"] - x0) > 1e-6)))
        proximity_list.append(float(np.abs(scaled(c["x"]) - scaled(x0)).sum()))
        # plausibility: mean L2 in scaled space to 5 nearest training leavers
        d = np.linalg.norm(leavers_train_scl - scaled(c["x"]), axis=1)
        plaus_list.append(float(np.sort(d)[:5].mean()))
        dprob_list.append(float(p0 - c["prob"]))
        cost_list.append(float(c["cost"]))
    # diversity: mean pairwise L2 in scaled space
    if len(cands) > 1:
        pairs = []
        S = [scaled(c["x"]) for c in cands]
        for i in range(len(S)):
            for j in range(i + 1, len(S)):
                pairs.append(np.linalg.norm(S[i] - S[j]))
        diversity = float(np.mean(pairs))
    else:
        diversity = 0.0

    return dict(
        success=True,
        n_k=len(cands),
        actionability=float(np.mean(feas_flags)),
        sparsity=float(np.mean(sparsity_list)),
        proximity=float(np.mean(proximity_list)),
        plausibility=float(np.mean(plaus_list)),
        diversity=diversity,
        mean_cost=float(np.mean(cost_list)),
        mean_dprob=float(np.mean(dprob_list)),
    )


# ─── run ──────────────────────────────────────────────────────────────
rows = []
for si, idx in enumerate(sel):
    x0 = X_test[idx]
    rh = gen_hard.generate(x0)
    rs = gen_soft.generate(x0)
    mh = per_sample_metrics(rh, gen_hard)
    ms = per_sample_metrics(rs, gen_hard)   # check against HARD rules for both
    mh["algo"] = "HR-CF"; mh["sample_idx"] = int(idx); mh["p0"] = float(rh["original_prob"])
    ms["algo"] = "soft-CF"; ms["sample_idx"] = int(idx); ms["p0"] = float(rs["original_prob"])
    rows.append(mh); rows.append(ms)
    if (si + 1) % 10 == 0:
        print(f"  [{si+1}/{len(sel)}] done")

df_per = pd.DataFrame(rows)

# ─── aggregate ────────────────────────────────────────────────────────
print("\n── summary (mean over high-risk leavers; CI = ±1 SE) ──")
agg_cols = ["success", "n_k", "actionability", "sparsity", "proximity",
            "plausibility", "diversity", "mean_cost", "mean_dprob"]
summary = df_per.groupby("algo")[agg_cols].agg(["mean", "std", "count"])
print(summary.round(3).to_string())

# per-constraint satisfaction rate — break down by category
cat_rates = {"HR-CF": {}, "soft-CF": {}}
for algo_name, gen in [("HR-CF", gen_hard), ("soft-CF", gen_soft)]:
    imm_ok = mono_ok = bound_ok = total = 0
    for idx in sel:
        x0 = X_test[idx]
        res = gen.generate(x0)
        if not res["success"]:
            continue
        for c in res["candidates"]:
            total += 1
            _, v = gen_hard._check_feasibility(c["x"], x0)
            imm_viol = any("immutable" in str(val) for val in v.values())
            mono_viol = any("up_only" in str(val) or "down_only" in str(val) for val in v.values())
            bound_viol = any("above" in str(val) or "below" in str(val)
                             or "non_integer" in str(val) or "non_quarter" in str(val)
                             for val in v.values())
            imm_ok += 0 if imm_viol else 1
            mono_ok += 0 if mono_viol else 1
            bound_ok += 0 if bound_viol else 1
    cat_rates[algo_name] = dict(
        immutable=imm_ok / max(total, 1),
        monotonic=mono_ok / max(total, 1),
        bounded=bound_ok / max(total, 1),
        total_cfs=total,
    )
print("\n── constraint satisfaction by category ──")
print(pd.DataFrame(cat_rates).round(3).to_string())

# write main comparison table
m = df_per.groupby("algo")[agg_cols].mean().round(4)
m["success_rate"] = df_per.groupby("algo")["success"].mean().round(4)
m.to_csv(OUT_TABLES / "table5b_hrcf_vs_dice.csv")
print(f"\nwrote {OUT_TABLES / 'table5b_hrcf_vs_dice.csv'}")

# ─── radar figure ────────────────────────────────────────────────────
def _to_rel(series, higher_better=True):
    """scale to [0,1] where 1 is best"""
    arr = np.asarray(series, dtype=float)
    if np.all(arr == arr[0]):
        return np.ones_like(arr)
    mn, mx = arr.min(), arr.max()
    rel = (arr - mn) / (mx - mn + 1e-12)
    return rel if higher_better else 1 - rel


radar_axes = [
    ("actionability", True),
    ("sparsity",     False),   # fewer changes is better
    ("proximity",    False),   # smaller distance is better
    ("plausibility", False),   # closer to manifold is better
    ("diversity",    True),    # more diversity is better
]
labels = [a for a, _ in radar_axes]
algos = ["soft-CF", "HR-CF"]
raw_vals = {a: [df_per[df_per["algo"] == a][m].mean() for m, _ in radar_axes] for a in algos}
# normalise per-axis
norm = []
for col_i, (m, hb) in enumerate(radar_axes):
    col = [raw_vals[a][col_i] for a in algos]
    norm.append(_to_rel(col, higher_better=hb))
norm = np.array(norm).T  # (algos, axes)

angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6.5, 6.5), subplot_kw=dict(polar=True))
for ai, a in enumerate(algos):
    vals = list(norm[ai]) + [norm[ai][0]]
    ax.plot(angles, vals, label=a, linewidth=2)
    ax.fill(angles, vals, alpha=0.15)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_ylim(0, 1.05)
ax.set_title("HR-CF vs soft-CF  (higher = better on each axis)")
ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig3a_hrcf_radar.png", dpi=140, bbox_inches="tight")
print(f"wrote {OUT_FIGS / 'fig3a_hrcf_radar.png'}")

# ─── a few example counterfactuals (sanity print) ────────────────────
print("\n── 3 example HR-CF counterfactuals ──")
for k, idx in enumerate(sel[:3]):
    x0 = X_test[idx]
    r = gen_hard.generate(x0)
    print(f"\n  [sample {idx}] original p={r['original_prob']:.3f}  success={r['success']}")
    for ci, c in enumerate(r["candidates"][:3]):
        print(f"    CF#{ci+1}: p={c['prob']:.3f}  cost=¥{c['cost']:.0f}  "
              f"feasible={c['feasible']}")
        for fn, dv in c["changes"].items():
            print(f"       {fn:<10} Δ = {dv:+.2f}")
