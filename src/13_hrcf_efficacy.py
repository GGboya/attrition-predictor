"""Phase 11 — HRCF surrogate validity.

HRCF's target classifier is the calibrated MT-MLP (see src/02b_hrcf_run.py:65)
because the full stack champion is not end-to-end differentiable after
isotonic. This phase validates that the MT-MLP is a faithful surrogate of
the Phase 6 stack champion in the region HRCF operates on.

Checks
------
1. Global agreement on the held-out test set
   - Pearson / Spearman correlation between stack_p and mtmlp_p
2. High-risk-region agreement (where HRCF actually runs)
   - Same correlations restricted to {i : stack_p[i] >= 0.30}
   - Top-10%, top-20%, top-30% agreement: Jaccard and Cohen's κ
3. Decision agreement at the Bal-Acc threshold
   - κ between (stack_p >= 0.135) and (mtmlp_p >= thr_mtmlp)
4. Bootstrap 95% CI for the high-risk-region Pearson ρ

Outputs
-------
src/tables/table20_hrcf_surrogate_validity.csv
src/figures/fig22_stack_vs_mtmlp.png
"""
from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import importlib.util, sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


RS = 42
N_BOOT = 1000
TARGET = "离职行为"
HIGH_RISK_CUT = 0.30
THR_BALACC = 0.135

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_FIGS = Path("src/figures"); OUT_FIGS.mkdir(exist_ok=True, parents=True)


# ─── load MT-MLP (HRCF target) ─────────────────────────────────────────
_spec = importlib.util.spec_from_file_location(
    "mt_model", Path(__file__).with_name("01a_mt_model.py"))
_m = importlib.util.module_from_spec(_spec); sys.modules["mt_model"] = _m
_spec.loader.exec_module(_m)
MTMlp = _m.MTMlp

with open("models/mt_mlp_calibrated.pkl", "rb") as f:
    bundle = pickle.load(f)
feat_cols = bundle["feat_cols"]
model = MTMlp(in_dim=bundle["in_dim"], n_ord_classes=5)
model.load_state_dict(bundle["state_dict"]); model.eval()
scaler_mean, scaler_scale = bundle["scaler_mean"], bundle["scaler_scale"]
iso_mtmlp = bundle["isotonic"]
thr_mtmlp = float(bundle["thr_f1"])
print(f"MT-MLP loaded: in_dim={bundle['in_dim']}  thr_f1={thr_mtmlp:.3f}")


# ─── data + stack probs ────────────────────────────────────────────────
df = pd.read_csv("data/processed/clean.csv")
test_idx = np.load("data/processed/test_idx.npy")
stack_te = np.load("data/processed/phase6_meta_test_probs.npy")
y_te = df[TARGET].values.astype(int)[test_idx]

X_te = df.iloc[test_idx][feat_cols].values.astype(np.float32)
with torch.no_grad():
    xs = (X_te - scaler_mean) / scaler_scale
    xt = torch.tensor(xs, dtype=torch.float32)
    logit, _ = model(xt)
    mtmlp_raw = torch.sigmoid(logit).cpu().numpy()
mtmlp_te = iso_mtmlp.predict(mtmlp_raw)
print(f"test N={len(y_te)}  stack-mean={stack_te.mean():.3f}  "
      f"mtmlp-mean={mtmlp_te.mean():.3f}")


# ─── 1. Global agreement ───────────────────────────────────────────────
r_global, p_global = pearsonr(stack_te, mtmlp_te)
rho_global, p_rho_global = spearmanr(stack_te, mtmlp_te)
print(f"\nGlobal (N={len(y_te)}): Pearson r={r_global:.4f} (p={p_global:.2e})  "
      f"Spearman ρ={rho_global:.4f} (p={p_rho_global:.2e})")


# ─── 2. High-risk region agreement ─────────────────────────────────────
hr_mask = stack_te >= HIGH_RISK_CUT
n_hr = int(hr_mask.sum())
print(f"\nHigh-risk (stack >= {HIGH_RISK_CUT}, N={n_hr}):")
if n_hr >= 20:
    r_hr, p_hr = pearsonr(stack_te[hr_mask], mtmlp_te[hr_mask])
    rho_hr, p_rho_hr = spearmanr(stack_te[hr_mask], mtmlp_te[hr_mask])
    print(f"  Pearson r={r_hr:.4f} (p={p_hr:.2e})  "
          f"Spearman ρ={rho_hr:.4f} (p={p_rho_hr:.2e})")

    # bootstrap CI for Pearson in high-risk region
    rng = np.random.default_rng(RS)
    boot_rs = []
    s_hr, m_hr = stack_te[hr_mask], mtmlp_te[hr_mask]
    for _ in range(N_BOOT):
        idx = rng.integers(0, n_hr, n_hr)
        if np.std(s_hr[idx]) < 1e-9 or np.std(m_hr[idx]) < 1e-9: continue
        boot_rs.append(pearsonr(s_hr[idx], m_hr[idx])[0])
    r_lo, r_hi = float(np.percentile(boot_rs, 2.5)), float(np.percentile(boot_rs, 97.5))
    print(f"  Pearson 95% CI = [{r_lo:.4f}, {r_hi:.4f}]  ({len(boot_rs)} boot reps)")
else:
    r_hr = rho_hr = r_lo = r_hi = float("nan")


# ─── 3. Top-k agreement ────────────────────────────────────────────────
def topk_jaccard(a, b, k):
    ai = set(np.argsort(-a)[:k]); bi = set(np.argsort(-b)[:k])
    return len(ai & bi) / len(ai | bi) if (ai | bi) else 0.0


rows_topk = []
for frac in (0.10, 0.20, 0.30):
    k = int(len(y_te) * frac)
    ja = topk_jaccard(stack_te, mtmlp_te, k)
    a_top = (np.argsort(-stack_te).argsort() < k).astype(int)
    b_top = (np.argsort(-mtmlp_te).argsort() < k).astype(int)
    kap = cohen_kappa_score(a_top, b_top)
    rows_topk.append({"top_frac": frac, "k": k, "jaccard": ja, "kappa": kap})
    print(f"  top-{int(frac*100)}%  k={k}   Jaccard={ja:.3f}   κ={kap:.3f}")


# ─── 4. Decision agreement at Bal-Acc threshold ────────────────────────
stack_pos = (stack_te >= THR_BALACC).astype(int)
mtmlp_pos = (mtmlp_te >= thr_mtmlp).astype(int)
kap_thr = cohen_kappa_score(stack_pos, mtmlp_pos)
agreement = float((stack_pos == mtmlp_pos).mean())
print(f"\nDecision agreement:  stack@τ={THR_BALACC}  vs  MT-MLP@τ={thr_mtmlp:.3f}")
print(f"  κ = {kap_thr:.3f}   raw agreement = {agreement:.3f}")


# ─── write table20 ─────────────────────────────────────────────────────
rows20 = [
    {"scope": "global_test", "n": len(y_te),
     "pearson_r": r_global, "spearman_rho": rho_global,
     "r_lo": float("nan"), "r_hi": float("nan"),
     "note": ""},
    {"scope": f"high_risk_stack>={HIGH_RISK_CUT}", "n": n_hr,
     "pearson_r": r_hr, "spearman_rho": rho_hr,
     "r_lo": r_lo, "r_hi": r_hi,
     "note": "bootstrap 1000 reps"},
]
for r in rows_topk:
    rows20.append({
        "scope": f"top_{int(r['top_frac']*100)}pct_agreement",
        "n": r["k"], "pearson_r": float("nan"), "spearman_rho": float("nan"),
        "r_lo": r["jaccard"], "r_hi": r["kappa"],
        "note": "r_lo=jaccard, r_hi=cohen_kappa",
    })
rows20.append({
    "scope": f"decision_agreement_thr",
    "n": len(y_te),
    "pearson_r": float("nan"), "spearman_rho": float("nan"),
    "r_lo": agreement, "r_hi": kap_thr,
    "note": f"stack@{THR_BALACC} vs mtmlp@{thr_mtmlp:.3f};  r_lo=agreement, r_hi=kappa",
})
pd.DataFrame(rows20).to_csv(OUT_TABLES / "table20_hrcf_surrogate_validity.csv",
                             index=False)
print(f"\nwrote {OUT_TABLES / 'table20_hrcf_surrogate_validity.csv'}")


# ─── figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11.5, 5))

# scatter: all + high-risk overlay
ax = axes[0]
ax.scatter(stack_te[~hr_mask], mtmlp_te[~hr_mask], s=10, alpha=0.35,
           color="tab:gray", label=f"low-risk (stack<{HIGH_RISK_CUT}, n={int((~hr_mask).sum())})")
ax.scatter(stack_te[hr_mask], mtmlp_te[hr_mask], s=22, alpha=0.75,
           color="crimson", label=f"high-risk (stack≥{HIGH_RISK_CUT}, n={n_hr})")
lim = max(stack_te.max(), mtmlp_te.max()) * 1.05
ax.plot([0, lim], [0, lim], "--", color="black", lw=1, alpha=0.6, label="y=x")
ax.axvline(HIGH_RISK_CUT, color="crimson", ls=":", alpha=0.4)
ax.axhline(thr_mtmlp, color="tab:blue", ls=":", alpha=0.4,
           label=f"MT-MLP τ={thr_mtmlp:.2f}")
ax.axvline(THR_BALACC, color="tab:blue", ls="--", alpha=0.4,
           label=f"Stack τ={THR_BALACC}")
ax.set_xlabel("Phase 6 stack probability (calibrated)")
ax.set_ylabel("MT-MLP probability (calibrated)")
ax.set_title(f"Stack vs MT-MLP on test\n"
             f"global r={r_global:.3f},  high-risk r={r_hr:.3f} [{r_lo:.2f},{r_hi:.2f}]",
             fontsize=10.5)
ax.legend(loc="lower right", fontsize=8.5)
ax.grid(alpha=0.3)

# rank scatter
ax = axes[1]
rank_stack = np.argsort(np.argsort(-stack_te))
rank_mtmlp = np.argsort(np.argsort(-mtmlp_te))
ax.scatter(rank_stack, rank_mtmlp, s=8, alpha=0.35, color="tab:blue")
top30 = int(len(y_te) * 0.30)
ax.axvline(top30, color="crimson", ls=":", alpha=0.5, label=f"top-30% cutoff")
ax.axhline(top30, color="crimson", ls=":", alpha=0.5)
ax.set_xlabel("Rank under Phase 6 stack")
ax.set_ylabel("Rank under MT-MLP")
ax.set_title(f"Rank agreement  (Spearman ρ={rho_global:.3f})", fontsize=10.5)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3)

plt.suptitle("HRCF surrogate validity — MT-MLP as differentiable proxy for Phase 6 stack",
             fontsize=11.5, y=1.02)
plt.tight_layout()
plt.savefig(OUT_FIGS / "fig22_stack_vs_mtmlp.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"wrote {OUT_FIGS / 'fig22_stack_vs_mtmlp.png'}")


# ─── pass criterion ─────────────────────────────────────────────────────
print("\n" + "=" * 72)
if np.isnan(r_hr):
    print(f"  SKIP — high-risk region has n<20 (n={n_hr})")
elif r_hr >= 0.85:
    print(f"  PASS — high-risk Pearson r = {r_hr:.4f} ≥ 0.85")
    print(f"         MT-MLP is a faithful surrogate for HRCF targeting.")
else:
    print(f"  WARNING — high-risk Pearson r = {r_hr:.4f} < 0.85")
    print(f"            Reviewers may question MT-MLP surrogacy; report honestly in paper.")
print("=" * 72)
