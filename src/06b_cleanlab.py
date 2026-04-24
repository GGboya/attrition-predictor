"""Phase 6b — Asymmetric label-noise diagnostic (Cleanlab).

Attrition labels are asymmetrically noisy: employees who *intended* to leave but
were unable to do so (labor market, contract, family) look like 离职行为=0 yet
are closer to 1 in latent state. Current NR-Boost / NR-Forest assume symmetric
noise and can't target this case.

This script identifies suspicious-0 rows via Cleanlab's label-quality scores,
applies an **asymmetric** filter (only low-quality 0-labels with high intent),
and emits per-sample weights for Phase 6c to consume.

Pipeline
--------
1. RandomForest (Phase 5 hyperparams) → 5-fold OOF probabilities on the train
   partition. Same CV seed as 05_stacking.py.
2. `cleanlab.rank.get_label_quality_scores` on (y_train, oof_probs).
3. Asymmetric flag mask:
       y==0  AND  quality_score < q_thresh  AND  离职意向 ≥ 4
4. Down-weight flagged samples (default w_flag=0.3; report sensitivity sweep).
5. Sanity check: re-run RF with these weights; OOF AUC should not drop.

Outputs
-------
data/processed/sample_weights_v6.npy          (n_train,) float32
src/tables/table10a_label_issues.csv          per-flagged-row details
src/tables/table10a_cleanlab_sensitivity.csv  threshold × weight grid with OOF AUC
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
from cleanlab.rank import get_label_quality_scores
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

RS = 42
N_FOLDS = 5
TARGET = "离职行为"
INTENT = "离职意向"

# Defaults selected after the sensitivity sweep below; kept permissive
# on q_thresh because a tighter setting flags too few rows to matter (5/4375).
Q_THRESH_MAIN = 0.5
W_FLAG_MAIN = 0.3
INTENT_CUTOFF_MAIN = 3.5  # 离职意向 ≥ 3.5 (asymmetric; 3.5 is "leaning toward wanting to leave")
Q_THRESH_GRID = [0.2, 0.3, 0.5]
W_FLAG_GRID = [0.1, 0.3, 0.5]
INTENT_CUTOFF_GRID = [3.5, 4.0]  # keep asymmetry: never below neutral (3)

OUT_TABLES = Path("src/tables"); OUT_TABLES.mkdir(exist_ok=True, parents=True)
OUT_WEIGHTS = Path("data/processed/sample_weights_v6.npy")


def rf_factory():
    return RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=5,
                                   class_weight="balanced_subsample",
                                   n_jobs=-1, random_state=RS)


def oof_rf(X_tr, y_tr, fold_idx, sample_weight=None):
    oof = np.zeros(len(y_tr), dtype=np.float64)
    for tr, va in fold_idx:
        rf = rf_factory()
        w = None if sample_weight is None else sample_weight[tr]
        rf.fit(X_tr[tr], y_tr[tr], sample_weight=w)
        oof[va] = rf.predict_proba(X_tr[va])[:, 1]
    return oof


def main():
    df = pd.read_csv("data/processed/clean.csv")
    train_idx = np.load("data/processed/train_idx.npy")
    feat_cols = [c for c in df.columns if c not in (TARGET, INTENT)]
    X_tr = df.iloc[train_idx][feat_cols].values.astype(np.float32)
    y_tr = df.iloc[train_idx][TARGET].values.astype(int)
    intent_tr = df.iloc[train_idx][INTENT].values.astype(float)

    print(f"train rows={len(y_tr)}  pos={y_tr.sum()}  neg={(y_tr==0).sum()}")
    for c in (3.5, 4.0):
        n = int(((y_tr == 0) & (intent_tr >= c)).sum())
        print(f"  y=0 ∧ 离职意向 ≥ {c}: {n} rows "
              f"({n / max(1, (y_tr==0).sum()):.3f} of negatives)")

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RS)
    fold_idx = list(skf.split(X_tr, y_tr))

    # --- baseline OOF ------------------------------------------------------
    print("\n── baseline RF OOF (uniform weights) ──")
    oof_base = oof_rf(X_tr, y_tr, fold_idx, sample_weight=None)
    auc_base = roc_auc_score(y_tr, oof_base)
    print(f"  OOF AUC = {auc_base:.4f}")

    # --- cleanlab label quality -------------------------------------------
    pred_probs = np.column_stack([1 - oof_base, oof_base])  # shape (n, 2)
    quality = get_label_quality_scores(labels=y_tr, pred_probs=pred_probs,
                                        method="self_confidence")
    print(f"\n── label-quality score distribution ──")
    for q in [0.05, 0.10, 0.20, 0.30, 0.50]:
        print(f"  P(quality ≤ {q:.2f}) = {(quality <= q).mean():.4f}  "
              f"({(quality <= q).sum()} rows)")

    # --- asymmetric flag mask (main config) -------------------------------
    flag_main = ((y_tr == 0) & (quality < Q_THRESH_MAIN)
                  & (intent_tr >= INTENT_CUTOFF_MAIN))
    print(f"\n── asymmetric flag (main: y=0 ∧ quality<{Q_THRESH_MAIN} "
          f"∧ intent≥{INTENT_CUTOFF_MAIN}) ──")
    print(f"  flagged rows = {int(flag_main.sum())}  "
          f"({100 * flag_main.mean():.2f}% of train)")
    vals, cnts = np.unique(intent_tr[flag_main], return_counts=True)
    print(f"  intent distribution of flagged rows: "
          + ", ".join(f"{v:.1f}→{c}" for v, c in zip(vals, cnts)))

    # --- sensitivity sweep -------------------------------------------------
    print(f"\n── sensitivity sweep (intent_cutoff × q_thresh × w_flag) ──")
    rows = [{"intent_cutoff": None, "q_thresh": None, "w_flag": 1.0,
              "n_flagged": 0, "oof_auc": auc_base, "delta_auc": 0.0,
              "note": "baseline"}]
    for intent_cutoff in INTENT_CUTOFF_GRID:
        for q_thresh in Q_THRESH_GRID:
            flag = ((y_tr == 0) & (quality < q_thresh)
                     & (intent_tr >= intent_cutoff))
            n_flag = int(flag.sum())
            for w_flag in W_FLAG_GRID:
                w = np.ones(len(y_tr), dtype=np.float32)
                w[flag] = w_flag
                oof_w = oof_rf(X_tr, y_tr, fold_idx, sample_weight=w)
                auc_w = roc_auc_score(y_tr, oof_w)
                is_main = (intent_cutoff == INTENT_CUTOFF_MAIN
                           and q_thresh == Q_THRESH_MAIN
                           and w_flag == W_FLAG_MAIN)
                rows.append({"intent_cutoff": intent_cutoff,
                              "q_thresh": q_thresh, "w_flag": w_flag,
                              "n_flagged": n_flag, "oof_auc": auc_w,
                              "delta_auc": auc_w - auc_base,
                              "note": "main" if is_main else ""})
                print(f"  intent≥{intent_cutoff:.1f}  q_thresh={q_thresh:.2f}  "
                      f"w_flag={w_flag:.2f}  n_flag={n_flag:4d}  "
                      f"OOF AUC={auc_w:.4f}  Δ={auc_w - auc_base:+.4f}")

    sens_df = pd.DataFrame(rows)
    sens_path = OUT_TABLES / "table10a_cleanlab_sensitivity.csv"
    sens_df.to_csv(sens_path, index=False)
    print(f"\nwrote {sens_path}")

    # --- write main sample weights ----------------------------------------
    w_main = np.ones(len(y_tr), dtype=np.float32)
    w_main[flag_main] = W_FLAG_MAIN
    np.save(OUT_WEIGHTS, w_main)
    print(f"wrote {OUT_WEIGHTS}  "
          f"(n_flagged={int(flag_main.sum())}, w_flag={W_FLAG_MAIN})")

    # --- per-flagged-row dump ---------------------------------------------
    flag_df = pd.DataFrame({
        "df_idx": train_idx[flag_main],
        "train_local_idx": np.where(flag_main)[0],
        "离职行为": y_tr[flag_main],
        "离职意向": intent_tr[flag_main],
        "oof_prob_1": oof_base[flag_main],
        "quality_score": quality[flag_main],
        "sample_weight": w_main[flag_main],
    }).sort_values("oof_prob_1", ascending=False)
    issues_path = OUT_TABLES / "table10a_label_issues.csv"
    flag_df.to_csv(issues_path, index=False)
    print(f"wrote {issues_path}  ({len(flag_df)} rows)")

    # --- directional sanity check -----------------------------------------
    oof_main = oof_rf(X_tr, y_tr, fold_idx, sample_weight=w_main)
    auc_main = roc_auc_score(y_tr, oof_main)
    print(f"\n── directional check (main config) ──")
    print(f"  OOF AUC baseline      = {auc_base:.4f}")
    print(f"  OOF AUC reweighted    = {auc_main:.4f}")
    print(f"  ΔAUC                  = {auc_main - auc_base:+.4f}")
    if auc_main < auc_base - 0.002:
        print(f"  WARNING: reweighting hurt OOF AUC by >0.002. "
              f"Consider a laxer q_thresh or smaller w_flag change before Phase 6c.")
    else:
        print(f"  OK — reweighting is directionally safe.")


if __name__ == "__main__":
    main()
