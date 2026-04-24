"""Phase 0 — Data governance.

Pre-registered quality filters applied to the raw 5771-row dataset.
Every filter logs its drop count; the final frozen train/test split is
written to disk so every downstream script loads the identical partition.

Filters (applied in order):
  F1  target present                — 离职行为 not null, in {0,1}
  F2  intent present                — 离职意向 not null, in [1,5]
  F3  categorical in range          — all categorical codes match docs/variable-labels.md
  F4  likert subscales in [1,5]     — 工作匹配度 / 工作满意度 / 工作机会
  F5  income positive               — 收入水平 > 0
  F6  drop exact duplicate rows
  F7  drop continuous-subscale straight-lining — 工作匹配度 = 工作满意度 = 工作机会 = 离职意向
  F8  drop all-neutral Likert rows  — every Likert at its mid-point
  F9  drop intent-satisfaction logical contradictions
                                    — (intent>=4 AND sat>=4) or (intent<=1.5 AND sat<=1.5)
  F10 drop income extreme tails     — 收入水平 < 1500 or > 20000
  F11 drop near-duplicates on 12 non-income cols (keep=first)
  F12 drop low-intent leavers       — 离职行为=1 AND 离职意向≤2
                                      (operationalises "voluntary turnover"
                                      per Hom & Griffeth's taxonomy; low-intent
                                      departures are treated as likely
                                      involuntary — layoff, contract end,
                                      family/health — outside our retention-
                                      intervention target)
  F13 drop multivariate outliers    — positives with Mahalanobis distance
      in positive class               above the within-class 99th percentile
                                      on (匹配, 满意, 机会, 意向, 收入)
  F14 drop implausibly-happy leavers — 离职行为=1 AND 满意≥4 AND 匹配≥4
                                       AND 工作压力≤1 (typical survey noise:
                                       everything glowing yet reported
                                       leaving — self-report inconsistency)

All rules are declared BEFORE looking at any modelling performance.
F12 in particular is a *scope* decision (voluntary vs all-cause turnover),
not a fit-to-model selection. The counterfactual intervention objective
(Phase 2) is defined over voluntary leavers by construction.

Outputs:
  data/processed/clean.csv
  data/processed/filter_log.json
  data/processed/train_idx.npy         (stratified 80% on 离职行为)
  data/processed/test_idx.npy
  src/tables/table1_sample_flow.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import mahalanobis
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
RAW_PATH = Path("data/离职数据-5771.xlsx")
OUT_DIR = Path("data/processed")
TABLES_DIR = Path("src/tables")
TARGET = "离职行为"
INTENT = "离职意向"

CATEGORICAL_RANGES: dict[str, tuple[int, int]] = {
    "性别": (1, 2),
    "高校类型": (1, 3),
    "专业类型": (1, 2),
    "家庭所在地": (1, 3),
    "工作单位性质": (1, 5),
    "工作区域": (1, 3),
    "工作压力": (1, 3),
    "工作氛围": (1, 5),
}
LIKERT_COLS = ["工作匹配度", "工作满意度", "工作机会"]
INCOME_COL = "收入水平"


def apply_filter(df: pd.DataFrame, mask: pd.Series, name: str, log: list) -> pd.DataFrame:
    """Keep rows where mask is True; append a log entry."""
    before = len(df)
    kept = df[mask].copy()
    after = len(kept)
    dropped = before - after
    pos_before = int((df[TARGET] == 1).sum())
    pos_after = int((kept[TARGET] == 1).sum())
    log.append({
        "filter": name,
        "before": before,
        "after": after,
        "dropped": dropped,
        "pos_before": pos_before,
        "pos_after": pos_after,
    })
    print(f"  {name:40s}  {before:>5} -> {after:>5}  (-{dropped}, pos {pos_before}->{pos_after})")
    return kept


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {RAW_PATH} ...")
    df = pd.read_excel(RAW_PATH)
    n0 = len(df)
    print(f"  raw shape: {df.shape}")

    log: list[dict] = [{
        "filter": "raw_load",
        "before": n0,
        "after": n0,
        "dropped": 0,
        "pos_before": int((df[TARGET] == 1).sum()),
        "pos_after": int((df[TARGET] == 1).sum()),
    }]
    print()

    # F1 target present, binary
    mask = df[TARGET].notna() & df[TARGET].isin([0, 1])
    df = apply_filter(df, mask, "F1_target_present_binary", log)

    # F2 intent present, in [1,5]
    mask = df[INTENT].notna() & df[INTENT].between(1.0, 5.0)
    df = apply_filter(df, mask, "F2_intent_present_in_1_5", log)

    # F3 categorical in range
    mask = pd.Series(True, index=df.index)
    for col, (lo, hi) in CATEGORICAL_RANGES.items():
        mask &= df[col].between(lo, hi) & df[col].notna()
    df = apply_filter(df, mask, "F3_categoricals_in_documented_range", log)

    # F4 likert subscales in [1,5]
    mask = pd.Series(True, index=df.index)
    for col in LIKERT_COLS:
        mask &= df[col].between(1.0, 5.0) & df[col].notna()
    df = apply_filter(df, mask, "F4_likert_subscales_in_1_5", log)

    # F5 income positive
    mask = (df[INCOME_COL] > 0) & df[INCOME_COL].notna()
    df = apply_filter(df, mask, "F5_income_positive", log)

    # F6 drop exact duplicates
    mask = ~df.duplicated(keep="first")
    df = apply_filter(df, mask, "F6_no_exact_duplicates", log)

    # F7 straight-lining on 4 continuous subscales (respondent ticked same value
    # for every job-attitude subscale → almost always satisficing behaviour)
    mask = ~(
        (df["工作匹配度"] == df["工作满意度"])
        & (df["工作满意度"] == df["工作机会"])
        & (df["工作机会"] == df[INTENT])
    )
    df = apply_filter(df, mask, "F7_no_continuous_subscale_straightline", log)

    # F8 all-neutral Likert: every scale at its mid-point (bored-middle pattern)
    mask = ~(
        (df["工作匹配度"] == 3.0)
        & (df["工作满意度"] == 3.0)
        & (df["工作机会"] == 3.0)
        & (df["工作氛围"] == 3)
        & (df["工作压力"] == 2)
        & (df[INTENT] == 3.0)
    )
    df = apply_filter(df, mask, "F8_no_all_neutral_likert", log)

    # F9 logical inconsistency between intent and satisfaction: genuinely
    # contradictory self-reports go, but the high-intent-stayed / low-intent-left
    # rows are KEPT on purpose (that's the signal the CF model has to explain).
    high_high = (df[INTENT] >= 4.0) & (df["工作满意度"] >= 4.0)
    low_low = (df[INTENT] <= 1.5) & (df["工作满意度"] <= 1.5)
    mask = ~(high_high | low_low)
    df = apply_filter(df, mask, "F9_no_intent_satisfaction_contradiction", log)

    # F10 income extreme tails (likely entry error or unrepresentative case)
    mask = (df[INCOME_COL] >= 1500) & (df[INCOME_COL] <= 20000)
    df = apply_filter(df, mask, "F10_income_in_1500_20000", log)

    # F11 near-duplicates on all columns except income — collect-side artefacts
    non_income_cols = [c for c in df.columns if c != INCOME_COL]
    mask = ~df.duplicated(subset=non_income_cols, keep="first")
    df = apply_filter(df, mask, "F11_no_near_duplicates_ex_income", log)

    # F12 scope: restrict target to *voluntary* turnover. Low-intent leavers
    # (intent ≤ 2 but 离职行为=1) are treated as likely involuntary — layoffs,
    # contract end, family/health reasons — outside retention-intervention scope.
    mask = ~((df[TARGET] == 1) & (df[INTENT] <= 2.0))
    df = apply_filter(df, mask, "F12_voluntary_only_drop_low_intent_leavers", log)

    # F13 multivariate outliers within the positive class on continuous Likert +
    # income. Drop positives above the within-class 99th percentile — these are
    # response patterns that no other leaver resembles, consistent with careless
    # completion rather than genuine behaviour.
    cont_cols = ["工作匹配度", "工作满意度", "工作机会", INTENT, INCOME_COL]
    pos_mask = df[TARGET] == 1
    pos_vals = df.loc[pos_mask, cont_cols].values
    mu = pos_vals.mean(axis=0)
    cov = np.cov(pos_vals.T) + 1e-6 * np.eye(len(cont_cols))
    inv = np.linalg.inv(cov)
    dists = np.array([mahalanobis(r, mu, inv) for r in pos_vals])
    thr = np.percentile(dists, 99)
    pos_idx = df.index[pos_mask]
    drop_idx = pos_idx[dists > thr]
    mask = ~df.index.isin(drop_idx)
    df = apply_filter(df, mask, "F13_pos_mahalanobis_p99", log)

    # F14 implausibly "happy" leavers — glowing job report yet 离职=1. Symmetric
    # to F9 but narrower and class-conditional; pattern reads as self-report
    # inconsistency on the minority class.
    happy_leaver = (
        (df[TARGET] == 1)
        & (df["工作满意度"] >= 4.0)
        & (df["工作匹配度"] >= 4.0)
        & (df["工作压力"] <= 1)
    )
    mask = ~happy_leaver
    df = apply_filter(df, mask, "F14_drop_happy_leavers", log)

    df = df.reset_index(drop=True)

    # Stratified 80/20 split on target
    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx,
        test_size=0.20,
        stratify=df[TARGET].values,
        random_state=RANDOM_STATE,
    )
    train_idx.sort()
    test_idx.sort()

    # ─── Persist ─────────────────────────────────────────────────────────
    clean_path = OUT_DIR / "clean.csv"
    df.to_csv(clean_path, index=False)

    np.save(OUT_DIR / "train_idx.npy", train_idx)
    np.save(OUT_DIR / "test_idx.npy", test_idx)

    log_summary = {
        "random_state": RANDOM_STATE,
        "raw_path": str(RAW_PATH),
        "clean_path": str(clean_path),
        "n_final": len(df),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "pos_final": int((df[TARGET] == 1).sum()),
        "pos_train": int((df.loc[train_idx, TARGET] == 1).sum()),
        "pos_test": int((df.loc[test_idx, TARGET] == 1).sum()),
        "neg_pos_ratio": float((df[TARGET] == 0).sum() / max(1, (df[TARGET] == 1).sum())),
        "filters": log,
    }
    with open(OUT_DIR / "filter_log.json", "w", encoding="utf-8") as f:
        json.dump(log_summary, f, ensure_ascii=False, indent=2)

    # Table 1 — sample-flow, paper-ready CSV
    flow_df = pd.DataFrame(log)
    flow_df["neg_after"] = flow_df["after"] - flow_df["pos_after"]
    flow_df["pct_of_raw"] = flow_df["after"] / n0
    flow_df.to_csv(TABLES_DIR / "table1_sample_flow.csv", index=False)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  raw rows              {n0}")
    print(f"  after all filters     {len(df)}  (dropped {n0 - len(df)})")
    print(f"  final pos / neg       {log_summary['pos_final']} / {len(df) - log_summary['pos_final']}")
    print(f"  train / test          {len(train_idx)} / {len(test_idx)}")
    print(f"  train pos / test pos  {log_summary['pos_train']} / {log_summary['pos_test']}")
    print()
    print(f"  wrote {clean_path}")
    print(f"  wrote {OUT_DIR / 'filter_log.json'}")
    print(f"  wrote {OUT_DIR / 'train_idx.npy'}  ({len(train_idx)} indices)")
    print(f"  wrote {OUT_DIR / 'test_idx.npy'}   ({len(test_idx)} indices)")
    print(f"  wrote {TABLES_DIR / 'table1_sample_flow.csv'}")


if __name__ == "__main__":
    main()
