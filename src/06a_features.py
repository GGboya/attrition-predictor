"""Phase 6a — Feature engineering module.

Pure module; NOT executed directly. Imported by 06c_stack_v4.py via the same
`importlib.util` pattern used elsewhere in the project (see 05_stacking.py).

Design rationale is in README.md §Phase 6 and the approved plan at
/Users/didi/.claude/plans/snappy-roaming-catmull.md.

Exports
-------
build_features(df, feat_cols, train_idx, test_idx, fold_idx)
    Leak-safe augmented feature matrix.
    Returns (X_train_aug, X_test_aug, feat_names).

Leak safety
-----------
- Likert cumulative dummies, interactions, gaps, composite index, and log-income
  are deterministic row-wise transforms → no leakage by construction.
- Income quintile-within-sector uses quintile boundaries estimated on TRAIN only.
- KFold target encoding for nominal cols: each train fold's validation slice
  sees group means computed on the fold's training slice only; test uses the
  all-train group mean.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

LIKERT_15_COLS = ["工作匹配度", "工作满意度", "工作机会", "工作氛围"]
STRESS_COL = "工作压力"          # 1–3
INCOME_COL = "收入水平"
SECTOR_COL = "工作单位性质"      # 1–5, nominal
REGION_COL = "工作区域"          # 1–3, nominal
MATCH_COL = "工作匹配度"
SAT_COL = "工作满意度"
OPP_COL = "工作机会"
ATMOS_COL = "工作氛围"
TARGET = "离职行为"


def _cumulative_dummies(col: pd.Series, thresholds=(2, 3, 4, 5)) -> pd.DataFrame:
    """x>=t for t in thresholds — 4 binary cols for a 1–5 Likert."""
    out = {}
    for t in thresholds:
        out[f"{col.name}>={t}"] = (col >= t).astype(np.float32)
    return pd.DataFrame(out, index=col.index)


def _kfold_target_encode(group_col: pd.Series, y: np.ndarray, fold_idx,
                         prior_strength: float = 20.0) -> np.ndarray:
    """Out-of-fold target mean encoding with smoothing toward the global prior.

    Parameters
    ----------
    group_col : pd.Series indexed 0..n_train-1 (already subsetted to train)
    y         : binary labels aligned with group_col
    fold_idx  : list of (tr_local, va_local) tuples, indices into 0..n_train-1
    prior_strength : Bayesian smoothing count — shrinks small groups toward global rate

    Returns
    -------
    np.ndarray shape (n_train,) — OOF target-mean encoding.
    """
    n = len(group_col)
    enc = np.full(n, np.nan, dtype=np.float32)
    global_mean = float(y.mean())
    g = group_col.values
    for tr, va in fold_idx:
        tr_y = y[tr]; tr_g = g[tr]
        for key in np.unique(g[va]):
            mask = tr_g == key
            if mask.sum() == 0:
                m = global_mean
            else:
                k = mask.sum()
                m = (tr_y[mask].sum() + prior_strength * global_mean) / (k + prior_strength)
            enc[va[g[va] == key]] = m
    # any fold that failed to assign (shouldn't happen with StratifiedKFold) → global
    enc[np.isnan(enc)] = global_mean
    return enc


def _test_target_encode(group_col_test: pd.Series, group_col_train: pd.Series,
                         y_train: np.ndarray, prior_strength: float = 20.0) -> np.ndarray:
    """Test-time target encoding using all-train group means with smoothing."""
    global_mean = float(y_train.mean())
    g_tr = group_col_train.values
    out = np.empty(len(group_col_test), dtype=np.float32)
    g_te = group_col_test.values
    lookup = {}
    for key in np.unique(g_tr):
        mask = g_tr == key
        k = mask.sum()
        lookup[key] = (y_train[mask].sum() + prior_strength * global_mean) / (k + prior_strength)
    for i, key in enumerate(g_te):
        out[i] = lookup.get(key, global_mean)
    return out


def _income_quintile_within_sector(income_tr: np.ndarray, sector_tr: np.ndarray,
                                    income_te: np.ndarray, sector_te: np.ndarray
                                    ) -> tuple[np.ndarray, np.ndarray]:
    """Rank income into within-sector quintiles.

    Quintile cut-points are estimated on TRAIN rows of each sector; TEST rows get
    digitized against those frozen cut-points.
    Returns (q_train, q_test) as float32 in {0,1,2,3,4}.
    """
    q_tr = np.zeros(len(income_tr), dtype=np.float32)
    q_te = np.zeros(len(income_te), dtype=np.float32)
    global_cuts = np.quantile(income_tr, [0.2, 0.4, 0.6, 0.8])
    for s in np.unique(sector_tr):
        mask_tr = sector_tr == s
        mask_te = sector_te == s
        if mask_tr.sum() < 10:
            cuts = global_cuts
        else:
            cuts = np.quantile(income_tr[mask_tr], [0.2, 0.4, 0.6, 0.8])
            # guard against degenerate (duplicate) cuts
            cuts = np.maximum.accumulate(cuts)
            if np.all(np.diff(cuts) == 0):
                cuts = global_cuts
        q_tr[mask_tr] = np.digitize(income_tr[mask_tr], cuts)
        if mask_te.any():
            q_te[mask_te] = np.digitize(income_te[mask_te], cuts)
    # sectors appearing only in test get global cuts
    unseen = ~np.isin(sector_te, np.unique(sector_tr))
    if unseen.any():
        q_te[unseen] = np.digitize(income_te[unseen], global_cuts)
    return q_tr, q_te


def build_features(df: pd.DataFrame, feat_cols: list[str], train_idx: np.ndarray,
                   test_idx: np.ndarray, fold_idx: list[tuple[np.ndarray, np.ndarray]]
                   ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Construct the leak-safe augmented feature matrix.

    Parameters
    ----------
    df         : cleaned dataframe (includes target)
    feat_cols  : 12 base feature columns (all df cols except 离职行为 / 离职意向)
    train_idx  : indices into df
    test_idx   : indices into df
    fold_idx   : list of (tr_local, va_local) index pairs into 0..len(train_idx)-1

    Returns
    -------
    X_train_aug : (n_train, F) float32
    X_test_aug  : (n_test,  F) float32
    feat_names  : list[str] of length F
    """
    df_tr = df.iloc[train_idx].reset_index(drop=True)
    df_te = df.iloc[test_idx].reset_index(drop=True)
    y_tr = df_tr[TARGET].values.astype(int)

    parts_tr: list[pd.DataFrame] = []
    parts_te: list[pd.DataFrame] = []

    # 1) base 12 columns, unchanged
    parts_tr.append(df_tr[feat_cols].astype(np.float32).reset_index(drop=True))
    parts_te.append(df_te[feat_cols].astype(np.float32).reset_index(drop=True))

    # 2) Likert cumulative dummies (4 cols × 4 thresholds = 16)
    for c in LIKERT_15_COLS:
        parts_tr.append(_cumulative_dummies(df_tr[c]))
        parts_te.append(_cumulative_dummies(df_te[c]))

    # 3) Interactions (top-5 compounded-dissatisfaction patterns)
    def _prod(a, b, name):
        return pd.DataFrame({name: (a * b).astype(np.float32)})
    interactions = [
        (SAT_COL, OPP_COL, "sat_x_opp"),
        (MATCH_COL, SAT_COL, "match_x_sat"),
        (MATCH_COL, OPP_COL, "match_x_opp"),
        (STRESS_COL, SAT_COL, "stress_x_sat"),
        (ATMOS_COL, OPP_COL, "atmos_x_opp"),
    ]
    for a, b, name in interactions:
        parts_tr.append(_prod(df_tr[a], df_tr[b], name).reset_index(drop=True))
        parts_te.append(_prod(df_te[a], df_te[b], name).reset_index(drop=True))

    # 4) Gap features — Mobley-style attitude gaps (linear combos trees need 2 splits to see)
    parts_tr.append(pd.DataFrame({
        "gap_opp_sat": (df_tr[OPP_COL] - df_tr[SAT_COL]).astype(np.float32),
        "gap_match_sat": (df_tr[MATCH_COL] - df_tr[SAT_COL]).astype(np.float32),
        "abs_match_sat": (df_tr[MATCH_COL] - df_tr[SAT_COL]).abs().astype(np.float32),
    }).reset_index(drop=True))
    parts_te.append(pd.DataFrame({
        "gap_opp_sat": (df_te[OPP_COL] - df_te[SAT_COL]).astype(np.float32),
        "gap_match_sat": (df_te[MATCH_COL] - df_te[SAT_COL]).astype(np.float32),
        "abs_match_sat": (df_te[MATCH_COL] - df_te[SAT_COL]).abs().astype(np.float32),
    }).reset_index(drop=True))

    # 5) Income transforms
    income_tr = df_tr[INCOME_COL].values.astype(np.float64)
    income_te = df_te[INCOME_COL].values.astype(np.float64)
    sector_tr = df_tr[SECTOR_COL].values.astype(int)
    sector_te = df_te[SECTOR_COL].values.astype(int)
    log_inc_tr = np.log1p(income_tr).astype(np.float32)
    log_inc_te = np.log1p(income_te).astype(np.float32)
    q_tr, q_te = _income_quintile_within_sector(income_tr, sector_tr, income_te, sector_te)
    parts_tr.append(pd.DataFrame({"log_income": log_inc_tr,
                                  "income_quintile_in_sector": q_tr}))
    parts_te.append(pd.DataFrame({"log_income": log_inc_te,
                                  "income_quintile_in_sector": q_te}))

    # 6) Composite attitudinal index
    comp_tr = df_tr[[MATCH_COL, SAT_COL, OPP_COL]].mean(axis=1).astype(np.float32).values
    comp_te = df_te[[MATCH_COL, SAT_COL, OPP_COL]].mean(axis=1).astype(np.float32).values
    parts_tr.append(pd.DataFrame({"attitude_composite": comp_tr}))
    parts_te.append(pd.DataFrame({"attitude_composite": comp_te}))

    # 7) KFold target encoding for nominal columns (leak-safe OOF on train, all-train for test)
    for col in (SECTOR_COL, REGION_COL):
        enc_tr = _kfold_target_encode(df_tr[col].reset_index(drop=True), y_tr, fold_idx)
        enc_te = _test_target_encode(df_te[col].reset_index(drop=True),
                                      df_tr[col].reset_index(drop=True), y_tr)
        name = f"te_{col}"
        parts_tr.append(pd.DataFrame({name: enc_tr}))
        parts_te.append(pd.DataFrame({name: enc_te}))

    X_tr_df = pd.concat(parts_tr, axis=1)
    X_te_df = pd.concat(parts_te, axis=1)
    assert list(X_tr_df.columns) == list(X_te_df.columns), "train/test schema drift"

    feat_names = list(X_tr_df.columns)
    return X_tr_df.values.astype(np.float32), X_te_df.values.astype(np.float32), feat_names
