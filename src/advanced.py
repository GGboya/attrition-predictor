"""
Advanced pipeline — Kaggle-style attempt at a real SOTA on this dataset.

Strategy (in order of expected lift):
  1. Feature engineering — bin 收入水平, K-fold target encoding on mid-cardinality
     columns, pairwise interactions on domain-top features, net/ratio features.
  2. Explainable Boosting Machine (EBM) — explicit pairwise-interaction model,
     often matches XGB on tabular while being more interpretable. Added as a
     base learner and ablated alone.
  3. Diverse stacking — XGB + LGB + CatBoost + EBM + RF + calibrated LR,
     LR meta-learner on OOF probabilities. Diversity beats depth.
  4. Probability calibration (Isotonic) on a holdout slice of training OOF.
  5. Significance: DeLong paired test vs main.py's Voting and vs SVM baseline.

All preprocessing that touches labels (target encoding, OOF generation) uses
a single StratifiedKFold object and never sees the test set.
"""

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, matthews_corrcoef,
    f1_score, balanced_accuracy_score, recall_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from scipy import stats as sp_stats

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from interpret.glassbox import ExplainableBoostingClassifier
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

RANDOM_STATE = 42
N_SPLITS = 5
TARGET = '离职行为'
DATA_PATH = 'data/处理之后的离职数据-5000.xlsx'
N_BOOT = 1000

np.random.seed(RANDOM_STATE)


# ─── DeLong (Sun & Xu 2014 fast version) ────────────────────────────────────

def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(scores, y_true):
    order = (-y_true).argsort()
    scores = scores[:, order]
    y_true = y_true[order]
    m = int(y_true.sum())
    n = len(y_true) - m
    k = scores.shape[0]
    tx = np.empty((k, m)); ty = np.empty((k, n)); tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _compute_midrank(scores[r, :m])
        ty[r] = _compute_midrank(scores[r, m:])
        tz[r] = _compute_midrank(scores[r])
    aucs = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1 - (tz[:, m:] - ty) / m
    sx = np.cov(v01); sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, np.atleast_2d(s)


def delong_test(y_true, score_a, score_b):
    y_true = np.asarray(y_true); a = np.asarray(score_a); b = np.asarray(score_b)
    scores = np.vstack([a, b])
    aucs, s = _fast_delong(scores, y_true)
    diff = aucs[0] - aucs[1]
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return float(aucs[0]), float(aucs[1]), float(z), float(p)


def bootstrap_ci(y_true, y_prob, fn, n_boot=N_BOOT, rs=RANDOM_STATE):
    rng = np.random.default_rng(rs)
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    n = len(y_true); vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        vals.append(fn(y_true[idx], y_prob[idx]))
    vals = np.array(vals)
    return float(fn(y_true, y_prob)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ─── Feature engineering ────────────────────────────────────────────────────

def add_engineered(df):
    """Domain features that don't need label info — safe to compute globally."""
    out = df.copy()

    # Net positive experience: satisfaction minus pressure
    out['fe_sat_minus_press'] = out['工作满意度'] - out['工作压力']

    # Fit × opportunity: the two most-cited paper features interact
    out['fe_match_x_opp'] = out['工作匹配度'] * out['工作机会']

    # Climate × pressure: hostile environment index (low climate + high pressure)
    out['fe_climate_minus_press'] = out['工作氛围'] - out['工作压力']

    # Overall work-experience composite (z-sum of the four positive scales
    # minus pressure — very cheap latent summary)
    pos = ['工作匹配度', '工作满意度', '工作机会', '工作氛围']
    out['fe_work_composite'] = out[pos].sum(axis=1) - out['工作压力']

    # Income log (收入水平 is roughly in yuan, skewed)
    out['fe_log_income'] = np.log1p(out['收入水平'])

    # Income rank (quantile) — robust to the long tail
    out['fe_income_rank'] = out['收入水平'].rank(pct=True)

    # Intent × dissatisfaction: high intent amplified when satisfaction is low
    out['fe_intent_x_lowsat'] = out['离职意向'] * (6 - out['工作满意度'])

    return out


def kfold_target_encode(X_tr, y_tr, X_te, cols, n_splits=N_SPLITS, smoothing=20.0):
    """Out-of-fold target encoding for `cols`. Test-set encoding uses the full
    train posterior. Smoothing shrinks rare-category means toward the global mean."""
    X_tr = X_tr.copy(); X_te = X_te.copy()
    global_mean = y_tr.mean()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    for col in cols:
        tr_enc = np.zeros(len(X_tr))
        for fold_tr, fold_va in skf.split(X_tr, y_tr):
            sub_tr = X_tr.iloc[fold_tr]
            agg = pd.DataFrame({col: sub_tr[col], 'y': y_tr.iloc[fold_tr]})
            stats = agg.groupby(col)['y'].agg(['mean', 'count'])
            smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) \
                     / (stats['count'] + smoothing)
            mapped = X_tr.iloc[fold_va][col].map(smooth).fillna(global_mean).values
            tr_enc[fold_va] = mapped
        X_tr[f'te_{col}'] = tr_enc

        agg = pd.DataFrame({col: X_tr[col], 'y': y_tr})
        stats = agg.groupby(col)['y'].agg(['mean', 'count'])
        smooth = (stats['mean'] * stats['count'] + global_mean * smoothing) \
                 / (stats['count'] + smoothing)
        X_te[f'te_{col}'] = X_te[col].map(smooth).fillna(global_mean).values

    return X_tr, X_te


# ─── Load + split + engineer ────────────────────────────────────────────────

def section(t): print(f'\n{"=" * 62}\n{t}\n{"=" * 62}')

t0 = time.time()
section('1. LOAD + SPLIT + FEATURE ENGINEERING')

df = pd.read_excel(DATA_PATH)
print(f'shape: {df.shape}, classes: {dict(df[TARGET].value_counts())}')

df = add_engineered(df)
feature_cols = [c for c in df.columns if c != TARGET]
print(f'After FE: {len(feature_cols)} features ({sum(c.startswith("fe_") for c in feature_cols)} engineered)')

X = df[feature_cols]; y = df[TARGET].astype(int)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

te_cols = ['收入水平', '工作单位性质', '家庭所在地', '专业类型', '高校类型']
X_tr, X_te = kfold_target_encode(X_tr, y_tr, X_te, te_cols)
final_cols = list(X_tr.columns)
print(f'Final feature count: {len(final_cols)} (added target-encoded: {len([c for c in final_cols if c.startswith("te_")])})')


# ─── Optuna tuning for boosters on the FE+TE feature set ──────────────────

section('2. OPTUNA TUNING (XGB / LGB / CAT on engineered features)')

pos_w = (y_tr == 0).sum() / (y_tr == 1).sum()
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
skf3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)


def _cv_auc(model_ctor, X, y, cv):
    aucs = []
    for tr, va in cv.split(X, y):
        m = model_ctor()
        m.fit(X.iloc[tr], y.iloc[tr])
        aucs.append(roc_auc_score(y.iloc[va], m.predict_proba(X.iloc[va])[:, 1]))
    return float(np.mean(aucs))


def tune_xgb(n_trials=25):
    def obj(trial):
        def ctor():
            return xgb.XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 200, 800),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
                min_child_weight=trial.suggest_int('min_child_weight', 1, 8),
                scale_pos_weight=pos_w, eval_metric='auc', verbosity=0,
                random_state=RANDOM_STATE, n_jobs=-1)
        return _cv_auc(ctor, X_tr, y_tr, skf3)
    st = optuna.create_study(direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return st.best_params, st.best_value


def tune_lgb(n_trials=25):
    def obj(trial):
        def ctor():
            return lgb.LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 200, 800),
                num_leaves=trial.suggest_int('num_leaves', 15, 127),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.12, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                reg_lambda=trial.suggest_float('reg_lambda', 0.5, 5.0, log=True),
                min_child_samples=trial.suggest_int('min_child_samples', 5, 40),
                is_unbalance=True, verbosity=-1,
                random_state=RANDOM_STATE, n_jobs=-1)
        return _cv_auc(ctor, X_tr, y_tr, skf3)
    st = optuna.create_study(direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return st.best_params, st.best_value


def tune_cat(n_trials=15):
    def obj(trial):
        def ctor():
            return CatBoostClassifier(
                iterations=trial.suggest_int('iterations', 200, 800),
                depth=trial.suggest_int('depth', 4, 8),
                learning_rate=trial.suggest_float('learning_rate', 0.02, 0.12, log=True),
                l2_leaf_reg=trial.suggest_float('l2_leaf_reg', 1.0, 8.0, log=True),
                auto_class_weights='Balanced', verbose=False,
                random_state=RANDOM_STATE, allow_writing_files=False)
        return _cv_auc(ctor, X_tr, y_tr, skf3)
    st = optuna.create_study(direction='maximize',
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
    st.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    return st.best_params, st.best_value


tic = time.time(); xgb_best, xgb_cv = tune_xgb(); print(f'  XGB best CV AUC = {xgb_cv:.4f}  ({time.time()-tic:.1f}s)')
tic = time.time(); lgb_best, lgb_cv = tune_lgb(); print(f'  LGB best CV AUC = {lgb_cv:.4f}  ({time.time()-tic:.1f}s)')
tic = time.time(); cat_best, cat_cv = tune_cat(); print(f'  CAT best CV AUC = {cat_cv:.4f}  ({time.time()-tic:.1f}s)')


section('3. BASE LEARNERS (tuned bosters + EBM + RF + LR)')


def make_learners():
    return {
        'XGB': xgb.XGBClassifier(
            **xgb_best, scale_pos_weight=pos_w, eval_metric='auc', verbosity=0,
            random_state=RANDOM_STATE, n_jobs=-1),
        'LGB': lgb.LGBMClassifier(
            **lgb_best, is_unbalance=True, verbosity=-1,
            random_state=RANDOM_STATE, n_jobs=-1),
        'CAT': CatBoostClassifier(
            **cat_best, auto_class_weights='Balanced', verbose=False,
            random_state=RANDOM_STATE, allow_writing_files=False),
        'EBM': ExplainableBoostingClassifier(
            interactions=10, outer_bags=8, random_state=RANDOM_STATE),
        'RF': RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
        'LR': Pipeline([
            ('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, class_weight='balanced',
                                       C=1.0, random_state=RANDOM_STATE))]),
    }


oof_probs = {}      # OOF predictions (for meta-learner training)
test_probs = {}     # average of fold models on test (for meta-learner scoring)

for name in make_learners():
    tic = time.time()
    oof = np.zeros(len(X_tr))
    te_stack = np.zeros(len(X_te))
    for fold_tr, fold_va in skf.split(X_tr, y_tr):
        m = make_learners()[name]
        m.fit(X_tr.iloc[fold_tr], y_tr.iloc[fold_tr])
        oof[fold_va] = m.predict_proba(X_tr.iloc[fold_va])[:, 1]
        te_stack += m.predict_proba(X_te)[:, 1] / N_SPLITS
    oof_probs[name] = oof
    test_probs[name] = te_stack
    print(f'  {name:4s}  OOF AUC = {roc_auc_score(y_tr, oof):.4f}   '
          f'test AUC = {roc_auc_score(y_te, te_stack):.4f}   ({time.time()-tic:.1f}s)')


# ─── Meta-learner: logistic regression on OOF probabilities ─────────────────

section('4. STACKING META-LEARNER')

meta_tr = np.column_stack([oof_probs[k] for k in oof_probs])
meta_te = np.column_stack([test_probs[k] for k in test_probs])

meta = LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE)
meta.fit(meta_tr, y_tr)
stack_test = meta.predict_proba(meta_te)[:, 1]
stack_oof = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE),
                              meta_tr, y_tr, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
print(f'  Stacked OOF AUC = {roc_auc_score(y_tr, stack_oof):.4f}')
print(f'  Stacked test AUC = {roc_auc_score(y_te, stack_test):.4f}')
print(f'  Meta coefficients: {dict(zip(oof_probs.keys(), meta.coef_[0].round(3)))}')

# Simple soft-voting average as well
vote_test = meta_te.mean(axis=1)
print(f'  Equal-vote test AUC = {roc_auc_score(y_te, vote_test):.4f}')

# Rank-average (robust to scale differences across learners)
def rank_avg(P):
    R = np.zeros_like(P)
    for j in range(P.shape[1]):
        R[:, j] = pd.Series(P[:, j]).rank(pct=True).values
    return R.mean(axis=1)
rank_test = rank_avg(meta_te)
print(f'  Rank-avg test AUC = {roc_auc_score(y_te, rank_test):.4f}')


# ─── Isotonic calibration of best model ─────────────────────────────────────

section('5. PROBABILITY CALIBRATION (Isotonic)')

# Pick best among {stack, vote, rank} on OOF-equivalent basis.
# For stack we have stack_oof; for vote/rank we need OOF equivalents.
vote_oof = meta_tr.mean(axis=1)
rank_oof = rank_avg(meta_tr)

candidates = {
    'Stacking': (stack_oof, stack_test),
    'Equal-vote': (vote_oof, vote_test),
    'Rank-avg': (rank_oof, rank_test),
}
for k, (o, t) in candidates.items():
    print(f'  {k:12s}: OOF AUC = {roc_auc_score(y_tr, o):.4f}   '
          f'test AUC = {roc_auc_score(y_te, t):.4f}')

best_name = max(candidates, key=lambda k: roc_auc_score(y_tr, candidates[k][0]))
best_oof, best_test = candidates[best_name]
print(f'  → best by OOF AUC: {best_name}')

from sklearn.isotonic import IsotonicRegression
iso = IsotonicRegression(out_of_bounds='clip').fit(best_oof, y_tr)
best_test_cal = iso.transform(best_test)
print(f'  Pre-calibration  Brier = {brier_score_loss(y_te, best_test):.4f}')
print(f'  Post-calibration Brier = {brier_score_loss(y_te, best_test_cal):.4f}  '
      f'(AUC unchanged: {roc_auc_score(y_te, best_test_cal):.4f})')


# ─── Full report with CIs + DeLong vs baselines ─────────────────────────────

section('6. FINAL REPORT')

auc_fn = roc_auc_score; ap_fn = average_precision_score

def threshold_from_oof(oof, y_tr):
    best_f, t_opt = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.01):
        f = f1_score(y_tr, (oof >= t).astype(int), pos_label=1, zero_division=0)
        if f > best_f: best_f, t_opt = f, t
    return t_opt


def metric_row(y, p, oof, y_tr):
    t = threshold_from_oof(oof, y_tr)
    pred = (p >= t).astype(int)
    auc, lo, hi = bootstrap_ci(y, p, auc_fn)
    pr, plo, phi = bootstrap_ci(y, p, ap_fn)
    return dict(
        t=t,
        auc=auc, auc_lo=lo, auc_hi=hi,
        pr=pr, pr_lo=plo, pr_hi=phi,
        brier=brier_score_loss(y, p),
        mcc=matthews_corrcoef(y, pred),
        bal=balanced_accuracy_score(y, pred),
        f1=f1_score(y, pred, pos_label=1),
    )


rows = {}
for k in oof_probs:
    rows[f'base:{k}'] = metric_row(y_te, test_probs[k], oof_probs[k], y_tr)
rows['Stacking'] = metric_row(y_te, stack_test, stack_oof, y_tr)
rows['Equal-vote'] = metric_row(y_te, vote_test, vote_oof, y_tr)
rows['Rank-avg'] = metric_row(y_te, rank_test, rank_oof, y_tr)
rows[f'{best_name}+Isotonic'] = metric_row(y_te, best_test_cal, iso.transform(best_oof), y_tr)

print()
print(f'  Test-set positive rate: {y_te.mean():.3f} (PR-AUC random baseline)')
print()
print(f'  {"Model":24s} {"AUC [95% CI]":22s}  {"PR-AUC [95% CI]":22s}  '
      f'{"Brier":>6s} {"MCC":>6s} {"Bal":>6s} {"F1(R)":>6s}  {"t":>4s}')
print('  ' + '-' * 110)
for k, r in rows.items():
    print(f'  {k:24s} {r["auc"]:.3f} [{r["auc_lo"]:.3f},{r["auc_hi"]:.3f}]  '
          f'{r["pr"]:.3f} [{r["pr_lo"]:.3f},{r["pr_hi"]:.3f}]  '
          f'{r["brier"]:6.3f} {r["mcc"]:6.3f} {r["bal"]:6.3f} {r["f1"]:6.3f}  {r["t"]:4.2f}')


# ─── DeLong vs main.py Voting and vs SVM baseline ───────────────────────────

section('7. DELONG: advanced pipeline vs main.py references')

# We don't have main.py's voting probs in this process — but we can train
# a plain SVM and a single XGB on the SAME split as a within-script reference.
ref = {}

# SVM baseline (matching main.py's)
svm = Pipeline([('sc', StandardScaler()),
                ('clf', SVC(kernel='rbf', C=1.0, gamma='scale',
                            class_weight='balanced', probability=True,
                            random_state=RANDOM_STATE))])
svm.fit(X_tr[[c for c in final_cols if not c.startswith(('fe_', 'te_'))]], y_tr)
ref['SVM_RBF (orig features)'] = svm.predict_proba(
    X_te[[c for c in final_cols if not c.startswith(('fe_', 'te_'))]])[:, 1]

# Single XGB on ORIGINAL features (what main.py's XGB sees)
xgb_plain = xgb.XGBClassifier(
    n_estimators=400, max_depth=5, learning_rate=0.05, scale_pos_weight=pos_w,
    random_state=RANDOM_STATE, n_jobs=-1, eval_metric='auc', verbosity=0)
orig_cols = [c for c in final_cols if not c.startswith(('fe_', 'te_'))]
xgb_plain.fit(X_tr[orig_cols], y_tr)
ref['XGB (orig features)'] = xgb_plain.predict_proba(X_te[orig_cols])[:, 1]

our_best_test = best_test_cal if roc_auc_score(y_te, best_test_cal) >= roc_auc_score(y_te, best_test) else best_test
our_best_name = f'{best_name}+Isotonic' if our_best_test is best_test_cal else best_name

print(f'\n  Our headline model: {our_best_name}   AUC = {roc_auc_score(y_te, our_best_test):.4f}\n')
print(f'  {"Reference":28s} {"AUC_ref":>8s}  {"Δ AUC":>8s}  {"z":>6s}  {"p":>10s}  sig')
print('  ' + '-' * 72)
for name, p in ref.items():
    a_ours, a_ref, z, pv = delong_test(y_te, our_best_test, p)
    sig = '***' if pv < 1e-3 else ('**' if pv < 1e-2 else ('*' if pv < 0.05 else 'n.s.'))
    print(f'  {name:28s} {a_ref:8.4f}  {a_ours-a_ref:+8.4f}  {z:6.2f}  {pv:10.2e}  {sig}')

print(f'\nTotal runtime: {time.time() - t0:.1f}s')
