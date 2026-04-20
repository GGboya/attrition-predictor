"""
Reproduce the BORF paper's reported numbers with CTGAN + Bayesian-tuned RF,
under TWO evaluation protocols:

  A) CLEAN: fit CTGAN on training minority ONLY; pristine test set.
     This is the honest protocol.

  B) LEAKY: fit CTGAN on the full minority, oversample everything, THEN split.
     Test set contains synthetic minority samples.
     This is the pattern many CTGAN-oversampling papers fall into.

If A stays around our diagnose.py numbers (AUC ~0.72, bal_acc ~0.69) but
B jumps close to paper's (AUC 0.69-ish, bal_acc 0.786, rec_0 0.892, rec_1 0.681),
that's empirical evidence the paper's bal_acc/recall numbers come from a
leakage-contaminated evaluation, not from a better model.

Runtime target: ~3-5 minutes (CTGAN fit dominates).
"""

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    precision_score, f1_score, roc_auc_score
)
import optuna
from ctgan import CTGAN

RANDOM_STATE = 42
TARGET = '离职行为'
DATA_PATH = 'data/处理之后的离职数据-5000.xlsx'
CTGAN_EPOCHS = 100
OPTUNA_TRIALS = 25

np.random.seed(RANDOM_STATE)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def section(t):
    print(f'\n{"=" * 62}\n{t}\n{"=" * 62}')


def eval_all(y_true, prob, t_opt=None):
    """Evaluate at best F1 threshold (CV-style, here just on the set)."""
    if t_opt is None:
        best_f1, t_opt = 0, 0.5
        for tt in np.arange(0.1, 0.9, 0.01):
            p = (prob >= tt).astype(int)
            f = f1_score(y_true, p, pos_label=1, zero_division=0)
            if f > best_f1:
                best_f1, t_opt = f, tt
    pred = (prob >= t_opt).astype(int)
    return {
        't': t_opt,
        'plain_acc': accuracy_score(y_true, pred),
        'bal_acc': balanced_accuracy_score(y_true, pred),
        'recall_0': recall_score(y_true, pred, pos_label=0),
        'recall_1': recall_score(y_true, pred, pos_label=1),
        'precision_1': precision_score(y_true, pred, pos_label=1, zero_division=0),
        'f1_1': f1_score(y_true, pred, pos_label=1),
        'auc': roc_auc_score(y_true, prob),
    }


def tune_rf(X_tr, y_tr, n_trials=OPTUNA_TRIALS):
    """Bayesian-optimised RandomForest on given train set, CV AUC objective."""
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def obj(trial):
        params = dict(
            n_estimators=trial.suggest_int('n_estimators', 100, 600),
            max_depth=trial.suggest_int('max_depth', 4, 20),
            min_samples_split=trial.suggest_int('min_samples_split', 2, 20),
            min_samples_leaf=trial.suggest_int('min_samples_leaf', 1, 10),
            max_features=trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.5, 0.7]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        aucs = []
        for tr_idx, va_idx in skf.split(X_tr, y_tr):
            m = RandomForestClassifier(**params)
            m.fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            p = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
            aucs.append(roc_auc_score(y_tr.iloc[va_idx], p))
        return np.mean(aucs)

    study = optuna.create_study(direction='maximize')
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    best = RandomForestClassifier(**study.best_params, random_state=RANDOM_STATE, n_jobs=-1)
    best.fit(X_tr, y_tr)
    return best, study.best_value


def ctgan_balance(X, y, discrete_cols, target_n_minority, epochs=CTGAN_EPOCHS):
    """Fit CTGAN on (X[y==1], y==1) rows and sample synthetic minority.

    Returns combined X_balanced, y_balanced. Also returns n synth generated.
    """
    minority = X[y == 1].copy()
    minority[TARGET] = 1
    t0 = time.time()
    ctgan = CTGAN(epochs=epochs, verbose=False)
    ctgan.fit(minority, discrete_columns=discrete_cols + [TARGET])
    n_need = max(0, target_n_minority - (y == 1).sum())
    synth = ctgan.sample(n_need)
    # Clip/round integer-coded columns to valid ranges (CTGAN may produce floats)
    for c in discrete_cols:
        synth[c] = synth[c].round().astype(int)
    synth[TARGET] = 1
    X_synth = synth[X.columns]
    y_synth = synth[TARGET].astype(int)
    X_out = pd.concat([X, X_synth], axis=0, ignore_index=True)
    y_out = pd.concat([y, y_synth], axis=0, ignore_index=True)
    print(f'  CTGAN: {epochs} epochs in {time.time()-t0:.1f}s, '
          f'generated {n_need} synthetic minority')
    return X_out, y_out


t_start = time.time()

# ─── Load ─────────────────────────────────────────────────────────────────────
section('LOAD')
df = pd.read_excel(DATA_PATH)
print(f'Shape: {df.shape}, class counts: {dict(df[TARGET].value_counts())}')
feature_cols = [c for c in df.columns if c != TARGET]

# Discrete columns per docs/variable-labels.md (integer-coded categoricals).
# 收入水平 is continuous; 离职意向 is 1-5 ordinal (we'll treat as discrete for CTGAN).
discrete_cols = ['性别', '高校类型', '专业类型', '家庭所在地', '工作单位性质',
                 '工作区域', '工作压力', '离职意向', '工作氛围',
                 # 工作匹配度/满意度/机会 may be continuous avg of sub-scales;
                 # keeping them out of discrete is safer for CTGAN.
                 ]

X = df[feature_cols].copy()
y = df[TARGET].astype(int).copy()

# ─── PROTOCOL A: Clean (train-only CTGAN, pristine test) ────────────────────
section('PROTOCOL A — CLEAN: CTGAN on train-minority only, pristine test')
X_tr_A, X_te_A, y_tr_A, y_te_A = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f'Train: {len(X_tr_A)}  ({(y_tr_A==1).sum()} pos)')
print(f'Test:  {len(X_te_A)}  ({(y_te_A==1).sum()} pos) [PRISTINE, no synthetic]')

n_majority_A = (y_tr_A == 0).sum()
X_tr_A_bal, y_tr_A_bal = ctgan_balance(X_tr_A, y_tr_A, discrete_cols,
                                        target_n_minority=n_majority_A)
print(f'Balanced train: {len(X_tr_A_bal)}  ({(y_tr_A_bal==1).sum()} pos)')

print('  tuning RF ...')
rf_A, cv_auc_A = tune_rf(X_tr_A_bal, y_tr_A_bal)
print(f'  train CV AUC: {cv_auc_A:.4f}')

prob_te_A = rf_A.predict_proba(X_te_A)[:, 1]
res_A_f1opt = eval_all(y_te_A, prob_te_A)
res_A_t05 = eval_all(y_te_A, prob_te_A, t_opt=0.5)

# ─── PROTOCOL B: Leaky (oversample BEFORE split; test has synthetic) ────────
section('PROTOCOL B — LEAKY: CTGAN on full minority, then split '
        '(test set contaminated with synthetic)')
# Fit CTGAN on ALL minority, oversample globally, then split.
X_all_bal, y_all_bal = ctgan_balance(X, y, discrete_cols,
                                      target_n_minority=(y == 0).sum())
X_tr_B, X_te_B, y_tr_B, y_te_B = train_test_split(
    X_all_bal, y_all_bal, test_size=0.2, stratify=y_all_bal, random_state=RANDOM_STATE
)
print(f'Train: {len(X_tr_B)}  ({(y_tr_B==1).sum()} pos)')
print(f'Test:  {len(X_te_B)}  ({(y_te_B==1).sum()} pos) [contains synthetic]')

print('  tuning RF ...')
rf_B, cv_auc_B = tune_rf(X_tr_B, y_tr_B)
print(f'  train CV AUC: {cv_auc_B:.4f}')

prob_te_B = rf_B.predict_proba(X_te_B)[:, 1]
res_B_f1opt = eval_all(y_te_B, prob_te_B)
res_B_t05 = eval_all(y_te_B, prob_te_B, t_opt=0.5)

# ─── Bonus: on Protocol B's model, evaluate against the PRISTINE test ──────
# i.e. we trained on dirty data but evaluate on the honest test split.
# This tells us whether B's "improvement" is real or a mirage of dirty test.
prob_te_B_on_A = rf_B.predict_proba(X_te_A)[:, 1]
res_B_on_A = eval_all(y_te_A, prob_te_B_on_A)

# ─── Report ──────────────────────────────────────────────────────────────────
section('SIDE-BY-SIDE (paper targets: plain~85 / bal 0.786 / rec_0 0.892 / '
        'rec_1 0.681 / AUC 0.69)')

rows = {
    'Paper BORF': dict(t=None, plain_acc=0.851, bal_acc=0.786, recall_0=0.892,
                       recall_1=0.681, precision_1=0.352, f1_1=0.46, auc=0.69),
    'A clean, F1-opt thresh':      res_A_f1opt,
    'A clean, t=0.5':              res_A_t05,
    'B LEAKY, F1-opt (dirty test)': res_B_f1opt,
    'B LEAKY, t=0.5 (dirty test)': res_B_t05,
    'B model on CLEAN test':       res_B_on_A,
}
cols = ['t', 'plain_acc', 'bal_acc', 'recall_0', 'recall_1', 'precision_1', 'f1_1', 'auc']
out = pd.DataFrame({k: {c: v.get(c) for c in cols} for k, v in rows.items()}).T
out = out[cols]
print(out.to_string(float_format=lambda x: f'{x:.3f}' if isinstance(x, float) else str(x)))

print(f'\nTotal elapsed: {time.time() - t_start:.1f}s')

# ─── Verdict ────────────────────────────────────────────────────────────────
section('READ THE TABLE')
print('If protocol B (LEAKY) matches paper numbers but protocol A (CLEAN) does not,')
print('the paper\'s reported bal_acc / recall_0 / recall_1 are an artifact of CTGAN')
print('leakage into the test set — NOT a model-quality win over what we have.')
print()
print('If BOTH A and B match paper numbers, then CTGAN really does buy something')
print('genuine, and we should adopt it for our own reporting.')
print()
print('If NEITHER matches, there is a third factor (sample size, split protocol,')
print('different feature construction) we haven\'t accounted for.')
