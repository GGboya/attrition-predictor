"""
Diagnostic pass before re-tuning: figure out WHERE the signal ceiling is,
not WHICH model is 0.01 AUC better. No Optuna. One baseline, fast.

Produces:
  - Per-feature single-variable AUC (where does the signal live?)
  - Class overlap per feature (KS statistic)
  - Test-set error profile: FN / FP vs population means
  - Threshold sweep: bal_acc / F1 / precision / recall vs threshold
  - Calibration curve + Brier score
  - Hard cases: the top-K false negatives (missed resigners) as a table

Target runtime: ~60-90s on 5K samples.
"""

import warnings
warnings.filterwarnings('ignore')

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, balanced_accuracy_score, precision_score, recall_score,
    f1_score, brier_score_loss, accuracy_score
)
from sklearn.calibration import calibration_curve
import xgboost as xgb

RANDOM_STATE = 42
TARGET = '离职行为'
DATA_PATH = 'data/处理之后的离职数据-5000.xlsx'

plt.rcParams.update({
    'font.sans-serif': ['Arial Unicode MS', 'SimHei', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'figure.dpi': 110,
})


def section(title):
    print(f'\n{"=" * 60}\n{title}\n{"=" * 60}')


t0 = time.time()

# ─── Load ─────────────────────────────────────────────────────────────────────
section('1. LOAD')
df = pd.read_excel(DATA_PATH)
print(f'Shape: {df.shape}')
vc = df[TARGET].value_counts().sort_index()
print(f'Class counts: 0={vc.get(0, 0)}, 1={vc.get(1, 0)}, ratio={vc.get(0,0)/max(vc.get(1,1),1):.2f}:1')

feature_cols = [c for c in df.columns if c != TARGET]
X = df[feature_cols]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ─── 2. Per-feature single-variable AUC ──────────────────────────────────────
# If the best single feature already gives AUC 0.68 and the model tops out
# at 0.73, the features themselves don't carry much more signal — more
# tuning won't help, better features / better data will.
section('2. PER-FEATURE SINGLE-VARIABLE AUC & KS')
rows = []
for c in feature_cols:
    # Direction-agnostic AUC: max(AUC, 1-AUC)
    try:
        auc = roc_auc_score(y_train, X_train[c])
        auc = max(auc, 1 - auc)
    except Exception:
        auc = np.nan
    ks = ks_2samp(X_train.loc[y_train == 0, c], X_train.loc[y_train == 1, c]).statistic
    rows.append({'feature': c, 'single_auc': auc, 'ks': ks})
feat_power = pd.DataFrame(rows).sort_values('single_auc', ascending=False).reset_index(drop=True)
print(feat_power.to_string(index=False, float_format='{:.3f}'.format))

# ─── 3. Baseline model (one, fixed params, no tuning) ────────────────────────
section('3. BASELINE XGBOOST (fixed params)')
spw = (y_train == 0).sum() / (y_train == 1).sum()
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    min_child_weight=3,
    reg_lambda=1.0,
    scale_pos_weight=spw,
    eval_metric='logloss',
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# 5-fold CV AUC on train — this is the ceiling we can report honestly
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_prob_train = cross_val_predict(model, X_train, y_train, cv=skf, method='predict_proba')[:, 1]
cv_auc = roc_auc_score(y_train, cv_prob_train)
print(f'CV AUC (train, 5-fold): {cv_auc:.4f}')

# Fit once on full train, predict on test
model.fit(X_train, y_train)
test_prob = model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_prob)
print(f'Test AUC:               {test_auc:.4f}')
print(f'Paper BORF AUC:         0.6900')
print(f'Best single-feature AUC: {feat_power.single_auc.iloc[0]:.4f} ({feat_power.feature.iloc[0]})')

# ─── 4. Threshold sweep ──────────────────────────────────────────────────────
# Where on the curve do we actually want to operate?
section('4. THRESHOLD SWEEP (test set)')
ts = np.arange(0.05, 0.96, 0.02)
sweep = []
for t in ts:
    pred = (test_prob >= t).astype(int)
    if pred.sum() == 0:
        prec = np.nan
    else:
        prec = precision_score(y_test, pred, pos_label=1, zero_division=0)
    sweep.append({
        't': t,
        'plain_acc': accuracy_score(y_test, pred),
        'bal_acc': balanced_accuracy_score(y_test, pred),
        'precision': prec,
        'recall_0': recall_score(y_test, pred, pos_label=0),
        'recall_1': recall_score(y_test, pred, pos_label=1),
        'f1': f1_score(y_test, pred, pos_label=1),
        'pos_rate': pred.mean(),
    })
sweep_df = pd.DataFrame(sweep)

# Trivial baseline: predict all zeros
trivial_acc = 1 - y_test.mean()
print(f'Trivial "predict all 0" plain accuracy: {trivial_acc:.3f}   '
      f'(AUC would be 0.5, recall_1 = 0)')
print()
# Key operating points
print(f'{"operating point":<28} {"t":>5} {"plain_acc":>9} {"bal_acc":>8} '
      f'{"rec_0":>7} {"rec_1":>7} {"prec":>7} {"f1":>6}')
print('-' * 80)

def show(label, row):
    print(f'  {label:<26} {row.t:>5.2f} {row.plain_acc:>9.3f} '
          f'{row.bal_acc:>8.3f} {row.recall_0:>7.3f} {row.recall_1:>7.3f} '
          f'{row.precision:>7.3f} {row.f1:>6.3f}')

show('max plain_acc',   sweep_df.loc[sweep_df.plain_acc.idxmax()])
show('max bal_acc',     sweep_df.loc[sweep_df.bal_acc.idxmax()])
show('max F1(resign)',  sweep_df.loc[sweep_df.f1.idxmax()])
show('match BORF rec_1 (0.681)',
     sweep_df.iloc[(sweep_df.recall_1 - 0.681).abs().argmin()])
show('match BORF rec_0 (0.892)',
     sweep_df.iloc[(sweep_df.recall_0 - 0.892).abs().argmin()])
# The BORF claim: recall_0=0.892 AND recall_1=0.681 at the SAME threshold.
# Can we get close? Measure "can we beat BOTH" at any single threshold.
feasible = sweep_df[(sweep_df.recall_0 >= 0.892) & (sweep_df.recall_1 >= 0.681)]
if len(feasible):
    show('BORF pareto-dominated', feasible.iloc[0])
else:
    print('  (no threshold simultaneously hits BORF rec_0>=0.892 AND rec_1>=0.681 — '
          'this is where the 5K-sample gap bites)')

# ─── 5. Error profile on test set ────────────────────────────────────────────
# At the F1-optimal threshold, what do FN (missed resigners) look like?
# If they look "average" on every feature, the data just doesn't carry
# enough signal to separate them — no model will fix that.
section('5. ERROR PROFILE AT F1-OPTIMAL THRESHOLD')
best_t = sweep_df.loc[sweep_df.f1.idxmax(), 't']
pred = (test_prob >= best_t).astype(int)
tn = (pred == 0) & (y_test == 0)
fp = (pred == 1) & (y_test == 0)
fn = (pred == 0) & (y_test == 1)
tp = (pred == 1) & (y_test == 1)
print(f'Threshold: {best_t:.2f}')
print(f'TP={tp.sum()}  FP={fp.sum()}  TN={tn.sum()}  FN={fn.sum()}')

profile = pd.DataFrame({
    'all_pos(y=1)': X_test[y_test == 1].mean(),
    'TP(correctly caught)': X_test[tp.values].mean(),
    'FN(missed)': X_test[fn.values].mean(),
    'all_neg(y=0)': X_test[y_test == 0].mean(),
    'FP(false alarm)': X_test[fp.values].mean(),
})
# Show features most different between TP and FN (what makes resigners hard to spot)
profile['TP_vs_FN_diff'] = (profile['TP(correctly caught)'] - profile['FN(missed)']).abs()
profile_sorted = profile.sort_values('TP_vs_FN_diff', ascending=False)
print('\n-- Feature means by group (sorted by |TP - FN|) --')
print(profile_sorted.drop(columns='TP_vs_FN_diff').to_string(float_format='{:.2f}'.format))

# ─── 6. Calibration ──────────────────────────────────────────────────────────
section('6. CALIBRATION')
brier = brier_score_loss(y_test, test_prob)
print(f'Brier score: {brier:.4f}   (lower is better; 0.25 is random)')
print(f'Mean predicted prob (y=1): {test_prob[y_test==1].mean():.3f}')
print(f'Mean predicted prob (y=0): {test_prob[y_test==0].mean():.3f}')
print(f'Actual positive rate:      {y_test.mean():.3f}')

# ─── 7. Hard FNs ─────────────────────────────────────────────────────────────
section('7. TOP-10 HARDEST FALSE NEGATIVES')
print('(Resigners the model was most confident were NOT resigning — inspect these)')
fn_idx = np.where(fn.values)[0]
if len(fn_idx) > 0:
    hardest = sorted(fn_idx, key=lambda i: test_prob[i])[:10]
    hard_df = X_test.iloc[hardest].copy()
    hard_df['predicted_prob'] = test_prob[hardest]
    hard_df['actual'] = 1
    print(hard_df.to_string(float_format='{:.2f}'.format))

# ─── 8. Plots ────────────────────────────────────────────────────────────────
section('8. PLOTS')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) per-feature AUC bars
ax = axes[0, 0]
f = feat_power.sort_values('single_auc')
ax.barh(f.feature, f.single_auc)
ax.axvline(0.5, color='k', linestyle='--', alpha=0.4, label='random')
ax.axvline(test_auc, color='r', linestyle='--', alpha=0.6, label=f'full model ({test_auc:.3f})')
ax.set_xlabel('single-feature AUC (direction-agnostic)')
ax.set_title('Per-feature signal strength')
ax.legend()

# (b) recall_0 vs recall_1 pareto — this is where BORF vs us is obvious
ax = axes[0, 1]
ax.plot(sweep_df.recall_0, sweep_df.recall_1, '-o', markersize=3,
        alpha=0.6, label='our operating curve')
# BORF reported point
ax.plot([0.892], [0.681], 'r*', markersize=22, label='BORF (0.892, 0.681)')
# Trivial classifier endpoints
ax.plot([1.0], [0.0], 's', color='grey', label='predict-all-0 (trivial)')
ax.plot([0.0], [1.0], 's', color='grey', alpha=0.5, label='predict-all-1')
# Iso-bal_acc lines at 0.786 (BORF) and our max
for ba, color in [(0.786, 'red'), (sweep_df.bal_acc.max(), 'blue')]:
    xs = np.linspace(0, 1, 100)
    ys = 2 * ba - xs
    mask = (ys >= 0) & (ys <= 1)
    ax.plot(xs[mask], ys[mask], '--', color=color, alpha=0.35,
            label=f'bal_acc={ba:.3f}')
ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
ax.set_xlabel('recall on class 0 (non-resigned)')
ax.set_ylabel('recall on class 1 (resigned)')
ax.set_title('Operating curve: can any threshold reach BORF?')
ax.legend(loc='lower left', fontsize=7)
ax.grid(alpha=0.3)

# (c) calibration curve
ax = axes[1, 0]
frac_pos, mean_pred = calibration_curve(y_test, test_prob, n_bins=10, strategy='quantile')
ax.plot(mean_pred, frac_pos, 'o-', label=f'XGBoost (Brier={brier:.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.4, label='perfectly calibrated')
ax.set_xlabel('mean predicted probability (bin)')
ax.set_ylabel('actual positive rate (bin)')
ax.set_title('Calibration curve')
ax.legend()
ax.grid(alpha=0.3)

# (d) predicted prob distribution by true class
ax = axes[1, 1]
ax.hist(test_prob[y_test == 0], bins=40, alpha=0.6, label='y=0 (stayed)', density=True)
ax.hist(test_prob[y_test == 1], bins=40, alpha=0.6, label='y=1 (resigned)', density=True)
ax.axvline(best_t, color='r', linestyle='--', alpha=0.6, label=f'F1-opt t={best_t:.2f}')
ax.set_xlabel('predicted probability of resignation')
ax.set_ylabel('density')
ax.set_title('Score distribution overlap (= AUC geometry)')
ax.legend()

plt.tight_layout()
plt.savefig('src/diagnose.png', bbox_inches='tight')
plt.close()
print('Saved: src/diagnose.png')

# ─── 9. Verdict ──────────────────────────────────────────────────────────────
section('9. VERDICT')
best_single = feat_power.single_auc.iloc[0]
ceiling_gap = test_auc - best_single
paper_gap = 0.786 - sweep_df.bal_acc.max()
print(f'Best single-feature AUC:     {best_single:.3f}')
print(f'Full-model test AUC:         {test_auc:.3f}')
print(f'Model lift over best single: +{ceiling_gap:.3f}')
print(f'Best threshold-tuned bal_acc: {sweep_df.bal_acc.max():.3f}  vs BORF 0.786  (gap {paper_gap:+.3f})')
print(f'\nElapsed: {time.time() - t0:.1f}s')

print('\nREAD-THE-PLOT GUIDE:')
print('  - If single-feature AUC >> model AUC:  feature interactions buy nothing (rare)')
print('  - If single-feature AUC ≈ model AUC:   model saturated; signal ceiling is real')
print('  - If calibration is off at high probs:  threshold tuning / isotonic will help')
print('  - If FN means ≈ population means:      resigners are indistinguishable in these 13 vars')
print('  - If score hists overlap heavily:      AUC physically cannot be much higher')
