"""Quick diagnostic: how much AUC does 离职意向 carry on the 5771 dataset?"""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import xgboost as xgb

RS = 42
df = pd.read_excel('data/离职数据-5771.xlsx')
print('shape:', df.shape, 'class:', dict(df['离职行为'].value_counts()))

y = df['离职行为'].astype(int)
feat_with = [c for c in df.columns if c != '离职行为']
feat_no_intent = [c for c in feat_with if c != '离职意向']

def run(cols, name):
    X = df[cols]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RS)
    pos = (ytr == 0).sum() / (ytr == 1).sum()
    skf = StratifiedKFold(5, shuffle=True, random_state=RS)
    m = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                         scale_pos_weight=pos, random_state=RS, n_jobs=-1,
                         eval_metric='auc', verbosity=0)
    cvp = cross_val_predict(m, Xtr, ytr, cv=skf, method='predict_proba', n_jobs=-1)[:, 1]
    cv_auc = roc_auc_score(ytr, cvp)
    m.fit(Xtr, ytr)
    te = roc_auc_score(yte, m.predict_proba(Xte)[:, 1])
    print(f'  {name:38s}  CV AUC={cv_auc:.4f}  Test AUC={te:.4f}')

print()
print('XGB (default untuned) on 5771:')
run(feat_with, 'WITH 离职意向 (13 feats)')
run(feat_no_intent, 'WITHOUT 离职意向 (12 feats)')

Xtr, Xte, ytr, yte = train_test_split(df[['离职意向']], y, test_size=0.2, stratify=y, random_state=RS)
lr = LogisticRegression().fit(Xtr, ytr)
intent_only = roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])
print(f'  {"Intent-only LR (1 feat)":38s}                    Test AUC={intent_only:.4f}')
