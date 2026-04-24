"""Shared statistical helpers.

Currently exposes:
  - _compute_midrank : midranks with ties averaged (1-based)
  - _fast_delong     : Sun & Xu 2014 fast DeLong covariance
  - delong_test      : paired two-sided AUC comparison
"""

import numpy as np
import scipy.stats as sp_stats


def _compute_midrank(x):
    """Midranks (average rank for ties), 1-based."""
    N = len(x)
    J = np.argsort(x)
    Z = x[J]
    T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N)
    T2[J] = T + 1
    return T2


def _fast_delong(scores, y_true):
    """Sun & Xu 2014 fast DeLong.

    scores: (k, n) — k predictors, n samples
    y_true: (n,) 0/1
    Returns aucs (k,) and delong covariance (k,k).
    """
    y_true = np.asarray(y_true).astype(int)
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    m = pos_mask.sum()
    n_neg = neg_mask.sum()
    if scores.ndim == 1:
        scores = scores[None, :]
    k = scores.shape[0]

    pos_scores = scores[:, pos_mask]
    neg_scores = scores[:, neg_mask]
    all_scores = np.concatenate([pos_scores, neg_scores], axis=1)

    tx = np.empty((k, m)); ty = np.empty((k, n_neg)); tz = np.empty((k, m + n_neg))
    for r in range(k):
        tx[r] = _compute_midrank(pos_scores[r])
        ty[r] = _compute_midrank(neg_scores[r])
        tz[r] = _compute_midrank(all_scores[r])

    aucs = (tz[:, :m].sum(axis=1) / (m * n_neg)) - (m + 1.0) / (2.0 * n_neg)
    v01 = (tz[:, :m] - tx) / n_neg
    v10 = 1.0 - (tz[:, m:] - ty) / m
    if k == 1:
        sx = np.array([[float(np.var(v01, ddof=1))]])
        sy = np.array([[float(np.var(v10, ddof=1))]])
    else:
        sx = np.cov(v01)
        sy = np.cov(v10)
    cov = sx / m + sy / n_neg
    return aucs, cov


def delong_test(y_true, score_a, score_b):
    """Compare two AUCs on the same test set. Returns (auc_a, auc_b, z, p)."""
    scores = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    aucs, cov = _fast_delong(scores, y_true)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return aucs[0], aucs[1], 0.0, 1.0
    z = diff / np.sqrt(var)
    p = 2 * (1 - sp_stats.norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p
