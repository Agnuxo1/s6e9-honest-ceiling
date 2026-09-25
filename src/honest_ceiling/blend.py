"""Rank-space blending with an honest (nested) estimate of its own score.

`nested_hillclimb` fits greedy forward-selection weights on 9 outer folds and scores the 10th, for every fold:
the reported AUC is out-of-fold *for the blend weights too*. The final weights are refit on all rows.
`paired_bootstrap` subsamples rows at a given size (e.g. the 229k-row private split) and reports the paired
difference between two score vectors: the quantity that decides who ranks above whom.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def rank01(a: np.ndarray) -> np.ndarray:
    return rankdata(a) / len(a)


def fast_auc(y: np.ndarray, s: np.ndarray) -> float:
    """Exact ROC-AUC via the Mann-Whitney statistic (ties get average ranks)."""
    r = rankdata(s)
    n_pos = float(y.sum())
    n_neg = len(y) - n_pos
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def hillclimb(O: np.ndarray, y: np.ndarray, rows: np.ndarray | None = None, n_iter: int = 60, step: float = 0.05,
              tol: float = 1e-7) -> tuple[np.ndarray, float]:
    """Greedy forward selection with replacement on rank-space streams O (n_rows, n_streams)."""
    rows = np.arange(len(y)) if rows is None else rows
    Or, yr = O[rows], y[rows]
    m = O.shape[1]
    aucs = [fast_auc(yr, Or[:, j]) for j in range(m)]
    j = int(np.argmax(aucs))
    w = np.zeros(m)
    w[j] = 1.0
    cur = Or[:, j].copy()
    best = aucs[j]
    for _ in range(n_iter):
        a, j = max((fast_auc(yr, cur + step * Or[:, j]), j) for j in range(m))
        if a <= best + tol:
            break
        best, cur = a, cur + step * Or[:, j]
        w[j] += step
    return w / w.sum(), best


def nested_hillclimb(O: np.ndarray, y: np.ndarray, fold: np.ndarray, **kw) -> dict:
    """Leave-one-fold-out hill-climb. Returns nested OOF scores, per-fold AUCs and the full-data weights."""
    nested = np.zeros(len(y))
    fold_w = []
    for k in np.unique(fold):
        w, _ = hillclimb(O, y, np.where(fold != k)[0], **kw)
        nested[fold == k] = O[fold == k] @ w
        fold_w.append(w)
    w_full, in_sample = hillclimb(O, y, None, **kw)
    per_fold = [fast_auc(y[fold == k], nested[fold == k]) for k in np.unique(fold)]
    return dict(nested_scores=nested, nested_auc=fast_auc(y, nested), fold_aucs=per_fold, weights=w_full,
                in_sample_auc=in_sample, fold_weights=np.array(fold_w))


def paired_bootstrap(y: np.ndarray, a: np.ndarray, b: np.ndarray, n: int, reps: int = 60, seed: int = 0) -> dict:
    """AUC(a) - AUC(b) on `reps` row subsamples of size n (without replacement)."""
    rng = np.random.default_rng(seed)
    d = np.empty(reps)
    for i in range(reps):
        idx = rng.choice(len(y), n, replace=False)
        d[i] = fast_auc(y[idx], a[idx]) - fast_auc(y[idx], b[idx])
    return dict(mean=float(d.mean()), std=float(d.std()), p_a_better=float((d > 0).mean()), n=n, reps=reps)
