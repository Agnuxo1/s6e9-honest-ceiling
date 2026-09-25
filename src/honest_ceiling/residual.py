"""What is left after the model: calibrated residual, whiteness tests and the oracle bound.

If the residual r = y - p(x) is white (uncorrelated between rows in every ordering and inside every key group),
no filter, reordering or feature built from x can recover it: the label is p(x) plus an independent coin flip.
The oracle bound answers "how high could AUC go if our probabilities were exactly right?": draw y' ~ Bernoulli(p)
and score p against y'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from .blend import fast_auc


def calibrated_probability(scores: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Two-way cross-fitted isotonic calibration of any score vector into probabilities."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    h = len(y) // 2
    p = np.zeros(len(y))
    for a, b in ((idx[:h], idx[h:]), (idx[h:], idx[:h])):
        p[b] = IsotonicRegression(out_of_bounds="clip").fit(scores[a], y[a]).predict(scores[b])
    return np.clip(p, 1e-4, 1 - 1e-4)


def acf(x: np.ndarray, lags=(1, 2, 3, 5, 10, 50)) -> dict[int, float]:
    x = x - x.mean()
    v = float((x * x).mean())
    return {int(l): float((x[:-l] * x[l:]).mean() / v) for l in lags}


def whiteness_report(r: np.ndarray, orderings: dict[str, np.ndarray], groups: dict[str, np.ndarray]) -> dict:
    """ACF of the residual under each ordering, and leave-one-out correlation inside each key grouping."""
    out = {"acf": {name: acf(r[np.argsort(v, kind="stable")]) for name, v in orderings.items()}, "group_loo_corr": {}}
    for name, key in groups.items():
        df = pd.DataFrame({"k": key, "r": r})
        g = df.groupby("k")["r"]
        s, n = g.transform("sum"), g.transform("size")
        m = n > 1
        loo = (s[m] - df.r[m]) / (n[m] - 1)
        out["group_loo_corr"][name] = float(np.corrcoef(df.r[m], loo)[0, 1])
    return out


def oracle_bound(p: np.ndarray, reps: int = 5, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    v = [fast_auc(rng.binomial(1, p), p) for _ in range(reps)]
    return dict(mean=float(np.mean(v)), std=float(np.std(v)), reps=reps)
