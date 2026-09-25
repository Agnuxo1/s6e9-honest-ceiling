from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from honest_ceiling.blend import fast_auc, hillclimb, nested_hillclimb, paired_bootstrap, rank01
from honest_ceiling.darwin import DarwinSearch, integer_columns, plant, program_key
from honest_ceiling.data import load, outer_folds
from honest_ceiling.features import build_base, fold_matrices, nested_target_encode
from honest_ceiling.residual import calibrated_probability, oracle_bound, whiteness_report


def test_folds_are_deterministic_and_stratified(synthetic_dir):
    _, _, y = load(synthetic_dir)
    f1, f2 = outer_folds(y), outer_folds(y)
    assert (f1 == f2).all() and set(np.unique(f1)) == set(range(10))
    rates = [y[f1 == k].mean() for k in range(10)]
    assert max(rates) - min(rates) < 0.02


def test_fast_auc_matches_sklearn():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 5000)
    s = rng.normal(size=5000) + y * 0.7
    s[:200] = 0.0  # ties
    assert abs(fast_auc(y, s) - roc_auc_score(y, s)) < 1e-12


def test_nested_target_encoding_never_sees_own_label():
    """With labels independent of the key, a leaky encoding would score AUC >> 0.5 on its own fit rows."""
    rng = np.random.default_rng(2)
    n = 20_000
    K = pd.DataFrame({"k_income_exact": rng.integers(0, 4000, n).astype(np.int32)})
    y = rng.integers(0, 2, n)
    enc_fit, _ = nested_target_encode(K, y, [K.iloc[:10]], seed=0, spec={"k_income_exact": [1.0]})
    assert abs(roc_auc_score(y, enc_fit["te_income_exact_a1"]) - 0.5) < 0.02


def test_fold_matrices_and_gbdt_leg(synthetic_dir, tmp_path):
    from honest_ceiling.train import train_leg
    train, test, y = load(synthetic_dir)
    X, K = build_base(train, test)
    fold = outer_folds(y)
    tr_idx, va_idx, Xtr, Xva, Xte = fold_matrices(X, K, y, fold, 0, len(train), seed=0)
    assert set(tr_idx).isdisjoint(va_idx) and len(Xte) == len(test) and list(Xtr.columns) == list(Xva.columns)
    rep = train_leg(X, K, y, len(train), len(test), tmp_path / "leg", model="lgbm", seed=0, threads=2, verbose=False)
    assert rep is not None and rep["oof_auc"] > 0.8


def test_nested_hillclimb_beats_or_matches_best_single():
    rng = np.random.default_rng(3)
    n = 30_000
    y = rng.integers(0, 2, n)
    signal = y + rng.normal(0, 1.2, n)
    O = np.column_stack([rank01(signal + rng.normal(0, 0.8, n)) for _ in range(3)])
    fold = rng.integers(0, 10, n)
    res = nested_hillclimb(O, y, fold)
    assert abs(res["weights"].sum() - 1) < 1e-9
    assert res["nested_auc"] >= max(fast_auc(y, O[:, j]) for j in range(3)) - 1e-4
    w, _ = hillclimb(O, y)
    assert (w > 0).sum() >= 2
    pb = paired_bootstrap(y, O @ w, O[:, 0], n=10_000, reps=10)
    assert pb["mean"] > 0


def test_residual_tools_and_oracle():
    rng = np.random.default_rng(4)
    p_true = rng.uniform(0.02, 0.9, 50_000)
    y = rng.binomial(1, p_true)
    p = calibrated_probability(p_true + rng.normal(0, 0.01, len(y)), y)
    r = y - p
    rep = whiteness_report(r, {"random": rng.permutation(len(y))}, {"key": rng.integers(0, 500, len(y))})
    assert all(abs(v) < 0.02 for v in rep["acf"]["random"].values())
    ob = oracle_bound(p_true)
    assert abs(ob["mean"] - fast_auc(y, p_true)) < 0.01


def test_darwin_rejects_noise_and_finds_planted_signal(synthetic_dir):
    train, _, y = load(synthetic_dir)
    fold = outer_folds(y)
    D = integer_columns(train)
    rng = np.random.default_rng(5)
    p = np.full(len(y), y.mean())
    # pure noise: nothing may be confirmed
    y_noise = rng.binomial(1, p)
    res = DarwinSearch(D, (y_noise - p).astype(float), fold, z=4.0).run(max_evals=300, seed=1)
    assert len(res["confirmed"]) == 0 and res["familywise_fp_expected"] < 0.05
    # strong planted structure on a single atom: the planted key itself must confirm
    prog = [("mod", "Age", 7)]
    y_pl, _ = plant(p, D, prog, effect=0.15, seed=2)
    s = DarwinSearch(D, (y_pl - p).astype(float), fold, z=4.0)
    assert s.confirm(program_key(prog, D)) > s.threshold
    found = s.run(max_evals=3000, seed=3, stop_on_confirm=True)
    assert len(found["confirmed"]) >= 1
