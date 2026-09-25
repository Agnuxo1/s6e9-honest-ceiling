"""Fold-safe GBDT legs.

Per outer fold: nested target encoding on outer-train, early stopping on a stratified 10 % slice of outer-train
(never on the scored fold), prediction of the outer-valid fold and of test. One checkpoint per fold.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from .data import CATS, N_OUTER, outer_folds
from .features import DEFAULT_DROP_KEYS, fold_matrices

LGBM_PARAMS = dict(objective="binary", metric="auc", learning_rate=0.02, num_leaves=32, max_depth=6,
                   min_data_in_leaf=20, feature_fraction=0.35, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l1=0.07, lambda_l2=2.0, max_bin=255, cat_smooth=20, verbosity=-1)
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="auc", tree_method="hist", learning_rate=0.02,
                  max_depth=6, min_child_weight=10, subsample=0.8, colsample_bytree=0.25, reg_lambda=2.0,
                  reg_alpha=0.07, max_bin=256, max_cat_to_onehot=8)


def _cuda_available() -> bool:
    try:
        import xgboost as xgb
        xgb.train({"device": "cuda", "tree_method": "hist"}, xgb.DMatrix(np.zeros((4, 1)), label=[0, 1, 0, 1]), 1)
        return True
    except Exception:
        return False


def fit_predict(model: str, seed: int, Xtr, ytr, Xes, yes, Xva, Xte, threads: int = -1, device: str | None = None,
                max_rounds: int = 6000, early_stopping: int = 150):
    if model == "lgbm":
        import lightgbm as lgb
        params = dict(LGBM_PARAMS, seed=seed, num_threads=threads)
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=CATS, free_raw_data=False)
        des = lgb.Dataset(Xes, yes, reference=dtr, categorical_feature=CATS)
        m = lgb.train(params, dtr, max_rounds, valid_sets=[des], callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        it = m.best_iteration
        return m.predict(Xva, num_iteration=it), m.predict(Xte, num_iteration=it), it
    if model == "xgb":
        import xgboost as xgb
        device = device or ("cuda" if _cuda_available() else "cpu")
        params = dict(XGB_PARAMS, seed=seed, device=device)
        if threads > 0:
            params["nthread"] = threads
        dtr = xgb.DMatrix(Xtr, ytr, enable_categorical=True)
        des = xgb.DMatrix(Xes, yes, enable_categorical=True)
        m = xgb.train(params, dtr, max_rounds, evals=[(des, "es")], early_stopping_rounds=early_stopping, verbose_eval=False)
        it = m.best_iteration + 1
        return (m.predict(xgb.DMatrix(Xva, enable_categorical=True), iteration_range=(0, it)),
                m.predict(xgb.DMatrix(Xte, enable_categorical=True), iteration_range=(0, it)), it)
    raise ValueError(f"unknown model {model!r}")


def train_leg(X, K, y, n_train: int, n_test: int, out_dir: str | Path, model: str = "xgb", seed: int = 42,
              outer_seed: int = 42, folds: list[int] | None = None, threads: int = -1, drop_keys=DEFAULT_DROP_KEYS,
              verbose: bool = True) -> dict | None:
    """Train one leg over the outer split; returns the report once all folds exist."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fold = outer_folds(y, N_OUTER, outer_seed)
    (out / "manifest.json").write_text(json.dumps(dict(model=model, seed=seed, outer_seed=outer_seed, n_outer=N_OUTER,
                                                       drop_keys=list(drop_keys)), indent=2))
    t0 = time.time()
    for k in (folds if folds is not None else range(N_OUTER)):
        ck = out / f"fold_{k}.npz"
        if ck.exists():
            continue
        tf = time.time()
        tr_idx, va_idx, Xtr, Xva, Xte = fold_matrices(X, K, y, fold, k, n_train, seed=seed * 1000 + k, drop_keys=drop_keys)
        ytr = y[tr_idx]
        es_fit, es_hold = next(StratifiedShuffleSplit(1, test_size=0.1, random_state=seed + k).split(np.zeros(len(ytr)), ytr))
        p_va, p_te, it = fit_predict(model, seed, Xtr.iloc[es_fit], ytr[es_fit], Xtr.iloc[es_hold], ytr[es_hold], Xva, Xte, threads)
        auc = float(roc_auc_score(y[va_idx], p_va))
        np.savez(ck, va_idx=va_idx, p_va=p_va.astype(np.float32), p_te=p_te.astype(np.float32), auc=auc, best_iter=it)
        if verbose:
            print(json.dumps(dict(fold=k, auc=round(auc, 6), best_iter=int(it), sec=round(time.time() - tf, 1))), flush=True)
    if not all((out / f"fold_{k}.npz").exists() for k in range(N_OUTER)):
        return None
    oof = np.zeros(n_train, np.float32)
    te = np.zeros(n_test, np.float32)
    aucs = []
    for k in range(N_OUTER):
        z = np.load(out / f"fold_{k}.npz")
        oof[z["va_idx"]] = z["p_va"]
        te += z["p_te"] / N_OUTER
        aucs.append(float(z["auc"]))
    np.save(out / "oof.npy", oof)
    np.save(out / "test.npy", te)
    rep = dict(model=model, seed=seed, outer_seed=outer_seed, oof_auc=float(roc_auc_score(y, oof)), fold_aucs=aucs,
               elapsed_sec=round(time.time() - t0, 1))
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    return rep
