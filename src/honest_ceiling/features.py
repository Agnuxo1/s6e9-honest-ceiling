"""Features for S6E9.

Two kinds of columns:

* **Unsupervised** (`build_base`): the generator's purchase recipe and range-anxiety score (reconstructed by
  Chris Deotte), income digits / remainders, frequencies over train+test, and lookups into the CC0 original
  10k-row table (its own labels are external data; the table is never used as extra training rows).
* **Supervised** (`nested_target_encode`, `nested_window_encode`): target statistics that are always
  *cross-fitted inside the rows they are given*. Called with the outer-train partition of a fold, a training
  row never sees its own label, and the outer-validation fold is never seen at all.

Integer keys are built once for train+test so encoding is a `bincount`, not a `groupby`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from .data import CATS, NUMS, RECIPE, TARGET

# key -> smoothing priors (m in (sum + m*prior) / (count + m))
TE_SPEC: dict[str, list[float]] = {
    "k_income_exact": [1.0, 10.0, 100.0],
    "k_income_nearest_orig": [1.0, 10.0],
    "k_income_100": [10.0],
    "k_income_1000": [10.0],
    "k_commute_exact": [1.0, 10.0],
    "k_commute_int": [10.0],
    "k_income_cell": [1.0, 10.0],
    "k_income1000_cell": [10.0],
    "k_income_subsidy": [10.0],
    "k_switch": [10.0],
    "k_inter4": [10.0],
    "k_inter_charge_city": [10.0],
}
# Ablation result (3 folds, +0.00012 AUC per fold when removed): the exact-income x recipe-cell encodings
# overfit. The final legs drop them.
DEFAULT_DROP_KEYS = ("k_income_cell", "k_income1000_cell")


def recipe_score(df: pd.DataFrame) -> np.ndarray:
    """Purchase score of the original generator (Deotte): buy iff score + noise > 5.5."""
    return (1.2 * df["Annual_Income_USD"].to_numpy(float) / 1e5
            + 0.6 * df["Environmental_Concern_Level"].to_numpy(float)
            + 2.0 * (df["Subsidy_Available"].to_numpy() == "Yes")
            - 1.0 * (df["Range_Anxiety_Level"].to_numpy() == "Medium")
            - 3.0 * (df["Range_Anxiety_Level"].to_numpy() == "High"))


def worry_score(df: pd.DataFrame) -> np.ndarray:
    """Range-anxiety score of the original generator: Low <= -25 < Medium <= 75 < High (plus noise)."""
    return (df["Daily_Commute_km"].to_numpy(float)
            - 5.0 * df["Charging_Stations_Near_Home"].to_numpy(float)
            - 5.0 * df["Charging_Stations_Near_Work"].to_numpy(float)
            - 150.0 * (df["Home_Charging_Possible"].to_numpy() == "Yes"))


def _original_tables(original_csv: str | Path | None):
    if original_csv is None or not Path(original_csv).exists():
        return None
    o = pd.read_csv(original_csv)
    o["y"] = (o[TARGET] == "Yes").astype(float)
    ok = o[["Annual_Income_USD", "Environmental_Concern_Level"]].notna().all(axis=1)
    o["recipe"] = np.where(ok, recipe_score(o.fillna({"Annual_Income_USD": 0, "Environmental_Concern_Level": 0})), np.nan)
    inc = o.dropna(subset=["Annual_Income_USD"]).copy()
    inc["Annual_Income_USD"] = inc["Annual_Income_USD"].round().astype(np.int64)
    g_inc = inc.groupby("Annual_Income_USD")["y"].agg(["mean", "size"])
    uniq = inc[~inc["Annual_Income_USD"].duplicated(keep=False)].set_index("Annual_Income_USD")
    com = o.dropna(subset=["Daily_Commute_km"]).copy()
    com["k"] = (com["Daily_Commute_km"] * 10).round().astype(np.int64)
    g_com = com.groupby("k")["y"].agg(["mean", "size"])
    support = np.sort(inc["Annual_Income_USD"].unique())
    return g_inc, uniq, g_com, support


def build_base(train: pd.DataFrame, test: pd.DataFrame, original_csv: str | Path | None = None,
               cache: str | Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unsupervised features X and integer keys K for train+test (train rows first). Reads no competition label."""
    if cache is not None and Path(cache).exists():
        full = pd.read_parquet(cache)
        kc = [c for c in full.columns if c.startswith("k_")]
        return full.drop(columns=kc), full[kc]
    df = pd.concat([train.drop(columns=[TARGET]), test], ignore_index=True)
    income = df["Annual_Income_USD"].round().astype(np.int64)
    commute = df["Daily_Commute_km"].astype(float)
    c10 = (commute * 10).round().astype(np.int64)

    X = pd.DataFrame(index=df.index)
    for c in NUMS:
        X[c] = df[c].astype(np.float32)
    for c in CATS:
        X[c] = pd.Categorical(df[c].astype(str))
    X["recipe_score"] = recipe_score(df).astype(np.float32)
    X["worry_score"] = worry_score(df).astype(np.float32)
    X["recipe_no_income"] = (X["recipe_score"] - 1.2 * X["Annual_Income_USD"] / 1e5).astype(np.float32)
    X["cars_x_income"] = (X["Number_of_Cars_Owned"] * X["Annual_Income_USD"] / 1e5).astype(np.float32)
    for p in (1, 10, 100, 1000):
        X[f"income_digit_{p}"] = ((income // p) % 10).astype(np.int8)
    for m in (10, 100, 1000):
        X[f"income_mod_{m}"] = (income % m).astype(np.int16)
        X[f"income_round_{m}"] = (income % m == 0).astype(np.int8)
    X["commute_tenth"] = (c10 % 10).astype(np.int8)
    X["commute_integer"] = np.floor(commute).astype(np.float32)
    X["commute_fraction"] = (commute - np.floor(commute)).astype(np.float32)
    X["cnt_income"] = income.map(income.value_counts()).astype(np.float32)
    X["cnt_income_1000"] = (income // 1000).map((income // 1000).value_counts()).astype(np.float32)
    X["cnt_commute"] = c10.map(c10.value_counts()).astype(np.float32)

    tables = _original_tables(original_csv)
    if tables is not None:
        g_inc, uniq, g_com, support = tables
        X["orig_y_income"] = income.map(g_inc["mean"]).astype(np.float32)
        X["orig_n_income"] = income.map(g_inc["size"]).fillna(0).astype(np.float32)
        X["orig_y_commute"] = c10.map(g_com["mean"]).astype(np.float32)
        X["orig_n_commute"] = c10.map(g_com["size"]).fillna(0).astype(np.float32)
        X["income_is_original"] = income.isin(support).astype(np.int8)
        pos = np.searchsorted(support, income.to_numpy())
        lo = support[np.clip(pos - 1, 0, len(support) - 1)]
        hi = support[np.clip(pos, 0, len(support) - 1)]
        nearest = np.where(np.abs(income.to_numpy() - lo) <= np.abs(hi - income.to_numpy()), lo, hi)
        X["nearest_orig_dist"] = (income.to_numpy() - nearest).astype(np.float32)
        Or = uniq.reindex(income.to_numpy())
        same = sum((df[c].to_numpy() == Or[c].to_numpy()).astype(np.int8) for c in RECIPE).astype(np.float32)
        olab = Or["y"].to_numpy(np.float32)
        X["orig_same_recipe"] = np.where(np.isnan(olab), np.nan, same).astype(np.float32)
        X["orig_label_same_cell"] = np.where(same == 3, olab, np.nan).astype(np.float32)
        X["orig_label_x_same"] = ((olab - 0.5) * same).astype(np.float32)
        X["orig_recipe_score"] = Or["recipe"].to_numpy(np.float32)
    else:
        nearest = income.to_numpy()

    cell = (df["Environmental_Concern_Level"].astype(str) + "|" + df["Subsidy_Available"].astype(str) + "|"
            + df["Range_Anxiety_Level"].astype(str))
    K = pd.DataFrame(index=df.index)
    K["k_income_exact"] = income.astype(str)
    K["k_income_nearest_orig"] = pd.Series(nearest, index=df.index).astype(str)
    K["k_income_100"] = (income // 100).astype(str)
    K["k_income_1000"] = (income // 1000).astype(str)
    K["k_commute_exact"] = c10.astype(str)
    K["k_commute_int"] = np.floor(commute).astype(int).astype(str)
    K["k_income_cell"] = K["k_income_exact"] + "|" + cell
    K["k_income1000_cell"] = K["k_income_1000"] + "|" + cell
    K["k_income_subsidy"] = K["k_income_exact"] + "|" + df["Subsidy_Available"].astype(str)
    K["k_switch"] = df["Subsidy_Available"].astype(str) + "|" + df["Range_Anxiety_Level"].astype(str)
    K["k_inter4"] = (df["Charging_Stations_Near_Home"].astype(str) + "|" + df["City_Type"].astype(str) + "|"
                     + df["Home_Charging_Possible"].astype(str) + "|" + df["Range_Anxiety_Level"].astype(str))
    K["k_inter_charge_city"] = (df["Charging_Stations_Near_Home"].astype(str) + "|"
                                + df["Charging_Stations_Near_Work"].astype(str) + "|" + df["City_Type"].astype(str))
    for c in K.columns:
        K[c] = K[c].astype("category").cat.codes.astype(np.int32)
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        pd.concat([X, K], axis=1).to_parquet(cache)
    return X, K


def _stats(keys: np.ndarray, y: np.ndarray, n_keys: int):
    return (np.bincount(keys, weights=y, minlength=n_keys),
            np.bincount(keys, minlength=n_keys).astype(np.float64))


def nested_target_encode(K_fit: pd.DataFrame, y_fit: np.ndarray, K_apply: list[pd.DataFrame], seed: int,
                         spec: dict[str, list[float]] | None = None, n_inner: int = 5):
    """Cross-fitted target encoding.

    Rows of `K_fit` are encoded from the *other* inner folds only; every frame in `K_apply` is encoded from all
    of `K_fit`. Also returns the key count in the fit partition (a leak-free frequency).
    """
    spec = TE_SPEC if spec is None else spec
    prior = float(y_fit.mean())
    enc_fit: dict[str, np.ndarray] = {}
    enc_apply: list[dict[str, np.ndarray]] = [dict() for _ in K_apply]
    inner = list(KFold(n_inner, shuffle=True, random_state=seed).split(np.zeros(len(y_fit))))
    for col, alphas in spec.items():
        kf = K_fit[col].to_numpy()
        n_keys = int(max(kf.max(), *(int(k[col].max()) for k in K_apply))) + 1
        s_all, c_all = _stats(kf, y_fit, n_keys)
        for a in alphas:
            out = np.empty(len(kf), dtype=np.float32)
            for fit_i, hold_i in inner:
                s, c = _stats(kf[fit_i], y_fit[fit_i], n_keys)
                p = float(y_fit[fit_i].mean())
                out[hold_i] = ((s + a * p) / (c + a))[kf[hold_i]]
            name = f"te_{col[2:]}_a{a:g}"
            enc_fit[name] = out
            m_all = (s_all + a * prior) / (c_all + a)
            for j, k in enumerate(K_apply):
                enc_apply[j][name] = m_all[k[col].to_numpy()].astype(np.float32)
        name = f"cnt_fit_{col[2:]}"
        enc_fit[name] = c_all[kf].astype(np.float32)
        for j, k in enumerate(K_apply):
            enc_apply[j][name] = c_all[k[col].to_numpy()].astype(np.float32)
    return pd.DataFrame(enc_fit), [pd.DataFrame(e) for e in enc_apply]


def fold_matrices(X: pd.DataFrame, K: pd.DataFrame, y: np.ndarray, fold: np.ndarray, k: int, n_train: int,
                  seed: int, drop_keys=DEFAULT_DROP_KEYS):
    """Feature matrices (train, valid, test) for outer fold k with every supervised statistic nested."""
    tr_idx = np.where(fold != k)[0]
    va_idx = np.where(fold == k)[0]
    spec = {c: v for c, v in TE_SPEC.items() if c not in set(drop_keys)}
    enc_tr, (enc_va, enc_te) = nested_target_encode(
        K.iloc[tr_idx].reset_index(drop=True), y[tr_idx],
        [K.iloc[va_idx].reset_index(drop=True), K.iloc[n_train:].reset_index(drop=True)], seed=seed, spec=spec)
    Xtr = pd.concat([X.iloc[tr_idx].reset_index(drop=True), enc_tr], axis=1)
    Xva = pd.concat([X.iloc[va_idx].reset_index(drop=True), enc_va], axis=1)
    Xte = pd.concat([X.iloc[n_train:].reset_index(drop=True), enc_te], axis=1)
    return tr_idx, va_idx, Xtr, Xva, Xte
