# Method

## 1. The data and what generates it

Kaggle Playground S6E9 has 668,665 training and 286,571 test rows, 13 features and a binary target
(`Will_Buy_EV`, 17.46 % positives). Like every Playground dataset it is synthetic: a generative model was trained on
a 10,000-row original table (CC0) and sampled. Three facts from the public discussion, all re-measured here, shape
everything that follows:

* **The purchase recipe.** The original label is `1[1.2·income/1e5 + 0.6·concern + 2·subsidy − 1·[anxiety=Medium]
  − 3·[anxiety=High] + ε > 5.5]` (reconstructed by Chris Deotte). This score alone, with no training, reaches
  ROC-AUC **0.93769** on the competition train set.
* **The income memory.** Only 13,214 distinct incomes appear across 668,665 rows; 97.9 % of them exist in the
  original table, where each value maps back to one or a few rows. The buy rate per exact income has about 2.4× the
  variance binomial sampling would give. Target-encoding the exact value is worth about +0.003 AUC.
* **No duplication.** No train row and no test row share all 13 feature values; the memory lives in the values,
  not in copies of rows.

## 2. One frozen split

Every model, every blend and every test uses `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)` over
`train.csv` sorted by `id`. This is the split of the public S6E9 OOF libraries (verified: identical fold ids for all
668,665 rows), so our out-of-fold vectors stack row for row with theirs and blend weights can be fit honestly.

## 3. Features

Unsupervised (computed once for train + test, read no label):

| Group | Columns |
|---|---|
| Raw | 7 numeric, 6 categorical |
| Generator | recipe score, range-anxiety score, recipe without income, cars × income |
| Income arithmetic | decimal digits (1, 10, 100, 1000), remainders mod 10/100/1000, round-number flags |
| Commute arithmetic | tenths digit, integer part, fractional part |
| Frequencies | counts of exact income, income//1000, exact commute over train + test |
| Original table | label mean and count at the exact income and commute, whether the income exists in the original, distance to the nearest original income, agreement of the recipe cell with the unique original row, that row's recipe score |

Supervised, always nested (`features.nested_target_encode`): smoothed target means with priors m ∈ {1, 10, 100} for
exact income, nearest original income, income//100, income//1000, exact commute, integer commute, income × subsidy,
subsidy × anxiety, a four-way charging/city/anxiety interaction and a charging/city interaction; plus the key count
inside the fit partition. For outer fold *k* the encoder receives only the outer-train rows; each of those rows is
encoded from the other four inner folds, and the outer-validation and test rows from all outer-train rows.

## 4. Legs

| Leg | Configuration | OOF AUC (per run) |
|---|---|---|
| XGBoost | depth 6, lr 0.02, colsample 0.25, hist, CUDA; early stopping on 10 % of outer-train | 0.94603-0.94607 |
| LightGBM | 32 leaves, depth 6, lr 0.02, feature fraction 0.35 | 0.94606-0.94608 |
| Logistic regression | logit-TE + one-hot + splines, L2, C = 0.5 | 0.94580 |
| MLP (PyTorch) | 512-256-128 SiLU, quantile-normal inputs | 0.94466-0.94502 |
| RealMLP (pytabkit) | defaults, 48 epochs | 0.94540 |

Each GBDT runs with three seeds (42, 101, 2026) on three outer splits (seeds 42, 7, 13): nine runs per family,
averaged in rank space. Averaging is worth about +0.00017 per family: XGBoost 0.94606 per run → **0.946234** for the
9-run average, LightGBM 0.94607 → **0.946241**. Per-run reports: [`results/legs.json`](../results/legs.json), family
averages: [`results/legs_summary.json`](../results/legs_summary.json).

## 5. Ablations (outer folds 0-2, XGBoost, mean Δ AUC over the three folds)

Reference **B** = the first run with all 74 features; reference **F** = the final configuration (B without the
exact-income × recipe-cell encodings, colsample 0.25).

| Change | Reference | Mean Δ AUC | Decision |
|---|---|---|---|
| drop exact-income × recipe-cell encodings | B | +0.00012 | adopted |
| … and colsample 0.25 (= F) | B | +0.00015 | adopted (final configuration) |
| LightGBM, drop exact-income × recipe-cell encodings | LightGBM B | +0.00014 | adopted |
| drop original-table lookups | B | −0.00001 | kept (near-neutral) |
| priors × 10 | B | −0.00009 | rejected |
| depth 8, lr 0.01 | B | −0.00005 | rejected |
| depth 5, lr 0.03 | B | +0.00002 | not adopted (within noise) |
| drop in-fold key counts | B | −0.00007 | rejected |
| centred-window target rates | B | −0.00000 | rejected |
| + income × city encoding | F | −0.00001 | rejected |
| + income × car-type / home-charging encodings | F | −0.00006 | rejected |
| + five income × column encodings | F | −0.00008 | rejected |
| no exact-income key, granularity ladder (different view) | B | −0.00034 | kept as a candidate leg; zero blend weight |

Raw fold AUCs: [`results/ablations.json`](../results/ablations.json).

## 6. Blending

Streams: the own GBDT families (and the zero-weight legs), plus the public CC0 six-view library (views A-F) and its
RealMLP-G, all on the same split, all converted to ranks. `blend.hillclimb` is greedy forward selection with
replacement (step 0.05, at most 60 steps). `blend.nested_hillclimb` fits it on 9 folds and scores the 10th for every
fold; that nested AUC is the number we report. Final weights (refit on all rows): own LightGBM 0.254, own XGBoost
0.028, public view A 0.282, RealMLP-G 0.225, view D 0.169, view F 0.028, view C 0.014, all other streams 0.

| Blend | Legs | Nested OOF AUC |
|---|---|---|
| v2 | own GBDT on one split × 3 seeds + library | 0.946397 |
| v3 | + second outer split | 0.946413 |
| v4 | + third outer split, RealMLP | **0.946420** |

## 7. Is the gain real at leaderboard scale?

A difference measured on 668,665 rows can vanish on the 229,257 private rows. `blend.paired_bootstrap` draws 60
subsamples of that size and scores both vectors on the *same* rows, which is how the private leaderboard compares
teams. v4 beats the public six-view ensemble by +0.000073 ± 0.000016 (P = 1.00) and v2 by +0.000023 ± 0.000005
(P = 1.00). At public-split size (57,314 rows) the v4-v2 difference is +0.000023 ± 0.000013 (P = 0.95): the
public leaderboard, which showed v4 0.00003 *below* v2, cannot resolve it. All four comparisons are stored in
[`results/residual_and_ceiling.json`](../results/residual_and_ceiling.json).

## 8. What is left: residual analysis

With p the two-way cross-fitted isotonic calibration of the nested blend and r = y − p:

* **Autocorrelation** of r in id, income, commute and prediction order stays within ±0.005 at lags 1-50; the ±2σ
  band of white noise is ±0.0024. The one departure, income order at small lags (−0.004 to −0.005), has the sign
  cross-fitting produces: a row's label enters the encodings of its same-income neighbours in other folds.
* **Same-key correlation.** The leave-one-out mean residual of rows with the same exact income correlates −0.032
  with a row's own residual: the income memory is fully extracted, not left behind.
* **Oracle bound.** If p were the true probability, the best achievable AUC would be 0.94660 ± 0.00028
  (10 draws of y′ ~ Bernoulli(p)). The nested blend reaches 0.94642.

## 9. Darwin-Cage search

See `src/honest_ceiling/darwin.py` for the exact algorithm.

* **Alphabet.** Atoms: a categorical column; or for a numeric column its raw integer value, `v mod m`
  (35 moduli from 2 to 10,000), `v // d` (18 divisors), a decimal digit, or `(v >> s) mod b`. Commute is scaled by
  10 to an integer. A program joins up to four atoms into one exact key; keys with more than 200,000 groups are
  rejected.
* **Fitness.** Shrunk group mean of r (prior 0, shrinkage 5) fit on 8 search folds, Pearson correlation with r on
  the 9th, averaged over the 9 rotations; the minimum over rotations is also recorded.
* **Evolution.** Population 80; elites (top 20 %) survive; offspring by mutation (45 %: add, drop or re-draw an
  atom, or re-draw a modulus), crossover (35 %) or fresh random programs (20 %).
* **Confirmation.** Programs with mean fitness > 0.006 and positive fitness on every rotation enter the hall of
  fame and are scored once on fold 9, which the search never sees. Threshold `z / sqrt(n)` with z = 3.
* **Null.** Random keys with 500 groups: fitness 0.00003 ± 0.0018.
* **Runs.** Two runs (seeds 2026 and 2027), 110 and 218 minutes, 47,679 distinct programs. Both ended with a
  `MemoryError` while other jobs shared the machine's 25 GB of RAM; the hall of fame is written every generation, so
  nothing evaluated was lost.

## 10. Power analysis

`scripts/power_analysis.py`: y′ ~ Bernoulli(p + d·s(key)) with s = ±1 drawn per group of a planted program, then the
same search on r′ = y′ − p with a 30-minute budget. For each (program, d) we record whether any program is confirmed,
and the AUC an oracle would gain by knowing the planted structure, AUC(y′, p + d·s) − AUC(y′, p). The search counts
as successful if it confirms the planted key or any finer key that contains it (for example, the exact income
contains income mod 97). Results: [`results/power/`](../results/power).

| Planted structure | Groups | Effect d | Oracle AUC gain | True program on reserved fold | Found in 30 min |
|---|---|---|---|---|---|
| A: income mod 97 | 97 | 0.002 | 0.0002 | 0.0043 (fails 0.012) | no (below resolution) |
| A: income mod 97 | 97 | 0.004 | 0.0007 | 0.0100 (fails 0.012) | no (below resolution) |
| A: income mod 97 | 97 | 0.008 | 0.0019 | 0.0197 (passes 0.012) | **yes** (235 s) |
| A: income mod 97 | 97 | 0.016 | 0.0049 | 0.0427 (passes 0.012) | **yes** (35 s) |
| B: income hundreds digit × city | 30 | 0.004 | 0.0008 | 0.0079 (fails 0.012) | no (below resolution) |
| B: income hundreds digit × city | 30 | 0.008 | 0.0021 | 0.0187 (passes 0.012) | **yes** (34 s) |
| C: commute mod 37 × age mod 7 | 259 | 0.008 | 0.0021 | 0.0153 (passes 0.012) | **no** (search miss) |
| C: commute mod 37 × age mod 7 | 259 | 0.016 | 0.0048 | 0.0390 (passes 0.012) | **yes** (118 s) |

Reading the table:

* **Resolution limit.** Structures worth ≤ 0.0008 AUC are below what a 66,866-row confirmation fold can resolve:
  even the true planted program does not clear 3σ there. No honest method could confirm them on this data.
* **Single-atom and coarse structures worth ≈ 0.002 AUC are found**, in 34 s to 4 min.
* **Search limit.** The two-atom, 259-group structure worth 0.0021 AUC was detectable (its true program scores 0.0153)
  but was *not* found within 30 minutes; at 0.0048 AUC it was found in 2 minutes. On the real residual the search ran
  for 5.5 hours (47,679 programs, versus about 5,800 per 30-minute power run), so deeper structures had far more
  budget there, but the power analysis only certifies what it tested: single-atom structures down to ≈ 0.002 AUC and
  two-atom structures down to ≈ 0.005 AUC.

