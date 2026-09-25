<p align="center">
  <img src="docs/assets/banner.svg" alt="Honest Ceiling: fold-safe ensembling and a Darwin-Cage residual search on Kaggle Playground S6E9" width="100%">
</p>

<p align="center">
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-5ee2c4"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-7c8cff">
  <a href="https://github.com/Agnuxo1/s6e9-honest-ceiling/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Agnuxo1/s6e9-honest-ceiling/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://www.kaggle.com/competitions/playground-series-s6e9"><img alt="Kaggle S6E9" src="https://img.shields.io/badge/Kaggle-Playground%20S6E9-20beff"></a>
  <a href="https://agnuxo1.github.io/s6e9-honest-ceiling/"><img alt="Report" src="https://img.shields.io/badge/report-GitHub%20Pages-0b0d12"></a>
</p>

**How high can an honest model go on a synthetic tabular competition, and how do you know when you have arrived?**
This repository answers both questions for [Kaggle Playground Series S6E9](https://www.kaggle.com/competitions/playground-series-s6e9)
(predict `Will_Buy_EV`, ROC-AUC, 668,665 training rows). It contains a fully fold-safe ensembling pipeline, an
evolutionary search that starts from noise and is rewarded only by structure it finds in the residual, a power
analysis of that search, and every number behind the claims.

## Results at a glance

| | Value | Evidence |
|---|---|---|
| Best honest blend, **nested** out-of-fold ROC-AUC | **0.946420** | [`results/blend_v4_report.json`](results/blend_v4_report.json) |
| Oracle bound (AUC if our probabilities were exactly right) | 0.94660 ± 0.00028 | [`results/residual_and_ceiling.json`](results/residual_and_ceiling.json) |
| Folds won against the public six-view ensemble | 10 / 10 | same file, `per_fold_v4_minus_six` |
| Paired gain at private-split size (229k rows, 60 subsamples) | +0.000073 ± 0.000016, P(better) = 1.00 | same file |
| Darwin-Cage programs evaluated / confirmed on the reserved fold | 47,679 / **0** | [`results/darwin_run*_hall.json`](results) |
| Public leaderboard, honest blend v4 | 0.94638 | [`results/submissions.json`](results/submissions.json) |

The honest blend sits within the sampling error of its own oracle bound. The residual behaves like independent
coin flips, and an unconstrained search over 47,679 feature programs found nothing that survives a fold it never
saw. The power analysis shows the search finds planted structures worth about 0.002-0.005 AUC, and where its limits are.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/ceiling_dark.png">
  <img alt="AUC ladder from the generator recipe to the nested blend, and the distance to the oracle bound" src="figures/ceiling_light.png">
</picture>

## Why every top score looks the same

The public top 200 is packed into 0.0003 of AUC. One positive row of the 57k-row public split is worth about
0.00002, so ranks 2 to 28 differ by roughly eight rows. The label is the generator's purchase recipe plus an
independent random draw per row; the only extra signal is that the generator copied incomes from a 10,000-row
original table, so an exact income value partially identifies an original row. Once a model target-encodes that
identity, nothing learnable remains. The first-placed public score (0.94945) comes with no published method. The
public discussion ([*Is there a leak?*](https://www.kaggle.com/competitions/playground-series-s6e9/discussion)) and
[megayak's notebook](https://www.kaggle.com/code/megayak/s6e9-0-94656-reading-the-public-split) showing that paired
submissions can read the public split suggest public-split information; such gains do not transfer to the private 80 %.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/leaderboard_dark.png">
  <img alt="Histogram of the public top-200 scores" src="figures/leaderboard_light.png">
</picture>

## What is inside

```
src/honest_ceiling/   installable package (pip install -e .)
  data.py             loading, the shared StratifiedKFold(10, seed 42) outer split
  features.py         generator recipe, digits, original-table lookups, nested target encoding
  train.py            fold-safe LightGBM / XGBoost legs, one checkpoint per fold
  blend.py            rank-space hill-climb with leave-one-fold-out evaluation, paired bootstrap
  residual.py         calibrated residual, whiteness tests, oracle bound
  darwin.py           Darwin-Cage search with a reserved confirmation fold, signal planting
  cli.py              honest-ceiling train | blend | residual | darwin
notebooks/            the two Kaggle notebooks (pipeline, Darwin Cage)
results/              every number quoted here, as JSON
figures/              light and dark versions of every figure
scripts/              figure rendering, power analysis
docs/                 method, results, reproduction guide, GitHub Pages report
tests/                synthetic-data tests (leakage, AUC, blending, search power), run in CI
```

## Quickstart

```bash
git clone https://github.com/Agnuxo1/s6e9-honest-ceiling && cd s6e9-honest-ceiling
pip install -e ".[gbdt]"
kaggle competitions download -c playground-series-s6e9 -p data && unzip -o data/*.zip -d data

# one fold-safe leg over the shared split (XGBoost uses CUDA when available)
honest-ceiling train --data-dir data --model xgb --seeds 42 101 2026 --outer-seeds 42 7 13
honest-ceiling train --data-dir data --model lgbm --seeds 42 101 2026 --outer-seeds 42 7 13

# nested hill-climb over every leg (+ optional public CC0 six-view OOF library on the same folds)
honest-ceiling blend --data-dir data --library path/to/s6e9-six-feature-views-oof-library

# what is left: whiteness tests, oracle bound, and the Darwin-Cage search
honest-ceiling residual --data-dir data --scores work/blend_oof_nested.npy
honest-ceiling darwin   --data-dir data --scores work/blend_oof_nested.npy --minutes 60
```

Full reproduction, timings and hardware: [docs/REPRODUCE.md](docs/REPRODUCE.md).

## Method in one screen

1. **One frozen split for everything.** `StratifiedKFold(10, shuffle=True, random_state=42)` over `train.csv` in id
   order, the split used by the public OOF libraries, so every out-of-fold file stacks row for row.
2. **Nested target encoding.** Exact income, nearest original income, income//100, //1000, exact commute and a few
   categorical interactions are encoded inside each outer-train partition with an inner 5-fold cross-fit. A training
   row never sees its own label; the outer fold is never seen at all.
3. **Early stopping without the scored fold.** A stratified 10 % slice of outer-train decides the number of trees.
4. **Averaging over splits.** Each GBDT family is trained on three outer splits (seeds 42, 7, 13) × three seeds and
   averaged in rank space: 0.94606 per run → 0.94623 per family, and +0.000023 for the final blend over one split.
5. **Nested blending.** Hill-climb weights are fit on 9 folds and scored on the 10th, for every fold, so the blend's
   reported AUC is itself out of fold.

Details and every ablation: [docs/METHOD.md](docs/METHOD.md).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/folds_dark.png">
  <img alt="Per-fold gain of the nested blend over the public six-view ensemble" src="figures/folds_light.png">
</picture>

## The Darwin-Cage experiment

The idea: do not tell the search what to look for or how. Start from random programs built out of the thirteen raw
columns (remainders, integer divisions, decimal digits, bit shifts, categorical values; up to four atoms), mutate and
recombine them, and reward each program only by how well its groups predict the **residual** of the best blend,
measured out of fold on nine search folds. The tenth fold is never touched by the search: a program is a discovery
only if its correlation there exceeds 3σ of the null (0.012 for 66,866 rows).

* Real residual: **47,679 programs, 27 hall-of-fame entries (6 distinct), 0 confirmed.** The best confirmation was
  0.0052 (z ≈ 1.3). Every entry that looked strong on the search folds collapsed on the reserved fold, which is the
  signature of multiple testing on white noise.
* Power analysis: labels are redrawn from the real calibrated probabilities plus a planted per-group shift of known
  size, and the same search runs for 30 minutes. It finds planted single-atom structures worth about 0.002 AUC in
  34 s to 4 min and a two-atom structure worth 0.005 AUC in 2 min; it **missed** a two-atom structure worth 0.002 AUC
  that was detectable in principle. Structures worth ≤ 0.0008 AUC cannot be confirmed on a 67k-row fold by any
  method. Full table: [docs/METHOD.md §10](docs/METHOD.md#10-power-analysis), data: [`results/power_summary.json`](results/power_summary.json).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/darwin_dark.png">
  <img alt="Darwin-Cage hall of fame versus planted signals" src="figures/darwin_light.png">
</picture>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/power_dark.png">
  <img alt="Power analysis of the Darwin-Cage search" src="figures/power_light.png">
</picture>

## What did not help (measured, not assumed)

| Idea | Result |
|---|---|
| Exact income × recipe-cell target encoding | −0.00012 per fold; removed from the final legs |
| Original 10k rows as extra training data | public evidence 0/5 folds; used only as a lookup table here |
| Centred-window target rates, stronger smoothing, deeper trees | neutral or negative on 3 folds |
| Income × city / car type / home-charging encodings | −0.00001 to −0.00008 on 3 folds |
| Own MLP, RealMLP (pytabkit, 48 epochs), logistic regression, ladder view | zero blend weight |
| Reservoir-computing leg (random tanh + complex-phase expansion, ridge readout) | fold-0 AUC 0.936-0.940, +0.00001 in blend (noise) |
| CatBoost with string identity columns | fold-0 AUC 0.9431 at 12 min per fold; dropped |

Raw numbers: [`results/ablations.json`](results/ablations.json), [`results/legs.json`](results/legs.json).

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="figures/residual_dark.png">
  <img alt="Autocorrelation of the residual under four orderings" src="figures/residual_light.png">
</picture>

## Open doors

A null result is only as good as the space it searched. The search alphabet here is integer transforms of single
columns and their joint keys, with group-mean readouts. Structures outside that space are untested, and several are
concrete enough to try: hierarchical (partial-pooling) encodings of the income identity, readouts that are not group
means, search over pairs of rows rather than single rows, neural program synthesis, and generator-level modelling of
how the synthetic rows were produced. Each is written up as a testable hypothesis with an acceptance criterion in
**[ROADMAP.md](ROADMAP.md)**. If you beat the numbers above honestly, please open an issue: we will add your result
and credit you.

## Provenance and licences

* Competition data: [Kaggle S6E9](https://www.kaggle.com/competitions/playground-series-s6e9), CC BY 4.0. Not
  redistributed here; download it with the Kaggle CLI.
* Original table: [itzzomkar / EV Adoption Behavior and Range Anxiety](https://www.kaggle.com/datasets/itzzomkar/ev-adoption-behavior-and-range-anxiety), CC0-1.0.
* Public OOF library used in the blend: [megayak / S6E9 Six Feature Views OOF Library](https://www.kaggle.com/datasets/megayak/s6e9-six-feature-views-oof-library), CC0-1.0.
* The generator's purchase and range-anxiety recipes were reconstructed by Chris Deotte in his public EDA notebook.
* No leaderboard probing, hidden labels or private sharing were used. Full statement: [NOTICE](NOTICE).

## Citation

```bibtex
@software{angulo_2026_honest_ceiling,
  author  = {Angulo de Lafuente, Francisco},
  title   = {Honest Ceiling: fold-safe ensembling and a Darwin-Cage residual search on Kaggle Playground S6E9},
  year    = {2026},
  url     = {https://github.com/Agnuxo1/s6e9-honest-ceiling},
  license = {Apache-2.0}
}
```

## Acknowledgements

Chris Deotte (generator recipes), megayak (six-view OOF library and the public-split analysis), Marc Maldonado Lorca
(original-row memory), Naji, Georgy Mamarin, Yusuke Hayashi and the S6E9 discussion community, whose careful public
measurements made an honest ceiling measurable.

Apache-2.0 © 2026 Francisco Angulo de Lafuente.
