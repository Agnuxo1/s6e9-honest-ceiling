# Open research directions

The Darwin-Cage search found no structure in the residual of the best honest blend. Its power analysis shows that,
inside its search space, it finds single-atom structures worth about 0.002 AUC and two-atom structures worth about
0.005 AUC, and that it can miss a two-atom structure worth 0.002. That space and that power are limited. This page
lists what lies outside it, each as a hypothesis with a concrete, falsifiable acceptance test. Contributions are
welcome (see [CONTRIBUTING.md](CONTRIBUTING.md)); an honest improvement will be added to the results with credit.

## The acceptance test (the same for every idea)

A proposal beats the ceiling only if **all** of these hold:

1. It is computed from `train.csv`, `test.csv` and CC0 public data only, with every supervised statistic fit inside
   the outer-train partition of `StratifiedKFold(10, shuffle=True, random_state=42)`.
2. Added to the published blend, it raises the **nested** out-of-fold AUC (weights fit on 9 folds, scored on the
   10th) in at least 8 of 10 folds.
3. The paired gain at private-split size (229,257-row subsamples, `blend.paired_bootstrap`) is positive with
   P(better) ≥ 0.95.
4. Or, as a cheaper screen: a feature whose group-mean residual correlation on the never-seen fold exceeds
   `z / sqrt(n)` with z = 3 after a family-wise correction for the number of candidates confirmed.

`honest-ceiling residual` and `honest_ceiling.darwin.DarwinSearch.confirm` implement the screen.

## Directions

### 1. Partial pooling of the income identity
**Hypothesis.** Fixed smoothing priors (m = 1, 10, 100) under- or over-shrink income groups of different sizes. An
empirical-Bayes or hierarchical encoding (beta-binomial per group, with the prior tied to the nearest original row
and the income neighbourhood) extracts the memory more precisely.
**Why it is plausible.** The residual shows a small negative correlation between rows that share an exact income
(−0.032). Part of that is the expected mechanics of cross-fitting, part may be mis-shrinkage.
**Test.** Replace `nested_target_encode` for the income keys by a partial-pooling encoder; the same-income residual
correlation should move toward 0 and the acceptance test above should pass.

### 2. Readouts that are not group means
**Hypothesis.** The search scores programs by the shrunk mean of the residual per key. Structures that change the
*shape* of the conditional distribution (variance, interaction with the predicted probability) are invisible to a
mean readout.
**Test.** Score programs by the out-of-fold AUC gain of a tiny model `logit(p) + f(key, logit(p))` instead of a
correlation. Keep the reserved fold.

### 3. Relations between rows
**Hypothesis.** The generator may have produced rows in related groups (near-duplicates in several columns that are
not exact copies). Single-row programs cannot see that.
**Test.** Build k-nearest-neighbour graphs over the raw columns (train + test, unsupervised), then search programs
over neighbourhood statistics (for example the out-of-fold residual of a row's neighbours).

### 4. Neural or LLM-guided program synthesis
**Hypothesis.** A random mutation operator explores a tiny fraction of the program space. A learned proposal
distribution (a small model trained on which programs scored well, or a language model proposing transformations
with a rationale) could reach deeper programs.
**Test.** Same fitness and reserved fold as `DarwinSearch`; report the number of evaluations to reach the planted
structures of the power analysis, and the result on the real residual.

### 5. Modelling the generator itself
**Hypothesis.** Playground data is sampled from a generative model trained on the original table. If the generator
family can be identified (for example a tabular diffusion or copula model), its sampling artefacts may be
predictable: which original row a synthetic row was conditioned on, and how much of that row's label it inherited.
**Test.** Fit candidate generators on the CC0 original table, compare the statistics of their samples with train,
and turn the best match into features for the label memory. Acceptance test as above.

### 6. A stronger neural leg
**Hypothesis.** A RealMLP or TabM leg trained to convergence (256 epochs, several seeds, on the same nested
encodings) reaches the public RealMLP's 0.9461 and earns blend weight our 48-epoch run (0.9454) did not.
**Test.** Train on the shared split; the nested blend weight of the leg must be positive and the acceptance test
must pass. Budget: about 20 GPU-hours on an RTX 3090.

### 7. Better use of the search budget
**Hypothesis.** The search spent its budget uniformly. With the power curve known, a sequential design (spend more
evaluations near programs whose search fitness is high but confirmation is not yet tested) lowers the minimum
detectable effect.
**Evidence.** In the power analysis a two-atom, 259-group structure worth 0.0021 AUC was detectable (its true
program scores 0.0153 on the reserved fold, threshold 0.012) but was not found in 30 minutes (5,778 programs).
**Test.** Re-run `scripts/power_analysis.py --plant C --effect 0.008` and show detection with the same budget.

## Engineering items

* Memory: the search ran out of RAM twice when other jobs shared the machine. Keys could be cached as `int32` and
  the fitness computed in chunks.
* Speed: `fast_auc` in the hill-climb is O(n log n) per candidate; an incremental AUC update would make nested
  blending of dozens of streams practical.
* A Kaggle-native version of the search that runs inside the 12-hour notebook limit.
