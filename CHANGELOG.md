# Changelog

## 1.0.0 — 2026-10-01

First public release, after the close of Kaggle Playground S6E9.

* `honest_ceiling` package: shared outer split, unsupervised features, nested target encoding, fold-safe
  LightGBM/XGBoost legs, nested hill-climb, paired bootstrap, residual tests, oracle bound, Darwin-Cage search with a
  reserved confirmation fold and signal planting; `honest-ceiling` CLI.
* Results: legs, ablations, blend v4 (nested OOF 0.946420), residual and ceiling analysis, two Darwin-Cage runs
  (47,679 programs, 0 confirmed), power analysis, submissions.
* Two Kaggle notebooks (pipeline; Darwin Cage) and a public dataset with our OOF/test library and search logs.
* Tests on synthetic data and GitHub Actions CI; GitHub Pages report; light and dark figures.
