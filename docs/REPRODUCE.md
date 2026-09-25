# Reproducing the results

## Environment

* Python 3.10-3.13, `pip install -e ".[gbdt]"` (numpy, pandas, scipy, scikit-learn, pyarrow, lightgbm, xgboost).
* Optional: CUDA for XGBoost (`device="cuda"` is picked automatically), PyTorch + `pytabkit` for the neural legs.
* Reference machine: RTX 3090 24 GB, 20 CPU threads, 25 GB RAM, Windows 10.

## Data

```bash
kaggle competitions download -c playground-series-s6e9 -p data && unzip -o data/playground-series-s6e9.zip -d data
# optional, CC0: original table and the public six-view OOF library
kaggle datasets download -d itzzomkar/ev-adoption-behavior-and-range-anxiety -p data/original --unzip
kaggle datasets download -d megayak/s6e9-six-feature-views-oof-library -p data/sixviews --unzip
```

Expected SHA-256: `train.csv` `eae9eaa4e6378df405e755f853771d7e26d212bd93258349fc797b771021946a`,
`test.csv` `539263f6caabc40afd5e2f0bc0ab16b10a2d1177c565fc71b866f0181d836b34`.

## Steps and timings (reference machine)

| Step | Command | Time |
|---|---|---|
| Features (cached) | first call of any command | 10 s |
| XGBoost, 3 seeds × 3 splits | `honest-ceiling train --model xgb --seeds 42 101 2026 --outer-seeds 42 7 13 --original data/original/EV_Adoption_and_Range_Anxiety_Dataset.csv` | ~45 min (GPU) |
| LightGBM, 3 seeds × 3 splits | same with `--model lgbm --threads 12` | ~80 min (CPU) |
| Nested blend | `honest-ceiling blend --library data/sixviews` | ~25 min |
| Residual tests | `honest-ceiling residual --scores work/blend_oof_nested.npy` | 1 min |
| Darwin search | `honest-ceiling darwin --scores work/blend_oof_nested.npy --minutes 220` | as budgeted |
| Power analysis | `python scripts/power_analysis.py --plant A --effect 0.004 ...` | ≤ 30 min per run |
| Figures | `python scripts/make_figures.py` | 30 s |

Randomness is seeded everywhere (outer split, inner cross-fit, model seeds, search). GPU XGBoost histograms are not
bit-exact across hardware; expect differences in the sixth decimal.

## Tests

```bash
pip install -e ".[test]" && pytest -q
```

The tests run on synthetic data with the competition schema: fold determinism and stratification, exact AUC,
absence of label leakage in the nested encoder, a full LightGBM leg, nested blending, residual tools, and that the
Darwin search rejects pure noise and finds a planted structure.
