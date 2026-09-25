"""honest-ceiling: fold-safe ensembling and a Darwin-Cage residual search for Kaggle Playground S6E9.

Modules
-------
data      loading, shared outer folds, paths
features  unsupervised features, integer keys and nested (cross-fitted) target encoding
train     GBDT legs (LightGBM / XGBoost) trained fold-safely on a shared split
blend     rank-space hill-climb with leave-one-fold-out ("nested") evaluation, paired bootstrap
residual  calibrated residual, whiteness tests, oracle AUC bound
darwin    genetic search for structure in the residual, with a reserved confirmation fold
"""

__version__ = "1.0.0"
