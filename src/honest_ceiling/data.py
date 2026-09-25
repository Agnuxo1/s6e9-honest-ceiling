"""Data loading and the shared outer split."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
ID = "id"
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
NUMS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
RECIPE = ["Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level"]

N_OUTER = 10
OUTER_SEED = 42  # StratifiedKFold(10, shuffle=True, random_state=42) over train.csv in id order: the split
                 # shared by the public S6E9 OOF libraries, so our OOF files stack row-for-row with theirs.


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(data_dir: str | Path):
    """Return (train, test, y) sorted by id. `data_dir` holds the competition train.csv and test.csv."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "train.csv").sort_values(ID).reset_index(drop=True)
    test = pd.read_csv(data_dir / "test.csv").sort_values(ID).reset_index(drop=True)
    y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
    return train, test, y


def outer_folds(y: np.ndarray, n_splits: int = N_OUTER, seed: int = OUTER_SEED) -> np.ndarray:
    """Fold id per training row for StratifiedKFold(n_splits, shuffle=True, random_state=seed)."""
    fold = np.full(len(y), -1, dtype=np.int8)
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for k, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[va] = k
    return fold


def find_competition_dir() -> Path | None:
    """Locate the competition data on Kaggle (legacy and namespaced input paths) or locally."""
    for p in [Path("/kaggle/input/playground-series-s6e9"), Path("/kaggle/input/competitions/playground-series-s6e9"),
              Path("data"), Path("../data")]:
        if (p / "train.csv").exists():
            return p
    return None
