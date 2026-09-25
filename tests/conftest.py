"""Synthetic S6E9-shaped data: same schema, a recipe-driven label and a memorised-income effect."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_frame(n: int, rng: np.random.Generator, with_target: bool, start_id: int = 0) -> pd.DataFrame:
    incomes = rng.integers(20_000, 150_000, size=400)  # a small repertoire -> repeated exact values
    df = pd.DataFrame({
        "id": np.arange(start_id, start_id + n),
        "Age": rng.integers(18, 63, n),
        "Annual_Income_USD": rng.choice(incomes, n).astype(float),
        "Daily_Commute_km": np.round(rng.gamma(2.0, 12.0, n), 1),
        "Number_of_Cars_Owned": rng.integers(0, 4, n),
        "Charging_Stations_Near_Home": rng.integers(0, 15, n),
        "Charging_Stations_Near_Work": rng.integers(0, 20, n),
        "Environmental_Concern_Level": rng.integers(1, 6, n).astype(float),
        "Gender": rng.choice(["Male", "Female", "Other"], n),
        "City_Type": rng.choice(["Urban", "Suburban", "Rural"], n),
        "Current_Car_Type": rng.choice(["Sedan", "SUV", "Truck", "Hatchback"], n),
        "Home_Charging_Possible": rng.choice(["Yes", "No"], n),
        "Subsidy_Available": rng.choice(["Yes", "No"], n),
        "Range_Anxiety_Level": rng.choice(["Low", "Medium", "High"], n),
    })
    if with_target:
        score = (1.2 * df.Annual_Income_USD / 1e5 + 0.6 * df.Environmental_Concern_Level + 2.0 * (df.Subsidy_Available == "Yes")
                 - 1.0 * (df.Range_Anxiety_Level == "Medium") - 3.0 * (df.Range_Anxiety_Level == "High"))
        memory = pd.Series(rng.normal(0, 0.6, size=len(incomes)), index=incomes).reindex(df.Annual_Income_USD.astype(int)).to_numpy()
        df["Will_Buy_EV"] = np.where(score + memory + rng.logistic(0, 0.6, n) > 4.0, "Yes", "No")
    return df


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory):
    rng = np.random.default_rng(0)
    d = tmp_path_factory.mktemp("s6e9")
    make_frame(12_000, rng, True).to_csv(d / "train.csv", index=False)
    make_frame(4_000, rng, False, start_id=12_000).to_csv(d / "test.csv", index=False)
    return d
