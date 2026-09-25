"""Power analysis of the Darwin-Cage search on the real S6E9 rows.

For a planted program and effect size d: y' ~ Bernoulli(p + d*s(key)), r' = y' - p, run the search with a fixed
budget and record whether it confirms the planted structure on the reserved fold. Also records the AUC an
oracle would gain from knowing the planted structure: AUC(y', p + d*s) - AUC(y', p).

usage: python power_analysis.py --effect 0.004 --plant A --minutes 25 --out results/power_A_0.004.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from honest_ceiling.blend import fast_auc  # noqa: E402
from honest_ceiling.darwin import DarwinSearch, integer_columns, plant, program_key  # noqa: E402
from honest_ceiling.data import load, outer_folds  # noqa: E402
from honest_ceiling.residual import calibrated_probability  # noqa: E402

PLANTS = {
    "A": [("mod", "Annual_Income_USD", 97)],                                   # 97 groups, one atom
    "B": [("digit", "Annual_Income_USD", 100), ("cat", "City_Type")],          # 30 groups, two atoms
    "C": [("mod", "Daily_Commute_km", 37), ("mod", "Age", 7)],                 # 259 groups, two atoms
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--blend-oof", required=True, help="nested OOF scores of the best blend (npy, train id order)")
    ap.add_argument("--effect", type=float, required=True)
    ap.add_argument("--plant", default="A", choices=list(PLANTS))
    ap.add_argument("--minutes", type=float, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    train, _, y = load(a.data_dir)
    fold = outer_folds(y)
    p = calibrated_probability(np.load(a.blend_oof), y)
    D = integer_columns(train)
    prog = PLANTS[a.plant]
    y2, shift = plant(p, D, prog, a.effect, seed=a.seed)
    oracle_gain = fast_auc(y2, p + shift) - fast_auc(y2, p)
    s = DarwinSearch(D, (y2 - p).astype(np.float64), fold)
    res = s.run(seconds=a.minutes * 60, seed=a.seed + 100, stop_on_confirm=True)
    planted_key = program_key(prog, D)
    found = [h for h in res["confirmed"]]
    out = dict(plant=a.plant, program=[list(t) for t in prog], effect=a.effect, n_groups=int(planted_key.max()) + 1,
               oracle_auc_gain=oracle_gain, planted_confirm_corr=s.confirm(planted_key),
               detected=bool(found), first_confirmed=found[0] if found else None, evaluated=res["evaluated"],
               seconds=res["seconds"], threshold=res["threshold"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
