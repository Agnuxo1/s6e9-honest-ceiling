"""Command line: `honest-ceiling <command>`.

  train     train one fold-safe GBDT leg over a shared outer split
  blend     nested hill-climb over every leg in a work dir (+ optional external OOF library)
  residual  whiteness tests and oracle bound for a score vector
  darwin    Darwin-Cage search on the residual of a score vector
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _base(args):
    from .data import load
    from .features import build_base
    train, test, y = load(args.data_dir)
    X, K = build_base(train, test, args.original, cache=Path(args.work_dir) / "base_features.parquet")
    return train, test, y, X, K


def cmd_train(args):
    from .train import train_leg
    train, test, y, X, K = _base(args)
    for seed in args.seeds:
        for os_ in args.outer_seeds:
            tag = f"{args.model}_s{seed}_os{os_}"
            rep = train_leg(X, K, y, len(train), len(test), Path(args.work_dir) / "legs" / tag, model=args.model,
                            seed=seed, outer_seed=os_, threads=args.threads)
            print(json.dumps({"tag": tag, **(rep or {})}))


def _load_legs(work_dir: Path):
    """Average every leg family (same model, any seed / outer split) in rank space."""
    from .blend import rank01
    fam: dict[str, list[Path]] = {}
    for d in sorted((work_dir / "legs").glob("*")):
        if (d / "oof.npy").exists():
            fam.setdefault(d.name.split("_s")[0], []).append(d)
    names, O, T = [], [], []
    for name, dirs in fam.items():
        O.append(np.mean([rank01(np.load(d / "oof.npy")) for d in dirs], axis=0))
        T.append(np.mean([rank01(np.load(d / "test.npy")) for d in dirs], axis=0))
        names.append(f"{name}x{len(dirs)}")
    return names, O, T


def cmd_blend(args):
    from .blend import nested_hillclimb, rank01
    from .data import load, outer_folds
    train, test, y = load(args.data_dir)
    fold = outer_folds(y)
    names, O, T = _load_legs(Path(args.work_dir))
    if args.library:
        lib = Path(args.library)
        so = pd.read_csv(lib / "oof_six_views.csv").set_index("id").loc[train["id"]]
        st = pd.read_csv(lib / "test_six_views.csv").set_index("id").loc[test["id"]]
        for c in [c for c in so.columns if c[:2] in ("A_", "B_", "C_", "D_", "E_", "F_")]:
            names.append(f"six:{c[0]}"); O.append(so[c].to_numpy(float)); T.append(st[c].to_numpy(float))
        g = lib / "oof_realmlp_g.csv"
        if g.exists():
            go = pd.read_csv(g).set_index("id").loc[train["id"]]
            gt = pd.read_csv(lib / "test_realmlp_g.csv").set_index("id").loc[test["id"]]
            names.append("six:G_realmlp10"); O.append(go["G_realmlp_10fold"].to_numpy(float)); T.append(gt["G_realmlp_10fold"].to_numpy(float))
    O = np.column_stack([rank01(o) for o in O])
    T = np.column_stack([rank01(t) for t in T])
    res = nested_hillclimb(O, y, fold)
    out = Path(args.work_dir)
    np.save(out / "blend_oof_nested.npy", res["nested_scores"])
    sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": rank01(T @ res["weights"])})
    sub.to_csv(out / "submission.csv", index=False)
    rep = dict(streams=names, weights=dict(zip(names, np.round(res["weights"], 4).tolist())), nested_auc=res["nested_auc"],
               in_sample_auc=res["in_sample_auc"], fold_aucs=res["fold_aucs"])
    (out / "blend_report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


def cmd_residual(args):
    from .data import load
    from .residual import calibrated_probability, oracle_bound, whiteness_report
    train, _, y = load(args.data_dir)
    p = calibrated_probability(np.load(args.scores), y)
    r = y - p
    inc = train["Annual_Income_USD"].round().astype(int).to_numpy()
    rep = whiteness_report(r, {"id": train["id"].to_numpy(), "income": inc, "commute": train["Daily_Commute_km"].to_numpy(),
                               "prediction": p}, {"income_exact": inc})
    rep["oracle_auc"] = oracle_bound(p)
    print(json.dumps(rep, indent=2))


def cmd_darwin(args):
    from .darwin import DarwinSearch, integer_columns
    from .data import load, outer_folds
    from .residual import calibrated_probability
    train, _, y = load(args.data_dir)
    p = calibrated_probability(np.load(args.scores), y)
    s = DarwinSearch(integer_columns(train), (y - p).astype(np.float64), outer_folds(y))
    null = s.null()
    res = s.run(seconds=args.minutes * 60, seed=args.seed, log_path=args.log)
    res["null_mean"], res["null_std"] = null
    Path(args.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("evaluated", "generations", "seconds", "threshold")} | {"confirmed": len(res["confirmed"])}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="honest-ceiling", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--data-dir", required=True, help="folder with competition train.csv / test.csv")
        p.add_argument("--work-dir", default="work")
    t = sub.add_parser("train"); common(t)
    t.add_argument("--model", choices=["xgb", "lgbm"], default="xgb")
    t.add_argument("--seeds", type=int, nargs="+", default=[42])
    t.add_argument("--outer-seeds", type=int, nargs="+", default=[42])
    t.add_argument("--threads", type=int, default=-1)
    t.add_argument("--original", default=None, help="CC0 original EV_Adoption_and_Range_Anxiety_Dataset.csv (optional)")
    t.set_defaults(fn=cmd_train)
    b = sub.add_parser("blend"); common(b)
    b.add_argument("--library", default=None, help="folder of the public six-view OOF library (optional)")
    b.set_defaults(fn=cmd_blend)
    r = sub.add_parser("residual"); common(r)
    r.add_argument("--scores", required=True, help="npy of OOF scores in train id order")
    r.set_defaults(fn=cmd_residual)
    d = sub.add_parser("darwin"); common(d)
    d.add_argument("--scores", required=True)
    d.add_argument("--minutes", type=float, default=30)
    d.add_argument("--seed", type=int, default=2026)
    d.add_argument("--out", default="darwin_result.json")
    d.add_argument("--log", default=None)
    d.set_defaults(fn=cmd_darwin)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
