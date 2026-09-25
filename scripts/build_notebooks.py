"""Generate the two Kaggle notebooks (also committed under notebooks/). Paths resolve on Kaggle and locally."""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Agnuxo1/s6e9-honest-ceiling"

SETUP = r'''
import glob, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def find(name, roots=("/kaggle/input", os.environ.get("HC_LOCAL", "."))):
    """First file called `name` under the given roots (handles legacy and namespaced Kaggle input paths)."""
    for r in roots:
        hits = sorted(glob.glob(f"{r}/**/{name}", recursive=True))
        if hits:
            return Path(hits[0])
    raise FileNotFoundError(name)

try:
    import honest_ceiling  # noqa: F401
except ImportError:  # the wheel ships inside the companion dataset; no internet needed
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", str(find("honest_ceiling-1.0.0-py3-none-any.whl"))], check=True)
import honest_ceiling
from honest_ceiling.data import load, outer_folds
from honest_ceiling.blend import rank01, fast_auc, nested_hillclimb, paired_bootstrap

COMP = find("train.csv").parent
LIB = find("oof_legs.csv").parent            # companion dataset: our OOF/test library, folds, results
print("honest_ceiling", honest_ceiling.__version__, "| competition data:", COMP, "| library:", LIB)
train, test, y = load(COMP)
fold = outer_folds(y)
print(f"train {train.shape}  test {test.shape}  positives {y.mean():.4f}")

plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "axes.edgecolor": "#e6e5e1",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                     "axes.titleweight": "semibold", "axes.titlelocation": "left", "font.size": 10})
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2"
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def pipeline_nb():
    cells = [
        md(f"""
# S6E9 · Fold-safe ensembling to the honest ceiling (nested OOF 0.94642)

**What this notebook shows, with code you can fork:**
1. one frozen outer split shared with the public OOF libraries,
2. **nested** target encoding of the income memory (a row never sees its own label; the scored fold is never seen),
3. early stopping on a slice of outer-train, never on the scored fold,
4. averaging each model family over **three outer splits × three seeds**,
5. a hill-climb blend whose own score is **out of fold** (weights fit on 9 folds, scored on the 10th),
6. a paired bootstrap at private-leaderboard size to check whether a gain would survive the private split.

The package, tests, figures and every number are in the open-source repository: [{REPO}]({REPO}) (Apache-2.0).
The companion notebook, *Darwin Cage on the residual*, asks what is left after this model.

> Honest framing: this pipeline does not beat the public pack. It reaches **0.946420 nested OOF**, within the sampling
> error of its own oracle bound (0.94660 ± 0.00028). Its value is that every number is out of fold, so it tells you
> what a blend is really worth before the private leaderboard does.
"""),
        md("## 1 · Setup\nThe wheel of the `honest_ceiling` package ships in the companion dataset, so the notebook runs offline."),
        code(SETUP),
        md("""
## 2 · The shared split
`StratifiedKFold(10, shuffle=True, random_state=42)` over `train.csv` in id order. It is the split of the public CC0
six-view OOF library, so our out-of-fold vectors stack row for row with theirs.
"""),
        code('''
lib = pd.read_csv(LIB / "oof_legs.csv")
assert (lib["id"].to_numpy() == train["id"].to_numpy()).all()
assert (lib["fold_s42"].to_numpy() == fold).all(), "fold ids differ from the published library"
print("fold ids identical to the published library for all", len(fold), "rows")
print(pd.Series(fold).value_counts().sort_index().to_dict())
'''),
        md("""
## 3 · Features and nested target encoding
`build_base` reads no label: the generator's purchase and range-anxiety recipes (reconstructed by Chris Deotte),
income digits and remainders, frequencies, and lookups into the CC0 original table. `fold_matrices` adds the
supervised columns for one outer fold, cross-fitted inside outer-train.

A quick leakage check first: with labels shuffled, a leaky encoder would still score well on its own fit rows.
"""),
        code('''
from honest_ceiling.features import build_base, fold_matrices, nested_target_encode
try:
    ORIG = find("EV_Adoption_and_Range_Anxiety_Dataset.csv")
except FileNotFoundError:
    ORIG = None
X, K = build_base(train, test, ORIG)
print("unsupervised features:", X.shape[1], "| integer keys:", K.shape[1])

rng = np.random.default_rng(0)
y_shuffled = rng.permutation(y)
enc_fit, _ = nested_target_encode(K.iloc[:len(train)], y_shuffled, [K.iloc[:5]], seed=0, spec={"k_income_exact": [1.0]})
print("AUC of the income encoding against shuffled labels (0.5 = no leakage):",
      round(fast_auc(y_shuffled, enc_fit["te_income_exact_a1"].to_numpy()), 4))
'''),
        md("""
## 4 · One fold-safe leg, live
XGBoost on one seed and the shared split. Early stopping uses a stratified 10 % slice of outer-train. This is one of
the 18 GBDT runs behind the final blend; the full set is loaded in the next section.
"""),
        code('''
from honest_ceiling.train import train_leg
t0 = time.time()
rep = train_leg(X, K, y, len(train), len(test), "leg_xgb_s42", model="xgb", seed=42, outer_seed=42)
print(f"OOF AUC {rep['oof_auc']:.6f} in {time.time()-t0:.0f} s")
published = json.load(open(LIB / "legs.json"))["final_xgb_s42"]
print("published run   ", round(published["oof_auc"], 6))
print("per-fold difference vs published (published run used CUDA; histogram construction differs between devices):",
      np.round(np.array(rep["fold_aucs"]) - np.array(published["fold_aucs"]), 6))
'''),
        md("""
## 5 · Averaging over outer splits
A single run's OOF depends on which rows its encoders and early stopping happened to see. Averaging the same model
over different outer splits (seeds 42, 7, 13) and seeds (42, 101, 2026) in rank space removes part of that variance.
"""),
        code('''
legs = json.load(open(LIB / "legs.json"))
single = np.mean([v["oof_auc"] for k, v in legs.items() if k.startswith("final_xgb")])
print(f"XGBoost: mean single-run OOF {single:.6f}  ->  9-run rank average {fast_auc(y, lib['xgb_x9'].to_numpy()):.6f}")
single = np.mean([v["oof_auc"] for k, v in legs.items() if k.startswith("final_lgbm")])
print(f"LightGBM: mean single-run OOF {single:.6f}  ->  9-run rank average {fast_auc(y, lib['lgbm_x9'].to_numpy()):.6f}")
'''),
        md("""
## 6 · Nested hill-climb
Streams: our legs plus the public CC0 six-view library (views A-F and RealMLP-G, same split). Weights are fit on
9 folds and scored on the 10th, for every fold, so the blend's AUC is itself out of fold.
Attach [megayak/s6e9-six-feature-views-oof-library](https://www.kaggle.com/datasets/megayak/s6e9-six-feature-views-oof-library) to reproduce this cell.
"""),
        code('''
SIX = find("oof_six_views.csv").parent
so = pd.read_csv(SIX / "oof_six_views.csv").set_index("id").loc[train["id"]]
go = pd.read_csv(SIX / "oof_realmlp_g.csv").set_index("id").loc[train["id"]]
names = ["xgb_x9", "lgbm_x9", "logreg_x1", "mlp_x2", "realmlp_x1", "ladderxgb_x1"]
O = [lib[n].to_numpy(float) for n in names]
for c in [c for c in so.columns if c[:2] in ("A_", "B_", "C_", "D_", "E_", "F_")]:
    names.append("six:" + c[0]); O.append(so[c].to_numpy(float))
names.append("six:G_realmlp"); O.append(go["G_realmlp_10fold"].to_numpy(float))
O = np.column_stack([rank01(o) for o in O])
t0 = time.time()
res = nested_hillclimb(O, y, fold)
print(f"nested OOF AUC {res['nested_auc']:.6f}   (in-sample {res['in_sample_auc']:.6f})   {time.time()-t0:.0f} s")
print(pd.Series(res["weights"], index=names).round(3).sort_values(ascending=False).to_string())
'''),
        code('''
ens = so["ensemble"].to_numpy(float)
d = np.array([fast_auc(y[fold == k], res["nested_scores"][fold == k]) - fast_auc(y[fold == k], ens[fold == k]) for k in range(10)])
fig, ax = plt.subplots(figsize=(9, 3.4))
ax.bar(range(10), d * 1e5, color=BLUE, width=0.62)
for i, v in enumerate(d * 1e5):
    ax.text(i, v + 0.2, f"+{v:.1f}" if v >= 0 else f"{v:.1f}", ha="center", fontsize=9, color="#52514e")
ax.set_xticks(range(10), [f"fold {k}" for k in range(10)]); ax.set_ylabel("AUC gain × 1e-5")
ax.set_title(f"Nested blend vs public six-view ensemble: {int((d > 0).sum())} of 10 folds")
plt.show()
'''),
        md("""
## 7 · Would the gain survive the private leaderboard?
The private split has 229,257 rows. Scoring both vectors on the *same* random subsamples of that size is how the
private leaderboard compares two teams.
"""),
        code('''
for n, label in ((229_257, "private-size"), (57_314, "public-size")):
    pb = paired_bootstrap(y, res["nested_scores"], ens, n=n, reps=40, seed=1)
    print(f"{label:12s} gain {pb['mean']:+.6f} ± {pb['std']:.6f}   P(better) {pb['p_a_better']:.2f}")
'''),
        md("""
## 8 · Submission
The test predictions use the same weights on the test ranks of every stream.
"""),
        code('''
st = pd.read_csv(SIX / "test_six_views.csv").set_index("id").loc[test["id"]]
gt = pd.read_csv(SIX / "test_realmlp_g.csv").set_index("id").loc[test["id"]]
tl = pd.read_csv(LIB / "test_legs.csv")
T = [tl[n].to_numpy(float) for n in names[:6]]
T += [st[c].to_numpy(float) for c in [c for c in so.columns if c[:2] in ("A_", "B_", "C_", "D_", "E_", "F_")]]
T += [gt["G_realmlp_10fold"].to_numpy(float)]
T = np.column_stack([rank01(t) for t in T])
sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": rank01(T @ res["weights"])})
sub.to_csv("submission.csv", index=False)
print(sub.head(), "\\nSpearman with the published v4 file:", round(sub["Will_Buy_EV"].corr(tl["blend_v4"], method="spearman"), 6))
'''),
        md(f"""
## What to take from it

* Every supervised number in a blend should be out of fold, **including the blend weights**. The nested score here
  is 0.0000047 below the in-sample one: small, but it is the honest one.
* Averaging a model over several outer splits is one of the few gains that remained at this saturation.
* A gain is only a gain if it survives a paired comparison at the size of the split that will judge it.

Code, tests, the full results and an open list of research directions: [{REPO}]({REPO}).
If you find an honest way above this ceiling, open an issue there: it will be added with credit.
"""),
    ]
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    return nb


def darwin_nb():
    cells = [
        md(f"""
# S6E9 · Darwin Cage on the residual: 47,679 programs, zero discoveries, and a power analysis

**The question.** After an honest model, is there *any* structure left that we have not thought of?

**The approach (Darwin Cage).** Do not tell the search what to look for or how. Start from random programs built out
of the thirteen raw columns (remainders, integer divisions, decimal digits, bit shifts, categorical values, up to four
atoms joined into one key), mutate and recombine them, and reward each program **only** by how well its groups predict
the residual of the best blend, measured out of fold. One fold is never touched by the search: a program counts as a
discovery only if it survives there.

**The result.** On the real residual: nothing survives. On synthetic labels with a planted structure of known size:
the same search finds it. That second part is what makes the first one mean something.

Repository with the package, tests and all logs: [{REPO}]({REPO}) (Apache-2.0).
"""),
        md("## 1 · Setup"),
        code(SETUP),
        md("""
## 2 · The residual
`blend_v4_nested` is the nested out-of-fold score of the best honest blend (see the companion pipeline notebook).
Two-way cross-fitted isotonic calibration turns it into a probability p; the residual is r = y − p.
"""),
        code('''
from honest_ceiling.residual import calibrated_probability, whiteness_report, oracle_bound
lib = pd.read_csv(LIB / "oof_legs.csv")
p = calibrated_probability(lib["blend_v4_nested"].to_numpy(float), y)
r = y - p
print(f"nested blend AUC {fast_auc(y, lib['blend_v4_nested'].to_numpy()):.6f}   residual mean {r.mean():+.5f}   std {r.std():.4f}")
'''),
        md("""
## 3 · Is the residual white?
If the residual were correlated between rows in some ordering (id, income, commute, predicted probability), a filter
or a feature could exploit it. The ±2σ band of white noise at this size is ±0.0024.
"""),
        code('''
inc = train["Annual_Income_USD"].round().astype(int).to_numpy()
rep = whiteness_report(r, {"id": train["id"].to_numpy(), "income": inc, "commute": train["Daily_Commute_km"].to_numpy(),
                           "prediction": p}, {"exact income": inc})
print(pd.DataFrame(rep["acf"]).round(4).to_string())
print("\\nleave-one-out residual correlation among rows sharing an exact income:", round(rep["group_loo_corr"]["exact income"], 4))
'''),
        md("""
Everything sits inside or near the white-noise band. The small negative values in income order, and the negative
same-income correlation, have the sign cross-fitting produces (a row's label enters its same-income neighbours'
encodings in other folds): the income memory has been extracted, not left behind.

## 4 · The oracle bound
If p were the *true* probability of every row, the best achievable AUC is the AUC of p against labels drawn from p.
"""),
        code('''
ob = oracle_bound(p, reps=10)
print(f"oracle bound {ob['mean']:.5f} ± {ob['std']:.5f}   vs achieved {fast_auc(y, lib['blend_v4_nested'].to_numpy()):.5f}")
'''),
        md("""
## 5 · The Darwin-Cage search, live
Fitness: shrunk group mean of r fit on 8 search folds, correlated with r on the 9th, averaged over the 9 rotations.
Fold 9 is reserved. Threshold for a discovery: 3σ of the null on that fold, `3 / sqrt(66,866) = 0.0116`.
This cell runs a short search (set `MINUTES`); the full runs (47,679 programs) are loaded in section 6.
"""),
        code('''
from honest_ceiling.darwin import DarwinSearch, integer_columns, plant, program_key
MINUTES = 15
D = integer_columns(train)
search = DarwinSearch(D, r.astype(np.float64), fold)
null_mean, null_std = search.null(reps=10)
print(f"null fitness (random keys): {null_mean:+.5f} ± {null_std:.5f}   threshold on the reserved fold: {search.threshold:.4f}")
res = search.run(seconds=MINUTES * 60, seed=11)
print(f"evaluated {res['evaluated']:,} programs in {res['seconds']:.0f} s | hall of fame {res['n_hall']} | confirmed {len(res['confirmed'])}"
      f" | expected family-wise false positives {res['familywise_fp_expected']:.3f}")
for h in res["hall"][:8]:
    print(f"  search {h['fit_mean']:.4f}  reserved fold {h['confirm']:+.4f}  {h['program']}")
'''),
        md("## 6 · The full runs"),
        code('''
hall = []
for f in ("darwin_run1_hall.json", "darwin_run2_hall.json"):
    j = json.load(open(LIB / f))
    hall += [(h["fit_mean"], h.get("confirm_fold9", h.get("confirm")), h["program"]) for h in j["hall"]]
    print(f, j["report"]["evaluated"], "programs,", j["report"]["hall"], "hall entries,", j["report"]["confirmed"], "confirmed")
fig, ax = plt.subplots(figsize=(9, 4))
ax.axhline(0.012, color=ORANGE, ls="--", lw=1.5); ax.text(0.0061, 0.0126, "discovery threshold (3σ, reserved fold)", fontsize=9, color="#52514e")
ax.axhspan(-0.008, 0.008, color=GREY, alpha=0.25, lw=0)
ax.scatter([h[0] for h in hall], [h[1] for h in hall], s=46, color=BLUE, edgecolor="white", zorder=3)
ax.set_xlabel("search fitness (9 search folds)"); ax.set_ylabel("correlation on the never-seen fold")
ax.set_title("Real residual: nothing that looks strong on the search folds survives the reserved fold")
plt.show()
'''),
        md("""
## 7 · Power: would the search have found something?
Labels are redrawn from the real probabilities plus a planted per-group shift of known size d on a program the search
does not know about. Same search, same reserved fold. Below, one planted structure live, then the full power table.
"""),
        code('''
prog = [("mod", "Annual_Income_USD", 97)]
y_pl, shift = plant(p, D, prog, effect=0.016, seed=7)
gain = fast_auc(y_pl, p + shift) - fast_auc(y_pl, p)
s2 = DarwinSearch(D, (y_pl - p).astype(np.float64), fold)
t0 = time.time()
found = s2.run(seconds=15 * 60, seed=107, stop_on_confirm=True)
print(f"planted income mod 97, d = 0.016 (oracle AUC gain {gain:.5f}) -> confirmed {len(found['confirmed'])} after "
      f"{found['evaluated']:,} programs, {time.time()-t0:.0f} s")
for h in found["confirmed"][:3]:
    print("  ", h["program"], "reserved-fold correlation", round(h["confirm"], 4))
'''),
        code('''
rows = [json.load(open(f)) for f in sorted(glob.glob(str(LIB / "power_[ABC]_*.json")))]
tab = pd.DataFrame([{"plant": r_["plant"], "groups": r_["n_groups"], "effect d": r_["effect"], "oracle AUC gain": r_["oracle_auc_gain"],
                     "found": r_["detected"], "programs": r_["evaluated"]} for r_ in rows])
print(tab.sort_values(["plant", "effect d"]).to_string(index=False))
'''),
        md(f"""
## 8 · What this does and does not show

* **Does:** inside its search space (integer transforms of single columns and their joint keys, read out through
  group means) there is no structure left in the residual of an honest blend that could be confirmed on unseen rows.
  The power table bounds how small a missed structure would have to be.
* **Does not:** exclude structures outside that space. Candidates worth testing, each with an acceptance criterion,
  are listed in the repository's [ROADMAP]({REPO}/blob/main/ROADMAP.md): partial-pooling encodings of the income
  identity, non-mean readouts, relations between rows, learned or LLM-guided program proposals, and modelling the
  synthetic generator itself.
* **Why a negative result is worth publishing:** 47,679 programs is more feature engineering than any person tries by
  hand. Knowing that none of it survives a held-out fold saves everyone that time, and the harness is reusable on any
  tabular competition: replace the residual and run it.

If you take one of the open directions and beat these numbers honestly, open an issue in the repository: it will be
added with credit.
"""),
    ]
    nb = nbf.v4.new_notebook(); nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    nbf.write(pipeline_nb(), out / "01_fold_safe_pipeline.ipynb")
    nbf.write(darwin_nb(), out / "02_darwin_cage_residual.ipynb")
    print("written", sorted(p.name for p in out.glob("*.ipynb")))
