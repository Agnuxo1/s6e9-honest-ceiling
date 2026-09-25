"""Render every figure of the report from results/*.json, in light and dark variants (PNG + SVG).

Palette: validated categorical slots (blue, orange, aqua; all-pairs CVD-safe in both modes), recessive grid,
text in text tokens, direct labels instead of colour-only identity.
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
FIG.mkdir(exist_ok=True)

THEMES = {
    "light": dict(surface="#fcfcfb", text="#0b0b0b", text2="#52514e", muted="#8a8984", grid="#e6e5e1",
                  s1="#2a78d6", s2="#eb6834", s3="#1baf7a", neutral="#b9b8b2"),
    "dark": dict(surface="#1a1a19", text="#ffffff", text2="#c3c2b7", muted="#8f8e86", grid="#34342f",
                 s1="#3987e5", s2="#d95926", s3="#199e70", neutral="#5b5a55"),
}


def style(ax, t, title, subtitle=None):
    ax.set_facecolor(t["surface"])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], labelsize=9, length=0)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=t["text"], fontsize=12.5, fontweight="semibold", pad=22 if subtitle else 10)
    if subtitle:
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, color=t["text2"], fontsize=9.5, va="bottom")
    ax.xaxis.label.set_color(t["text2"]); ax.yaxis.label.set_color(t["text2"])


SITE = ROOT / "docs" / "figures"   # GitHub Pages serves docs/ only
SITE.mkdir(parents=True, exist_ok=True)


def save(fig, name, mode):
    for ext in ("png", "svg"):
        fig.savefig(FIG / f"{name}_{mode}.{ext}", dpi=180, facecolor=fig.get_facecolor(), bbox_inches="tight")
    fig.savefig(SITE / f"{name}_{mode}.png", dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def fig_leaderboard(mode):
    t = THEMES[mode]
    lb = json.load(open(RES / "leaderboard_public_20260924.json"))["rows"]
    s = np.array([r["score"] for r in lb])
    body = s[s < 0.9480]
    fig, ax = plt.subplots(figsize=(8.2, 3.9), facecolor=t["surface"])
    vals, counts = np.unique(np.round(body, 5), return_counts=True)
    ax.bar(vals, counts, width=0.0000085, color=t["s1"], linewidth=0)
    for x, lab, dy, ha in ((0.94659, "our public anchor\n0.94659 (rank 28)", 42, "left"), (0.94638, "our honest blend\n0.94638 (v4)", 42, "left")):
        ax.axvline(x, color=t["text2"], linewidth=1, linestyle=(0, (3, 3)))
        ax.text(x + 0.0000012, dy, lab, color=t["text2"], fontsize=8.5, va="bottom", ha=ha)
    ax.text(0.99, 0.95, "rank 1 at 0.94945 is off this scale\n(public-split probing; see report)", transform=ax.transAxes,
            ha="right", va="top", color=t["text2"], fontsize=8.5)
    ax.set_ylim(0, 78)
    ax.set_xlabel("public ROC-AUC (top 200 teams, 2026-09-24)")
    ax.set_ylabel("teams")
    ax.ticklabel_format(axis="x", useOffset=False)
    style(ax, t, "The public top 200 sits inside 0.0003 of AUC", "141 teams share 0.94655-0.94657; one public positive row is worth ~0.00002")
    save(fig, "leaderboard", mode)


def fig_ladder(mode):
    t = THEMES[mode]
    r = json.load(open(RES / "residual_and_ceiling.json"))
    L = r["ladder"]
    legs = json.load(open(RES / "legs.json"))
    xgb = np.mean([v["oof_auc"] for k, v in legs.items() if k.startswith("final_xgb")])
    rows = [("generator recipe, no training", L["recipe_only_no_training"], t["neutral"]),
            ("LightGBM on raw columns", L["raw_lightgbm_no_identity"], t["neutral"]),
            ("one fold-safe XGBoost run (mean of 9 runs)", xgb, t["neutral"]),
            ("XGBoost averaged over 3 splits × 3 seeds", json.load(open(RES / "legs_summary.json"))["xgb_x9"]["oof_auc_rank_average"], t["neutral"]),
            ("public six-view ensemble (CC0)", L["six_view_ensemble_public"], t["neutral"]),
            ("our nested blend v4", L["fresh_blend_v4_nested"], t["s1"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0), facecolor=t["surface"], gridspec_kw=dict(width_ratios=[1, 1.25]))
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    ax.scatter([v for _, v, _ in rows], y, s=64, c=[c for *_, c in rows], zorder=3, edgecolors=t["surface"], linewidths=2)
    for yi, (lab, v, _) in zip(y, rows):
        ax.text(0.9368, yi + 0.28, f"{lab}  {v:.5f}", color=t["text"], fontsize=8.5)
    ax.set_yticks([]); ax.set_xlim(0.9365, 0.9475); ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlabel("out-of-fold ROC-AUC (668,665 rows)")
    style(ax, t, "Where the AUC comes from")
    ax.grid(axis="x", color=t["grid"], linewidth=0.8); ax.grid(axis="y", visible=False)
    ax = axes[1]
    o = r["oracle_auc"]
    ax.axvspan(o["mean"] - 2 * o["std"], o["mean"] + 2 * o["std"], color=t["s2"], alpha=0.16, linewidth=0)
    ax.axvline(o["mean"], color=t["s2"], linewidth=2)
    ax.text(o["mean"] + 0.000004, 0.84, f"oracle bound\n{o['mean']:.5f} ± {2*o['std']:.5f} (2σ)", color=t["text"], fontsize=8.5, transform=ax.get_xaxis_transform())
    for v, lab, yy in ((L["six_view_ensemble_public"], "six-view ensemble", 0.30), (L["fresh_blend_v2_nested"], "blend v2", 0.48), (L["fresh_blend_v4_nested"], "blend v4", 0.66)):
        ax.scatter([v], [yy], s=64, color=t["s1"] if "v4" in lab else t["neutral"], zorder=3, edgecolors=t["surface"], linewidths=2, transform=ax.get_xaxis_transform())
        ax.text(v, yy + 0.07, f"{lab} {v:.6f}", color=t["text"], fontsize=8.5, ha="center", transform=ax.get_xaxis_transform())
    ax.set_xlim(0.94620, 0.94725); ax.set_yticks([])
    ax.ticklabel_format(axis="x", useOffset=False)
    ax.set_xlabel("ROC-AUC, zoomed")
    style(ax, t, "Distance to our own ceiling")
    ax.grid(axis="x", color=t["grid"], linewidth=0.8); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "ceiling", mode)


def fig_folds(mode):
    t = THEMES[mode]
    r = json.load(open(RES / "residual_and_ceiling.json"))
    d = np.array(r["per_fold_v4_minus_six"]) * 1e5
    pb = r["paired_private_size_v4_vs_six"]
    fig, ax = plt.subplots(figsize=(8.2, 3.5), facecolor=t["surface"])
    ax.bar(np.arange(10), d, width=0.62, color=t["s1"], linewidth=0)
    for i, v in enumerate(d):
        ax.text(i, v + 0.25, f"+{v:.1f}", ha="center", color=t["text2"], fontsize=8.5)
    ax.axhline(0, color=t["text2"], linewidth=1)
    ax.set_xticks(range(10), [f"fold {k}" for k in range(10)])
    ax.set_ylabel("AUC gain × 10⁻⁵")
    style(ax, t, "Nested blend v4 vs the public six-view ensemble: 10 of 10 folds",
          f"weights fit on 9 folds, scored on the 10th; private-size bootstrap +{pb['mean']*1e5:.1f} ± {pb['std']*1e5:.1f} ×10⁻⁵, P(better) = {pb['p_a_better']:.2f}")
    save(fig, "folds", mode)


def fig_weights(mode):
    t = THEMES[mode]
    w = json.load(open(RES / "blend_v4_report.json"))["weights"]
    pretty = {"own:final_lgbmx9": "own LightGBM (3 splits × 3 seeds)", "own:final_xgbx9": "own XGBoost (3 splits × 3 seeds)",
              "six:A": "public view A (LightGBM)", "six:C": "public view C", "six:D": "public view D (no exact key)",
              "six:F": "public view F (init-score)", "six:G_realmlp10": "public RealMLP-G", "six:B": "public view B",
              "six:E": "public view E", "own:final_realmlpx1": "own RealMLP", "own:final_mlpx2": "own MLP",
              "own:final_logregx1": "own logistic regression", "own:final_ladderxgbx1": "own ladder-view XGBoost"}
    items = sorted(w.items(), key=lambda kv: kv[1])
    fig, ax = plt.subplots(figsize=(8.2, 4.4), facecolor=t["surface"])
    y = np.arange(len(items))
    cols = [t["s1"] if k.startswith("own") else t["neutral"] for k, _ in items]
    ax.barh(y, [v for _, v in items], height=0.62, color=cols, linewidth=0)
    for yi, (k, v) in zip(y, items):
        ax.text(v + 0.004, yi, f"{pretty.get(k, k)}  {v:.3f}", va="center", color=t["text"], fontsize=8.5)
    ax.set_yticks([]); ax.set_xlim(0, 0.42)
    ax.set_xlabel("hill-climb weight (rank space)")
    style(ax, t, "Final blend weights: own legs 28 %, public CC0 library 72 %", "blue = trained here; grey = public OOF library on the same folds; zero-weight legs listed for completeness")
    ax.grid(axis="x", color=t["grid"], linewidth=0.8); ax.grid(axis="y", visible=False)
    save(fig, "weights", mode)


def fig_residual(mode):
    t = THEMES[mode]
    r = json.load(open(RES / "residual_and_ceiling.json"))
    acf = r["acf"]
    band = r["acf_null_2sigma"]
    fig, ax = plt.subplots(figsize=(8.2, 3.6), facecolor=t["surface"])
    ax.axhspan(-band, band, color=t["neutral"], alpha=0.25, linewidth=0)
    ax.text(-0.45, band + 0.0003, "±2σ band of white noise", color=t["text2"], fontsize=8.5, va="bottom", ha="left")
    lags = list(acf["id"].keys())
    x = np.arange(len(lags))
    colors = [t["s1"], t["s2"], t["s3"], t["text2"]]
    for (name, vals), c, off in zip(acf.items(), colors, (-0.18, -0.06, 0.06, 0.18)):
        v = [vals[l] for l in lags]
        ax.plot(x + off, v, "o-", color=c, linewidth=2, markersize=6, markeredgecolor=t["surface"], markeredgewidth=1.5)
        ax.text(x[-1] + off + 0.12, v[-1], f"by {name}", color=t["text"], fontsize=8.5, va="center")
    ax.axhline(0, color=t["text2"], linewidth=1)
    ax.set_xticks(x, [f"lag {l}" for l in lags]); ax.set_xlim(-0.5, len(lags) + 0.6)
    ax.set_ylim(-0.012, 0.012)
    ax.set_ylabel("autocorrelation of y − p")
    g = r["group_loo_corr"]["income_exact"]
    style(ax, t, "No exploitable structure left in the residual",
          f"only departure: income order (negative), the expected sign of cross-fitting; same-income leave-one-out correlation {g:+.3f}")
    save(fig, "residual", mode)


def fig_darwin(mode):
    t = THEMES[mode]
    hall = json.load(open(RES / "darwin_run1_hall.json"))["hall"] + json.load(open(RES / "darwin_run2_hall.json"))["hall"]
    seen, pts = set(), []
    for h in hall:
        k = (round(h["fit_mean"], 6), round(h.get("confirm_fold9", h.get("confirm")), 6))
        if k not in seen:
            seen.add(k); pts.append((h["fit_mean"], h.get("confirm_fold9", h.get("confirm"))))
    power = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "power" / "*.json")))]
    fig, ax = plt.subplots(figsize=(8.2, 4.2), facecolor=t["surface"])
    thr = 0.012
    ax.axhline(thr, color=t["s2"], linewidth=1.5, linestyle=(0, (4, 3)))
    ax.text(0.0205, thr - 0.0012, "confirmation threshold\n(3σ on the reserved fold)", color=t["text2"], fontsize=8.5, ha="right", va="top")
    ax.axhspan(-thr / 3 * 2, thr / 3 * 2, color=t["neutral"], alpha=0.22, linewidth=0)
    ax.scatter(*zip(*pts), s=46, color=t["s1"], edgecolors=t["surface"], linewidths=1.5, zorder=3)
    ax.text(pts[0][0] + 0.0002, pts[0][1] - 0.0045, f"real residual: 27 hall-of-fame entries ({len(pts)} distinct programs)\nfrom 47,679 evaluated; 0 confirmed", color=t["text"], fontsize=8.5)
    det = [(p["first_confirmed"]["fit_mean"], p["first_confirmed"]["confirm"], p) for p in power if p["detected"]]
    if det:
        ax.scatter([d[0] for d in det], [d[1] for d in det], s=64, marker="D", color=t["s3"], edgecolors=t["surface"], linewidths=1.5, zorder=4)
        for fx, fy, p in det:
            below = p["plant"] == "C"
            ax.text(fx, fy + (-0.0011 if below else 0.0011), f"planted {p['plant']} ({p['effect']})", color=t["text"], fontsize=8,
                    va="top" if below else "bottom", ha="center")
    ax.set_xlim(0.0055, 0.0210)
    ax.set_xlabel("search fitness (mean out-of-fold correlation on 9 search folds)")
    ax.set_ylabel("correlation on the never-seen fold")
    style(ax, t, "Darwin-Cage search: what survives the reserved fold", "blue = real residual (nothing survives); aqua diamonds = synthetic structures planted to test the search (found)")
    save(fig, "darwin", mode)


def fig_power(mode):
    t = THEMES[mode]
    power = [json.load(open(f)) for f in sorted(glob.glob(str(RES / "power" / "*.json")))]
    fig, ax = plt.subplots(figsize=(8.2, 3.7), facecolor=t["surface"])
    markers = {"A": ("o", t["s1"]), "B": ("s", t["s2"]), "C": ("D", t["s3"])}
    labels = {"A": "A: income mod 97 (97 groups)", "B": "B: income hundreds digit × city (30)", "C": "C: commute mod 37 × age mod 7 (259)"}
    for plant_id in ("A", "B", "C"):
        pp = sorted([p for p in power if p["plant"] == plant_id], key=lambda p: p["effect"])
        if not pp:
            continue
        m, c = markers[plant_id]
        xs = [max(p["oracle_auc_gain"], 1e-6) for p in pp]
        ys = [1 if p["detected"] else 0 for p in pp]
        yoff = {"A": 0.06, "B": 0.0, "C": -0.06}[plant_id]
        ax.plot(xs, np.array(ys) + yoff, m + "-", color=c, linewidth=2, markersize=8, markeredgecolor=t["surface"], markeredgewidth=1.5)
        ax.text(xs[-1] * 1.12, ys[-1] + yoff, labels[plant_id], color=t["text"], fontsize=8.5, va="center")
    ax.set_xscale("log")
    ax.set_yticks([0, 1], ["missed", "found"])
    ax.set_ylim(-0.4, 1.4)
    ax.set_xlabel("AUC an oracle would gain from the planted structure (log scale)")
    style(ax, t, "Power analysis: how large a hidden structure the search can find",
          "labels drawn from the real calibrated probabilities plus a planted per-group shift; 30-minute budget per run")
    save(fig, "power", mode)


if __name__ == "__main__":
    for mode in THEMES:
        for f in (fig_leaderboard, fig_ladder, fig_folds, fig_weights, fig_residual, fig_darwin, fig_power):
            f(mode)
    print(sorted(p.name for p in FIG.glob("*.png")))
