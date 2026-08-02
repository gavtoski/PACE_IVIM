#!/usr/bin/env python3
"""Compare released against retrained networks, side by side.

Answers one question: after retraining from scratch, does the ranking
still hold?

Reads two directories of inference results and reports signal RMSE and
per parameter RMSE for each, with the ranking in both. Conventional
methods have no weights to retrain, so they are carried across from the
released side unchanged and act as a fixed reference.

Usage
-----
    python scripts/compare_retrained.py
    python scripts/compare_retrained.py --snrs 25
    python scripts/compare_retrained.py --no-figure
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pace import common as C

NEURAL = ["pace", "cnn_fusion", "dnn"]
CONVENTIONAL = ["lsq", "nnls", "map"]
PARAMS = ["Dpar", "Dint", "Fint", "Dmv", "Fmv", "S0"]


def frames_for(results_dir, models, snrs, paths, seeds=None):
    """Signal and parameter RMSE frames read from one results directory.

    seeds restricts both sides to the same runs. Without it the released
    side would average five seeds at SNR 25 while a retrained side with
    only seed 19 averages one, and the two columns would describe
    different populations.
    """
    p = dataclasses.replace(paths, results_dir=Path(results_dir))
    available = {m for m, _ in C.discover_results(p)}
    models = [m for m in models if m in available]
    if not models:
        return None, None, []
    sig = C.build_signal_rmse_df(models, snrs, p, warn_nonphysical=False)
    par = C.build_param_rmse_df(models, snrs, paths=p)
    if seeds is not None:
        sig = sig[sig.seed.isin(seeds)]
        par = par[par.seed.isin(seeds)]
    return sig, par, models


def mean_by_model(df, column="rmse"):
    if df is None or df.empty:
        return {}
    return df.groupby("model")[column].mean().to_dict()


def print_ranking(metric, released, retrained, fmt="{:.5f}"):
    """Rank all six methods, best first, after retraining.

    Neural models take their retrained value, conventional methods their
    released one, since they have no weights to retrain. This is the
    ranking a reader would see having retrained everything that can be
    retrained.
    """
    combined, source = {}, {}
    for m, v in released.items():
        combined[m] = v
        source[m] = "conventional" if m in CONVENTIONAL else "released"
    for m, v in retrained.items():
        combined[m] = v
        source[m] = "retrained"

    order = sorted(combined, key=combined.get)
    print()
    print(f"  {metric}")
    print(f"  {'#':<3} {'method':<12} {'RMSE':>12} {'source':<13} {'vs released':>12}")
    print("  " + "-" * 57)
    for i, m in enumerate(order, 1):
        v = combined[m]
        if m in retrained and m in released:
            a = released[m]
            delta = f"{(v - a) / a:>11.1%}"
        else:
            delta = f"{'':>12}"
        print(f"  {i:<3} {C.label_of(m):<12} {fmt.format(v):>12} "
              f"{source[m]:<13}{delta}")

    top2 = order[:2]
    lead = set(top2) == {"pace", "cnn_fusion"}
    print(f"      top two: {', '.join(C.label_of(m) for m in top2)}"
          f"   {'PACE and CNN Fusion lead' if lead else 'note: not both leaders'}")
    return order


def figure(sig_rel, sig_ret, par_rel, par_ret, paths):
    """Grouped bars, released beside retrained, for each metric."""
    fig, axes = plt.subplots(1, 1 + len(PARAMS), figsize=(22, 4.5))

    panels = [("Signal RMSE", sig_rel, sig_ret)]
    for p in PARAMS:
        panels.append((C.PARAM_LATEX.get(p, p),
                       mean_by_model(par_rel[par_rel.param == p])
                       if par_rel is not None and not par_rel.empty else {},
                       mean_by_model(par_ret[par_ret.param == p])
                       if par_ret is not None and not par_ret.empty else {}))

    for ax, (title, rel, ret) in zip(axes, panels):
        models = C.ordered_models(set(rel) | set(ret))
        x = np.arange(len(models))
        w = 0.38

        ax.bar(x - w / 2, [rel.get(m, 0) for m in models], w,
               color=[C.color_of(m) for m in models],
               edgecolor="white", linewidth=0.6, alpha=0.95, label="released")
        ax.bar(x + w / 2, [ret.get(m, np.nan) for m in models], w,
               color=[C.color_of(m) for m in models],
               edgecolor="black", linewidth=0.8, alpha=0.55, hatch="///",
               label="retrained")

        ax.set_xticks(x)
        ax.set_xticklabels([C.label_of(m) for m in models], rotation=40,
                           ha="right", fontsize=8)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", labelsize=8)
        C.clean_spines(ax)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04), fontsize=11)
    fig.suptitle("Released against retrained. Solid bars are the released "
                 "weights, hatched bars retrained from scratch.",
                 fontsize=12, y=1.10)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(
        description="Compare released against retrained networks.")
    ap.add_argument("--released", default="results/synthetic")
    ap.add_argument("--retrained", default="results/synthetic_from_retrained")
    ap.add_argument("--snrs", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None,
                    help="Restrict both sides to these seeds. Defaults to "
                         "the seeds present on the retrained side, so the "
                         "two columns cover the same runs.")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args()

    paths = C.load_paths()
    rel_dir = paths.repo_root / args.released
    ret_dir = paths.repo_root / args.retrained

    for d, label in [(rel_dir, "--released"), (ret_dir, "--retrained")]:
        if not d.is_dir():
            print(f"[ERROR] {label} directory not found: {d}", file=sys.stderr)
            return 2

    snrs = args.snrs or sorted({s for _, s in C.discover_results(
        dataclasses.replace(paths, results_dir=ret_dir))})

    print("=" * 60)
    print(" Released against retrained")
    print("=" * 60)
    print(f" released  {args.released}")
    print(f" retrained {args.retrained}")
    print(f" SNR       {snrs}")

    # Default to whatever seeds the retrained side actually has, so the
    # comparison is like for like rather than five seeds against one.
    ret_paths = dataclasses.replace(paths, results_dir=ret_dir)
    seeds = args.seeds or sorted({sd for seeds_at in
                                  C.discover_results(ret_paths).values()
                                  for sd in seeds_at})
    print(f" seeds     {seeds}")

    sig_rel, par_rel, all_rel = frames_for(
        rel_dir, NEURAL + CONVENTIONAL, snrs, paths, seeds)
    sig_ret, par_ret, all_ret = frames_for(ret_dir, NEURAL, snrs, paths, seeds)

    if not all_ret:
        print(f"\n[ERROR] no retrained results under {ret_dir}",
              file=sys.stderr)
        print("        Generate them first:", file=sys.stderr)
        print("          python scripts/reproduce_synthetic_results.py \\\\",
              file=sys.stderr)
        print("            --checkpoints checkpoints/synthetic_retrained \\\\",
              file=sys.stderr)
        print(f"            --out {args.retrained}", file=sys.stderr)
        return 2

    print(f" models    released {all_rel}")
    print(f"           retrained {all_ret}")

    if sig_rel is not None and sig_ret is not None:
        n_rel = sig_rel.groupby("model").size().to_dict()
        n_ret = sig_ret.groupby("model").size().to_dict()
        shared = sorted(set(n_rel) & set(n_ret))
        bad = [m for m in shared if n_rel[m] != n_ret[m]]
        print(f" runs      released {[n_rel[m] for m in shared]} "
              f"retrained {[n_ret[m] for m in shared]} for {shared}")
        if bad:
            print(f" [WARN] unequal run counts for {bad}; the two columns "
                  f"do not cover the same population")

    s_rel, s_ret = mean_by_model(sig_rel), mean_by_model(sig_ret)

    print()
    print("=" * 60)
    print(" Ranking after retraining, best first")
    print("=" * 60)
    print(" Neural models use their retrained value. Conventional methods")
    print(" have no weights to retrain and carry their released value.")

    metrics = [("Signal RMSE", s_rel, s_ret, "{:.5f}")]
    for p in PARAMS:
        metrics.append((
            f"{p} RMSE",
            mean_by_model(par_rel[par_rel.param == p]) if par_rel is not None else {},
            mean_by_model(par_ret[par_ret.param == p]) if par_ret is not None else {},
            "{:.3e}" if p.startswith("D") else "{:.5f}"))

    orders = {}
    for name, rel, ret, fmt in metrics:
        if rel:
            orders[name] = print_ranking(name, rel, ret, fmt)

    print()
    print("=" * 60)
    print(" Summary: position of each method, 1 is best")
    print("=" * 60)
    names = [n for n, _, _, _ in metrics if n in orders]
    header = "".join(f"{n.split()[0]:>8}" for n in names)
    print(f"  {'method':<12}{header}")
    print("  " + "-" * (12 + 8 * len(names)))
    for m in C.ordered_models(set().union(*orders.values())):
        cells = "".join(f"{orders[n].index(m) + 1:>8}" if m in orders[n]
                        else f"{'-':>8}" for n in names)
        print(f"  {C.label_of(m):<12}{cells}")

    if not args.no_figure:
        fig = figure(s_rel, s_ret, par_rel, par_ret, paths)
        written = C.save_figure(fig, "released_vs_retrained", paths,
                                ["png"], dpi=200)
        plt.close(fig)
        print()
        for w in written:
            print(f" figure: {w.relative_to(paths.repo_root)}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
