#!/usr/bin/env python3
"""Generate the synthetic evaluation figures.

Reads the committed inference results under results/synthetic and writes
figures to figures/. Needs no GPU and no torch.

Figures
-------
1    Signal fit quality      1x3: RMSE against SNR, ranked mean, residual
                             against b-value
2.1  Parameter RMSE          2x3 bar grid, fixed model order
2.2  Bland Altman            3x5, method against ground truth
3.1  Lesion CNR              2x5, predicted CNR and CNR RMSE against SNR

Conventions follow the analysis notebook: triangles with dashed
connectors for the conventional fitters, circles with solid connectors
for the learned models, linear axes anchored at zero, and a fixed model
order with the proposed methods first.

Usage
-----
    python scripts/make_manuscript_figures.py
    python scripts/make_manuscript_figures.py --figures 1 2.1
    python scripts/make_manuscript_figures.py --snrs 25 30 35
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pace import common as C


# Font sizes for Figure 1.
FS_TITLE, FS_LABEL, FS_TICK = 15, 14, 12
FS_LEGEND, FS_ANNOT, FS_PANEL = 11, 11, 18
FS_TITLE_PAD = 10

# Output resolution per figure, matching FIG1_DPI, FIG2_DPI, FIG22_DPI
# and FIG3_DPI in the analysis. Figure 1 alone is saved at 600.
DPI = {"fig1": 600, "fig2_1": 300, "fig2_2": 300, "fig3_1": 300}

# Figure 2.2.
BA_SNR, BA_SEED_INDEX = 25, 0
BA_N_PLOT, BA_ALPHA, BA_SIZE = 3000, 0.20, 5

# Parameters carrying tissue contrast. S0 is a scaling term.
CNR_PARAMS = ["Dpar", "Dint", "Fint", "Dmv", "Fmv"]

TOP_PARAMS = ["Dpar", "Dint", "Fint"]
BOTTOM_PARAMS = ["Dmv", "Fmv", "S0"]


def _plot_models(df, models=None):
    """Models present in a frame, in the canonical plot order."""
    if df is None or df.empty:
        return []
    have = set(df["model"].unique())
    return [m for m in C.ordered_models(have)
            if models is None or m in models]


# ===========================================================
# Figure 1
# ===========================================================

def _signal_stats(df, model, snr):
    sub = df[(df.model == model) & (df.snr == snr)]["rmse"]
    if sub.empty:
        return np.nan, 0.0, 0
    return float(sub.mean()), float(sub.std()), len(sub)


def _residual_by_bvalue(model, snrs, paths, tissues=None):
    """Mean signal residual against b-value, pooled over tissue, SNR, seed.

    One residual vector is accumulated per (SNR, seed, tissue) group, then
    averaged, then collapsed onto unique b-values because the protocol
    repeats several acquisitions.
    """
    tissues = tissues or list(C.TISSUE_ORDER)
    found = C.discover_results(paths)
    per_config, bvals = [], None

    for snr in snrs:
        for seed in sorted(found.get((model, snr), {})):
            pred = C.load_result(model, snr, seed, paths)
            if not pred.has_gt or pred.tissue is None:
                continue
            bvals = pred.bvals
            sig, ref = pred.signal(), pred.gt_signal()
            labels, ok = pred.labels(), pred.valid()
            for t in tissues:
                tm = (labels == t) & ok
                if tm.sum() < 5:
                    continue
                per_config.append(np.mean(sig[tm] - ref[tm], axis=0))

    if not per_config or bvals is None:
        return None, None

    mean_res = np.mean(np.stack(per_config, axis=0), axis=0)
    b_unique = np.unique(bvals)
    collapsed = np.array([np.mean(mean_res[bvals == b]) for b in b_unique])
    return b_unique, collapsed


def figure1(df_sig, snrs, models, paths, show_residual=True):
    ratios = [1.0, 0.8, 1.0] if show_residual else [1.0, 1.0]
    fig, axarr = plt.subplots(1, len(ratios), figsize=(18, 5.5),
                              gridspec_kw={"width_ratios": ratios,
                                           "wspace": 0.32})
    if show_residual:
        ax_line, ax_bar, ax_curve = axarr
    else:
        ax_line, ax_bar = axarr
        ax_curve = None

    # ---- a: RMSE against SNR ----
    for m in models:
        xs, mu, sd, ns = [], [], [], []
        for snr in snrs:
            a, b, n = _signal_stats(df_sig, m, snr)
            if np.isfinite(a):
                xs.append(snr); mu.append(a); sd.append(b); ns.append(n)
        if not xs:
            continue
        xs_a, mu_a = np.array(xs, float), np.array(mu)
        yerr = np.where(np.array(ns) > 1, np.array(sd), 0.0)
        col = C.color_of(m)
        if len(xs_a) >= 2:
            ax_line.plot(xs_a, mu_a, color=col, linestyle=C.linestyle_of(m),
                         lw=1.0, alpha=0.45, zorder=3)
        ax_line.errorbar(xs_a, mu_a, yerr=yerr,
                         fmt=C.marker_of(m), markersize=C.marker_size_of(m),
                         capsize=3, color=col, lw=0, elinewidth=1.2,
                         label=C.label_of(m), zorder=5)

    ax_line.set_xlabel("SNR (dB)", fontsize=FS_LABEL)
    ax_line.set_ylabel("Signal RMSE", fontsize=FS_LABEL)
    ax_line.set_title("Signal RMSE\n vs Noise Level", fontsize=FS_TITLE,
                      fontweight="bold", pad=FS_TITLE_PAD)
    ax_line.set_xticks(sorted(snrs))
    ax_line.set_ylim(bottom=0)
    ax_line.tick_params(axis="both", labelsize=FS_TICK)
    C.clean_spines(ax_line)

    # ---- b: ranked mean over the same SNR pool ----
    bars = []
    for m in models:
        sub = df_sig[(df_sig.model == m) & (df_sig.snr.isin(snrs))]["rmse"]
        if sub.empty:
            continue
        bars.append({"model": m, "mean": float(sub.mean()),
                     "std": float(sub.std()), "n": len(sub)})
    bars.sort(key=lambda d: d["mean"])

    x = np.arange(len(bars))
    h = [d["mean"] for d in bars]
    e = [d["std"] if d["n"] > 1 else 0.0 for d in bars]
    ax_bar.bar(x, h, width=0.65, yerr=e, capsize=3,
               color=[C.color_of(d["model"]) for d in bars],
               edgecolor="white", linewidth=0.6, alpha=0.88)
    y_top = max((a + b for a, b in zip(h, e)), default=1.0)
    for i, (a, b) in enumerate(zip(h, e)):
        ax_bar.text(i, a + b + y_top * 0.02, f"{a:.4f}", ha="center",
                    va="bottom", fontsize=FS_ANNOT, color="dimgray")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([C.label_of(d["model"]) for d in bars],
                           fontsize=FS_TICK, rotation=40, ha="right")
    ax_bar.set_ylabel("Signal RMSE", fontsize=FS_LABEL)
    ax_bar.set_title("Averaged Signal RMSE\nover SNR range", fontsize=FS_TITLE,
                     fontweight="bold", pad=FS_TITLE_PAD)
    ax_bar.set_ylim(bottom=0, top=y_top * 1.12)
    ax_bar.tick_params(axis="y", labelsize=FS_TICK)
    C.clean_spines(ax_bar)

    # ---- c: residual against b-value ----
    if ax_curve is not None:
        for m in models:
            b_unique, res = _residual_by_bvalue(m, snrs, paths)
            if b_unique is None:
                continue
            col = C.color_of(m)
            if len(b_unique) >= 2:
                ax_curve.plot(b_unique, res, color=col,
                              linestyle=C.linestyle_of(m), lw=1.0,
                              alpha=0.45, zorder=3)
            ax_curve.plot(b_unique, res, marker=C.marker_of(m),
                          markersize=C.marker_size_of(m), color=col,
                          linestyle="None", label=C.label_of(m), zorder=5)
        ax_curve.axhline(0, color="black", ls="-", lw=1.0, alpha=0.4, zorder=1)
        ax_curve.set_xlabel("b-value (s/mm$^2$)", fontsize=FS_LABEL)
        ax_curve.set_ylabel("Signal Residual (Pred \u2212 GT)",
                            fontsize=FS_LABEL)
        ax_curve.set_title("Averaged Signal Residual\nvs b-value",
                           fontsize=FS_TITLE, fontweight="bold",
                           pad=FS_TITLE_PAD)
        ax_curve.tick_params(axis="both", labelsize=FS_TICK)
        C.clean_spines(ax_curve)

    panels = [ax_line, ax_bar] + ([ax_curve] if ax_curve is not None else [])
    for ax, letter in zip(panels, ["a)", "b)", "c)"]):
        ax.annotate(letter, xy=(0.0, 1.0), xycoords="axes fraction",
                    xytext=(-10, 16), textcoords="offset points",
                    fontsize=FS_PANEL, fontweight="bold", color="black",
                    ha="left", va="bottom", annotation_clip=False)

    handles, labels = ax_line.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.15), ncol=6, fontsize=FS_LEGEND,
               frameon=False, handletextpad=0.4, columnspacing=1.4)
    with warnings.catch_warnings():
        # The panel letters sit outside their axes, which tight_layout
        # reports as incompatible. The rect argument already reserves the
        # margin, so the result is correct.
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout(rect=(0, 0.09, 1, 1))
    return fig, bars


# ===========================================================
# Figure 2.1
# ===========================================================

def _param_bar_data(df, param, models, snrs):
    """Collapse tissues within each (SNR, seed), then average over those.

    Matching the notebook matters here. Averaging every row at once would
    weight each (SNR, seed) group by how many tissues survived the minimum
    voxel count, and would take a standard deviation over the wrong
    population.
    """
    rows = []
    for m in models:
        sub = df[(df.model == m) & (df.param == param) & (df.snr.isin(snrs))]
        if sub.empty:
            continue
        seed_means = sub.groupby(["snr", "seed"])["rmse"].mean()
        rows.append({
            "model": m,
            "mean": float(seed_means.mean()),
            "std": float(seed_means.std()) if len(seed_means) > 1 else 0.0,
            "n": len(seed_means),
        })
    return rows


def figure21(df_par, snrs, models, include_s0=True):
    bottom = BOTTOM_PARAMS if include_s0 else BOTTOM_PARAMS[:-1]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for r, plist in enumerate([TOP_PARAMS, bottom]):
        for c in range(3):
            ax = axes[r, c]
            if c >= len(plist):
                ax.set_visible(False)
                continue
            p = plist[c]
            bars = _param_bar_data(df_par, p, models, snrs)
            if not bars:
                ax.text(0.5, 0.5, f"No data for {p}", transform=ax.transAxes,
                        ha="center", fontsize=10)
                continue

            x = np.arange(len(bars))
            h = [d["mean"] for d in bars]
            e = [d["std"] for d in bars]
            ax.bar(x, h, width=0.65, yerr=e, capsize=3,
                   color=[C.color_of(d["model"]) for d in bars],
                   edgecolor="white", linewidth=0.6, alpha=0.88)

            y_top = max(a + b for a, b in zip(h, e))
            for i, (a, b) in enumerate(zip(h, e)):
                txt = f"{a:.2e}" if p.startswith("D") else f"{a:.4f}"
                ax.text(i, a + b + y_top * 0.03, txt, ha="center",
                        va="bottom", fontsize=6.5, color="dimgray")

            ax.set_xticks(x)
            ax.set_xticklabels([C.label_of(d["model"]) for d in bars],
                               fontsize=8.5, rotation=30, ha="right")
            ax.set_ylim(bottom=0, top=y_top * 1.18)
            ax.set_title(C.PARAM_LATEX.get(p, p), fontsize=13,
                         fontweight="bold")
            if c == 0:
                ax.set_ylabel("Parameter RMSE", fontsize=10)
            C.clean_spines(ax)

    fig.suptitle(f"Per Parameter RMSE (SNR {min(snrs)}\u2013{max(snrs)} dB, "
                 f"All Tissues Pooled)", fontsize=14, fontweight="bold",
                 y=1.01)
    fig.tight_layout()
    return fig


# ===========================================================
# Figure 2.2
# ===========================================================

def figure22(methods, params, snr, paths, seed_index=0):
    nrow, ncol = len(methods), len(params)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.4 * nrow),
                             squeeze=False)
    rng = np.random.default_rng(99)
    found = C.discover_results(paths)

    for r, m in enumerate(methods):
        seeds = sorted(found.get((m, snr), {}))
        if seed_index >= len(seeds):
            for c in range(ncol):
                axes[r, c].text(0.5, 0.5, "No data",
                                transform=axes[r, c].transAxes,
                                ha="center", fontsize=9, color="gray")
            continue

        pred = C.load_result(m, snr, seeds[seed_index], paths)
        col = C.color_of(m)
        dark = C.darken_hex(col, 0.4)
        labels = pred.labels() if pred.tissue is not None else None

        for c, p in enumerate(params):
            ax = axes[r, c]
            a = pred.params[p].astype(np.float64)
            b = pred.gt[p].astype(np.float64)
            valid = np.isfinite(a) & np.isfinite(b) & (a != 0)
            if labels is not None:
                valid &= np.isin(labels, C.TISSUE_ORDER)
            a, b = a[valid], b[valid]
            if len(a) < 10:
                continue

            mean_v, diff_v = (a + b) / 2.0, a - b
            bias = float(np.mean(diff_v))
            sd = float(np.std(diff_v))
            lo, hi = bias - 1.96 * sd, bias + 1.96 * sd

            idx = np.arange(len(mean_v))
            if len(idx) > BA_N_PLOT:
                idx = rng.choice(idx, size=BA_N_PLOT, replace=False)
            ax.scatter(mean_v[idx], diff_v[idx], s=BA_SIZE, alpha=BA_ALPHA,
                       color=col, edgecolors="none", rasterized=True, zorder=2)

            ax.axhline(bias, color=dark, ls="-", lw=3.0, zorder=5)
            ax.axhline(hi, color=dark, ls="-", lw=1.0, zorder=5)
            ax.axhline(lo, color=dark, ls="-", lw=1.0, zorder=5)
            ax.axhline(0, color="gray", ls=":", lw=0.7, alpha=0.5, zorder=4)

            ax.text(0.98, 0.97, f"Bias={bias:.3g}", transform=ax.transAxes,
                    fontsize=7, color=dark, ha="right", va="top",
                    fontweight="bold")
            ax.text(0.98, 0.88, f"LOA+={hi:.3g}", transform=ax.transAxes,
                    fontsize=6.5, color=dark, ha="right", va="top")
            ax.text(0.98, 0.80, f"LOA\u2212={lo:.3g}", transform=ax.transAxes,
                    fontsize=6.5, color=dark, ha="right", va="top")

            if r == 0:
                ax.set_title(C.PARAM_LATEX.get(p, p), fontsize=12,
                             fontweight="bold")
            if r == nrow - 1:
                ax.set_xlabel("(Method + GT) / 2", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"{C.label_of(m)}\nMethod \u2212 GT",
                              fontsize=9, fontweight="bold")
            C.clean_spines(ax)
            ax.tick_params(labelsize=7)

    elems = []
    for m in methods:
        dark = C.darken_hex(C.color_of(m), 0.5)
        lbl = C.label_of(m)
        elems += [
            Line2D([0], [0], marker="o", color="w", markerfacecolor=dark,
                   markersize=7, label=lbl),
            Line2D([0], [0], color=dark, ls="-", lw=3.0, label=f"{lbl} Bias"),
            Line2D([0], [0], color=dark, ls="-", lw=1.0, label=f"{lbl} LOA"),
        ]
    fig.legend(handles=elems, loc="upper center", ncol=len(elems),
               fontsize=7.5, bbox_to_anchor=(0.5, 1.02), frameon=False,
               columnspacing=0.8, handletextpad=0.3)
    fig.suptitle(f"Bland-Altman: Method vs Ground Truth (SNR {snr} dB)",
                 fontsize=14, fontweight="bold", y=1.06)
    fig.tight_layout()
    return fig


# ===========================================================
# Figure 3.1
# ===========================================================

def _cnr_rmse_at_snr(df, param, model, snr, gt_val):
    v = df[(df.model == model) & (df.param == param) &
           (df.snr == snr)]["cnr_pred"].to_numpy(dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.sqrt(np.mean((v - gt_val) ** 2)))


def figure31(df_cnr, gt_cnr, params, snrs, models):
    fig, axes = plt.subplots(2, len(params), figsize=(18, 7), squeeze=False)

    for c, p in enumerate(params):
        ax_top, ax_bot = axes[0, c], axes[1, c]
        gt_val = gt_cnr.get(p, np.nan)

        heights = {m: [_cnr_rmse_at_snr(df_cnr, p, m, s, gt_val) for s in snrs]
                   for m in models}

        for m in models:
            v = df_cnr[(df_cnr.model == m) & (df_cnr.param == p) &
                       (df_cnr.snr.isin(snrs))]["cnr_pred"].to_numpy(float)
            v = v[np.isfinite(v)]
            finite = [x for x in heights[m] if np.isfinite(x)]
            if v.size == 0 or not finite:
                continue
            col = C.color_of(m)
            ax_top.scatter(float(np.mean(v)), float(np.mean(finite)),
                           s=70 if m in C.CONVENTIONAL_MODELS else 60,
                           marker=C.marker_of(m), color=col,
                           edgecolors=C.darken_hex(col, 0.5), linewidths=1.0,
                           label=C.label_of(m), zorder=3)

        if np.isfinite(gt_val):
            ax_top.axvline(gt_val, color="black", ls="--", lw=1.0, alpha=0.5,
                           zorder=2)
        ax_top.set_title(C.PARAM_LATEX.get(p, p), fontsize=13,
                         fontweight="bold")
        ax_top.set_xlabel("Mean Predicted CNR", fontsize=8.5)
        if c == 0:
            ax_top.set_ylabel("CNR RMSE", fontsize=10)
        ax_top.set_ylim(bottom=0)
        ax_top.tick_params(labelsize=7)
        C.clean_spines(ax_top)

        for m in models:
            xs = [s for s, v in zip(snrs, heights[m]) if np.isfinite(v)]
            ys = [v for v in heights[m] if np.isfinite(v)]
            if not xs:
                continue
            col = C.color_of(m)
            if len(xs) >= 2:
                ax_bot.plot(xs, ys, color=col, linestyle=C.linestyle_of(m),
                            lw=1.0, alpha=0.45, zorder=3)
            ax_bot.plot(xs, ys, marker=C.marker_of(m),
                        markersize=C.marker_size_of(m), color=col,
                        linestyle="None", zorder=5)

        ax_bot.set_xlabel("SNR (dB)", fontsize=8.5)
        ax_bot.set_xticks(sorted(snrs))
        if c == 0:
            ax_bot.set_ylabel("CNR RMSE", fontsize=10)
        ax_bot.set_ylim(bottom=0)
        ax_bot.tick_params(labelsize=7)
        C.clean_spines(ax_bot)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="center right", frameon=False, fontsize=9)
    fig.suptitle("WMH vs NAWM Lesion CNR", fontsize=14, fontweight="bold",
                 y=1.02)
    fig.tight_layout(rect=(0, 0, 0.92, 1))
    return fig


# ===========================================================
# Main
# ===========================================================

def main():
    ap = argparse.ArgumentParser(description="Generate synthetic figures.")
    ap.add_argument("--figures", nargs="+", default=["1", "2.1", "2.2", "3.1"],
                    choices=["1", "2.1", "2.2", "3.1"])
    ap.add_argument("--snrs", nargs="+", type=int, default=None)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--format", nargs="+", default=["png", "pdf"])
    ap.add_argument("--no-residual", action="store_true",
                    help="Figure 1 without panel c.")
    ap.add_argument("--no-s0", action="store_true",
                    help="Figure 2.1 without the S0 panel.")
    args = ap.parse_args()

    paths = C.load_paths()
    found = C.discover_results(paths)
    if not found:
        print(f"[ERROR] no results under {paths.results_dir}", file=sys.stderr)
        return 2

    avail_m = sorted({m for m, _ in found})
    avail_s = sorted({s for _, s in found})
    models = args.models or avail_m
    snrs = sorted(args.snrs or avail_s)

    bad = set(models) - set(avail_m)
    if bad:
        print(f"[ERROR] no results for {sorted(bad)}. Available: {avail_m}",
              file=sys.stderr)
        return 2

    print(f"[DATA] {paths.results_dir}")
    print(f"       models {models}")
    print(f"       SNR    {snrs}")

    need_sig = "1" in args.figures
    need_par = "2.1" in args.figures
    need_cnr = "3.1" in args.figures

    print("[LOAD] building analysis frames")
    df_sig = C.build_signal_rmse_df(models, snrs, paths) if need_sig else None
    df_par = (C.build_param_rmse_df(models, snrs, paths=paths)
              if need_par else None)
    df_cnr = C.build_cnr_df(models, snrs, paths=paths) if need_cnr else None

    order_sig = _plot_models(df_sig, models)
    order_par = _plot_models(df_par, models)
    order_cnr = _plot_models(df_cnr, models)

    written, ranking = [], None

    if "1" in args.figures:
        print("[FIG ] fig1_signal_rmse")
        fig, ranking = figure1(df_sig, snrs, order_sig, paths,
                               show_residual=not args.no_residual)
        written += C.save_figure(fig, "fig1_signal_rmse", paths, args.format,
                                 dpi=DPI["fig1"])
        plt.close(fig)

    if "2.1" in args.figures:
        print("[FIG ] fig2_1_param_rmse")
        fig = figure21(df_par, snrs, order_par, include_s0=not args.no_s0)
        written += C.save_figure(fig, "fig2_1_param_rmse", paths, args.format,
                                 dpi=DPI["fig2_1"])
        plt.close(fig)

    if "2.2" in args.figures:
        print("[FIG ] fig2_2_bland_altman")
        methods = [m for m in ["dnn", "cnn_fusion", "pace"] if m in models]
        fig = figure22(methods, CNR_PARAMS, BA_SNR, paths, BA_SEED_INDEX)
        written += C.save_figure(fig, "fig2_2_bland_altman", paths,
                                 args.format, dpi=DPI["fig2_2"])
        plt.close(fig)

    if "3.1" in args.figures:
        print("[FIG ] fig3_1_cnr_rmse")
        gt_cnr = C.ground_truth_cnr(paths=paths)
        fig = figure31(df_cnr, gt_cnr, CNR_PARAMS, snrs, order_cnr)
        written += C.save_figure(fig, "fig3_1_cnr_rmse", paths, args.format,
                                 dpi=DPI["fig3_1"])
        plt.close(fig)

    for w in written:
        print(f"       {w.relative_to(paths.repo_root)} "
              f"({w.stat().st_size // 1024} KB)")

    if ranking:
        print(f"\n{'=' * 60}")
        print(f"  Signal RMSE ranking (SNR {snrs})")
        print(f"{'=' * 60}")
        for i, d in enumerate(ranking, 1):
            sd = f"\u00b1 {d['std']:.5f}" if d["n"] > 1 else "(1 seed)"
            print(f"  {i}. {C.label_of(d['model']):>14s}  "
                  f"RMSE = {d['mean']:.5f} {sd}  (n={d['n']})")

    if df_par is not None and not df_par.empty:
        print(f"\n{'=' * 70}")
        print(f"  Parameter RMSE ranking (SNR {snrs})")
        print(f"{'=' * 70}")
        for p in TOP_PARAMS + BOTTOM_PARAMS:
            bars = sorted(_param_bar_data(df_par, p, order_par, snrs),
                          key=lambda d: d["mean"])
            if not bars:
                continue
            print(f"\n  {p}:")
            for i, d in enumerate(bars, 1):
                sd = f"\u00b1 {d['std']:.6f}" if d["n"] > 1 else "(1 seed)"
                print(f"    {i}. {C.label_of(d['model']):>14s}  "
                      f"RMSE = {d['mean']:.6f} {sd}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
