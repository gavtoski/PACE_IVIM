#!/usr/bin/env python3
"""Model comparison: signal fit, parameter accuracy, and lesion contrast.

Usage:
    python make_model_comparison.py
    python make_model_comparison.py --snr 25 --no-tick-names
    python make_model_comparison.py --with-ivim3c-sigmoid
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent

PARAMETER_KEYS = ["Dpar", "Dint", "Dmv", "Fint", "Fmv"]

TISSUE_NAMES = ["GM", "NAWM", "WMH"]

PARAMETER_DISPLAY = {
    "Dpar": {
        "title": "$D_\\mathrm{par}$",
        "scale": 1e3,
        "unit": "$\\times 10^{-3}$ mm$^2$/s",
    },
    "Dint": {
        "title": "$D_\\mathrm{int}$",
        "scale": 1e3,
        "unit": "$\\times 10^{-3}$ mm$^2$/s",
    },
    "Dmv": {
        "title": "$D_\\mathrm{mv}$",
        "scale": 1e3,
        "unit": "$\\times 10^{-3}$ mm$^2$/s",
    },
    "Fint": {"title": "$f_\\mathrm{int}$", "scale": 1.0, "unit": "fraction"},
    "Fmv": {"title": "$f_\\mathrm{mv}$", "scale": 1.0, "unit": "fraction"},
}

COMPARED_MODELS = [
    {
        "label": "PACE",
        "tick": "PACE",
        "color": "#7030A0",
        "marker": "o",
        "model": "pace",
        "tree": "pace_nob0",
    },
    {
        "label": "PACE + spatial prior",
        "tick": "PACE\n+prior",
        "color": "#2E7D32",
        "marker": "^",
        "model": "pace",
        "tree": "pace_ansc_rel_a0.1",
    },
    {
        "label": "Signal-only DNN",
        "tick": "DNN",
        "color": "#5B9BD5",
        "marker": "s",
        "model": "dnn",
        "tree": "dnn_softmax_on",
    },
    {
        "label": "IVIM3C",
        "tick": "IVIM\n3C",
        "color": "#1F3864",
        "marker": "D",
        "model": "ivim3brain",
        "tree": "synthetic",
    },
    {
        "label": "LSQ",
        "tick": "LSQ",
        "color": "#17A398",
        "marker": "v",
        "model": "lsq",
        "tree": "synthetic",
    },
    {
        "label": "NNLS",
        "tick": "NNLS",
        "color": "#D4A017",
        "marker": "P",
        "model": "nnls",
        "tree": "synthetic",
    },
    {
        "label": "MAP",
        "tick": "MAP",
        "color": "#8C5A2B",
        "marker": "X",
        "model": "map",
        "tree": "synthetic",
    },
]

PUBLISHED_DNN = {
    "label": "Signal-only DNN, as published",
    "tick": "DNN\npublished",
    "color": "#9DC3E6",
    "marker": "8",
    "model": "dnn",
    "tree": "dnn_softmax_off",
}

KERNEL_5X5 = {
    "label": "PACE + spatial prior, 5x5",
    "tick": "PACE\n+prior 5x5",
    "color": "#E8590C",
    "marker": ">",
    "model": "pace",
    "tree": "pace_ansc_rel_a0.1_p5",
}

IVIM3C_SIGMOID = {
    "label": "IVIM3C-sigmoid",
    "tick": "IVIM\n3C-s",
    "color": "#B8860B",
    "marker": "*",
    "model": "ivim3brain_sigmoid",
    "tree": "synthetic",
}

DISTRIBUTION_MODELS = ["PACE + spatial prior", "Signal-only DNN"]


def snr_phrase(snrs):
    """How every title states its noise range, so all three agree."""
    if len(snrs) > 1:
        return f"averaged over SNR {min(snrs)} to {max(snrs)} dB"
    return f"SNR = {snrs[0]} dB"


def load_analysis_module(path):
    """Load analyse_arms.py without requiring it to be installed."""
    module_spec = importlib.util.spec_from_file_location("analyse_arms", path)
    analysis_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(analysis_module)
    return analysis_module


def available_seeds(results_root, result_tree, model_name, snr):
    """Every seed present for one model at one SNR."""
    result_dir = results_root / result_tree / f"{model_name}_snr{snr}"
    if not result_dir.is_dir():
        return []
    return sorted(
        {
            int(path.name.split("seed")[1].split(".")[0])
            for path in result_dir.glob("*seed*.h5")
        }
    )


def collect_model(
    analysis_module,
    common_module,
    data_paths,
    results_root,
    model_name,
    result_tree,
    snrs,
):
    """Seed mean per SNR, and the mean over SNRs."""
    per_snr, by_seed = {}, {}
    for snr in snrs:
        seed_results = []
        for seed in available_seeds(results_root, result_tree, model_name, snr):
            try:
                seed_results.append(
                    analysis_module.one_seed(
                        common_module,
                        data_paths,
                        results_root,
                        result_tree,
                        model_name,
                        snr,
                        seed,
                        None,
                    )
                )
            except Exception as error:
                print(
                    f"  [warn] {result_tree} snr {snr} seed {seed}: {error}",
                    file=sys.stderr,
                )
        if not seed_results:
            continue
        by_seed[snr] = seed_results
        if len(seed_results) == 1:
            per_snr[snr] = seed_results[0]
        else:
            per_snr[snr] = analysis_module._mean_of(seed_results)

    if not per_snr:
        return {}, {}, None

    blocks = list(per_snr.values())
    if len(blocks) == 1:
        overall = blocks[0]
    else:
        overall = analysis_module._mean_of(blocks)

    return per_snr, by_seed, overall


def tissue_values(
    common_module,
    data_paths,
    results_root,
    model_name,
    result_tree,
    snr,
    seed,
    tissue_name,
):
    """Predicted and true values for every voxel of one tissue."""
    import dataclasses

    model_paths = dataclasses.replace(
        data_paths, results_dir=results_root / result_tree
    )
    result = common_module.load_result(model_name, snr, seed, paths=model_paths)
    labels = np.asarray(
        common_module.decode_tissue_array(result.tissue, result.lesion)
    ).reshape(-1)
    selected = labels == tissue_name

    predicted = {}
    truth = {}
    for parameter in PARAMETER_KEYS:
        prediction = np.asarray(result.params[parameter], dtype=float).reshape(-1)
        reference = np.asarray(result.gt[parameter], dtype=float).reshape(-1)
        usable = selected & np.isfinite(prediction) & np.isfinite(reference)
        predicted[parameter] = prediction[usable]
        truth[parameter] = reference[usable]

    return predicted, truth


def cnr_rmse(per_snr_results, parameter):
    """Root mean square distance of the predicted CNR from the truth."""
    predicted = np.asarray(
        [result["cnr_pred"][parameter] for result in per_snr_results], dtype=float
    )
    truth = np.asarray(
        [result["cnr_true"][parameter] for result in per_snr_results], dtype=float
    )
    usable = np.isfinite(predicted) & np.isfinite(truth)
    if not usable.any():
        return np.nan
    return float(np.sqrt(np.mean((predicted[usable] - truth[usable]) ** 2)))


def tidy_axis(axis, tick_count=5):
    """Cap the tick count and keep markers off the frame."""
    from matplotlib.ticker import MaxNLocator

    axis.yaxis.set_major_locator(MaxNLocator(nbins=tick_count))
    axis.margins(y=0.16)


def plot_model_scatter(axis, parameter, collected, getter, tick_names):
    """One marker per model, colour and shape carrying the identity."""
    for position, (model, metrics) in enumerate(collected):
        value = getter(metrics, parameter)
        if value is None or not np.isfinite(value):
            continue
        axis.plot(
            position,
            value,
            model["marker"],
            color=model["color"],
            markersize=10,
            linestyle="",
            zorder=3,
        )

    axis.set_xlim(-0.6, len(collected) - 0.4)
    axis.set_xticks(range(len(collected)))
    if tick_names:
        axis.set_xticklabels([model["tick"] for model, _ in collected], fontsize=8)
    else:
        axis.set_xticklabels([])
    tidy_axis(axis)
    axis.grid(alpha=0.18, linewidth=0.5)
    for spine_name in ("top", "right"):
        axis.spines[spine_name].set_visible(False)


def plot_distribution_panel(axis, parameter, distributions, tissue_name):
    """Predicted spread against the ground-truth spread."""
    display = PARAMETER_DISPLAY[parameter]
    scale = display["scale"]

    reference = None
    for _, _, truth_values in distributions:
        if truth_values.size:
            reference = truth_values
            break

    if reference is None or not reference.size:
        axis.set_axis_off()
        return

    centre = float(np.mean(reference)) * scale
    spread = float(np.std(reference)) * scale
    if spread <= 0:
        spread = max(abs(centre) * 0.1, 1e-6)
    bins = np.linspace(centre - 4.0 * spread, centre + 4.0 * spread, 80)

    axis.hist(
        reference * scale,
        bins=bins,
        density=True,
        color="#bbbbbb",
        alpha=0.55,
        zorder=1,
    )

    for model, predicted_values, _ in distributions:
        if not predicted_values.size:
            continue
        axis.hist(
            predicted_values * scale,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.7,
            color=model["color"],
            zorder=3,
        )

    axis.set_yticks([])
    axis.set_xlabel(f"{display['title']} ({display['unit']})", fontsize=8.5)
    for spine_name in ("top", "right", "left"):
        axis.spines[spine_name].set_visible(False)


def legend_entries(collected, line_module):
    """One handle per model, in the order the panels use."""
    return [
        line_module(
            [],
            [],
            marker=model["marker"],
            color=model["color"],
            linestyle="",
            markersize=9,
            label=model["label"],
        )
        for model, _ in collected
    ]


def write_figure(figure, output_root, stem, dpi):
    """Save as PNG and as TIFF, in separate folders."""
    written_files = []
    for extension, subfolder in (("png", "png"), ("tiff", "tiff")):
        output_dir = output_root / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / f"{stem}.{extension}"

        save_options = {}
        if extension == "tiff":
            save_options["pil_kwargs"] = {"compression": "tiff_lzw"}

        figure.savefig(output_file, dpi=dpi, bbox_inches="tight", **save_options)
        written_files.append(output_file)
    return written_files


def plot_signal_seeds(axis, collected, per_seed, snr, reference_label,
                      tick_names):
    """Per seed signal RMSE at one level, against a reference mean."""
    jitter_rng = np.random.default_rng(0)

    def values_for(label):
        points = per_seed.get(label, {}).get(snr, [])
        values = np.asarray(
            [result["signal_rmse"]["all"] for result in points], dtype=float
        )
        return values[np.isfinite(values)]

    reference_values = values_for(reference_label)
    if reference_values.size:
        axis.axhline(
            float(reference_values.mean()),
            color="#9a9a9a",
            linewidth=0.8,
            linestyle=(0, (4, 3)),
            alpha=0.8,
            zorder=1,
        )

    for position, (model, _) in enumerate(collected):
        values = values_for(model["label"])
        if not values.size:
            continue
        if values.size > 1:
            axis.plot(
                position + jitter_rng.uniform(-0.18, 0.18, values.size),
                values,
                model["marker"],
                color=model["color"],
                markersize=3.5,
                alpha=0.35,
                linestyle="",
                zorder=2,
            )
        axis.plot(
            position + (0.32 if values.size > 1 else 0.0),
            float(values.mean()),
            model["marker"],
            color=model["color"],
            markersize=8,
            linestyle="",
            zorder=4,
        )

    axis.set_xlim(-0.6, len(collected) - 0.4)
    axis.set_xticks(range(len(collected)))
    axis.set_xticklabels(
        [model["tick"] for model, _ in collected] if tick_names else [],
        fontsize=8,
    )
    axis.set_ylabel("Signal RMSE")
    axis.set_title(f"Per seed, {snr} dB", fontsize=10)
    axis.text(
        0.015,
        0.985,
        f"Dashed line: {reference_label} mean",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
        color="#555555",
    )
    axis.grid(alpha=0.2, linewidth=0.5)


def build_signal_figure(
    plt, collected, per_snr, per_seed, snrs, seed_snr,
    reference_label="PACE", signal_key="all", tick_names=True
):
    """Figure 1."""
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))

    for model, _ in collected:
        points = per_snr[model["label"]]
        x_values = [snr for snr in snrs if snr in points]
        y_values = [points[snr]["signal_rmse"][signal_key] for snr in x_values]
        if not x_values:
            continue
        axes[0].plot(
            x_values,
            y_values,
            "-",
            marker=model["marker"],
            color=model["color"],
            markersize=5,
            linewidth=1.4,
            label=model["label"],
        )

    axes[0].set_xticks(snrs)
    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Signal RMSE")
    axes[0].set_title("Signal RMSE mean", fontsize=10)
    axes[0].grid(alpha=0.2, linewidth=0.5)

    plot_signal_seeds(
        axes[1], collected, per_seed, seed_snr, reference_label, tick_names
    )

    for axis in axes:
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)

    axes[0].legend(fontsize=8, frameon=False, ncol=2)
    figure.suptitle(
        f"Signal reconstruction accuracy, {snr_phrase(snrs)}", fontsize=12, y=0.98
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.12, wspace=0.24)
    return figure


def build_parameter_figure(plt, line_module, collected, tick_names, snr_note):
    """Figure 2."""
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.6))
    flat_axes = axes.ravel()

    for axis, parameter in zip(flat_axes, PARAMETER_KEYS):
        display = PARAMETER_DISPLAY[parameter]
        plot_model_scatter(
            axis,
            parameter,
            collected,
            lambda metrics, key: (
                metrics["param_rmse"]["mean"][key] * PARAMETER_DISPLAY[key]["scale"]
            ),
            tick_names,
        )
        axis.set_title(f"{display['title']}  ({display['unit']})", fontsize=11, pad=6)

    for axis_index in (0, 3):
        flat_axes[axis_index].set_ylabel("Parameter RMSE", fontsize=9.5)

    legend_axis = flat_axes[len(PARAMETER_KEYS)]
    legend_axis.set_axis_off()
    legend_axis.legend(
        handles=legend_entries(collected, line_module),
        loc="center",
        frameon=False,
        fontsize=10.5,
        labelspacing=1.0,
    )

    figure.suptitle(f"IVIM parameter accuracy, {snr_note}", fontsize=12, y=0.972)
    figure.subplots_adjust(
        left=0.075, right=0.985, top=0.915, bottom=0.065, wspace=0.27, hspace=0.34
    )
    return figure


def build_contrast_figure(
    plt,
    line_module,
    patch_module,
    collected,
    per_snr,
    snrs,
    distributions,
    tissue_names,
    tick_names,
    snr_note,
):
    """Figure 3."""
    row_count = 1 + len(tissue_names)
    figure, axes = plt.subplots(
        row_count,
        len(PARAMETER_KEYS),
        figsize=(16.0, 3.6 + 2.4 * len(tissue_names)),
        gridspec_kw={"height_ratios": [1.0] + [1.0] * len(tissue_names)},
        squeeze=False,
    )

    for column, parameter in enumerate(PARAMETER_KEYS):
        axis = axes[0][column]

        values = []
        for model, _ in collected:
            points = per_snr[model["label"]]
            results = [points[snr] for snr in snrs if snr in points]
            values.append(cnr_rmse(results, parameter) if results else np.nan)

        for position, (model, _) in enumerate(collected):
            if not np.isfinite(values[position]):
                continue
            axis.plot(
                position,
                values[position],
                model["marker"],
                color=model["color"],
                markersize=10,
                linestyle="",
                zorder=3,
            )

        finite = [value for value in values if np.isfinite(value)]
        if finite:
            axis.set_ylim(0.0, max(finite) * 1.25)

        axis.set_xlim(-0.6, len(collected) - 0.4)
        axis.set_xticks(range(len(collected)))
        if tick_names:
            axis.set_xticklabels([model["tick"] for model, _ in collected], fontsize=8)
        else:
            axis.set_xticklabels([])
        tidy_axis(axis, tick_count=4)
        axis.grid(alpha=0.18, linewidth=0.5)
        for spine_name in ("top", "right"):
            axis.spines[spine_name].set_visible(False)

        axis.set_title(PARAMETER_DISPLAY[parameter]["title"], fontsize=11, pad=6)

        for row, tissue_name in enumerate(tissue_names, start=1):
            plot_distribution_panel(
                axes[row][column],
                parameter,
                distributions[tissue_name][parameter],
                tissue_name,
            )

    axes[0][0].set_ylabel("Lesion CNR RMSE", fontsize=9.5)
    for row, tissue_name in enumerate(tissue_names, start=1):
        axes[row][0].set_ylabel(f"{tissue_name} density", fontsize=9.5)
        if row < len(tissue_names):
            for column in range(len(PARAMETER_KEYS)):
                axes[row][column].set_xlabel("")

    handles = legend_entries(collected, line_module)
    if tissue_names:
        handles.append(
            patch_module(facecolor="#bbbbbb", alpha=0.55,
                         label="Ground truth distribution")
        )

    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9.5,
        bbox_to_anchor=(0.5, -0.02),
    )

    figure.suptitle(f"IVIM lesion contrast accuracy, {snr_note}", fontsize=12, y=0.975)
    figure.subplots_adjust(
        left=0.065, right=0.99, top=0.915 if tissue_names else 0.80,
        bottom=0.14 if tissue_names else 0.24, wspace=0.28, hspace=0.38
    )
    return figure


def main(argv=None):
    parser = argparse.ArgumentParser(description="Model comparison figures.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--analyser", default=str(SCRIPT_DIR / "analyse_arms.py"))
    parser.add_argument(
        "--snrs",
        type=int,
        nargs="+",
        default=[20, 25, 30, 35],
        help="Noise levels. Figure 1 draws them as a "
        "sweep; figures 2 and 3 average over them, "
        "each level averaged over its own seeds "
        "first so a level with more seeds does not "
        "outweigh one with fewer.",
    )
    parser.add_argument(
        "--distribution-snr",
        type=int,
        default=25,
        help="Noise level for the distribution row, "
        "which needs one run rather than an "
        "average: pooling predictions across runs "
        "would narrow every curve and manufacture "
        "the effect the row shows.",
    )
    parser.add_argument(
        "--tissue",
        nargs="+",
        default=["WMH", "NAWM"],
        choices=["GM", "NAWM", "WMH"],
        help="Tissues to draw distributions for, one row "
        "each. WMH and NAWM by default: a contrast "
        "can be inflated by widening the lesion or "
        "by narrowing what it is measured against, "
        "and one row cannot tell those apart.",
    )
    parser.add_argument(
        "--by-tissue",
        action="store_true",
        help="Figure 1: break the signal RMSE down by "
        "tissue and macro average it, instead of "
        "pooling every valid voxel.",
    )
    parser.add_argument(
        "--no-tick-names",
        action="store_true",
        help="Drop the model names from the scatter x "
        "axes and let the legend carry them.",
    )
    parser.add_argument(
        "--with-5x5",
        action="store_true",
        help="Add the spatial prior on a 5x5 window. Off "
        "by default: at five seeds it is a wash "
        "against 3x3 and an eighth model crowds "
        "these panels.",
    )
    parser.add_argument(
        "--with-published-dnn",
        action="store_true",
        help="Add the DNN as published, with sigmoid "
        "fractions and unexpanded bounds. The "
        "default DNN row is the retrained one, "
        "which matches PACE on the fraction "
        "parameterisation and IVIM3C on the bound "
        "expansion.",
    )
    parser.add_argument(
        "--with-ivim3c-sigmoid",
        action="store_true",
        help="Add the sigmoid variant of IVIM3C. Off by "
        "default: it is a variant rather than a "
        "separate method.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=["signal", "parameter", "contrast"],
        default=["signal", "parameter", "contrast"],
    )
    parser.add_argument("--select", nargs="+", default=None,
                        metavar="TREE/MODEL",
                        help="Draw only these, named by result tree and "
                             "model, such as pace_nob0/pace.")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument(
        "--out", default="figures/model_comparison"
    )
    args = parser.parse_args(argv)

    repository = Path(args.repo).expanduser().resolve()
    if not (repository / "pace" / "common.py").is_file():
        print(f"[ERROR] not a PACE_IVIM repository: {repository}", file=sys.stderr)
        return 2

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})

    analysis_module = load_analysis_module(args.analyser)
    sys.path.insert(0, str(repository))
    from pace import common as common_module

    data_paths = common_module.load_paths()
    results_root = repository / "results"

    models = list(COMPARED_MODELS)
    if args.with_5x5:
        models.insert(2, KERNEL_5X5)
    if args.with_published_dnn:
        models.insert(3, PUBLISHED_DNN)
    if args.with_ivim3c_sigmoid:
        models.insert(4, IVIM3C_SIGMOID)

    if args.select:
        wanted = set(args.select)
        models = [model for model in models
                  if f"{model['tree']}/{model['model']}" in wanted]

    sweep_snrs = sorted(set(args.snrs) | {args.distribution_snr})

    print(f"[repo] {repository}")
    print(
        f"[snr ] {', '.join(map(str, args.snrs))}, "
        f"distributions at {args.distribution_snr} dB"
    )

    collected = []
    per_snr = {}
    per_seed = {}
    for model in models:
        points, seed_points, overall = collect_model(
            analysis_module,
            common_module,
            data_paths,
            results_root,
            model["model"],
            model["tree"],
            sweep_snrs,
        )
        if overall is None:
            print(
                f"  [warn] {model['tree']}/{model['model']} has nothing",
                file=sys.stderr,
            )
            continue
        collected.append((model, overall))
        per_snr[model["label"]] = points
        per_seed[model["label"]] = seed_points
        print(f"    {model['label']:18} " f"{len(points)}/{len(sweep_snrs)} SNR")

    coverage = {tuple(sorted(per_snr[model["label"]])) for model, _ in collected}
    if len(coverage) > 1:
        print(
            "  [warn] models differ in SNR coverage, so the averages "
            "are not all over the same noise levels",
            file=sys.stderr,
        )

    if len(collected) < 2:
        print("[ERROR] fewer than two models available", file=sys.stderr)
        return 1

    tissue_names = list(args.tissue)
    if not any(model["label"] in DISTRIBUTION_MODELS
               for model, _ in collected):
        tissue_names = []

    distributions = {
        tissue_name: {parameter: [] for parameter in PARAMETER_KEYS}
        for tissue_name in tissue_names
    }
    if "contrast" in args.only:
        for model, _ in collected:
            if model["label"] not in DISTRIBUTION_MODELS:
                continue
            seeds = available_seeds(
                results_root, model["tree"], model["model"], args.distribution_snr
            )
            if not seeds:
                continue
            for tissue_name in tissue_names:
                try:
                    predicted, truth = tissue_values(
                        common_module,
                        data_paths,
                        results_root,
                        model["model"],
                        model["tree"],
                        args.distribution_snr,
                        seeds[0],
                        tissue_name,
                    )
                except Exception as error:
                    print(
                        f"  [warn] {model['tree']} {tissue_name} "
                        f"distribution: {error}",
                        file=sys.stderr,
                    )
                    continue
                for parameter in PARAMETER_KEYS:
                    distributions[tissue_name][parameter].append(
                        (model, predicted[parameter], truth[parameter])
                    )
    print()

    output_root = repository / args.out
    snr_note = snr_phrase(args.snrs)
    tick_names = not args.no_tick_names
    written_files = []

    if "signal" in args.only:
        figure = build_signal_figure(
            plt,
            collected,
            per_snr,
            per_seed,
            args.snrs,
            args.distribution_snr,
            signal_key="mean" if args.by_tissue else "all",
            tick_names=tick_names,
        )
        written_files += write_figure(figure, output_root, "fig1_signal", args.dpi)
        plt.close(figure)

    if "parameter" in args.only:
        figure = build_parameter_figure(plt, Line2D, collected, tick_names, snr_note)
        written_files += write_figure(figure, output_root, "fig2_parameter", args.dpi)
        plt.close(figure)

    if "contrast" in args.only:
        figure = build_contrast_figure(
            plt,
            Line2D,
            Patch,
            collected,
            per_snr,
            args.snrs,
            distributions,
            tissue_names,
            tick_names,
            snr_note,
        )
        written_files += write_figure(figure, output_root, "fig3_contrast", args.dpi)
        plt.close(figure)

    for output_file in written_files:
        print(f"[write] {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
