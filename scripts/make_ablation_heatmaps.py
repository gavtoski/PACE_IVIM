#!/usr/bin/env python3
"""Architecture ablation heatmaps.

Usage:
    python make_ablation_heatmaps.py
    python make_ablation_heatmaps.py --snr 25 --with-dnn
    python make_ablation_heatmaps.py --saturate 0.25
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PARAMETER_KEYS = ["Dpar", "Dint", "Dmv", "Fint", "Fmv"]

PARAMETER_SCALE = {"Dpar": 1e-3, "Dint": 1e-3, "Dmv": 1e-3,
                   "Fint": 1.0, "Fmv": 1.0}

D_UNIT = r"($10^{-3}$ mm$^2$/s)"

HEADER_LINE_IN = 0.25


def three_sig(value):
    """Three significant figures, the manuscript's convention."""
    import math
    if value is None or not np.isfinite(value):
        return "-"
    if value == 0:
        return "0.00"
    digits = max(0, 2 - int(math.floor(math.log10(abs(value)))))
    return f"{value:.{digits}f}"


PARAMETER_LABELS = {"Dpar": "$D_\\mathrm{par}$", "Dint": "$D_\\mathrm{int}$",
        "Dmv": "$D_\\mathrm{mv}$", "Fint": "$f_\\mathrm{int}$",
        "Fmv": "$f_\\mathrm{mv}$"}

BASELINE_ARM = ("PACE", "pace", "pace_nob0")
ABLATION_ARMS = [
    ("PACE, with b0",         "pace",       "released_v1_with_b0"),
    ("PACE-concat",           "cnn_fusion", "cnn_nob0"),
    ("PACE, gates closed",    "pace",       "pace_nob0_gate0"),
    ("PACE, no CE",           "pace",       "pace_nob0_noce"),
    ("PACE, 5x5 patch",       "pace",       "pace_nob0_p5"),
    ("PACE, neighbour input", "pace",       "pace_nb_anatselect"),
    ("PACE, no anatomy",      "pace",       "pace_nbonly"),
]
DNN_ARM = ("Signal-only DNN", "dnn", "dnn_softmax_on")

CONFIG_COLUMNS = ["Spatial input", "Fusion", "Gate", "CE", "Patch"]
ARCHITECTURE_CONFIG = {
    "PACE":                  ["T1+FLAIR",         "attention",   "yes", "yes", "3x3"],
    "PACE, with b0":         ["T1+FLAIR+b0",      "attention",   "yes", "yes", "3x3"],
    "PACE-concat":           ["T1+FLAIR",         "concatenate", "yes", "yes", "3x3"],
    "PACE, gates closed":    ["T1+FLAIR",         "attention",   "no",  "yes", "3x3"],
    "PACE, no CE":           ["T1+FLAIR",         "attention",   "yes", "no",  "3x3"],
    "PACE, 5x5 patch":       ["T1+FLAIR",         "attention",   "yes", "yes", "5x5"],
    "PACE, neighbour input": ["T1+FLAIR+IVIM_nb", "attention",   "yes", "yes", "3x3"],
    "PACE, no anatomy":      ["IVIM_nb",          "attention",   "yes", "yes", "3x3"],
    "Signal-only DNN":       ["none",             "none",        "no",  "no",  "none"],
}


def load_analysis_module(path):
    module_spec = importlib.util.spec_from_file_location("analyse_arms", path)
    analysis_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(analysis_module)
    return analysis_module


def collect_results(analysis_module, repository, arm_definitions, snr_values):
    sys.path.insert(0, str(repository))
    from pace import common as common_module
    data_paths = common_module.load_paths()
    results_root = repository / "results"
    results_by_arm, seed_counts = {}, {}
    for label, model, tree in arm_definitions:
        snr_mean_results, all_seed_results, total_seed_count = [], [], 0
        for snr in snr_values:
            result_dir = results_root / tree / f"{model}_snr{snr}"
            if not result_dir.is_dir():
                continue
            available_seed_numbers = sorted({
                int(path.name.split("seed")[1].split(".")[0])
                for path in result_dir.glob("*seed*.h5")
            })
            seed_results = []
            for seed in available_seed_numbers:
                try:
                    seed_results.append(
                        analysis_module.one_seed(
                            common_module, data_paths, results_root, tree, model,
                            snr, seed, None
                        )
                    )
                except Exception as exc:
                    print(f"  [warn] {label} snr {snr} seed {seed}: {exc}",
                          file=sys.stderr)
            if seed_results:
                snr_mean_results.append(
                    analysis_module._mean_of(seed_results)
                    if len(seed_results) > 1 else seed_results[0]
                )
                all_seed_results.extend(seed_results)
                total_seed_count += len(seed_results)
        if not snr_mean_results:
            print(f"  [warn] {label}: nothing under results/{tree}",
                  file=sys.stderr)
            continue
        results_by_arm[label] = (
            analysis_module._mean_of(snr_mean_results)
            if len(snr_mean_results) > 1 else snr_mean_results[0]
        )
        results_by_arm[label]["_every"] = all_seed_results
        seed_counts[label] = total_seed_count
    return results_by_arm, seed_counts


def cnr_rmse(results, parameter):
    """Root mean square distance of the predicted CNR from the truth."""
    predicted = np.asarray(
        [result["cnr_pred"][parameter] for result in results], dtype=float
    )
    truth = np.asarray(
        [result["cnr_true"][parameter] for result in results], dtype=float
    )
    usable = np.isfinite(predicted) & np.isfinite(truth)
    if not usable.any():
        return np.nan
    return float(np.sqrt(np.mean((predicted[usable] - truth[usable]) ** 2)))


def calculate_color_weight(value, reference, truth=None, saturation=0.2):
    """Signed, saturating colour weight in [-1, 1]."""
    if not (np.isfinite(value) and np.isfinite(reference)):
        return 0.0
    if truth is not None:
        value, reference = abs(value - truth), abs(reference - truth)
    if reference == 0:
        return 0.0
    relative_change = (value - reference) / abs(reference)
    return float(np.clip(relative_change / saturation, -1.0, 1.0))


def draw_heatmap(
    plt,
    row_labels,
    column_labels,
    cell_text,
    color_weights,
    title,
    output_stem,
    dpi,
    baseline_label,
    shade_first_line_only=False,
):
    """Configuration block on the left, coloured metrics on the right."""
    from matplotlib.colors import LinearSegmentedColormap
    color_map = LinearSegmentedColormap.from_list(
        "bwr", ["#1f4e79", "#ffffff", "#c00000"])
    row_count = len(row_labels)
    metric_count = len(column_labels)
    config_count = len(CONFIG_COLUMNS)

    header_lines = max(str(label).count("\n") + 1 for label in column_labels)
    extra_header_in = HEADER_LINE_IN * (header_lines - 1)
    base_height = 0.62 * row_count + 1.9
    figure_height = base_height + extra_header_in

    def pad_header(label):
        label = str(label)
        return label + "\n " * (header_lines - 1 - label.count("\n"))

    column_labels = [pad_header(label) for label in column_labels]
    config_labels = [pad_header(label) for label in CONFIG_COLUMNS]

    figure, (config_axis, metric_axis) = plt.subplots(
        1,
        2,
        figsize=(
            1.35 * metric_count + 1.45 * config_count + 3.4,
            figure_height,
        ),
        gridspec_kw={
            "width_ratios": [1.45 * config_count, 1.35 * metric_count],
            "wspace": 0.04,
        },
    )

    config_axis.imshow(
        np.zeros((row_count, config_count)),
        cmap="Greys",
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    for row_index, row_label in enumerate(row_labels):
        config_values = ARCHITECTURE_CONFIG.get(
            row_label, [""] * config_count
        )
        for column_index in range(config_count):
            config_axis.text(
                column_index,
                row_index,
                config_values[column_index],
                ha="center",
                va="center",
                fontsize=8,
            )
    config_axis.set_xticks(range(config_count))
    config_axis.set_xticklabels(config_labels, fontsize=9)
    config_axis.xaxis.set_ticks_position("top")
    config_axis.xaxis.set_label_position("top")
    config_axis.set_yticks(range(row_count))
    config_axis.set_yticklabels(row_labels, fontsize=9)
    config_axis.set_xticks(np.arange(-0.5, config_count, 1), minor=True)
    config_axis.set_yticks(np.arange(-0.5, row_count, 1), minor=True)
    config_axis.grid(which="minor", color="#cccccc", lw=0.8)

    metric_axis.imshow(
        np.asarray(color_weights),
        cmap=color_map,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )
    for row_index in range(row_count):
        for column_index in range(metric_count):
            color_weight = color_weights[row_index][column_index]
            text = cell_text[row_index][column_index]
            if shade_first_line_only and "\n" in text:
                first_line, second_line = text.split("\n", 1)
                metric_axis.text(
                    column_index, row_index - 0.16, first_line,
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(color_weight) > 0.6 else "black",
                )
                metric_axis.text(
                    column_index, row_index + 0.22, second_line,
                    ha="center", va="center", fontsize=7.5,
                    color="white" if abs(color_weight) > 0.6 else "#333333",
                )
            else:
                metric_axis.text(
                    column_index, row_index, text,
                    ha="center", va="center", fontsize=8.5,
                    color="white" if abs(color_weight) > 0.6 else "black",
                )
    metric_axis.set_xticks(range(metric_count))
    metric_axis.set_xticklabels(column_labels, fontsize=10)
    metric_axis.xaxis.set_ticks_position("top")
    metric_axis.xaxis.set_label_position("top")
    metric_axis.set_yticks([])
    metric_axis.set_xticks(np.arange(-0.5, metric_count, 1), minor=True)
    metric_axis.set_yticks(np.arange(-0.5, row_count, 1), minor=True)
    metric_axis.grid(which="minor", color="white", lw=1.5)

    for axis in (config_axis, metric_axis):
        axis.tick_params(which="minor", length=0)
        axis.tick_params(which="major", length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)

    if baseline_label in row_labels:
        baseline_row = row_labels.index(baseline_label)
        for axis, width in (
            (config_axis, config_count),
            (metric_axis, metric_count),
        ):
            axis.add_patch(
                plt.Rectangle(
                    (-0.5, baseline_row - 0.5),
                    width,
                    1,
                    facecolor="#e8e8e8",
                    zorder=0,
                )
            )
            axis.plot(
                [-0.5, width - 0.5],
                [baseline_row + 0.5, baseline_row + 0.5],
                color="#555555",
                lw=1.1,
                clip_on=False,
                zorder=5,
            )

    title_artist = figure.suptitle(title, fontsize=12, y=0.98)
    top_margin = (0.14 * base_height + extra_header_in) / figure_height
    figure.subplots_adjust(left=0.20, right=0.99, top=1.0 - top_margin,
                           bottom=0.03)

    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    content_boxes = [axis.get_tightbbox(renderer)
                     for axis in (config_axis, metric_axis)]
    content_left = min(box.x0 for box in content_boxes)
    content_right = max(box.x1 for box in content_boxes)
    title_artist.set_x(0.5 * (content_left + content_right)
                       / figure.bbox.width)

    written_files = []
    for extension, subfolder in (("png", "png"), ("tiff", "tiff")):
        output_dir = output_stem.parent / subfolder
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"{output_stem.name}.{extension}"
        figure.savefig(
            output_file,
            dpi=dpi,
            bbox_inches="tight",
            pil_kwargs={"compression": "tiff_lzw"}
            if extension == "tiff" else None,
        )
        written_files.append(output_file)
    plt.close(figure)
    return written_files


def main(argv=None):
    parser = argparse.ArgumentParser(description="Ablation heatmaps.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--analyser", default=str(SCRIPT_DIR / "analyse_arms.py"))
    parser.add_argument("--snr", type=int, default=25)
    parser.add_argument("--all-snr", action="store_true")
    parser.add_argument("--with-dnn", action="store_true",
                    help="Add the signal-only DNN. Off by default: it is "
                         "a different architecture, not an ablation of "
                         "this one.")
    parser.add_argument("--saturate", type=float, default=0.2,
                    help="Relative difference from PACE at which the "
                         "colour saturates. 0.2 means twenty percent "
                         "better or worse reaches full blue or red.")
    parser.add_argument("--select", nargs="+", default=None,
                        metavar="TREE/MODEL",
                        help="Draw only these, named by result tree and "
                             "model, such as pace_nob0/pace.")
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--out", default="figures/ablation/x")
    args = parser.parse_args(argv)

    repository = Path(args.repo).expanduser().resolve()
    if not (repository / "pace" / "common.py").is_file():
        print(f"[ERROR] not a PACE_IVIM repository: {repository}", file=sys.stderr)
        return 2

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})

    analysis_module = load_analysis_module(args.analyser)
    snr_values = [20, 25, 30, 35] if args.all_snr else [args.snr]
    arm_definitions = (
        [BASELINE_ARM]
        + ABLATION_ARMS
        + ([DNN_ARM] if args.with_dnn else [])
    )

    if args.select:
        wanted = set(args.select)
        arm_definitions = [BASELINE_ARM] + [
            arm for arm in arm_definitions[1:]
            if f"{arm[2]}/{arm[1]}" in wanted
        ]

    print(f"[repo] {repository}")
    print(f"[snr ] {', '.join(map(str, snr_values))}")
    results_by_arm, seed_counts = collect_results(
        analysis_module,
        repository,
        arm_definitions,
        snr_values,
    )
    if BASELINE_ARM[0] not in results_by_arm:
        print(f"[ERROR] baseline {BASELINE_ARM[0]} missing", file=sys.stderr)
        return 1
    row_order = [
        label
        for label, _, _ in arm_definitions
        if label in results_by_arm
    ]
    for label in row_order:
        print(f"    {label:24} {seed_counts[label]} file(s)")
    print()

    baseline_results = results_by_arm[BASELINE_ARM[0]]
    output_stem = repository / args.out

    column_labels = ["Signal"] + [
        PARAMETER_LABELS[parameter]
        + ("\n" + D_UNIT if PARAMETER_SCALE[parameter] != 1.0 else "")
        for parameter in PARAMETER_KEYS
    ]
    scales = [1.0] + [PARAMETER_SCALE[p] for p in PARAMETER_KEYS]
    cell_text, color_weights = [], []
    for label in row_order:
        arm_results = results_by_arm[label]
        values = [arm_results["signal_rmse"]["mean"]] + [
            arm_results["param_rmse"]["mean"][parameter]
            for parameter in PARAMETER_KEYS
        ]
        reference_values = [baseline_results["signal_rmse"]["mean"]] + [
            baseline_results["param_rmse"]["mean"][parameter]
            for parameter in PARAMETER_KEYS
        ]
        cell_text.append([
            three_sig(value / scale)
            for value, scale in zip(values, scales)
        ])
        color_weights.append([
            calculate_color_weight(
                value, reference, None, args.saturate
            )
            for value, reference in zip(values, reference_values)
        ])
    rmse_files = draw_heatmap(
        plt,
        row_order,
        column_labels,
        cell_text,
        color_weights,
        "Architecture ablation: signal and parameter RMSE",
        output_stem.parent / "ablation_rmse",
        args.dpi,
        BASELINE_ARM[0],
    )

    cell_text, color_weights = [], []
    baseline_cnr_rmse = {
        parameter: cnr_rmse(baseline_results["_every"], parameter)
        for parameter in PARAMETER_KEYS
    }
    for label in row_order:
        arm_results = results_by_arm[label]
        row_text, row_weights = [], []
        for parameter in PARAMETER_KEYS:
            cnr_value = cnr_rmse(arm_results["_every"], parameter)
            row_text.append(three_sig(cnr_value))
            row_weights.append(
                calculate_color_weight(
                    cnr_value, baseline_cnr_rmse[parameter], None, args.saturate
                )
            )
        cell_text.append(row_text)
        color_weights.append(row_weights)
    cnr_files = draw_heatmap(
        plt,
        row_order,
        [PARAMETER_LABELS[parameter] for parameter in PARAMETER_KEYS],
        cell_text,
        color_weights,
        "Architecture ablation: lesion CNR RMSE",
        output_stem.parent / "ablation_cnr",
        args.dpi,
        BASELINE_ARM[0],
    )

    for output_file in rmse_files + cnr_files:
        print(f"[write] {output_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
