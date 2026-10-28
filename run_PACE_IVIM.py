#!/usr/bin/env python3
"""Reproduce the synthetic results of the PACE manuscript.

PACE estimates three compartment IVIM parameters from diffusion weighted
brain MRI, conditioning a physics informed neural network on co-registered
anatomical context (T1 and FLAIR) through gated cross attention.

DISCLAIMER
    This work is under peer review. The method, code and reported numbers
    may change before publication. Until then, please cite the preprint.

If you use this code, please cite:

    Hoang N, Faiyaz A, Mensah EA, Uddin MN, Zhong J, Schifitto G.
    Improved three component IVIM fitting using physics-informed neural
    network with multimodal spatial prior. arXiv:2610.28873 [eess.IV];
    2026. https://arxiv.org/abs/2610.28873

The IVIM3C comparator and the three compartment model follow:

    Voorter PHM, Backes WH, Gurney-Champion OJ, Wong S-M, Staals J,
    van Oostenbrugge RJ, van der Thiel MM, Jansen JFA, Drenthen GS.
    Improving microstructural integrity, interstitial fluid, and blood
    microcirculation images from multi-b-value diffusion MRI using
    physics-informed neural networks in cerebrovascular disease. Magnetic
    Resonance in Med. 2023;90(4):1657-71. doi: 10.1002/mrm.29753.

Spatial information in deep learning IVIM estimation:

    Kaandorp MPT, Zijlstra F, Karimi D, Gholipour A, While PT.
    Incorporating spatial information in deep learning parameter
    estimation with application to the intravoxel incoherent motion model
    in diffusion-weighted MRI. Medical Image Analysis. 2025;101:103414.
    doi: 10.1016/j.media.2024.103414.

Usage:
    python run_PACE_IVIM.py
    python run_PACE_IVIM.py --models PACE DNN LSQ
    python run_PACE_IVIM.py --ablations
    python run_PACE_IVIM.py --conventional
    python run_PACE_IVIM.py --list
"""

import argparse
import contextlib
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent
CONFIG = REPO / "configs" / "arms.json"
SCRIPTS = REPO / "scripts"
RESULTS = REPO / "results"

PARAMS = ("Dpar", "Dint", "Dmv", "Fint", "Fmv")
DIFFUSIVITIES = {"Dpar", "Dint", "Dmv"}
CONVENTIONAL_SEED = 19
# A CPU run against Metal results differed by at most 2.06 percent in any one
# file, from the backend's noise stream.
COMPARE_TOLERANCE = 0.10
RESULT_FILE_RE = re.compile(r"^inferred_seed(?P<seed>\d+)\.h5$")
RUN_DIR_RE = re.compile(r"^(?P<model>.+)_snr(?P<snr>\d+)_seed(?P<seed>\d+)$")


# Registry

@dataclass
class Arm:
    """One model or ablation arm, as listed in configs/arms.json."""
    name: str
    label: str
    group: str
    kind: str
    model: str
    tree: str
    checkpoint_tree: str
    test_file: str
    infer_flags: list = field(default_factory=list)
    train_flags: list = field(default_factory=list)

    @property
    def key(self):
        return f"{self.tree}/{self.model}"


def load_registry():
    """Return (settings, models, ablations) from configs/arms.json."""
    doc = json.loads(CONFIG.read_text(encoding="utf-8"))
    lists = {}
    for group, singular in (("models", "model"), ("ablations", "ablation")):
        lists[group] = [
            Arm(name=entry["name"], label=entry["label"], group=singular,
                kind=entry["kind"], model=entry["model"], tree=entry["tree"],
                checkpoint_tree=entry.get("checkpoint_tree", entry["tree"]),
                test_file=entry.get("test_file", doc["test_file"]),
                infer_flags=list(entry.get("infer_flags", [])),
                train_flags=list(entry.get("train_flags", [])))
            for entry in doc[group]
        ]
    return doc, lists["models"], lists["ablations"]


def data_paths():
    """Resolved data locations, honoring --data-dir and configs/paths.yaml."""
    sys.path.insert(0, str(REPO))
    from pace import common
    return common.load_paths()


def test_path(arm, paths):
    """The test file an arm is run on, inside the data folder."""
    if arm.test_file == Path(paths.synthetic_test).name:
        return Path(paths.synthetic_test)
    return Path(paths.data_dir) / arm.test_file


def shown(path):
    """A path relative to the repository when inside it, absolute otherwise."""
    path = Path(path)
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


def pick_arms(names, pool, what):
    """Arms named in names, case insensitive, in registry order."""
    wanted = {name.lower() for name in names}
    known = {arm.name.lower() for arm in pool}
    unknown = sorted(wanted - known)
    if unknown:
        raise SystemExit(f"[ERROR] unknown {what}: {', '.join(unknown)}. "
                         f"Choose from: {', '.join(a.name for a in pool)}")
    return [arm for arm in pool if arm.name.lower() in wanted]


def print_registry(models, ablations):
    for title, pool in (("Models (default: all)", models),
                        ("Ablations (off unless --ablations)", ablations)):
        print(title)
        for arm in pool:
            print(f"  {arm.name:<16} {arm.label:<24} {arm.kind:<13} {arm.key}")
        print()


# Arguments

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the synthetic results of the PACE manuscript.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run with --list to see every model and ablation name.")
    parser.add_argument("--models", nargs="+", default=None, metavar="NAME",
                        help="Models to report, by name. Default: all seven.")
    parser.add_argument("--ablations", nargs="*", default=None, metavar="NAME",
                        help="Add ablation arms at the ablation SNR. Given "
                             "alone, adds all of them.")
    parser.add_argument("--conventional", action="store_true",
                        help="Refit LSQ, NNLS and MAP instead of reading the "
                             "fits that come with the data. Slower.")
    parser.add_argument("--data-dir", type=Path, default=None, metavar="DIR",
                        help="Folder holding the synthetic data files. "
                             "Default: data/synthetic in this repository, or "
                             "the folder set in configs/paths.yaml.")
    parser.add_argument("--snrs", nargs="+", type=int, default=None,
                        help="Noise levels for the models. Default: 20 25 30 35.")
    parser.add_argument("--seeds", nargs="+", type=int, default=None,
                        help="Restrict to these seeds. Default: every seed "
                             "with a checkpoint.")
    parser.add_argument("--device", default=None,
                        help="cuda, mps or cpu. Default: whichever is present.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Rerun inference even where results exist.")
    parser.add_argument("--skip-inference", action="store_true",
                        help="Only compute metrics and figures from the "
                             "results already present.")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--compare-to", type=Path, default=None, metavar="DIR",
                        help="A results folder from an earlier run, to "
                             "compare the new results against.")
    parser.add_argument("--skip-flag-check", action="store_true",
                        help="Do not compare inference flags with the "
                             "architecture recorded beside each checkpoint.")
    parser.add_argument("--list", action="store_true",
                        help="List every model and ablation, then exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan and the commands, then exit.")
    return parser.parse_args(argv)


# Plan

def checkpoint_seeds(arm, snr):
    """Seeds with a checkpoint for one arm at one SNR."""
    root = REPO / "checkpoints" / arm.checkpoint_tree
    seeds = []
    if root.is_dir():
        for run in root.iterdir():
            match = RUN_DIR_RE.match(run.name)
            if (match and run.is_dir() and match["model"] == arm.model
                    and int(match["snr"]) == snr and any(run.glob("*.pt"))):
                seeds.append(int(match["seed"]))
    return sorted(seeds)


def result_seeds(root, arm, snr):
    """Seeds with an inference file for one arm at one SNR."""
    folder = root / arm.tree / f"{arm.model}_snr{snr}"
    if not folder.is_dir():
        return []
    return sorted(int(m["seed"]) for m in
                  (RESULT_FILE_RE.match(f.name) for f in folder.iterdir()) if m)


def keep_seeds(seeds, wanted):
    return [s for s in seeds if wanted is None or s in wanted]


@dataclass
class Plan:
    """What this run reports, and what it has to compute first."""
    main: list
    ablation: list
    main_snrs: list
    ablation_snr: int
    jobs: dict = field(default_factory=dict)
    problems: list = field(default_factory=list)


def build_plan(args, settings, models, ablations):
    main = pick_arms(args.models, models, "model") if args.models else list(models)
    if args.ablations is None:
        ablation = []
    elif not args.ablations:
        ablation = list(ablations)
    else:
        ablation = pick_arms(args.ablations, ablations, "ablation")

    plan = Plan(main=main, ablation=ablation,
                main_snrs=sorted(args.snrs or settings["snrs"]),
                ablation_snr=int(settings["ablation_snr"]))

    work = [(arm, snr) for arm in main for snr in plan.main_snrs]
    work += [(arm, plan.ablation_snr) for arm in ablation]
    if ablation:
        baseline = next(arm for arm in models if arm.name == "PACE")
        if (baseline, plan.ablation_snr) not in work:
            work.append((baseline, plan.ablation_snr))

    for arm, snr in work:
        if arm.kind == "conventional":
            if args.conventional:
                plan.jobs[(arm.name, snr)] = (arm, snr, [CONVENTIONAL_SEED])
            elif not result_seeds(RESULTS, arm, snr):
                plan.problems.append(
                    f"{arm.name} at {snr} dB: no results under results/"
                    f"{arm.tree}. They come with the data download, or "
                    f"pass --conventional to fit them.")
            continue
        if args.skip_inference:
            if not keep_seeds(result_seeds(RESULTS, arm, snr), args.seeds):
                plan.problems.append(f"{arm.name} at {snr} dB: no results under "
                                     f"results/{arm.tree} to report")
            continue
        seeds = keep_seeds(checkpoint_seeds(arm, snr), args.seeds)
        if not seeds:
            plan.problems.append(f"{arm.name} at {snr} dB: no checkpoints under "
                                 f"checkpoints/{arm.checkpoint_tree}")
            continue
        plan.jobs[(arm.name, snr)] = (arm, snr, seeds)
    return plan


def data_problems(plan, paths):
    """Data files the plan needs that are missing."""
    needed = {Path(paths.bvals), Path(paths.synthetic_test)}
    for arm in plan.main + plan.ablation:
        needed.add(test_path(arm, paths))
    for arm, _, _ in plan.jobs.values():
        needed.add(test_path(arm, paths))
    return [f"missing data file: {shown(path)}"
            for path in sorted(needed) if not path.is_file()]


def print_plan(plan, args, paths):
    print(f"[data ] {shown(paths.data_dir)}")
    print("[plan ] models    " + (", ".join(a.name for a in plan.main) or "none")
          + f"  at {', '.join(map(str, plan.main_snrs))} dB")
    if plan.ablation:
        print("[plan ] ablations " + ", ".join(a.name for a in plan.ablation)
              + f"  at {plan.ablation_snr} dB")
    if args.skip_inference:
        print("[plan ] inference skipped; reading existing results")
    else:
        print(f"[plan ] {len(plan.jobs)} inference job(s)")
    print()


# Flag check

def flag_value(flags, name):
    """Value after a flag, True for a bare flag, None when absent."""
    if name not in flags:
        return None
    index = flags.index(name)
    if index + 1 < len(flags) and not flags[index + 1].startswith("--"):
        return flags[index + 1]
    return True


def read_summary(arm):
    """Scalar fields of the first summary beside one of the arm's checkpoints."""
    import numpy as np
    root = REPO / "checkpoints" / arm.checkpoint_tree
    for run in sorted(root.glob(f"{arm.model}_snr*_seed*")):
        files = sorted(run.glob("summary_*.npz"))
        if not files:
            continue
        doc = np.load(files[0], allow_pickle=True)
        out = {}
        for key in doc.files:
            if doc[key].size == 1:
                out[key] = doc[key].item()
        return out
    return {}


def flag_problems(arm, summary):
    """Disagreements between the inference flags and the recorded architecture.

    Only fields the summary records are checked. The neighbor oracle
    options are not recorded, so they rest on the reference comparison.
    """
    flags, problems = arm.infer_flags, []
    spatial = bool(summary.get("spatial_on", True))

    def expect(condition, message):
        if not condition:
            problems.append(message)

    if spatial and "use_b0" in summary:
        has = flag_value(flags, "--no-b0") is not None
        expect(has == (not summary["use_b0"]),
               f"summary use_b0={summary['use_b0']}, flags "
               f"{'include' if has else 'omit'} --no-b0")
    if spatial and "use_struct" in summary:
        has = flag_value(flags, "--no-struct") is not None
        expect(has == (not summary["use_struct"]),
               f"summary use_struct={summary['use_struct']}, flags "
               f"{'include' if has else 'omit'} --no-struct")
    neighbors = summary.get("use_ivim_nb")
    if neighbors is None and "b0_channels" in summary:
        neighbors = int(summary["b0_channels"]) > 1
    if spatial and neighbors is not None:
        has = flag_value(flags, "--ivim-nb") is not None
        expect(has == bool(neighbors),
               f"summary records neighbor input={bool(neighbors)}, flags "
               f"{'include' if has else 'omit'} --ivim-nb")
    if spatial and "use_anat_token" in summary:
        value = flag_value(flags, "--anat-token")
        token_on = value is None or str(value) != "0"
        expect(token_on == bool(summary["use_anat_token"]),
               f"summary use_anat_token={summary['use_anat_token']}, flags "
               f"give {token_on}")
    if "widen_bounds" in summary:
        value = flag_value(flags, "--voorter-bounds")
        if value is not True:
            given = 0.0 if value is None else float(value)
            expect(abs(given - float(summary["widen_bounds"])) < 1e-9,
                   f"summary widen_bounds={summary['widen_bounds']}, flags "
                   f"give {given}")
    if "patch_size" in summary:
        value = flag_value(flags, "--patch-size")
        given = 3 if value in (None, True) else int(value)
        expect(given == int(summary["patch_size"]),
               f"summary patch_size={summary['patch_size']}, flags give {given}")
    if not spatial and "use_softmax_fractions" in summary:
        value = flag_value(flags, "--softmax-fractions")
        given = value is not None and str(value) != "0"
        expect(given == bool(summary["use_softmax_fractions"]),
               f"summary use_softmax_fractions="
               f"{summary['use_softmax_fractions']}, flags give {given}")
    return problems


def check_flags(plan):
    """Return a list of problems across every network arm in the plan."""
    problems, seen = [], set()
    for arm, _, _ in plan.jobs.values():
        if arm.kind != "network" or arm.name in seen:
            continue
        seen.add(arm.name)
        summary = read_summary(arm)
        if not summary:
            print(f"[flags] {arm.name}: no summary file, not checked")
            continue
        found = flag_problems(arm, summary)
        print(f"[flags] {arm.name}: "
              + ("consistent" if not found else f"{len(found)} problem(s)"))
        problems += [f"{arm.name}: {p}" for p in found]
    return problems


# Inference

def inference_command(arm, snr, seeds, args, paths):
    """The command for one job, or None for a kind with no inference."""
    tail = (["--device", args.device] if args.device else [])
    seed_args = ["--seeds", *map(str, seeds)]
    if arm.kind == "network":
        return [sys.executable, str(SCRIPTS / "reproduce_synthetic_results.py"),
                "--models", arm.model, "--snrs", str(snr), *seed_args,
                "--checkpoints", f"checkpoints/{arm.checkpoint_tree}",
                "--out", f"results/{arm.tree}", *arm.infer_flags, *tail,
                *(["--overwrite"] if args.overwrite else [])]
    if arm.kind == "ivim3c":
        return [sys.executable, str(SCRIPTS / "IVIM3C_infer.py"),
                "--checkpoints", f"checkpoints/{arm.checkpoint_tree}",
                "--test-h5", str(test_path(arm, paths)), "--out", f"results/{arm.tree}",
                "--snrs", str(snr), *seed_args, *tail,
                *(["--overwrite"] if args.overwrite else [])]
    return None


def conventional_commands(plan):
    """One fit per SNR covering every selected conventional method."""
    by_snr = {}
    for arm, snr, _ in plan.jobs.values():
        if arm.kind == "conventional":
            by_snr.setdefault(snr, []).append(arm)
    commands = []
    for snr, arms in sorted(by_snr.items()):
        methods = [arm.name.upper() for arm in arms]
        commands.append((f"{'/'.join(methods)} at {snr} dB", [
            sys.executable, str(SCRIPTS / "fit_conventional.py"),
            "--snr", str(snr), "--seed", str(CONVENTIONAL_SEED),
            "--out", f"results/{arms[0].tree}", "--methods", *methods]))
    return commands


def run_command(label, command, dry_run):
    """Run one command from the repository root. Returns True on success."""
    shown = " ".join(["python"] + [str(Path(c).relative_to(REPO))
                                   if c.startswith(str(REPO)) else c
                                   for c in command[1:]])
    print(f"[run  ] {label}")
    print(f"        {shown}")
    if dry_run:
        return True
    sys.stdout.flush()
    start = time.time()
    code = subprocess.run(command, cwd=REPO).returncode
    status = "ok" if code == 0 else f"FAILED, exit {code}"
    print(f"[{'done ' if code == 0 else 'FAIL '}] {label}: {status}, "
          f"{time.time() - start:.0f}s")
    print()
    return code == 0


def run_inference(plan, args, paths):
    """Run every job; return the names of arms with a failed job."""
    failed = set()
    for arm, snr, seeds in plan.jobs.values():
        command = inference_command(arm, snr, seeds, args, paths)
        if command is None:
            continue
        label = f"{arm.name} at {snr} dB, seeds {', '.join(map(str, seeds))}"
        if not run_command(label, command, args.dry_run):
            failed.add(arm.name)
    for label, command in conventional_commands(plan):
        if not run_command(label, command, args.dry_run):
            failed.update(arm.name for arm, _, _ in plan.jobs.values()
                          if arm.kind == "conventional")
    return failed


# Phantom guard

def phantom_problems(arms_snrs, reference_h5):
    """Result files whose tissue labels differ from the scoring test set.

    Every metric takes its ground truth from the test set, so a result
    produced on a different phantom, or in a different voxel order, would
    be scored against the wrong truth without any error.
    """
    import h5py
    import numpy as np
    with h5py.File(reference_h5, "r") as handle:
        tissue = np.asarray(handle["Tissue"][:]).ravel()
        lesion = np.asarray(handle["Lesion"][:]).ravel()

    bad = {}
    for arm, snr in arms_snrs:
        for seed in result_seeds(RESULTS, arm, snr):
            path = RESULTS / arm.tree / f"{arm.model}_snr{snr}" / f"inferred_seed{seed}.h5"
            with h5py.File(path, "r") as handle:
                if "Tissue" not in handle or "Lesion" not in handle:
                    continue
                same = (np.array_equal(np.asarray(handle["Tissue"][:]).ravel(), tissue)
                        and np.array_equal(np.asarray(handle["Lesion"][:]).ravel(), lesion))
            if not same:
                bad.setdefault(arm.name, []).append(f"{snr} dB seed {seed}")
    return bad


# Metrics

def load_engine():
    """The metric functions of scripts/analyse_arms.py and the pace helpers."""
    sys.path.insert(0, str(REPO))
    from pace import common
    spec = importlib.util.spec_from_file_location("analyse_arms",
                                                  SCRIPTS / "analyse_arms.py")
    analyse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyse)
    return analyse, common, common.load_paths()


def cnr_rmse(results, parameter):
    """Root mean square distance of the predicted lesion CNR from the truth,
    defined as in the figure scripts."""
    import numpy as np
    predicted = np.asarray([r["cnr_pred"][parameter] for r in results], float)
    truth = np.asarray([r["cnr_true"][parameter] for r in results], float)
    usable = np.isfinite(predicted) & np.isfinite(truth)
    if not usable.any():
        return float("nan")
    return float(np.sqrt(np.mean((predicted[usable] - truth[usable]) ** 2)))


def arm_metrics(engine, arm, snrs, seeds_wanted, group, root=RESULTS):
    """Summary metrics for one arm, averaged as its manuscript figure does.

    Seeds are averaged within an SNR, then SNRs equally. group decides
    the remaining two choices as the matching figure makes them: lesion
    CNR RMSE pools the SNR means for the model comparison and every seed
    for the ablations, and signal RMSE is pooled over voxels for the first
    and averaged over tissues for the second. The seed filter does not
    apply to conventional fits, which have a single seed.
    """
    analyse, common, base = engine
    if arm.kind == "conventional":
        seeds_wanted = None
    per_snr, every = [], []
    for snr in snrs:
        seeds = keep_seeds(result_seeds(root, arm, snr), seeds_wanted)
        rows = [analyse.one_seed(common, base, root, arm.tree, arm.model,
                                 snr, seed, None) for seed in seeds]
        if rows:
            per_snr.append(analyse._mean_of(rows) if len(rows) > 1 else rows[0])
            every.extend(rows)
    if not per_snr:
        return None
    overall = analyse._mean_of(per_snr) if len(per_snr) > 1 else per_snr[0]
    pool = every if group == "ablation" else per_snr
    signal_key = "mean" if group == "ablation" else "all"
    return {
        "files": len(every),
        "snrs": len(per_snr),
        "signal": overall["signal_rmse"][signal_key],
        "rmse": {p: overall["param_rmse"]["mean"][p] for p in PARAMS},
        "cnr": {p: cnr_rmse(pool, p) for p in PARAMS},
    }


def three_sig(value):
    """Three significant figures in plain decimal notation."""
    import math
    if value is None or not math.isfinite(value):
        return "-"
    if value == 0:
        return "0.00"
    digits = max(0, 2 - int(math.floor(math.log10(abs(value)))))
    return f"{value:.{digits}f}"


def print_table(title, rows, columns, getter):
    width = max([len(label) for label, _ in rows] + [10]) + 2
    print(title)
    print(" " * width + "".join(f"{c:>10}" for c in columns))
    for label, metrics in rows:
        print(f"{label:<{width}}" + "".join(
            f"{three_sig(getter(metrics, c)):>10}" for c in columns))
    print()


def scaled(metrics, parameter):
    factor = 1e3 if parameter in DIFFUSIVITIES else 1.0
    return metrics["rmse"][parameter] * factor


def report_tables(title, rows):
    """Print the parameter and contrast tables for one group of arms."""
    if not rows:
        return
    print_table(f"{title}: signal and parameter RMSE "
                f"(diffusivities in 10^-3 mm^2/s)", rows,
                ["Signal", *PARAMS],
                lambda m, c: m["signal"] if c == "Signal" else scaled(m, c))
    print_table(f"{title}: lesion CNR RMSE", rows, list(PARAMS),
                lambda m, c: m["cnr"][c])


def write_csv(path, groups):
    """Raw values, diffusivity RMSE in mm^2/s."""
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["group", "name", "label", "tree", "model", "snrs",
                         "files", "signal_rmse"]
                        + [f"{p}_rmse" for p in PARAMS]
                        + [f"{p}_cnr_rmse" for p in PARAMS])
        for group, entries in groups:
            for arm, metrics in entries:
                writer.writerow(
                    [group, arm.name, arm.label, arm.tree, arm.model,
                     metrics["snrs"], metrics["files"], metrics["signal"]]
                    + [metrics["rmse"][p] for p in PARAMS]
                    + [metrics["cnr"][p] for p in PARAMS])


def write_full_report(engine, path, plan, main_arms, ablation_arms, baseline, args):
    """The complete analyse_arms report for the arms that were reported."""
    analyse = engine[0]
    seed_args = ["--seeds", *map(str, args.seeds)] if args.seeds else []
    sections = []
    if main_arms:
        snr_args = (["--all-snr", *map(str, plan.main_snrs)]
                    if len(plan.main_snrs) > 1 else ["--snr", str(plan.main_snrs[0])])
        sections.append(("MODELS", main_arms, snr_args))
    if ablation_arms:
        sections.append(("ABLATIONS", [baseline] + ablation_arms,
                         ["--snr", str(plan.ablation_snr)]))
    with open(path, "w", encoding="utf-8") as handle:
        for title, arms, snr_args in sections:
            trees = sorted({arm.tree for arm in arms})
            models = sorted({arm.model for arm in arms})
            argv = ["--repo", str(REPO), "--trees", *trees, "--models", *models,
                    *snr_args, *seed_args]
            handle.write(f"{'#' * 60}\n# {title}\n{'#' * 60}\n\n")
            handle.flush()
            with contextlib.redirect_stdout(handle):
                analyse.main(argv)
            handle.write("\n")


# Figures

def figure_commands(plan, main_arms, ablation_arms):
    commands = []
    if len(main_arms) >= 2:
        snrs = plan.main_snrs
        distribution = 25 if 25 in snrs else snrs[0]
        commands.append(("model comparison figures", [
            sys.executable, str(SCRIPTS / "make_model_comparison.py"),
            "--snrs", *map(str, snrs), "--distribution-snr", str(distribution),
            "--select", *[arm.key for arm in main_arms]]))
    elif main_arms:
        print("[skip ] model comparison figures need at least two models")
    if ablation_arms:
        commands.append(("ablation heatmaps", [
            sys.executable, str(SCRIPTS / "make_ablation_heatmaps.py"),
            "--snr", str(plan.ablation_snr),
            "--select", *[arm.key for arm in ablation_arms]]))
    return commands


# Reference comparison

def compare_with_reference(engine, arms_snrs, reference_root):
    """Largest relative difference in parameter RMSE, per arm, per file."""
    analyse, common, base = engine
    worst = {}
    for arm, snr in arms_snrs:
        shared = sorted(set(result_seeds(RESULTS, arm, snr))
                        & set(result_seeds(reference_root, arm, snr)))
        for seed in shared:
            new = analyse.one_seed(common, base, RESULTS, arm.tree, arm.model,
                                   snr, seed, None)
            old = analyse.one_seed(common, base, reference_root, arm.tree,
                                   arm.model, snr, seed, None)
            for p in PARAMS:
                a, b = new["param_rmse"]["mean"][p], old["param_rmse"]["mean"][p]
                if b:
                    gap = abs(a - b) / abs(b)
                    entry = worst.setdefault(arm.name, [0.0, 0])
                    entry[0] = max(entry[0], gap)
            if arm.name in worst:
                worst[arm.name][1] += 1
    return worst


def table_gaps(engine, rows, snrs, group, seeds, reference_root):
    """Largest relative difference in the reported parameter RMSE, per arm.

    These are the values the summary table prints, averaged over seeds and
    SNRs, which is what a reader compares against the manuscript.
    """
    gaps = {}
    for arm, metrics in rows:
        reference = arm_metrics(engine, arm, snrs, seeds, group, root=reference_root)
        if not reference:
            continue
        gaps[arm.name] = max(
            abs(metrics["rmse"][p] - reference["rmse"][p]) / abs(reference["rmse"][p])
            for p in PARAMS if reference["rmse"][p])
    return gaps


def print_comparison(arms, worst, table, reference_root):
    print(f"Comparison with {reference_root}")
    print("  largest relative difference in parameter RMSE, in the reported "
          "table and in the")
    print(f"  worst single file; the worst file is judged, tolerance "
          f"{COMPARE_TOLERANCE:.0%}")
    print(f"  {'':<16} {'table':>8}  {'worst file':>10}")
    for arm in arms:
        if arm.name not in worst or arm.name not in table:
            print(f"  {arm.name:<16} no matching reference files")
            continue
        gap, files = worst[arm.name]
        if arm.kind == "ivim3c":
            status = "not judged, external comparator"
        else:
            status = "OK" if gap <= COMPARE_TOLERANCE else "MISMATCH"
        print(f"  {arm.name:<16} {table[arm.name]:8.3%}  {gap:10.3%}  "
              f"over {files} file(s)  {status}")
    print()


# Main

def main(argv=None):
    if sys.version_info < (3, 10):
        print("[ERROR] Python 3.10 or newer is required.", file=sys.stderr)
        return 2
    args = parse_args(argv)
    settings, models, ablations = load_registry()
    if args.data_dir:
        os.environ["PACE_DATA_DIR"] = str(args.data_dir.expanduser().resolve())
    if args.list:
        print_registry(models, ablations)
        return 0

    paths = data_paths()
    plan = build_plan(args, settings, models, ablations)
    print_plan(plan, args, paths)
    problems = plan.problems + data_problems(plan, paths)
    if problems:
        for problem in problems:
            print(f"[ERROR] {problem}", file=sys.stderr)
        return 2
    if not args.skip_inference and not args.skip_flag_check:
        flag_errors = check_flags(plan)
        if flag_errors:
            for problem in flag_errors:
                print(f"[ERROR] {problem}", file=sys.stderr)
            print("[ERROR] fix configs/arms.json, or pass --skip-flag-check",
                  file=sys.stderr)
            return 2
        print()

    failed = set() if args.skip_inference else run_inference(plan, args, paths)
    if args.dry_run:
        for label, command in figure_commands(plan, plan.main, plan.ablation):
            run_command(label, command, True)
        return 0

    baseline = next(arm for arm in models if arm.name == "PACE")
    report_pairs = [(arm, snr) for arm in plan.main for snr in plan.main_snrs]
    report_pairs += [(arm, plan.ablation_snr) for arm in plan.ablation]
    if plan.ablation:
        report_pairs.append((baseline, plan.ablation_snr))
    report_pairs = list({(arm.name, snr): (arm, snr)
                         for arm, snr in report_pairs}.values())

    bad = phantom_problems(report_pairs, paths.synthetic_test)
    for name, where in bad.items():
        print(f"[ERROR] {name}: tissue labels differ from the test set at "
              f"{', '.join(where)}; excluded from metrics and figures",
              file=sys.stderr)
    excluded = failed | set(bad)
    main_arms = [arm for arm in plan.main if arm.name not in excluded]
    ablation_arms = [arm for arm in plan.ablation if arm.name not in excluded]
    if plan.ablation and baseline.name in excluded:
        print("[ERROR] the PACE baseline failed, so ablations cannot be "
              "reported", file=sys.stderr)
        ablation_arms = []

    engine = load_engine()
    main_rows = [(arm, arm_metrics(engine, arm, plan.main_snrs, args.seeds,
                                   "model"))
                 for arm in main_arms]
    ablation_rows = []
    if ablation_arms:
        ablation_rows = [(arm, arm_metrics(engine, arm, [plan.ablation_snr],
                                           args.seeds, "ablation"))
                         for arm in [baseline] + ablation_arms]
    main_rows = [(a, m) for a, m in main_rows if m]
    ablation_rows = [(a, m) for a, m in ablation_rows if m]

    print("=" * 72)
    report_tables(f"Models, averaged over {', '.join(map(str, plan.main_snrs))} dB",
                  [(arm.label, m) for arm, m in main_rows])
    report_tables(f"Ablations at {plan.ablation_snr} dB",
                  [(arm.label, m) for arm, m in ablation_rows])

    RESULTS.mkdir(exist_ok=True)
    write_csv(RESULTS / "summary.csv", [("model", main_rows),
                                        ("ablation", ablation_rows)])
    write_full_report(engine, RESULTS / "report.txt", plan,
                      [a for a, _ in main_rows],
                      [a for a, _ in ablation_rows if a is not baseline],
                      baseline, args)

    figures_ok = True
    if not args.no_figures:
        for label, command in figure_commands(plan, [a for a, _ in main_rows],
                                              [a for a, _ in ablation_rows
                                               if a is not baseline]):
            figures_ok &= run_command(label, command, False)

    if args.compare_to:
        reference = args.compare_to.expanduser().resolve()
        worst = compare_with_reference(engine, report_pairs, reference)
        table = table_gaps(engine, ablation_rows, [plan.ablation_snr],
                           "ablation", args.seeds, reference)
        table.update(table_gaps(engine, main_rows, plan.main_snrs, "model",
                                args.seeds, reference))
        compared = {a.name: a for a, _ in main_rows + ablation_rows}
        print_comparison(list(compared.values()), worst, table, reference)

    print("=" * 72)
    print(f"[out  ] {(RESULTS / 'summary.csv').relative_to(REPO)}")
    print(f"[out  ] {(RESULTS / 'report.txt').relative_to(REPO)}")
    if not args.no_figures:
        print("[out  ] figures/")
    if excluded:
        print(f"[warn ] not reported: {', '.join(sorted(excluded))}")
    return 0 if not excluded and figures_ok else 1


if __name__ == "__main__":
    sys.exit(main())
