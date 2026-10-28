#!/usr/bin/env python3
"""Reproduce the synthetic PACE-IVIM results (models and ablations).

PACE: physics-informed three-compartment IVIM fitting with gated
cross-attention to T1w/FLAIR. Optional anatomy-weighted spatial prior.

Citation: Hoang et al. (2026), arXiv:2610.28873.
Comparator: Voorter et al. (2023), doi:10.1002/mrm.29753.
Related work: Kaandorp et al. (2025), doi:10.1016/j.media.2024.103414.
See README.md for setup, data, commands, and the full citation.
"""

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

# Repository layout and fixed settings:
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

# Zenodo record holding the data and weights; each archive unpacks into
# data/, checkpoints/ and results/ at the repository root:
ZENODO_RECORD = "23746536"
RECORD_URL = f"https://zenodo.org/records/{ZENODO_RECORD}"
DOWNLOAD_BASE = os.environ.get("PACE_DOWNLOAD_BASE", f"{RECORD_URL}/files")
ARCHIVES = {
    "core": ("PACE_IVIM_core.zip",
             "9edea2ea87fd1348844f03f8386128bd64fe34d86608ba648ed49d28653a4f55",
             "test set, b-values, weights of the four networks, conventional fits"),
    "ablations": ("PACE_IVIM_ablations.zip",
                  "c7381b64a404a5d6ffcf9c7f7b2cb590516f03f2d1c9ad633f881ce0f39c4575",
                  "5x5 test set and weights of the seven ablation arms"),
    "training": ("PACE_IVIM_training.zip",
                 "b6623c8260e4b9a7117e909c5d3a41b6300559b16ad02c1e3dc10bf2a8bb615e",
                 "training and validation sets, 1.7 GB"),
}
ARCHIVE_ROOTS = ("data/", "checkpoints/", "results/")


# Read the model and ablation registry from configs/arms.json:
@dataclass
class Arm:
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
    """Return (settings, models, ablations)."""
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


def pick_arms(names, pool, what):
    """Arms matching names, case insensitive, in registry order."""
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


# Helpers to check file paths:
def data_paths():
    """Resolved data locations, so it follows --data-dir and configs/paths.yaml."""
    sys.path.insert(0, str(REPO))
    from pace import common
    return common.load_paths()


def test_path(arm, paths):
    if arm.test_file == Path(paths.synthetic_test).name:
        return Path(paths.synthetic_test)
    return Path(paths.data_dir) / arm.test_file


def shown(path):
    path = Path(path)
    return str(path.relative_to(REPO)) if path.is_relative_to(REPO) else str(path)


# Parse command line options:
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Reproduce the synthetic PACE-IVIM results.",
        epilog="Use --list for model and ablation names.")
    p.add_argument("--models", nargs="+", metavar="NAME", help="Models (default: all seven)")
    p.add_argument("--ablations", nargs="*", metavar="NAME", help="Add selected or all ablations")
    p.add_argument("--conventional", action="store_true", help="Refit LSQ/NNLS/MAP")
    p.add_argument("--data-dir", type=Path, metavar="DIR", help="Synthetic data directory")
    p.add_argument("--snrs", nargs="+", type=int, help="SNRs (default: 20 25 30 35)")
    p.add_argument("--seeds", nargs="+", type=int, help="Restrict to these seeds")
    p.add_argument("--device", help="cuda, mps, or cpu (default: auto)")
    p.add_argument("--overwrite", action="store_true", help="Rerun inference")
    p.add_argument("--skip-inference", action="store_true", help="Reuse existing results")
    p.add_argument("--no-figures", action="store_true", help="Skip figures")
    p.add_argument("--compare-to", type=Path, metavar="DIR", help="Compare with earlier results")
    p.add_argument("--no-download", action="store_true", help="Disable automatic downloads")
    p.add_argument("--download-only", nargs="+", choices=sorted(ARCHIVES), metavar="PART",
                   help="Download core, ablations, or training and exit")
    p.add_argument("--list", action="store_true", help="List models and ablations")
    p.add_argument("--dry-run", action="store_true", help="Print planned commands only")
    return p.parse_args(argv)


# Plan the run: which arms, SNRs and seeds, and what is missing:
def checkpoint_seeds(arm, snr):
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
    folder = root / arm.tree / f"{arm.model}_snr{snr}"
    if not folder.is_dir():
        return []
    return sorted(int(m["seed"]) for m in
                  (RESULT_FILE_RE.match(f.name) for f in folder.iterdir()) if m)


def keep_seeds(seeds, wanted):
    return [s for s in seeds if wanted is None or s in wanted]


@dataclass
class Plan:
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

    # Ablations are scored against PACE, so PACE runs at the ablation SNR too.
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


# Download missing data and weights from Zenodo, verified by SHA256:
def part_present(part, paths):
    data = Path(paths.data_dir)
    if part == "core":
        return (Path(paths.synthetic_test).is_file() and Path(paths.bvals).is_file()
                and (REPO / "checkpoints" / "pace_nob0").is_dir())
    if part == "ablations":
        return ((data / "synthetic_IVIM_test_2D_p5.h5").is_file()
                and (REPO / "checkpoints" / "pace_nbonly").is_dir())
    return (data / "synthetic_IVIM_train_2D.h5").is_file()


def needed_parts(args, paths):
    wanted = ["core"] + (["ablations"] if args.ablations is not None else [])
    return [part for part in wanted if not part_present(part, paths)]


def extract_archive(path):
    """Unpack into the repository, refusing any path outside the known folders."""
    with zipfile.ZipFile(path) as zf:
        for item in zf.infolist():
            name = item.filename
            dest = (REPO / name).resolve()
            if (not name.startswith(ARCHIVE_ROOTS) or ".." in Path(name).parts
                    or not dest.is_relative_to(REPO)):
                raise RuntimeError(f"{path.name} holds an unexpected path: {name}")
        zf.extractall(REPO)


def download_archive(part):
    """Download one archive, verify its SHA256, unpack it, delete the zip."""
    name, digest, about = ARCHIVES[part]
    print(f"[fetch] {name}: {about}")
    tmp_dir = Path(tempfile.mkdtemp(prefix="pace_download_"))
    zip_path = tmp_dir / name
    try:
        sha = hashlib.sha256()
        req = urllib.request.Request(f"{DOWNLOAD_BASE}/{name}?download=1",
                                     headers={"User-Agent": "run_PACE_IVIM"})
        with urllib.request.urlopen(req, timeout=60) as resp, \
                open(zip_path, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done, step_shown = 0, -1
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                sha.update(chunk)
                done += len(chunk)
                step = int(20 * done / total) if total else done >> 26
                if step != step_shown:
                    step_shown = step
                    size = f" of {total / 1e6:.0f}" if total else ""
                    print(f"\r        {done / 1e6:.0f}{size} MB", end="", flush=True)
        print()
        if sha.hexdigest() != digest:
            raise RuntimeError(f"{name} failed its checksum; the download is "
                               f"incomplete or altered")
        extract_archive(zip_path)
        print(f"[done ] {name} unpacked into the repository")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def fetch(parts):
    """Download each part; on failure, print the manual route."""
    for part in parts:
        try:
            download_archive(part)
        except (urllib.error.URLError, OSError, RuntimeError,
                zipfile.BadZipFile) as error:
            print(f"[ERROR] could not download {ARCHIVES[part][0]}: {error}",
                  file=sys.stderr)
            print(f"        Download it from {RECORD_URL}, then unzip it inside "
                  f"this repository:", file=sys.stderr)
            print(f"        unzip {ARCHIVES[part][0]} -d {REPO}", file=sys.stderr)
            return False
    print()
    return True


# Run inference, one command per arm and SNR:
def inference_command(arm, snr, seeds, args, paths):
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
    cmds = []
    for snr, arms in sorted(by_snr.items()):
        methods = [arm.name.upper() for arm in arms]
        cmds.append((f"{'/'.join(methods)} at {snr} dB", [
            sys.executable, str(SCRIPTS / "fit_conventional.py"),
            "--snr", str(snr), "--seed", str(CONVENTIONAL_SEED),
            "--out", f"results/{arms[0].tree}", "--methods", *methods]))
    return cmds


def run_command(label, command, dry_run):
    cmd_text = " ".join(["python"] + [str(Path(c).relative_to(REPO))
                                      if c.startswith(str(REPO)) else c
                                      for c in command[1:]])
    print(f"[run  ] {label}")
    print(f"        {cmd_text}")
    if dry_run:
        return True
    sys.stdout.flush()
    t0 = time.time()
    code = subprocess.run(command, cwd=REPO).returncode
    status = "ok" if code == 0 else f"FAILED, exit {code}"
    print(f"[{'done ' if code == 0 else 'FAIL '}] {label}: {status}, "
          f"{time.time() - t0:.0f}s")
    print()
    return code == 0


def run_inference(plan, args, paths):
    """Run every job; return the names of arms with a failed job."""
    failed = set()
    for arm, snr, seeds in plan.jobs.values():
        cmd = inference_command(arm, snr, seeds, args, paths)
        if cmd is None:
            continue
        label = f"{arm.name} at {snr} dB, seeds {', '.join(map(str, seeds))}"
        if not run_command(label, cmd, args.dry_run):
            failed.add(arm.name)
    for label, cmd in conventional_commands(plan):
        if not run_command(label, cmd, args.dry_run):
            failed.update(arm.name for arm, _, _ in plan.jobs.values()
                          if arm.kind == "conventional")
    return failed


# Exclude results whose tissue labels do not match the test set, since every
# metric takes its ground truth from the test set:
def phantom_problems(arms_snrs, reference_h5):
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


# Compute metrics with scripts/analyse_arms.py, the same code the figures use:
def load_engine():
    sys.path.insert(0, str(REPO))
    from pace import common
    spec = importlib.util.spec_from_file_location("analyse_arms",
                                                  SCRIPTS / "analyse_arms.py")
    analyse = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyse)
    return analyse, common, common.load_paths()


def cnr_rmse(results, parameter):
    import numpy as np
    predicted = np.asarray([r["cnr_pred"][parameter] for r in results], float)
    truth = np.asarray([r["cnr_true"][parameter] for r in results], float)
    usable = np.isfinite(predicted) & np.isfinite(truth)
    if not usable.any():
        return float("nan")
    return float(np.sqrt(np.mean((predicted[usable] - truth[usable]) ** 2)))


def arm_metrics(engine, arm, snrs, seeds_wanted, group, root=RESULTS):
    """Seeds averaged within an SNR, then SNRs equally, as in the figures."""
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
    # The ablation heatmap pools every seed and averages signal over tissues;
    # the model comparison pools the SNR means and pools signal over voxels.
    pool = every if group == "ablation" else per_snr
    signal_key = "mean" if group == "ablation" else "all"
    return {
        "files": len(every),
        "snrs": len(per_snr),
        "signal": overall["signal_rmse"][signal_key],
        "rmse": {p: overall["param_rmse"]["mean"][p] for p in PARAMS},
        "cnr": {p: cnr_rmse(pool, p) for p in PARAMS},
    }


# Print and save the summary tables and the full report:
def three_sig(value):
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
    if not rows:
        return
    print_table(f"{title}: signal and parameter RMSE "
                f"(diffusivities in 10^-3 mm^2/s)", rows,
                ["Signal", *PARAMS],
                lambda m, c: m["signal"] if c == "Signal" else scaled(m, c))
    print_table(f"{title}: lesion CNR RMSE", rows, list(PARAMS),
                lambda m, c: m["cnr"][c])


def write_csv(path, groups):
    """Raw values; diffusivity RMSE in mm^2/s."""
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


# Draw figures for the selected arms:
def figure_commands(plan, main_arms, ablation_arms):
    cmds = []
    if len(main_arms) >= 2:
        snrs = plan.main_snrs
        dist_snr = 25 if 25 in snrs else snrs[0]
        cmds.append(("model comparison figures", [
            sys.executable, str(SCRIPTS / "make_model_comparison.py"),
            "--snrs", *map(str, snrs), "--distribution-snr", str(dist_snr),
            "--select", *[arm.key for arm in main_arms]]))
    elif main_arms:
        print("[skip ] model comparison figures need at least two models")
    if ablation_arms:
        cmds.append(("ablation heatmaps", [
            sys.executable, str(SCRIPTS / "make_ablation_heatmaps.py"),
            "--snr", str(plan.ablation_snr),
            "--select", *[arm.key for arm in ablation_arms]]))
    return cmds


# Compare results with an earlier run (--compare-to), per file and per table:
def compare_with_reference(engine, arms_snrs, reference_root):
    """Largest relative difference in parameter RMSE in any single file, per arm."""
    analyse, common, base = engine
    worst = {}
    for arm, snr in arms_snrs:
        seeds = sorted(set(result_seeds(RESULTS, arm, snr))
                       & set(result_seeds(reference_root, arm, snr)))
        for seed in seeds:
            cur = analyse.one_seed(common, base, RESULTS, arm.tree, arm.model,
                                   snr, seed, None)
            ref = analyse.one_seed(common, base, reference_root, arm.tree,
                                   arm.model, snr, seed, None)
            for p in PARAMS:
                a, b = cur["param_rmse"]["mean"][p], ref["param_rmse"]["mean"][p]
                if b:
                    gap = abs(a - b) / abs(b)
                    entry = worst.setdefault(arm.name, [0.0, 0])
                    entry[0] = max(entry[0], gap)
            if arm.name in worst:
                worst[arm.name][1] += 1
    return worst


def table_gaps(engine, rows, snrs, group, seeds, reference_root):
    """Largest relative difference in the reported table values, per arm."""
    gaps = {}
    for arm, metrics in rows:
        ref = arm_metrics(engine, arm, snrs, seeds, group, root=reference_root)
        if not ref:
            continue
        gaps[arm.name] = max(
            abs(metrics["rmse"][p] - ref["rmse"][p]) / abs(ref["rmse"][p])
            for p in PARAMS if ref["rmse"][p])
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


# Main:
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

    # Fetch whatever this run needs, then plan it.
    paths = data_paths()
    if args.download_only:
        return 0 if fetch(args.download_only) else 1
    if not args.no_download:
        missing = needed_parts(args, paths)
        if missing and not fetch(missing):
            return 2
    plan = build_plan(args, settings, models, ablations)
    print_plan(plan, args, paths)
    problems = plan.problems + data_problems(plan, paths)
    if problems:
        for problem in problems:
            print(f"[ERROR] {problem}", file=sys.stderr)
        return 2

    # Inference.
    failed = set() if args.skip_inference else run_inference(plan, args, paths)
    if args.dry_run:
        for label, command in figure_commands(plan, plan.main, plan.ablation):
            run_command(label, command, True)
        return 0

    # Keep only arms that ran and match the test set.
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

    # Metrics, tables and the full report.
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

    # Figures.
    fig_ok = True
    if not args.no_figures:
        for label, command in figure_commands(plan, [a for a, _ in main_rows],
                                              [a for a, _ in ablation_rows
                                               if a is not baseline]):
            fig_ok &= run_command(label, command, False)

    # Optional comparison with an earlier run.
    if args.compare_to:
        ref_root = args.compare_to.expanduser().resolve()
        worst = compare_with_reference(engine, report_pairs, ref_root)
        table = table_gaps(engine, ablation_rows, [plan.ablation_snr],
                           "ablation", args.seeds, ref_root)
        table.update(table_gaps(engine, main_rows, plan.main_snrs, "model",
                                args.seeds, ref_root))
        compared = {a.name: a for a, _ in main_rows + ablation_rows}
        print_comparison(list(compared.values()), worst, table, ref_root)

    print("=" * 72)
    print(f"[out  ] {(RESULTS / 'summary.csv').relative_to(REPO)}")
    print(f"[out  ] {(RESULTS / 'report.txt').relative_to(REPO)}")
    if not args.no_figures:
        print("[out  ] figures/")
    if excluded:
        print(f"[warn ] not reported: {', '.join(sorted(excluded))}")
    return 0 if not excluded and fig_ok else 1


if __name__ == "__main__":
    sys.exit(main())
