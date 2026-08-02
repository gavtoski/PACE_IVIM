#!/usr/bin/env python3
"""Retrain a synthetic model from scratch.

This is the only script here that trains. It builds a network, runs the
optimiser, and writes a new checkpoint. Contrast with
reproduce_synthetic_results.py, which only loads weights that already
exist.

    INPUTS                                       OUTPUT
    data/synthetic/synthetic_IVIM_train_2D.h5  ─┐
    data/synthetic/synthetic_IVIM_val_2D.h5    ─┴─▶ checkpoints/
                                                     synthetic_retrained/

The training set is not distributed with this repository. It is 136 MB and
is needed only to retrain, not to reproduce any published result. It is
available from the authors on request.

On reproducibility
------------------
A retrained network will not match the released weights. Training is far
more stochastic than inference: dropout, data shuffling and noise
injection all draw from a random stream that differs by compute backend,
and those differences compound over 35 epochs. What should agree is where
training lands. --compare reports the retrained best_val_mse against the
value recorded when the released checkpoint was trained.

Architecture comes from configs/models.json, the same source
reproduce_synthetic_results.py reads, so a retrained network cannot
silently differ in configuration from the released one.

Output goes to checkpoints/synthetic_retrained. The script refuses to
write into checkpoints/synthetic, so the released weights cannot be
overwritten.

Usage
-----
    python scripts/train_synthetic.py --models pace --snrs 25 --seeds 19 --epochs 2
    python scripts/train_synthetic.py --models pace --snrs 25 --seeds 19
    python scripts/train_synthetic.py --models pace --snrs 25 --seeds 19 --compare
"""

from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pace import common as C

# Reuse the in-memory loader rather than reading the H5 per voxel. scripts/
# is on sys.path when a script here is run directly.
from reproduce_synthetic_results import InMemoryIVIMDataset

# Loss weights and schedule, from the training driver that produced the
# released checkpoints. These are training hyperparameters rather than
# architecture, so they live here rather than in the manifest. The values
# are confirmed by the summary files shipped alongside each checkpoint,
# which record alpha_ce, alpha_recon and alpha_penalty as they were used.
TRAIN_CONFIG = {
    "recon_all": False,
    "alpha_ce": 0.2,
    "alpha_recon": 0.0,
    "alpha_penalty": 0.1,
    "latent_dropout_p": 0.3,
    "modality_dropout_p": 0.15,
}

DEFAULTS = {
    "epochs": 35,
    "lr": 5e-4,
    "batch_size": 256,
    "ivim_latent_dim": 32,
    "scheduler_delay": 20,
    "scheduler_warmup": 5,
    "recon_warmup_start": 5,
    "recon_warmup_length": 5,
}

# The DNN configuration switches off the spatial pathway entirely, so the
# dropout terms that act on it are zeroed too.
DNN_OVERRIDES = {"latent_dropout_p": 0.0, "modality_dropout_p": 0.0}


def _reject_if_inside(out_root, protected, label):
    """Refuse an output path that is the protected tree or sits inside it.

    Comparing for equality alone is not enough: a subdirectory such as
    checkpoints/synthetic/logs would pass an equality test and still
    overwrite released material. is_relative_to catches both cases, and
    resolve() first normalises traversal, trailing slashes and absolute
    paths so they cannot be used to slip past.
    """
    a, b = out_root.resolve(), protected.resolve()
    if a == b or a.is_relative_to(b):
        print(f"[ERROR] refusing to write to {a}, which is inside "
              f"{label}. That directory holds released material. "
              f"Choose another --out.", file=sys.stderr)
        return True
    return False


def released_summary(ck, paths):
    """Best validation MSE recorded when the released checkpoint trained."""
    d = paths.checkpoints_dir / "logs" / ck.public_name
    files = sorted(d.glob("summary_*.npz"))
    if not files:
        return None
    s = np.load(files[0], allow_pickle=True)
    out = {}
    for k in ["best_val_mse", "best_epoch", "best_val_acc", "best_val_ce",
              "use_ordered_diffusion", "use_softmax_fractions"]:
        if k in s.files:
            out[k] = s[k].item() if s[k].ndim == 0 else s[k]
    return out


def retrained_summary(out_dir):
    files = sorted(Path(out_dir).glob("summary_*.npz"))
    if not files:
        return None
    s = np.load(files[0], allow_pickle=True)
    return {k: (s[k].item() if s[k].ndim == 0 else s[k])
            for k in ["best_val_mse", "best_epoch", "best_val_acc",
                      "best_val_ce"] if k in s.files}


def resolve_config(ck):
    """Assemble the full keyword set for one training run.

    Three sources, kept separate on purpose:

      ARCH_CONFIGS   what the network is made of, per model
      the manifest   the three flags that change what it computes
      TRAIN_CONFIG   loss weights and dropout, shared by all models

    The flags come from the manifest rather than from a local dictionary
    so that a network trained here cannot silently differ in
    configuration from the released one. That is the failure this split
    is guarding against: in the original analysis these flags were set
    from three separate places and one of them was easy to miss.
    """
    arch = C.ARCH_CONFIGS[ck.cfg_name]

    cfg = dict(TRAIN_CONFIG)

    for key in ["spatial_on", "use_struct", "use_b0", "parallel_heads",
                "recon_struct", "recon_b0", "detach_recon",
                "detach_spatial_delta", "use_anat_token"]:
        cfg[key] = arch[key]

    cfg["use_ordered_diffusion"] = ck.use_ordered_diffusion
    cfg["use_softmax_fractions"] = ck.use_softmax_fractions
    cfg["fusion_mode"] = ck.fusion_mode

    # Without a spatial pathway there is nothing for these to drop out.
    if not arch["spatial_on"]:
        cfg.update(DNN_OVERRIDES)

    return cfg


def seed_everything(seed):
    """Seed every generator the training loop draws from."""
    import random
    import torch
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_datasets(paths, need_spatial):
    """Read the training and validation sets into memory.

    Reading the H5 once and slicing arrays is roughly two orders of
    magnitude faster than the per voxel access the default loader uses.
    """
    train_h5 = paths.data_dir / "synthetic_IVIM_train_2D.h5"
    t0 = time.time()
    tr = InMemoryIVIMDataset(str(train_h5), use_struct=need_spatial,
                             use_b0=need_spatial)
    va = InMemoryIVIMDataset(str(paths.synthetic_val),
                             use_struct=need_spatial, use_b0=need_spatial)
    return tr, va, time.time() - t0


def train_one(ck, paths, args, datasets):
    """Train one configuration. Returns the output directory."""
    import torch
    from pace.learn_ivim import learn_IVIM

    out_root = paths.repo_root / args.out
    out_dir = out_root / ck.public_name
    if out_dir.is_dir() and any(out_dir.glob("*.pt")) and not args.overwrite:
        print(f"  [skip] {ck.public_name} already trained")
        return out_dir

    cfg = resolve_config(ck)
    tr, va = datasets
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(ck.seed)

    t0 = time.time()
    net, history = learn_IVIM(
        train_h5=str(paths.data_dir / "synthetic_IVIM_train_2D.h5"),
        val_h5=str(paths.synthetic_val),
        bvals=C.load_bvals(paths),
        seed=ck.seed,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        out_dir=str(out_dir),
        train_dataset=tr, val_dataset=va,
        pretrained_path=None,
        ivim_latent_dim=DEFAULTS["ivim_latent_dim"],
        use_three_compartment=True, fitS0=True,
        noise_level=float(ck.snr_db),
        test_mode=True,
        run_tag=ck.cfg_name,
        scheduler_delay=args.scheduler_delay,
        scheduler_warmup=args.scheduler_warmup,
        device=str(C.resolve_device(args.device)),
        recon_warmup_start=DEFAULTS["recon_warmup_start"],
        recon_warmup_length=DEFAULTS["recon_warmup_length"],
        **cfg,
    )
    elapsed = time.time() - t0

    best = int(np.argmin(history["val_mse"]))
    print(f"  trained in {elapsed / 60:.1f} min | "
          f"best_val_mse={history['val_mse'][best]:.6g} at epoch {best}")

    del net
    return out_dir


def print_plan(todo, args):
    """Announce what is about to be trained, and with which flags."""
    print(f"[DATA  ] synthetic_IVIM_train_2D.h5")
    print(f"[OUT   ] {args.out}")
    print(f"[EPOCHS] {args.epochs}"
          + ("" if args.epochs == DEFAULTS["epochs"]
             else f"  (released weights used {DEFAULTS['epochs']})"))
    print(f"[RUN   ] {len(todo)} configuration(s)")
    print()
    for ck in todo:
        print(f"  {ck.public_name:<26} cfg={ck.cfg_name:<18} "
              f"fusion={ck.fusion_mode:<10} "
              f"ordered={ck.use_ordered_diffusion} "
              f"softmax={ck.use_softmax_fractions}")
    print()


def print_resolved_config(ck):
    """Show every keyword that would reach learn_IVIM, and its source."""
    arch = C.ARCH_CONFIGS[ck.cfg_name]
    cfg = resolve_config(ck)

    print(f"  Resolved configuration for {ck.public_name}:")
    print()
    print("  from ARCH_CONFIGS")
    for k in sorted(arch):
        print(f"    {k:<24} {arch[k]}")
    print("  from configs/models.json")
    for k in ["fusion_mode", "use_ordered_diffusion", "use_softmax_fractions"]:
        print(f"    {k:<24} {cfg[k]}")
    print("  from TRAIN_CONFIG")
    for k in sorted(TRAIN_CONFIG):
        print(f"    {k:<24} {cfg[k]}")
    print()
    print("  Re-run without --dry-run to train.")


def print_comparison(done, paths, args):
    """Show where retraining landed against the released training run."""
    print()
    print("=" * 72)
    print(" Retrained against the released training run")
    print("=" * 72)
    print(" A retrained network will not match the released weights.")
    print(" Training draws on a random stream that differs by backend, and")
    print(" 35 epochs of it compound. What should agree is roughly where")
    print(" training lands.")
    print()

    for ck, out_dir in done:
        ref = released_summary(ck, paths)
        new = retrained_summary(out_dir)
        if not ref or not new:
            print(f" {ck.public_name}: no summary to compare")
            continue

        print(f" {ck.public_name}")
        print(f"   {'':<18} {'released':>14} {'retrained':>14}")
        for k in ["best_val_mse", "best_val_acc", "best_val_ce", "best_epoch"]:
            if k in ref and k in new:
                a, b = ref[k], new[k]
                fmt = "{:>14.6g}" if isinstance(a, float) else "{:>14}"
                print(f"   {k:<18} " + fmt.format(a) + " " + fmt.format(b))

        if args.epochs != DEFAULTS["epochs"]:
            print(f"   note: trained for {args.epochs} epochs against "
                  f"{DEFAULTS['epochs']} for the released weights, so these "
                  f"are not directly comparable")
        print()

    print("=" * 72)


def preflight(args, paths, manifest):
    """Validate the request. Returns the checkpoints to train, or None.

    Three things can go wrong before any work starts: the output path
    could overwrite released material, the training set could be absent,
    or the selection could match nothing. Each prints its own reason.
    """
    out_root = paths.repo_root / args.out
    for protected, label in [(paths.checkpoints_dir, "checkpoints/synthetic"),
                             (paths.results_dir, "results/synthetic")]:
        if _reject_if_inside(out_root, protected, label):
            return None

    train_h5 = paths.data_dir / "synthetic_IVIM_train_2D.h5"
    if not train_h5.is_file():
        print(f"[ERROR] training set not found: {train_h5}", file=sys.stderr)
        print("        It is not distributed with this repository. See the",
              file=sys.stderr)
        print("        data availability note in README.md.", file=sys.stderr)
        return None

    todo = [ck for ck in manifest.checkpoints
            if (args.models is None or ck.model in args.models)
            and (args.snrs is None or ck.snr_db in args.snrs)
            and (args.seeds is None or ck.seed in args.seeds)]
    if not todo:
        print("[ERROR] no checkpoints match the selection.", file=sys.stderr)
        print(f"        models {manifest.neural_models}", file=sys.stderr)
        print(f"        SNR    {manifest.snrs()}", file=sys.stderr)
        return None

    return todo


def main():
    ap = argparse.ArgumentParser(
        description="Retrain a synthetic model from scratch.")
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--snrs", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"],
                    help="Reduce for a quick check. Released weights used 35.")
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--scheduler-delay", type=int,
                    default=DEFAULTS["scheduler_delay"])
    ap.add_argument("--scheduler-warmup", type=int,
                    default=DEFAULTS["scheduler_warmup"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="checkpoints/synthetic_retrained")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="Compare against the released training summaries.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and the resolved configuration.")
    args = ap.parse_args()

    paths = C.load_paths()
    manifest = C.load_manifest(paths)

    todo = preflight(args, paths, manifest)
    if todo is None:
        return 2

    print_plan(todo, args)

    if args.dry_run:
        print_resolved_config(todo[0])
        return 0

    try:
        import torch  # noqa: F401
    except ImportError:
        print("[ERROR] torch is required. pip install torch", file=sys.stderr)
        return 2

    dev = C.resolve_device(args.device)
    print(f"[DEVICE] {dev}")

    # An unpatched learn_IVIM hardcodes cuda or cpu and would silently
    # ignore a Metal device, so say so rather than let the run look like
    # it is using the GPU when it is not.
    import inspect
    from pace.learn_ivim import learn_IVIM as _li
    if "device" not in inspect.signature(_li).parameters:
        print(f"[WARN  ] pace/learn_ivim.py does not accept a device "
              f"argument, so it will select cuda or cpu on its own and "
              f"ignore {dev}.")

    # The spatial models need the structural and b0 patches; the signal only
    # model does not. Load once per requirement rather than per run.
    need_spatial = any(C.ARCH_CONFIGS[ck.cfg_name]["spatial_on"] for ck in todo)
    print("[LOAD  ] reading the training and validation sets into memory")
    tr, va, secs = load_datasets(paths, need_spatial)
    print(f"         {len(tr):,} train, {len(va):,} val, {secs:.1f}s")
    print()

    done, failed = [], []
    for i, ck in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {ck.public_name}")
        try:
            out_dir = train_one(ck, paths, args, (tr, va))
            done.append((ck, out_dir))
        except Exception as e:
            failed.append((ck.public_name, f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {type(e).__name__}: {e}")

    print()
    print(f"[DONE] {len(done)} trained, {len(failed)} failed")
    for name, err in failed:
        print(f"   {name}: {err}")

    if args.compare and done:
        print_comparison(done, paths, args)

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
