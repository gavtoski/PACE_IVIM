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

from reproduce_synthetic_results import PreloadedIVIMDataset

TRAIN_CONFIG = {
    "recon_all": False,
    "alpha_ce": 0.2,
    "alpha_recon": 0.0,
    "alpha_penalty": 0.1,
    "alpha_prior": 0.0,
    "prior_sigma_scale": 2.0,
    "prior_paired_only": False,
    "use_ivim_nb": False,
    "ivim_nb_keep_center": False,
    "ivim_nb_oracle": False,
    "ivim_nb_anat_oracle": False,
    "alpha_ansc": 0.0,
    "ansc_target_mode": "rel",
    "ansc_tau": 0.5,
    "ansc_b0_floor": 0.05,
    "val_gt_metrics": False,
    "val_noise": False,
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

DNN_OVERRIDES = {"latent_dropout_p": 0.0, "modality_dropout_p": 0.0}


def _reject_if_inside(out_root, protected, label):
    """Refuse an output path that is the protected tree or sits inside it."""
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


def resolve_config(ck, args=None):
    """Assemble the full keyword set for one training run."""
    arch = C.ARCH_CONFIGS[ck.cfg_name]

    cfg = dict(TRAIN_CONFIG)

    if args is not None:
        if getattr(args, "alpha_ce", None) is not None:
            cfg["alpha_ce"] = float(args.alpha_ce)
        if getattr(args, "permute_ce", False):
            cfg["permute_ce_labels"] = True
        if getattr(args, "val_gt_metrics", None) is not None:
            cfg["val_gt_metrics"] = bool(args.val_gt_metrics)

        if getattr(args, "val_noise", None) is not None:
            cfg["val_noise"] = bool(args.val_noise)

        if getattr(args, "alpha_ansc", None) is not None:
            cfg["alpha_ansc"] = float(args.alpha_ansc)
        if getattr(args, "ansc_target_mode", None) is not None:
            cfg["ansc_target_mode"] = str(args.ansc_target_mode)
        if getattr(args, "ansc_tau", None) is not None:
            cfg["ansc_tau"] = float(args.ansc_tau)

        if getattr(args, "ivim_nb_anat_oracle", None) is not None:
            cfg["ivim_nb_anat_oracle"] = bool(args.ivim_nb_anat_oracle)

        if getattr(args, "ivim_nb_oracle", None) is not None:
            cfg["ivim_nb_oracle"] = bool(args.ivim_nb_oracle)

        if getattr(args, "use_ivim_nb", None) is not None:
            cfg["use_ivim_nb"] = bool(args.use_ivim_nb)

        if getattr(args, "ivim_nb_keep_center", None) is not None:
            cfg["ivim_nb_keep_center"] = bool(args.ivim_nb_keep_center)

        if getattr(args, "prior_paired_only", None) is not None:
            cfg["prior_paired_only"] = bool(args.prior_paired_only)

        if getattr(args, "alpha_prior", None) is not None:
            cfg["alpha_prior"] = float(args.alpha_prior)
        if getattr(args, "prior_sigma_scale", None) is not None:
            cfg["prior_sigma_scale"] = float(args.prior_sigma_scale)

        if getattr(args, "alpha_recon", None) is not None:
            cfg["alpha_recon"] = float(args.alpha_recon)

    for key in ["spatial_on", "use_struct", "use_b0", "parallel_heads",
                "recon_struct", "recon_b0", "detach_recon",
                "detach_spatial_delta", "use_anat_token"]:
        cfg[key] = arch[key]

    if getattr(args, "anat_token", None) is not None:
        cfg["use_anat_token"] = bool(int(args.anat_token))

    cfg["use_ordered_diffusion"] = ck.use_ordered_diffusion
    cfg["use_softmax_fractions"] = ck.use_softmax_fractions

    if args is not None:
        if getattr(args, "plain_head", False):
            cfg["use_ordered_diffusion"] = False
            cfg["use_softmax_fractions"] = False
        if getattr(args, "ordered_diffusion", None) is not None:
            cfg["use_ordered_diffusion"] = bool(args.ordered_diffusion)
        if getattr(args, "softmax_fractions", None) is not None:
            cfg["use_softmax_fractions"] = bool(args.softmax_fractions)
    cfg["fusion_mode"] = ck.fusion_mode

    if not arch["spatial_on"]:
        cfg.update(DNN_OVERRIDES)

    if args is not None and getattr(args, "no_struct", False):
        if cfg.get("spatial_on", False):
            cfg["use_struct"] = False
            cfg["recon_struct"] = False

    if args is not None and getattr(args, "no_b0", False):
        if cfg.get("spatial_on", False):
            cfg["use_b0"] = False
            cfg["recon_b0"] = False

    if args is not None and cfg.get("spatial_on", False):
        if getattr(args, "freeze_gates", None) is not None:
            cfg["freeze_gates_value"] = float(args.freeze_gates)
            subset = getattr(args, "freeze_gates_subset", None)
            if subset:
                cfg["freeze_gates_subset"] = list(subset)
        if getattr(args, "no_detach_delta", False):
            cfg["detach_spatial_delta"] = False

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


def resolve_data_paths(paths, args=None):
    """Training and validation h5 paths."""
    def _abs(value):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else paths.repo_root / candidate

    train_h5 = paths.data_dir / "synthetic_IVIM_train_2D.h5"
    val_h5 = Path(paths.synthetic_val)

    if args is None:
        return train_h5, val_h5

    explicit_train = getattr(args, "train_h5", None)
    explicit_val = getattr(args, "val_h5", None)
    if explicit_train or explicit_val:
        if not (explicit_train and explicit_val):
            raise SystemExit(
                "--train-h5 and --val-h5 must be given together: training "
                "on one dataset while validating on the released one is "
                "almost never what was meant.")
        return _abs(explicit_train), _abs(explicit_val)

    data_dir = _abs(args.data_dir) if getattr(args, "data_dir", None) else paths.data_dir
    ps = getattr(args, "patch_size", None)
    suffix = f"_p{int(ps)}" if ps else ""

    if getattr(args, "data_dir", None) or suffix:
        return (data_dir / f"synthetic_IVIM_train_2D{suffix}.h5",
                data_dir / f"synthetic_IVIM_val_2D{suffix}.h5")
    return train_h5, val_h5


def load_datasets(paths, need_spatial, args=None):
    """Read the training and validation sets into memory."""
    train_h5, val_h5 = resolve_data_paths(paths, args)
    t0 = time.time()
    _nb = bool(getattr(args, "use_ivim_nb", False) or False)
    _nb_c = bool(getattr(args, "ivim_nb_keep_center", False) or False)
    _nb_o = bool(getattr(args, "ivim_nb_oracle", False) or False)
    _nb_a = bool(getattr(args, "ivim_nb_anat_oracle", False) or False)
    _ansc = float(getattr(args, "alpha_ansc", 0.0) or 0.0) > 0
    tr = PreloadedIVIMDataset(str(train_h5), use_struct=need_spatial,
                             use_b0=need_spatial, use_ivim_nb=_nb,
                             ivim_nb_keep_center=_nb_c,
                             ivim_nb_oracle=_nb_o,
                             ivim_nb_anat_oracle=_nb_a, use_ansc=_ansc)
    va = PreloadedIVIMDataset(str(val_h5),
                             use_struct=need_spatial, use_b0=need_spatial,
                             use_ivim_nb=_nb, ivim_nb_keep_center=_nb_c,
                             ivim_nb_oracle=_nb_o,
                             ivim_nb_anat_oracle=_nb_a, use_ansc=_ansc)
    if _nb:
        print(f"[DATA  ] spatial token carries neighbouring IVIM curves, "
              f"{tr.B0.shape[1]} channels"
              + ("" if _nb_c else ", centre zeroed")
              + (", same class oracle" if _nb_o else ""))

    if tr.patch_size != va.patch_size:
        raise RuntimeError(
            f"train and validation patch sizes differ: {tr.patch_size} "
            f"vs {va.patch_size}. Training on one and validating on the "
            f"other would fail inside the first epoch.")

    asked = getattr(args, "patch_size", None) if args is not None else None
    if asked and int(asked) != tr.patch_size:
        raise RuntimeError(
            f"--patch-size {int(asked)} but {Path(train_h5).name} holds "
            f"{tr.patch_size}x{tr.patch_size} patches.")

    print(f"[DATA  ] {Path(train_h5).name}, {Path(val_h5).name} "
          f"| patch {tr.patch_size}x{tr.patch_size}")
    return tr, va, time.time() - t0


def train_one(ck, paths, args, datasets):
    """Train one configuration."""
    import torch
    from pace.learn_ivim import learn_IVIM

    out_root = paths.repo_root / args.out
    out_dir = out_root / ck.public_name
    if out_dir.is_dir() and any(out_dir.glob("*.pt")) and not args.overwrite:
        print(f"  [skip] {ck.public_name} already trained")
        return out_dir

    cfg = resolve_config(ck, args)
    tr, va = datasets
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(ck.seed)

    t0 = time.time()
    _train_h5, _val_h5 = resolve_data_paths(paths, args)
    net, history = learn_IVIM(
        train_h5=str(_train_h5),
        val_h5=str(_val_h5),
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
        widen_bounds=float(getattr(args, "voorter_bounds", 0.0) or 0.0),
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


def print_resolved_config(ck, args=None):
    """Show every keyword that would reach learn_IVIM, and its source."""
    arch = C.ARCH_CONFIGS[ck.cfg_name]
    cfg = resolve_config(ck, args)

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
    """Validate the request."""
    out_root = paths.repo_root / args.out
    for protected, label in [(paths.checkpoints_dir, "checkpoints/synthetic"),
                             (paths.results_dir, "results/synthetic")]:
        if _reject_if_inside(out_root, protected, label):
            return None

    train_h5, _val_h5 = resolve_data_paths(paths, args)
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

    extra = getattr(args, "add_seeds", None)
    if extra:
        import dataclasses

        pool = [ck for ck in manifest.checkpoints
                if (args.models is None or ck.model in args.models)
                and (args.snrs is None or ck.snr_db in args.snrs)]
        have = {(ck.model, ck.snr_db, ck.seed) for ck in todo}
        seen_pair = set()
        added = []
        for ck in pool:
            if (ck.model, ck.snr_db) in seen_pair:
                continue
            seen_pair.add((ck.model, ck.snr_db))
            for seed in extra:
                if (ck.model, ck.snr_db, int(seed)) in have:
                    continue
                clone = dataclasses.replace(
                    ck, seed=int(seed),
                    public_name=f"{ck.model}_snr{ck.snr_db}_seed{int(seed)}")
                added.append(clone)
        if added:
            print(f"[SEEDS ] {len(added)} entr(y|ies) synthesised, not in "
                  f"the manifest:")
            for ck in added:
                print(f"    {ck.public_name}")
            todo = todo + added

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
    ap.add_argument("--alpha-ce", dest="alpha_ce", type=float, default=None,
                    help="Override the CE weight. 0.0 removes the term "
                         "entirely (ablation arm C).")
    ap.add_argument("--permute-ce", dest="permute_ce", action="store_true",
                    help="Shuffle tissue labels within each training batch. "
                         "Keeps the CE gradient, removes its label content "
                         "(ablation arm B).")
    ap.add_argument("--freeze-gates", dest="freeze_gates", type=float,
                    default=None,
                    help="Freeze every spatial gate at this sigmoid value. "
                         "0.0 disconnects the fusion block, 1.0 forces it "
                         "into every parameter head.")
    ap.add_argument("--freeze-gates-subset", dest="freeze_gates_subset",
                    nargs="+", default=None,
                    choices=["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"],
                    help="Freeze only these gates and leave the rest free. "
                         "Default: all six.")
    ap.add_argument("--no-detach-delta", dest="no_detach_delta",
                    action="store_true",
                    help="Attach z_ivim inside z_fused. Required whenever "
                         "gates are frozen at 1.0, or the IVIM encoder "
                         "receives no gradient at all.")
    ap.add_argument("--anat-token", dest="anat_token", type=int,
                    choices=[0, 1], default=None,
                    help="1 fuses T1 and FLAIR into one anatomy token, "
                         "which is what the released configs do. 0 keeps "
                         "them as separate attention tokens, so the two "
                         "can be weighted independently: T1 separates "
                         "grey matter from white and FLAIR separates "
                         "lesion from normal white matter, and one shared "
                         "weight cannot serve both. Default follows the "
                         "architecture config.")
    ap.add_argument("--val-gt-metrics", dest="val_gt_metrics",
                    action="store_true", default=None,
                    help="Log parameter RMSE, signal RMSE and contrast "
                         "error against ground truth every epoch, and "
                         "noise the validation input at the training SNR "
                         "while keeping the clean signal as the target. "
                         "Answers whether more training improves the "
                         "parameters or only the signal fit. Does not "
                         "change which epoch is checkpointed.")
    ap.add_argument("--val-noise", dest="val_noise",
                    action="store_true", default=None,
                    help="Noise the validation input at the training SNR, "
                         "so validation runs on the condition training "
                         "and testing use. This changes val_mse, which "
                         "drives early stopping, the scheduler and "
                         "checkpoint selection, so an arm trained with it "
                         "is not comparable to results/synthetic or the "
                         "archives. The reconstruction target is the same "
                         "array either way, so this stays noisy in and "
                         "noisy out rather than becoming supervision.")
    ap.add_argument("--ivim-nb", dest="use_ivim_nb",
                    action="store_true", default=None,
                    help="Put the 3x3 neighbourhood of full IVIM curves in "
                         "the spatial token in place of the b0 patch. The "
                         "IVIM encoder still fits the centre voxel alone, "
                         "so the token supplies context rather than a "
                         "second copy of the measurement. Needs a phantom "
                         "augmented by add_neighbour_signals.py.")
    ap.add_argument("--ivim-nb-keep-centre", dest="ivim_nb_keep_center",
                    action="store_true", default=None,
                    help="Leave the centre voxel in the neighbourhood "
                         "token rather than zeroing it. Off by default: "
                         "the IVIM encoder already holds that curve, so "
                         "including it lets the network satisfy the token "
                         "without reading a neighbour, which the metrics "
                         "could not distinguish from success.")
    ap.add_argument("--alpha-ansc", dest="alpha_ansc", type=float,
                    default=None,
                    help="Weight on anatomy weighted neighbour signal "
                         "consistency. The network fits a second signal "
                         "target built from its anatomically similar "
                         "neighbours as well as its own measurement. The "
                         "two combine to an interpolation at "
                         "beta_eff = a/(1+a), so this is the only knob. "
                         "Unlike the tissue prior it names no parameter "
                         "value and so has no prototype to collapse onto, "
                         "though it can still shrink the maps indirectly, "
                         "which is what the arm measures. Read the sd "
                         "ratio before the RMSE.")
    ap.add_argument("--ansc-target", dest="ansc_target_mode",
                    choices=["decay", "rel", "abs", "b0"], default=None,
                    help="What the neighbours contribute. decay: their "
                         "stored curves, which carry no amplitude and "
                         "separate the planted pairs at only 1.4 times "
                         "the target noise. rel: (B0_j/B0_i) times the "
                         "decay, 8.1, the main candidate and in the same "
                         "units as the centre loss. abs: B0_j times the "
                         "decay, 8.7, diagnostic only since it is in "
                         "different units from the centre term. b0: the "
                         "mean neighbour amplitude ratio scaling the "
                         "centre's own curve, 9.9, the control that says "
                         "whether neighbour decay adds anything beyond "
                         "local brightness.")
    ap.add_argument("--ansc-tau", dest="ansc_tau", type=float, default=None,
                    help="Width of the anatomical similarity. Small "
                         "weights only very similar neighbours, large "
                         "approaches a flat average. The separation "
                         "diagnostic is flat from about 0.25 upward.")
    ap.add_argument("--ivim-nb-anat-oracle", dest="ivim_nb_anat_oracle",
                    action="store_true", default=None,
                    help="Average the k neighbours nearest the centre in "
                         "T1-FLAIR space, k calibrated against the mean "
                         "same class count. Uses no label at selection "
                         "time, so unlike --ivim-nb-oracle its ceiling is "
                         "reachable: the gap from here to a trained "
                         "network is the architecture rather than the "
                         "part of the label anatomy cannot see.")
    ap.add_argument("--ivim-nb-oracle", dest="ivim_nb_oracle",
                    action="store_true", default=None,
                    help="Average only the neighbours whose true class "
                         "matches the centre's. A diagnostic and not a "
                         "method: it reads ground truth at inference. It "
                         "answers whether neighbour curves carry usable "
                         "information at all, which a pooled arm cannot, "
                         "since pooling averages across a lesion boundary "
                         "by construction.")
    ap.add_argument("--prior-paired-only", dest="prior_paired_only",
                    action="store_true", default=None,
                    help="Apply the tissue prior only to voxels flagged "
                         "Paired in the phantom, the twenty percent whose "
                         "clean signals agree while their parameters "
                         "differ. The prior is where its gain is measured "
                         "and nowhere else, so the within class spread on "
                         "the remaining voxels is set by the signal. Read "
                         "the coverage on the Prior line to confirm the "
                         "mask arrived.")
    ap.add_argument("--alpha-prior", dest="alpha_prior", type=float,
                    default=None,
                    help="Weight on the tissue conditional prior, which "
                         "penalises parameters improbable under the tissue "
                         "the anatomy implies. Zero, the default, leaves "
                         "the loss as it was. The prior sits around 0.13 "
                         "when the parameters match their class, against a "
                         "converged signal MSE near 6e-6, so useful values "
                         "are small: read the Prior line in the log and "
                         "choose a weight that makes the two comparable.")
    ap.add_argument("--prior-sigma-scale", dest="prior_sigma_scale",
                    type=float, default=None,
                    help="How far the class conditional sigmas are "
                         "inflated. 1.0 penalises any departure from the "
                         "class mean and risks reproducing it; larger "
                         "values only penalise implausible parameters. "
                         "Default 2.0.")
    ap.add_argument("--alpha-recon", dest="alpha_recon", type=float,
                    default=None,
                    help="Override the reconstruction weight. The released "
                         "models use 0.0, so the term is off. It ramps in "
                         "from the warmup epoch rather than applying at "
                         "full weight from the start.")
    ap.add_argument("--no-b0", dest="no_b0", action="store_true",
                    help="Drop the b0 encoder, keeping T1, FLAIR, attention "
                         "and per parameter gating. The mirror of "
                         "--no-struct: isolates the b0 neighbourhood rather "
                         "than the anatomical pair.")
    ap.add_argument("--no-struct", dest="no_struct", action="store_true",
                    help="Drop the T1 and FLAIR encoders, keeping the b0 "
                         "neighbourhood, attention and per parameter "
                         "gating. Isolates the anatomical contribution "
                         "from spatial context in general.")
    ap.add_argument("--train-h5", dest="train_h5", default=None,
                    help="Training h5. Overrides --data-dir and "
                         "--patch-size, and assumes nothing about how the "
                         "file is named. Must be given with --val-h5.")
    ap.add_argument("--val-h5", dest="val_h5", default=None,
                    help="Validation h5. See --train-h5.")
    ap.add_argument("--data-dir", dest="data_dir", default=None,
                    help="Directory holding the synthetic h5 files. "
                         "Relative paths resolve against the repository "
                         "root. Default: the released data directory.")
    ap.add_argument("--patch-size", dest="patch_size", type=int, default=None,
                    help="Selects the _pN files, for example 5 for the "
                         "patch size ablation. It does not configure the "
                         "network: the architecture is built from the "
                         "patch size recorded in the data, and a mismatch "
                         "between the two stops the run.")
    ap.add_argument("--add-seeds", dest="add_seeds", nargs="+", type=int,
                    default=None,
                    help="Train at seeds the manifest does not list, by "
                         "cloning an existing entry at the same model and "
                         "SNR. For replicating an ablation where the "
                         "released grid has only one seed. Nothing is "
                         "written back to the manifest.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="checkpoints/synthetic_retrained")
    ap.add_argument("--plain-head", dest="plain_head", action="store_true",
                    help="Build the parameter heads the way DNN does: no "
                         "ordered diffusion constraint and no softmax "
                         "simplex. Lets a spatial arm be compared with DNN "
                         "without the head differing as well.")
    ap.add_argument("--ordered-diffusion", dest="ordered_diffusion",
                    type=int, choices=[0, 1], default=None,
                    help="Force the ordered diffusion constraint on or off, "
                         "overriding both the manifest and --plain-head.")
    ap.add_argument("--softmax-fractions", dest="softmax_fractions",
                    type=int, choices=[0, 1], default=None,
                    help="Force the softmax simplex on or off, overriding "
                         "both the manifest and --plain-head.")
    ap.add_argument("--voorter-bounds", dest="voorter_bounds", nargs="?",
                    type=float, const=0.3, default=0.0,
                    help="Widen the sigmoid bounds by this fraction of "
                         "their range, floor clamped at zero. Bare flag "
                         "gives 0.3, the factor Voorter et al. use. Only "
                         "affects models that squash fractions through "
                         "the sigmoid, so DNN and --plain-head arms. "
                         "Inference needs the same flag.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--compare", action="store_true",
                    help="Compare against the released training summaries.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print the plan and the resolved configuration.")
    args = ap.parse_args()

    if getattr(args, "no_b0", False) and getattr(args, "no_struct", False):
        print("[ERROR] --no-b0 and --no-struct together leave the spatial "
              "pathway with no inputs. Use neither and set --freeze-gates 0.0 "
              "instead, which is the arm that removes spatial context "
              "without removing the encoders.", file=sys.stderr)
        return 2

    if args.permute_ce and args.alpha_ce is not None and args.alpha_ce == 0.0:
        print("[ERROR] --permute-ce with --alpha-ce 0 has no effect. "
              "Drop one of the two.", file=sys.stderr)
        return 2

    if args.freeze_gates is not None:
        if not 0.0 <= args.freeze_gates <= 1.0:
            print(f"[ERROR] --freeze-gates must lie in [0, 1], got "
                  f"{args.freeze_gates}.", file=sys.stderr)
            return 2
        if args.freeze_gates >= 1.0 and not args.no_detach_delta \
                and not args.freeze_gates_subset:
            print("[ERROR] --freeze-gates 1.0 with detachment on leaves the "
                  "IVIM encoder without gradient. Add --no-detach-delta.",
                  file=sys.stderr)
            return 2

    if args.no_detach_delta and args.freeze_gates is None:
        print("[warn] --no-detach-delta without --freeze-gates changes the "
              "published configuration on its own.", file=sys.stderr)

    paths = C.load_paths()
    manifest = C.load_manifest(paths)

    todo = preflight(args, paths, manifest)
    if todo is None:
        return 2

    print_plan(todo, args)

    if args.plain_head or args.ordered_diffusion is not None \
            or args.softmax_fractions is not None:
        print("[ABLATION] head parameterisation overridden. A checkpoint "
              "trained this way does not match its manifest entry, so "
              "inference on it needs the same flags.")

    if args.dry_run:
        print_resolved_config(todo[0], args)
        return 0

    try:
        import torch
    except ImportError:
        print("[ERROR] torch is required. pip install torch", file=sys.stderr)
        return 2

    dev = C.resolve_device(args.device)
    print(f"[DEVICE] {dev}")

    import inspect
    from pace.learn_ivim import learn_IVIM as _li
    if "device" not in inspect.signature(_li).parameters:
        print(f"[WARN  ] pace/learn_ivim.py does not accept a device "
              f"argument, so it will select cuda or cpu on its own and "
              f"ignore {dev}.")

    need_spatial = any(C.ARCH_CONFIGS[ck.cfg_name]["spatial_on"] for ck in todo)
    print("[LOAD  ] reading the training and validation sets into memory")
    tr, va, secs = load_datasets(paths, need_spatial, args)
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
