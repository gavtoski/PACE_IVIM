#!/usr/bin/env python3
"""Train and evaluate IVIM3brain-NET on the PACE synthetic data.

This runs the authors' own code rather
than a reimplementation of it:

    Voorter et al. Improving microstructural integrity, interstitial
    fluid, and blood microcirculation images from multi-b-value
    diffusion MRI using physics-informed neural networks in
    cerebrovascular disease. Magn Reson Med 2023;90:1657-1671.
    https://github.com/paulienvoorter/IVIM3brain-NET

pace/PaulienVoorter_deep.py is a verbatim port of their deep.py, with
four environment fixes documented in its header and no change to the
network, the constraints or the loss.

WHAT IS MATCHED AND WHAT IS THEIRS
----------------------------------
Matched to PACE, because these are properties of the experiment rather
than of either method:

    training and test data     the same synthetic h5 files
    noise                      the same Rician injection at the same SNR,
                               using the identical expression from
                               deep_models.clean_ivim_signals
    b-values                   read from the same bval file
    epoch budget               35, as in TRAIN_CONFIG
    seeds                      the same grid as the released manifest
    parameter bounds           the Voorter Table S1 values that
                               pace/hyperparams.py already uses

Left as the authors published them, because they are the method:

    architecture               depth 2, parallel heads, batch norm,
                               dropout 0.1, width = number of b-values
    constraint scheme          sigmoidabs, a sigmoid on each diffusivity
                               within its own range and an absolute value
                               on each fraction
    bounds expansion           the authors widen the sigmoid range by 30
                               percent of its span to stay off the flat
                               tails, which is part of their design
    optimiser and rate         Adam at 3e-5 with their plateau scheduler
    batch size                 128
    early stopping             patience 10 on validation loss

--match-pace-training switches the optimiser, rate and batch size to the
PACE values if the reverse comparison is ever wanted. The default is the
authors' own settings, which is the fairer starting point and the easier
one to defend in a response letter.

INVERSION RECOVERY IS OFF
Their net_pars defaults IR to True, correcting for an inversion recovery
preparation. The PACE synthetic data carries no inversion recovery, so
leaving it on would have the network correcting for a preparation that
is not in the signal. It is off here, and --ir turns it back on.

Outputs land in the same layout the rest of the analysis expects, so the
existing figure scripts pick the model up with no changes:

    checkpoints/<out>/ivim3brain_snr<SNR>_seed<SEED>/
    results/<out>/ivim3brain_snr<SNR>/inferred_seed<SEED>.h5

Usage, from the repository root:
    python scripts/IVIM3C_train.py --dry-run
    python scripts/IVIM3C_train.py --snrs 25 --seeds 19 --epochs 2
    python scripts/IVIM3C_train.py
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

MODEL = "ivim3brain"

DEFAULT_SNRS = [20, 25, 30, 35]
DEFAULT_SEEDS = [19, 22, 35, 42, 63]
SEEDED_SNR = 25


def grid(snrs, seeds):
    """(snr, seed) pairs matching the released manifest layout."""
    out = []
    for snr in snrs:
        for seed in seeds:
            if snr == SEEDED_SNR or seed == seeds[0]:
                out.append((snr, seed))
    return out


def load_signals(h5_path, key="IVIM_cube"):
    import h5py
    with h5py.File(h5_path, "r") as f:
        return np.asarray(f[key][:], dtype=np.float64)


def load_labels(h5_path):
    """Tissue and lesion labels, needed for the CNR analysis downstream."""
    import h5py
    out = {}
    with h5py.File(h5_path, "r") as f:
        for key in ("Tissue", "Lesion"):
            if key in f:
                out[key] = np.asarray(f[key][:])
    return out


def normalise_to_b0(X, bvals):
    """Divide by the median b0, as the PACE pipeline does."""
    b0 = np.abs(bvals) < 1e-3
    if not b0.any():
        raise ValueError("no b0 volume found in the b-value list")
    s0 = np.median(X[:, b0], axis=1, keepdims=True)
    s0 = np.clip(s0, 1e-6, None)
    return np.nan_to_num(X / s0, nan=0.0, posinf=0.0, neginf=0.0)


def add_rician_noise(X, snr_db, rng):
    """Rician noise at a given SNR in dB."""
    snr_linear = 10.0 ** (float(snr_db) / 20.0)
    sigma = 1.0 / max(snr_linear, 1e-8)
    n1 = rng.normal(0.0, sigma, X.shape)
    n2 = rng.normal(0.0, sigma, X.shape)
    return np.nan_to_num(np.sqrt((X + n1) ** 2 + n2 ** 2), nan=0.0)


def build_arg(bvals, args, seed):
    """Assemble the argument object Voorter's code expects."""
    from pace import PaulienVoorter_deep as V
    import torch

    class Arg:
        pass

    arg = Arg()
    arg.train_pars = V.train_pars("optim")
    arg.net_pars = V.net_pars("optim")

    if getattr(args, "con", None):
        arg.net_pars.con = str(args.con)
    arg.rel_times = V.rel_times()
    arg.sim = V.sim()
    arg.fit = V.lsqfit()

    from pace.hyperparams import net_pars as pace_net_pars
    pace = pace_net_pars()
    cons_min = np.array(pace.cons_min, dtype=float)
    cons_max = np.array(pace.cons_max, dtype=float)
    if not args.no_bounds_expansion:
        span = 0.3 * (cons_max - cons_min)
        cons_min = cons_min - span
        cons_max = cons_max + span
    arg.net_pars.cons_min = cons_min
    arg.net_pars.cons_max = cons_max

    arg.net_pars.IR = bool(args.ir)
    arg.sim.IR = bool(args.ir)

    arg.net_pars.width = 0

    arg.train_pars.max_epochs = int(args.epochs)
    arg.train_pars.select_best = True
    arg.sim.jobs = 1
    arg.fit.do_fit = False

    if args.match_pace_training:
        arg.train_pars.optim = "adam"
        arg.train_pars.lr = 1e-3
        arg.train_pars.batch_size = 256

    if args.device:
        arg.train_pars.device = torch.device(args.device)
        arg.train_pars.use_cuda = args.device.startswith("cuda")
    else:
        if torch.cuda.is_available():
            dev = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        arg.train_pars.device = torch.device(dev)
        arg.train_pars.use_cuda = dev.startswith("cuda")

    return arg


def describe(arg, bvals):
    n = arg.net_pars
    t = arg.train_pars
    print(f"[NET   ] depth={n.depth} width={len(bvals) if n.width == 0 else n.width} "
          f"parallel={n.parallel} dropout={n.dropout} batch_norm={n.batch_norm} "
          f"con={n.con} fitS0={n.fitS0} IR={n.IR}")
    print(f"[TRAIN ] optim={t.optim} lr={t.lr} batch={t.batch_size} "
          f"epochs<={t.max_epochs} patience={t.patience} split={t.split} "
          f"device={t.device}")
    names = ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"]
    print("[BOUNDS] " + "  ".join(
        f"{k}[{lo:.4g},{hi:.4g}]"
        for k, lo, hi in zip(names, n.cons_min, n.cons_max)))


def run_one(snr, seed, bvals, X_train_clean, X_test_clean, labels, args):
    """Train on the noisy training split, then infer on the noisy test
    split.
    """
    from pace import PaulienVoorter_deep as V
    import torch

    ck_dir = REPO / "checkpoints" / args.out / f"{MODEL}_snr{snr}_seed{seed}"
    res_dir = REPO / "results" / args.out / f"{MODEL}_snr{snr}"
    out_h5 = res_dir / f"inferred_seed{seed}.h5"
    ck_path = ck_dir / f"trained_net_{MODEL}_{seed}_noise{snr}dB.pt"

    if out_h5.is_file() and ck_path.is_file() and not args.overwrite:
        print(f"[skip  ] snr{snr} seed{seed} already done")
        return 0, 0.0

    ck_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    V.set_seed(seed)
    rng = np.random.default_rng(seed)
    arg = build_arg(bvals, args, seed)

    X_train = add_rician_noise(X_train_clean, snr, rng)
    X_test = add_rician_noise(X_test_clean, snr, rng)

    print(f"\n{'=' * 68}")
    print(f"  IVIM3brain-NET | SNR {snr} dB | seed {seed}")
    print(f"  train {X_train.shape}  test {X_test.shape}")
    print("=" * 68, flush=True)
    describe(arg, bvals)

    t0 = time.time()
    net = V.learn_IVIM(X_train, bvals, arg)
    elapsed = (time.time() - t0) / 60.0

    torch.save(net.state_dict(), ck_path)

    params = V.predict_IVIM(X_test, bvals, net, arg)
    Dpar, Fmv, Dmv, Dint, Fint, S0 = params

    import h5py
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("Dpar_pred", data=np.asarray(Dpar, dtype=np.float32))
        f.create_dataset("Dint_pred", data=np.asarray(Dint, dtype=np.float32))
        f.create_dataset("Fint_pred", data=np.asarray(Fint, dtype=np.float32))
        f.create_dataset("Dmv_pred", data=np.asarray(Dmv, dtype=np.float32))
        f.create_dataset("Fmv_pred", data=np.asarray(Fmv, dtype=np.float32))
        f.create_dataset("S0_pred", data=np.asarray(S0, dtype=np.float32))
        f.create_dataset("bvals", data=np.asarray(bvals, dtype=np.float32))
        stacked = np.vstack([Dpar, Dint, Fint, Dmv, Fmv, S0])
        valid = np.isfinite(stacked).all(axis=0) & (np.abs(stacked).sum(axis=0) > 0)
        f.create_dataset("Valid_mask", data=valid.astype(np.uint8))
        for key, val in labels.items():
            f.create_dataset(key, data=val)

    kept = 100.0 * valid.mean()
    print(f"  {kept:.1f}% voxels kept  |  {elapsed:.1f} min  ->  "
          f"{out_h5.relative_to(REPO)}")
    return 0, elapsed


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train and evaluate IVIM3brain-NET on the PACE data.")
    p.add_argument("--snrs", nargs="+", type=int, default=DEFAULT_SNRS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--epochs", type=int, default=35,
                   help="Epoch budget, matching TRAIN_CONFIG. "
                        "Default: %(default)s")
    p.add_argument("--out", default="ivim3brain",
                   help="Directory name under checkpoints/ and results/. "
                        "Default: %(default)s")
    p.add_argument("--con", default=None,
                   choices=["sigmoid", "sigmoidabs", "abs", "none"],
                   help="Constraint scheme. Their class default, and the "
                        "one used for the published comparison, is "
                        "sigmoidabs: sigmoid on the diffusivities, "
                        "absolute value on the fractions, so the fractions "
                        "have no ceiling. sigmoid bounds the fractions too, "
                        "which matches the parameterisation DNN uses and "
                        "isolates the architecture from the fraction "
                        "scheme. Omit to leave their default in place.")
    p.add_argument("--device", default=None,
                   help="cuda:0, mps or cpu. Default: best available.")
    p.add_argument("--ir", action="store_true",
                   help="Enable their inversion recovery correction. Off "
                        "by default: the PACE synthetic signal carries no "
                        "inversion recovery preparation.")
    p.add_argument("--no-bounds-expansion", action="store_true",
                   help="Skip the authors' 30 percent widening of the "
                        "sigmoid range. Their design includes it, so this "
                        "is for diagnosis rather than the headline run.")
    p.add_argument("--match-pace-training", action="store_true",
                   help="Use the PACE optimiser, rate and batch size "
                        "instead of the authors' own.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and stop.")
    p.add_argument("--keep-going", action="store_true",
                   help="Continue after a failed run instead of stopping.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        from pace import common as C
        import importlib
        importlib.import_module("pace.PaulienVoorter_deep")
    except ImportError as exc:
        print(f"[ERROR] cannot import from pace: {exc}", file=sys.stderr)
        print("        PaulienVoorter_deep.py belongs in pace/.",
              file=sys.stderr)
        return 2

    paths = C.load_paths()
    bvals = np.asarray(C.load_bvals(paths), dtype=np.float64)

    train_h5 = Path(paths.data_dir) / "synthetic_IVIM_train_2D.h5"
    test_h5 = Path(paths.data_dir) / "synthetic_IVIM_test_2D.h5"
    for pth in (train_h5, test_h5):
        if not pth.is_file():
            print(f"[ERROR] missing {pth}", file=sys.stderr)
            return 2

    plan = grid(args.snrs, args.seeds)

    print(f"[REPO  ] {REPO}")
    print(f"[DATA  ] {train_h5.name}, {test_h5.name}")
    print(f"[BVALS ] {len(bvals)} values, {bvals.min():.0f} to {bvals.max():.0f}")
    print(f"[GRID  ] {len(plan)} run(s)")
    for snr, seed in plan:
        print(f"    SNR {snr:<4} seed {seed}")
    print(f"[OUT   ] checkpoints/{args.out}, results/{args.out}")
    print()

    if args.dry_run:
        print("Dry run. Nothing was trained.")
        return 0

    X_train_clean = normalise_to_b0(load_signals(train_h5), bvals)
    X_test_clean = normalise_to_b0(load_signals(test_h5), bvals)
    labels = load_labels(test_h5)
    print(f"[LOAD  ] {X_train_clean.shape[0]:,} train, "
          f"{X_test_clean.shape[0]:,} test voxels")

    results = []
    t_start = time.time()
    for snr, seed in plan:
        try:
            code, elapsed = run_one(snr, seed, bvals, X_train_clean,
                                    X_test_clean, labels, args)
        except Exception as exc:
            print(f"\n[FAIL  ] SNR {snr} seed {seed}: {exc}", file=sys.stderr)
            code, elapsed = 1, 0.0
            if not args.keep_going:
                results.append((snr, seed, code, elapsed))
                break
        results.append((snr, seed, code, elapsed))

    total = (time.time() - t_start) / 60.0
    ok = sum(1 for r in results if r[2] == 0)
    print(f"\n{'=' * 68}")
    print(f"  {ok} of {len(plan)} run(s) completed in {total:.1f} min")
    print("=" * 68)
    for snr, seed, code, elapsed in results:
        status = "ok  " if code == 0 else "FAIL"
        print(f"  {status}  SNR {snr:<4} seed {seed:<4} {elapsed:6.1f} min")

    return 0 if ok == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
