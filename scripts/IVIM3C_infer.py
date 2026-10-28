#!/usr/bin/env python3
"""Re-run trained IVIM3brain-NET checkpoints on a different test set."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

MODEL = "ivim3brain"

DEFAULT_SNRS = [20, 25, 30, 35]
DEFAULT_SEEDS = [19, 22, 35, 42, 63]
SEEDED_SNR = 25


def grid(snrs, seeds):
    """(snr, seed) pairs matching the released manifest layout."""
    return [(snr, seed) for snr in snrs for seed in seeds
            if snr == SEEDED_SNR or seed == seeds[0]]


def abs_path(value):
    p = Path(value)
    return p if p.is_absolute() else REPO / p


def find_checkpoint(ck_root, snr, seed):
    """Locate one trained network, handling both on disk layouts."""
    nested = ck_root / f"{MODEL}_snr{snr}_seed{seed}"
    if nested.is_dir():
        hits = sorted(nested.glob("*.pt"))
        if hits:
            return hits[0]
    flat = ck_root / f"{MODEL}_snr{snr}_seed{seed}.pt"
    return flat if flat.is_file() else None


def run_one(snr, seed, bvals, X_test_clean, labels, args, T):
    """Score one checkpoint against the chosen test set."""
    from pace import PaulienVoorter_deep as V
    import torch
    import h5py

    ck_root = abs_path(args.checkpoints)
    ck_path = find_checkpoint(ck_root, snr, seed)
    if ck_path is None:
        print(f"[miss  ] snr{snr} seed{seed}: no checkpoint under "
              f"{ck_root}", file=sys.stderr)
        return 2, 0.0

    res_dir = abs_path(args.out) / f"{MODEL}_snr{snr}"
    out_h5 = res_dir / f"inferred_seed{seed}.h5"
    if out_h5.is_file() and not args.overwrite:
        print(f"[skip  ] snr{snr} seed{seed} already written")
        return 0, 0.0
    res_dir.mkdir(parents=True, exist_ok=True)

    V.set_seed(seed)
    rng = np.random.default_rng(seed)
    arg = T.build_arg(bvals, args, seed)

    if args.match_training_noise and args.n_train:
        rng.normal(0.0, 1.0, (int(args.n_train), len(bvals)))
        rng.normal(0.0, 1.0, (int(args.n_train), len(bvals)))

    X_test = T.add_rician_noise(X_test_clean, snr, rng)

    bvals_t = torch.FloatTensor(np.asarray(bvals)).to(arg.train_pars.device)
    net = V.Net(bvals_t, arg.net_pars, arg.rel_times).to(arg.train_pars.device)
    state = torch.load(ck_path, map_location=arg.train_pars.device,
                       weights_only=True)
    net.load_state_dict(state, strict=True)
    net.eval()

    print(f"\n{'=' * 68}")
    print(f"  IVIM3brain-NET | SNR {snr} dB | seed {seed}")
    print(f"  weights {ck_path.relative_to(REPO)}")
    print("=" * 68, flush=True)

    t0 = time.time()
    params = V.predict_IVIM(X_test, bvals, net, arg)
    elapsed = (time.time() - t0) / 60.0
    Dpar, Fmv, Dmv, Dint, Fint, S0 = params

    with h5py.File(out_h5, "w") as f:
        for name, arr in (("Dpar_pred", Dpar), ("Dint_pred", Dint),
                          ("Fint_pred", Fint), ("Dmv_pred", Dmv),
                          ("Fmv_pred", Fmv), ("S0_pred", S0)):
            f.create_dataset(name, data=np.asarray(arr, dtype=np.float32))
        f.create_dataset("bvals", data=np.asarray(bvals, dtype=np.float32))
        stacked = np.vstack([Dpar, Dint, Fint, Dmv, Fmv, S0])
        valid = np.isfinite(stacked).all(axis=0) & (np.abs(stacked).sum(axis=0) > 0)
        f.create_dataset("Valid_mask", data=valid.astype(np.uint8))
        for key, val in labels.items():
            f.create_dataset(key, data=val)

    print(f"  {100.0 * valid.mean():.1f}% voxels kept  |  {elapsed:.1f} min  "
          f"->  {out_h5.relative_to(REPO)}")
    return 0, elapsed


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Score trained IVIM3brain-NET checkpoints on a test set.")
    p.add_argument("--checkpoints", default="checkpoints/ivim3brain",
                   help="Directory holding the trained networks. "
                        "Default: %(default)s")
    p.add_argument("--test-h5", dest="test_h5", required=True,
                   help="Test set to score against. Required, so a "
                        "re-scoring is always deliberate.")
    p.add_argument("--out", required=True,
                   help="Output directory under results/. Required rather "
                        "than defaulted, so this cannot overwrite the "
                        "original results/ivim3brain tree.")
    p.add_argument("--snrs", nargs="+", type=int, default=DEFAULT_SNRS)
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--device", default=None)
    p.add_argument("--ir", action="store_true",
                   help="Enable the inversion recovery correction. Must "
                        "match whatever the checkpoints were trained with.")
    p.add_argument("--no-bounds-expansion", action="store_true",
                   help="Must match whatever the checkpoints were trained "
                        "with, since it changes the output scaling.")
    p.add_argument("--match-pace-training", action="store_true",
                   help="Kept only so build_arg sees the same flags as at "
                        "training time. It has no effect on inference.")
    p.add_argument("--match-training-noise", action="store_true",
                   help="Advance the generator past a training sized draw "
                        "before drawing test noise, reproducing the "
                        "realisation the original run scored on.")
    p.add_argument("--n-train", type=int, default=570000,
                   help="Training set size, for --match-training-noise. "
                        "Default: %(default)s")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan and stop.")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--epochs", type=int, default=35, help=argparse.SUPPRESS)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    try:
        from pace import common as C
        import IVIM3C_train as T
    except ImportError as exc:
        print(f"[ERROR] cannot import: {exc}", file=sys.stderr)
        print("        IVIM3C_train.py must sit in scripts/ and "
              "PaulienVoorter_deep.py in pace/.", file=sys.stderr)
        return 2

    paths = C.load_paths()
    bvals = np.asarray(C.load_bvals(paths), dtype=np.float64)

    test_h5 = abs_path(args.test_h5)
    if not test_h5.is_file():
        print(f"[ERROR] test set not found: {test_h5}", file=sys.stderr)
        return 2

    out_root = abs_path(args.out)
    if out_root.resolve() == (REPO / "results" / "ivim3brain").resolve():
        print("[ERROR] refusing to write over the original results tree. "
              "Choose a different --out.", file=sys.stderr)
        return 2

    plan = grid(args.snrs, args.seeds)
    print(f"[REPO  ] {REPO}")
    print(f"[WEIGHT] {abs_path(args.checkpoints).relative_to(REPO)}")
    print(f"[DATA  ] {test_h5}")
    print(f"[OUT   ] {out_root.relative_to(REPO)}")
    print(f"[RUN   ] {len(plan)} checkpoint(s)")
    for snr, seed in plan:
        print(f"    SNR {snr:<4} seed {seed}")
    print()

    if args.dry_run:
        print("Dry run. Nothing was written.")
        return 0

    X_test_clean = T.normalise_to_b0(T.load_signals(test_h5), bvals)
    labels = T.load_labels(test_h5)
    print(f"[LOAD  ] {X_test_clean.shape[0]:,} test voxels")

    results = []
    t_start = time.time()
    for snr, seed in plan:
        try:
            code, elapsed = run_one(snr, seed, bvals, X_test_clean,
                                    labels, args, T)
        except Exception as exc:
            print(f"\n[FAIL  ] SNR {snr} seed {seed}: {exc}", file=sys.stderr)
            code, elapsed = 1, 0.0
            if not args.keep_going:
                results.append((snr, seed, code, elapsed))
                break
        results.append((snr, seed, code, elapsed))
        if code != 0 and not args.keep_going:
            print("[STOP  ] use --keep-going to continue past this.",
                  file=sys.stderr)
            break

    ok = sum(1 for r in results if r[2] == 0)
    missing = sum(1 for r in results if r[2] == 2)
    failed = sum(1 for r in results if r[2] == 1)

    print(f"\n{'=' * 68}")
    line = f"  {ok} of {len(plan)} written"
    if missing:
        line += f", {missing} checkpoint(s) absent"
    if failed:
        line += f", {failed} failed"
    print(line + f" in {(time.time() - t_start) / 60.0:.1f} min")
    print("=" * 68)
    status = {0: "ok  ", 1: "FAIL", 2: "miss"}
    for snr, seed, code, elapsed in results:
        print(f"  {status[code]}  SNR {snr:<4} seed {seed:<4} "
              f"{elapsed:5.1f} min")
    return 0 if ok == len(plan) else 1


if __name__ == "__main__":
    sys.exit(main())
