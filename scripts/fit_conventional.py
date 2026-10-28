#!/usr/bin/env python3
"""Fit LSQ, NNLS and MAP on the noised test phantom, and write result H5s."""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PARAMS = ("Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0")


def add_rician_noise(signals, snr_db, seed):
    """Rician noise, the same model clean_ivim_signals uses."""
    rng = np.random.default_rng(seed)
    sigma = 1.0 / (10.0 ** (snr_db / 20.0))
    n1 = rng.normal(0, sigma, signals.shape).astype(np.float32)
    n2 = rng.normal(0, sigma, signals.shape).astype(np.float32)
    noisy = np.sqrt((signals + n1) ** 2 + n2 ** 2)
    return np.nan_to_num(noisy, nan=0.0, posinf=0.0, neginf=0.0)


def saturation(res, tissue, lesion, valid, bounds):
    """Share of voxels pinned at each fraction ceiling, overall and in WMH."""
    hi = bounds[1]
    wmh = valid & (tissue == 3) & (lesion != 0)
    out = {}
    for name, ceil in (("Fint", hi[2]), ("Fmv", hi[4])):
        near = np.isclose(np.asarray(res[name]).reshape(-1), ceil,
                          rtol=1e-3, atol=1e-6)
        out[name] = (float(near[valid].mean()) if valid.any() else float("nan"),
                     float(near[wmh].mean()) if wmh.any() else float("nan"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Fit the conventional models on the noised phantom.")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--methods", nargs="+", default=["LSQ", "NNLS", "MAP"],
                    help="Run LSQ before MAP so MAP can reuse its result.")
    ap.add_argument("--out", default="results/conventional")
    ap.add_argument("--snr", type=int, default=25)
    ap.add_argument("--seed", type=int, default=19)
    ap.add_argument("--njobs", type=int, default=4)
    ap.add_argument("--test-h5", default=None)
    args = ap.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(repo))
    import h5py
    from pace import common as C
    from pace import conventional_models as CM

    paths = C.load_paths()
    test_h5 = (Path(args.test_h5).expanduser() if args.test_h5
               else Path(paths.synthetic_test))
    bvals = np.asarray(C.load_bvals(paths), dtype=np.float32)
    out_root = repo / args.out
    B = CM.DEFAULT_BOUNDS

    with h5py.File(test_h5, "r") as f:
        clean = f["IVIM_cube"][:].astype(np.float32)
        tissue = np.asarray(f["Tissue"][:]).ravel()
        lesion = np.asarray(f["Lesion"][:]).ravel()
    n, nb = clean.shape

    if nb != len(bvals):
        print(f"[ERROR] {nb} b-values in the file against {len(bvals)} "
              f"in the bval file", file=sys.stderr)
        return 1

    sigma = 1.0 / (10.0 ** (args.snr / 20.0))
    noisy = add_rician_noise(clean, args.snr, args.seed)

    print(f"[test  ] {test_h5}")
    print(f"[shape ] {n:,} voxels, {nb} b-values")
    print(f"[noise ] Rician SNR {args.snr} dB, sigma {sigma:.4f}, "
          f"seed {args.seed}, b0 median "
          f"{np.median(noisy[:, bvals < 1.0]):.4f}")
    print(f"[bounds] fint <= {B[1][2]}, fmv <= {B[1][4]}")
    print()

    lsq_result = None
    for method in args.methods:
        m = method.upper()
        t0 = time.time()
        print(f"[{m}] fitting")

        if m == "LSQ":
            res = CM.fit_least_squares_array(
                bvals, noisy, fitS0=True, bounds=B, njobs=args.njobs,
                verbose=True, average_repeats=True)
            lsq_result = res
        elif m == "NNLS":
            res = CM.fit_NNLS_array(
                bvals, noisy, bounds=B, verbose=True, average_repeats=True)
        elif m == "MAP":
            if lsq_result is None:
                print("  no LSQ in this run, MAP fits its own")
            res = CM.fit_MAP_array(
                bvals, noisy, fitS0=True, bounds=B, njobs=args.njobs,
                verbose=True, average_repeats=True,
                lsq_precomputed=lsq_result, split_prior=False)
        else:
            print(f"[ERROR] unknown method {method}", file=sys.stderr)
            return 1

        valid = np.asarray(res["S0"]).reshape(-1) > 0

        save = {f"{p}_pred": np.asarray(res[p], dtype=np.float32).reshape(-1)
                for p in PARAMS}
        save["Tissue"] = tissue
        save["Lesion"] = lesion
        save["Valid_mask"] = valid.astype(np.uint8)
        save["bvals"] = bvals.astype(np.float32)

        bad = {k: len(v) for k, v in save.items()
               if k != "bvals" and len(v) != n}
        if bad:
            print(f"[ERROR] {m}: inconsistent lengths {bad}, expected {n}",
                  file=sys.stderr)
            return 1

        out_dir = out_root / f"{m.lower()}_snr{args.snr}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"inferred_seed{args.seed}.h5"
        with h5py.File(out_path, "w") as f:
            for k, v in save.items():
                f.create_dataset(k, data=v, compression="gzip")

        print(f"  {int(valid.sum()):,}/{n:,} voxels kept "
              f"({100 * valid.mean():.1f}%)  {time.time() - t0:.0f}s"
              f"  -> {out_path}")
        for name, (allv, wmhv) in saturation(res, tissue, lesion,
                                             valid, B).items():
            print(f"  {name} pinned at the ceiling: "
                  f"{100 * allv:.1f}% of valid voxels, "
                  f"{100 * wmhv:.1f}% of WMH")
        print()

    print(f"Read these with results_dir pointed at {args.out}, "
          f"model name in lower case.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
