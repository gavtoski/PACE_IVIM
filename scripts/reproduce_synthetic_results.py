#!/usr/bin/env python3
"""Reproduce the synthetic inference results from the released checkpoints.

Usage:
    python scripts/reproduce_synthetic_results.py --models pace --snrs 25 --seeds 19
    python scripts/reproduce_synthetic_results.py
    python scripts/reproduce_synthetic_results.py --compare
    python scripts/reproduce_synthetic_results.py --device cpu
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pace import common as C

PRED_KEYS = ["Dpar_pred", "Fint_pred", "Dint_pred", "Fmv_pred", "Dmv_pred",
             "S0_pred"]


from pace.deep_models import IVIMDataset, PreloadedIVIMDataset


def _reject_if_inside(out_root, protected, label):
    """Refuse an output path that is the protected tree or sits inside it."""
    a, b = out_root.resolve(), protected.resolve()
    if a == b or a.is_relative_to(b):
        print(f"[ERROR] refusing to write to {a}, which is inside "
              f"{label}. That directory holds released material. "
              f"Choose another --out.", file=sys.stderr)
        return True
    return False


def _forward_pass(net, loader, cfg, bvals, dev, ck, n_batches, quiet):
    """Run every batch through the network."""
    import torch
    from pace.deep_models import clean_ivim_signals

    preds = {k: [] for k in PRED_KEYS}
    tissue, lesion, valid_chunks = [], [], []

    torch.manual_seed(ck.seed)

    t0 = time.time()

    with torch.no_grad():
        for bi, batch in enumerate(loader):
            _report_progress(bi, n_batches, t0, quiet)

            SP_in = (batch["struct_patch"].to(dev)
                     if cfg["spatial_on"] and cfg["use_struct"] else None)
            B0_in = (batch["b0_patch"].to(dev)
                     if cfg["spatial_on"] and cfg["use_b0"] else None)

            X, SP_clean, B0_clean, vm = clean_ivim_signals(
                batch["x_true"].to(dev), SP_in, B0_in, bvals,
                inject_noise=True, noise_type="rician",
                noise_level=float(ck.snr_db),
            )
            SP = SP_clean if cfg["spatial_on"] else None
            B0 = B0_clean if cfg["spatial_on"] else None

            vm_np = vm.cpu().numpy().astype(bool)
            valid_chunks.append(vm_np)
            tissue.append(batch["tissue"].numpy())
            lesion.append(batch["lesion"].numpy())
            if X.shape[0] < 1:
                continue

            out = net(ivim_true_signal=X, structural_patch=SP, b0_patch=B0)
            Dpar, Fint, Dint, Fmv, Dmv, S0 = out["ivim_params"]

            preds["Dpar_pred"].append(Dpar.cpu().numpy().ravel())
            preds["Fmv_pred"].append(Fmv.cpu().numpy().ravel())
            preds["Dmv_pred"].append(Dmv.cpu().numpy().ravel())
            preds["S0_pred"].append(S0.cpu().numpy().ravel())
            if Fint is not None:
                preds["Fint_pred"].append(Fint.cpu().numpy().ravel())
            if Dint is not None:
                preds["Dint_pred"].append(Dint.cpu().numpy().ravel())


    if not quiet:
        print(flush=True)
    return preds, tissue, lesion, valid_chunks, time.time() - t0


def _report_progress(bi, n_batches, t0, quiet):
    """Progress line."""
    if quiet or not (bi == 0 or (bi + 1) % 10 == 0 or bi + 1 == n_batches):
        return
    done = (bi + 1) / n_batches
    el = time.time() - t0
    print(f"\r    batch {bi + 1}/{n_batches}  {el:5.1f}s elapsed, "
          f"{el / max(done, 1e-9) - el:5.1f}s remaining", end="", flush=True)


def _assemble(preds, tissue, lesion, valid_chunks, bvals):
    """Concatenate the per batch arrays into the output dictionary."""
    valid = np.concatenate(valid_chunks)
    n = len(valid)

    save = {}
    for key, chunks in preds.items():
        if not chunks:
            continue
        kept = np.concatenate(chunks)
        if len(kept) != int(valid.sum()):
            raise ValueError(
                f"{key}: {len(kept)} predictions for {int(valid.sum())} "
                f"valid voxels")
        full = np.zeros(n, dtype=np.float32)
        full[valid] = kept
        save[key] = full

    save["Tissue"] = np.concatenate(tissue)
    save["Lesion"] = np.concatenate(lesion)
    save["Valid_mask"] = valid.astype(np.uint8)
    save["bvals"] = np.asarray(bvals, dtype=np.float32)

    for k in ["Tissue", "Lesion"]:
        if len(save[k]) != n:
            raise ValueError(f"{k} has {len(save[k])} rows, expected {n}")
    return save


def resolve_test_path(paths, args=None):
    """Test h5 path, honouring the ablation flags."""
    if args is None:
        return Path(paths.synthetic_test)

    def _abs(value):
        candidate = Path(value)
        return candidate if candidate.is_absolute() else paths.repo_root / candidate

    explicit = getattr(args, "test_h5", None)
    if explicit:
        return _abs(explicit)

    data_dir = _abs(args.data_dir) if getattr(args, "data_dir", None) else None
    ps = getattr(args, "patch_size", None)
    if data_dir is None and not ps:
        return Path(paths.synthetic_test)

    data_dir = data_dir or paths.data_dir
    suffix = f"_p{int(ps)}" if ps else ""
    return data_dir / f"synthetic_IVIM_test_2D{suffix}.h5"


def run_one(ck, paths, device, batch_size, out_root, overwrite,
            quiet=False, checkpoint_dir=None, no_struct=False, args=None):
    """Run one checkpoint over the test set and write an H5."""
    import h5py
    import torch
    from torch.utils.data import DataLoader
    from pace.deep_models import clean_ivim_signals

    out_dir = out_root / f"{ck.model}_snr{ck.snr_db}"
    out_path = out_dir / f"inferred_seed{ck.seed}.h5"
    if out_path.is_file() and not overwrite:
        if not quiet:
            print(f"  [skip] exists: {out_path.name}")
        return None

    cfg = C.ARCH_CONFIGS[ck.cfg_name]
    arch_overrides = None
    if no_struct and cfg.get("spatial_on", False):
        arch_overrides = {"use_struct": False, "recon_struct": False}
        cfg = dict(cfg)
        cfg.update(arch_overrides)

    if getattr(args, "no_b0", False) and cfg.get("spatial_on", False):
        arch_overrides = dict(arch_overrides or {})
        arch_overrides.update(use_b0=False, recon_b0=False)
        cfg = dict(cfg)
        cfg.update(arch_overrides)

    if getattr(args, "use_ivim_nb", False) and cfg.get("spatial_on", False):
        arch_overrides = dict(arch_overrides or {})
        _bv = C.load_bvals(paths)
        arch_overrides.update(b0_channels=len(_bv))
        cfg = dict(cfg)
        cfg.update(arch_overrides)

    _tok = getattr(args, "anat_token", None)
    if _tok is not None and cfg.get("spatial_on", False):
        arch_overrides = dict(arch_overrides or {})
        arch_overrides["use_anat_token"] = bool(int(_tok))
        cfg = dict(cfg)
        cfg.update(arch_overrides)

    _widen = float(getattr(args, "voorter_bounds", 0.0) or 0.0)
    if _widen:
        arch_overrides = dict(arch_overrides or {})
        arch_overrides["widen_bounds"] = _widen
        cfg = dict(cfg)
        cfg.update(arch_overrides)
        if not quiet:
            print(f"  [BOUNDS] {ck.public_name}: widened by {_widen:g}")

    if getattr(args, "plain_head", False):
        arch_overrides = dict(arch_overrides or {})
        arch_overrides.update(use_ordered_diffusion=False,
                              use_softmax_fractions=False)
        cfg = dict(cfg)
        cfg.update(arch_overrides)
        if not quiet:
            print(f"  [HEAD] {ck.public_name}: plain head, overriding the "
                  f"manifest")

    _head = {}
    if getattr(args, "ordered_diffusion", None) is not None:
        _head["use_ordered_diffusion"] = bool(args.ordered_diffusion)
    if getattr(args, "softmax_fractions", None) is not None:
        _head["use_softmax_fractions"] = bool(args.softmax_fractions)
    if _head:
        arch_overrides = dict(arch_overrides or {})
        arch_overrides.update(_head)
        cfg = dict(cfg)
        cfg.update(arch_overrides)
        if not quiet:
            said = ", ".join(f"{k.replace('use_', '')}={v}"
                             for k, v in sorted(_head.items()))
            print(f"  [HEAD] {ck.public_name}: {said}")
    bvals = C.load_bvals(paths)
    dev = C.resolve_device(device)

    net, _ = C.load_net(ck.model, ck.snr_db, ck.seed, bvals=bvals,
                        device=device, paths=paths, strict=True,
                        checkpoint_dir=checkpoint_dir,
                        arch_overrides=arch_overrides)

    test_h5 = resolve_test_path(paths, args)
    ds = PreloadedIVIMDataset(str(test_h5),
                             use_struct=cfg["use_struct"],
                             use_b0=cfg["use_b0"],
                             use_ivim_nb=getattr(args, "use_ivim_nb", False),
                             ivim_nb_keep_center=getattr(
                                 args, "ivim_nb_keep_center", False),
                             ivim_nb_oracle=getattr(
                                 args, "ivim_nb_oracle", False),
                             ivim_nb_anat_oracle=getattr(
                                 args, "ivim_nb_anat_oracle", False))

    net_ps = getattr(net, "patch_size", None)
    if net_ps is not None and cfg.get("spatial_on", False) \
            and int(net_ps) != ds.patch_size:
        raise RuntimeError(
            f"{ck.public_name} was built for {int(net_ps)}x{int(net_ps)} "
            f"patches but {Path(test_h5).name} holds "
            f"{ds.patch_size}x{ds.patch_size}.")
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)
    n_batches = len(loader)
    if not quiet:
        print(f"  {len(ds):,} voxels in {n_batches} batches on {dev}",
              flush=True)

    preds, tissue, lesion, valid_chunks, elapsed = _forward_pass(
        net, loader, cfg, bvals, dev, ck, n_batches, quiet)

    ds.close()
    if not quiet:
        print(flush=True)

    save = _assemble(preds, tissue, lesion, valid_chunks, bvals)
    n_total = len(save["Valid_mask"])
    n_kept = int(save["Valid_mask"].sum())

    lengths = {len(v) for k, v in save.items() if k != "bvals"}
    if lengths != {n_total}:
        raise ValueError(f"{ck.public_name}: inconsistent array lengths "
                         f"{sorted(lengths)}, expected all {n_total}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for k, v in save.items():
            f.create_dataset(k, data=v, compression="gzip")

    if not quiet:
        print(f"  {n_kept:,}/{n_total:,} voxels kept "
              f"({100 * n_kept / max(n_total, 1):.1f}%)  "
              f"{elapsed:.1f}s  -> {out_path.name}")

    del net
    return {"public_name": ck.public_name, "n_kept": n_kept,
            "n_total": n_total, "seconds": elapsed, "path": out_path}


def compare(ck, paths, out_root):
    """Compare a regenerated result against the committed reference."""
    import h5py

    ref_path = (paths.results_dir / f"{ck.model}_snr{ck.snr_db}"
                / f"inferred_seed{ck.seed}.h5")
    new_path = (out_root / f"{ck.model}_snr{ck.snr_db}"
                / f"inferred_seed{ck.seed}.h5")
    if not (ref_path.is_file() and new_path.is_file()):
        return None

    rows = []
    with h5py.File(ref_path) as fr, h5py.File(new_path) as fn:
        for key in PRED_KEYS:
            if key not in fr or key not in fn:
                continue
            a = fr[key][:].astype(np.float64)
            b = fn[key][:].astype(np.float64)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size == 0 or b.size == 0:
                continue
            ma, mb = a.mean(), b.mean()
            rel = abs(ma - mb) / max(abs(ma), 1e-12)
            rows.append({
                "param": key.replace("_pred", ""),
                "ref_mean": ma, "new_mean": mb, "rel_diff": rel,
                "ref_n": a.size, "new_n": b.size,
            })
    return rows


def preflight(args, paths, manifest):
    """Validate the request."""
    out_root = Path(args.out)
    if not out_root.is_absolute():
        out_root = paths.repo_root / out_root

    for protected, label in [(paths.results_dir, "results/synthetic"),
                             (paths.checkpoints_dir, "checkpoints/synthetic")]:
        if _reject_if_inside(out_root, protected, label):
            return out_root, None

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
        return out_root, None

    try:
        import torch
    except ImportError:
        print("[ERROR] torch is required. pip install torch", file=sys.stderr)
        return out_root, None

    return out_root, todo


def print_comparison(todo, paths, out_root, weights_note=None):
    """Report the regenerated output against the committed reference."""
    print()
    print("=" * 72)
    print(" Regenerated against the committed reference")
    print("=" * 72)
    if weights_note:
        print(f" Weights: {weights_note}")
        print(" These are different weights to the ones that produced the")
        print(" reference, so this compares two trained networks, not two")
        print(" runs of the same one.")
        print()
    print(" Voxel level equality is not expected. Rician noise is injected")
    print(" at inference and the random stream depends on the backend, so")
    print(" the two runs retain different voxels and see different noise.")
    print(" Close agreement in the means is the meaningful check.")
    print()

    worst = 0.0
    for ck in todo:
        rows = compare(ck, paths, out_root)
        if not rows:
            continue
        print(f" {ck.public_name}")
        print(f"   {'param':<6} {'reference':>12} {'regenerated':>12} "
              f"{'rel diff':>10} {'n ref':>8} {'n new':>8}")
        for r in rows:
            worst = max(worst, r["rel_diff"])
            print(f"   {r['param']:<6} {r['ref_mean']:>12.6g} "
                  f"{r['new_mean']:>12.6g} {r['rel_diff']:>9.2%} "
                  f"{r['ref_n']:>8,} {r['new_n']:>8,}")
        print()

    print(f" largest relative difference in any mean: {worst:.2%}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce the synthetic inference results from the released checkpoints.")
    ap.add_argument("--models", nargs="+", default=None,
                    help="Neural models to run. Default: all in the manifest.")
    ap.add_argument("--snrs", nargs="+", type=int, default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--voorter-bounds", dest="voorter_bounds", nargs="?",
                    type=float, const=0.3, default=0.0,
                    help="Score a checkpoint trained with --voorter-bounds. "
                         "The factor must match training or the model is "
                         "clipped at inference.")
    ap.add_argument("--add-seeds", dest="add_seeds", nargs="+", type=int,
                    default=None,
                    help="Score seeds the manifest does not list, by "
                         "cloning an existing entry at the same model and "
                         "SNR. Pairs with the same flag on "
                         "train_synthetic.py. Nothing is written back to "
                         "the manifest.")
    ap.add_argument("--device", default=None,
                    help="cuda, mps, cpu or auto. Default: auto.")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--out", default="results/synthetic_regenerated",
                    help="Output directory, relative to the repository root.")
    ap.add_argument("--plain-head", dest="plain_head", action="store_true",
                    help="Score a checkpoint trained with --plain-head on "
                         "train_synthetic.py: no ordered diffusion "
                         "constraint and no softmax simplex. The flag must "
                         "match training or the weights will not load.")
    ap.add_argument("--ordered-diffusion", dest="ordered_diffusion",
                    type=int, choices=[0, 1], default=None,
                    help="Force the ordered diffusion constraint on or off, "
                         "overriding both the manifest and --plain-head. "
                         "Must match training: this changes no parameter "
                         "shape, so a mismatch loads cleanly and returns "
                         "wrong values.")
    ap.add_argument("--softmax-fractions", dest="softmax_fractions",
                    type=int, choices=[0, 1], default=None,
                    help="Force the softmax simplex on or off, overriding "
                         "both the manifest and --plain-head. Without it the "
                         "fractions need a range, so pair it with "
                         "--voorter-bounds.")
    ap.add_argument("--anat-token", dest="anat_token", type=int,
                    choices=[0, 1], default=None,
                    help="1 fuses T1 and FLAIR into one anatomy token, 0 "
                         "keeps them separate. Must match the flag the "
                         "checkpoint was trained with: the two forms have "
                         "different modules and a different number of "
                         "spatial tokens, so a mismatch will not load. The "
                         "value is recorded as use_anat_token in the "
                         "summary npz beside each checkpoint.")
    ap.add_argument("--ivim-nb", dest="use_ivim_nb", action="store_true",
                    help="The spatial token carries neighbouring IVIM "
                         "curves rather than the b0 patch. Must match how "
                         "the checkpoint was trained: it was built with a "
                         "46 channel encoder and will not load without "
                         "this.")
    ap.add_argument("--ivim-nb-keep-centre", dest="ivim_nb_keep_center",
                    action="store_true",
                    help="Leave the centre voxel in that token rather "
                         "than zeroing it. Must match training.")
    ap.add_argument("--ivim-nb-anat-oracle", dest="ivim_nb_anat_oracle",
                    action="store_true",
                    help="Average the k neighbours nearest the centre in "
                         "T1-FLAIR space. Must match training: nothing in "
                         "the checkpoint records which selection was "
                         "used, so the run command is the only record.")
    ap.add_argument("--ivim-nb-oracle", dest="ivim_nb_oracle",
                    action="store_true",
                    help="Average only the neighbours whose true class "
                         "matches the centre's. A diagnostic, not a "
                         "method: it reads ground truth at inference. "
                         "Must match training.")
    ap.add_argument("--no-b0", dest="no_b0", action="store_true",
                    help="Build the b0 free architecture: no b0 encoder, "
                         "T1 and FLAIR retained. The mirror of --no-struct, "
                         "and required for checkpoints trained with the "
                         "matching train_synthetic flag.")
    ap.add_argument("--no-struct", dest="no_struct", action="store_true",
                    help="Build the anatomy free architecture: no T1 or "
                         "FLAIR encoders, b0 neighbourhood and gating "
                         "retained. Required for checkpoints trained with "
                         "the matching train_synthetic flag, which carry "
                         "no anatomical weights to load.")
    ap.add_argument("--test-h5", dest="test_h5", default=None,
                    help="Test h5. Overrides --data-dir and --patch-size, "
                         "and assumes nothing about how the file is named.")
    ap.add_argument("--data-dir", dest="data_dir", default=None,
                    help="Directory holding the synthetic h5 files. "
                         "Relative paths resolve against the repository "
                         "root. Default: the released data directory.")
    ap.add_argument("--patch-size", dest="patch_size", type=int, default=None,
                    help="Selects the _pN test split. It does not configure "
                         "anything: the architecture comes from the "
                         "checkpoint, and a mismatch against the data stops "
                         "the run.")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--checkpoints", default=None,
                    help="Read weights from this directory instead of the "
                         "released ones. The architecture still comes from "
                         "the manifest, so the two differ only in weights.")
    ap.add_argument("--compare", action="store_true",
                    help="Compare against results/synthetic afterwards.")
    args = ap.parse_args()

    if getattr(args, "no_b0", False) and getattr(args, "no_struct", False):
        print("[ERROR] --no-b0 and --no-struct together leave the spatial "
              "pathway with no inputs. No checkpoint was trained that way.",
              file=sys.stderr)
        return 2

    paths = C.load_paths()
    manifest = C.load_manifest(paths)

    out_root, todo = preflight(args, paths, manifest)
    if todo is None:
        return 2

    dev = C.resolve_device(args.device)
    print(f"[DATA  ] {resolve_test_path(paths, args).name}")
    print(f"[WEIGHT] {args.checkpoints or 'checkpoints/synthetic (released)'}")
    print(f"[DEVICE] {dev}")
    print(f"[OUT   ] {out_root.relative_to(paths.repo_root)}")
    print(f"[RUN   ] {len(todo)} checkpoint(s)")
    print()

    done, failed = [], []
    for i, ck in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {ck.public_name}  "
              f"cfg={ck.cfg_name} fusion={ck.fusion_mode} "
              f"ordered={ck.use_ordered_diffusion} "
              f"softmax={ck.use_softmax_fractions}")
        try:
            r = run_one(ck, paths, args.device, args.batch_size, out_root,
                        args.overwrite, checkpoint_dir=args.checkpoints,
                        no_struct=args.no_struct,
                        args=args)
            if r:
                done.append(r)
        except Exception as e:
            failed.append((ck.public_name, f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {type(e).__name__}: {e}")

    print()
    print(f"[DONE] {len(done)} written, {len(failed)} failed")
    for name, err in failed:
        print(f"   {name}: {err}")

    if args.compare:
        print_comparison(todo, paths, out_root, args.checkpoints)

    print()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
