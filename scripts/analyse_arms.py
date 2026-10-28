#!/usr/bin/env python3
"""Compare result trees on signal RMSE, parameter RMSE and CNR, per tissue.

Usage:
    python analyse_arms.py --trees synthetic
    python analyse_arms.py --trees synthetic --by-tissue
    python analyse_arms.py --trees synthetic abl_noce abl_gate0
    python analyse_arms.py --trees abl_p5 --test-h5 data/synthetic/synthetic_IVIM_test_2D_p5.h5
"""

import argparse
import dataclasses
import re
import sys
from pathlib import Path

import numpy as np

PARAMS = ("Dpar", "Dint", "Dmv", "Fint", "Fmv")
TISSUES = ("GM", "NAWM", "WMH")
DIR_RE = re.compile(r"^(?P<model>.+)_snr(?P<snr>\d+)$")
FILE_RE = re.compile(r"^inferred_seed(?P<seed>\d+)\.h5$")

GROUPS = (("diffusivities", ("Dpar", "Dint", "Dmv")),
          ("fractions", ("Fint", "Fmv")))

OVERLAP_BINS = {
    "GM": 80,
    "NAWM": 80,
    "WMH": 40,
}

_OVERLAP_EDGES = {}
_OVERLAP_REFERENCE = {}


def shared_edges(truth, bins):
    """Bin edges over the central 99 percent of the truth, widened by half
    that range on each side.
    """
    lower, upper = np.percentile(truth, [0.5, 99.5])
    margin = 0.5 * (upper - lower)
    return np.linspace(lower - margin, upper + margin, bins + 1)


def overlap_area(first, second, edges):
    """Shared area under two density histograms on the same edges."""
    low, high = edges[0], edges[-1]
    first_density, _ = np.histogram(np.clip(first, low, high), edges,
                                    density=True)
    second_density, _ = np.histogram(np.clip(second, low, high), edges,
                                     density=True)
    return float(np.sum(np.minimum(first_density, second_density)
                        * np.diff(edges)))


def prepare_overlap(lab, gt_):
    """Set the shared bins and the truth halves reference, once per run."""
    if _OVERLAP_EDGES:
        return
    generator = np.random.default_rng(0)
    for t in TISSUES:
        indices = np.flatnonzero(lab == t)
        if len(indices) < 20:
            continue
        generator.shuffle(indices)
        first, second = np.array_split(indices, 2)
        _OVERLAP_REFERENCE[t] = {}
        for p in PARAMS:
            edges = shared_edges(gt_[p][lab == t], OVERLAP_BINS[t])
            _OVERLAP_EDGES[(t, p)] = edges
            _OVERLAP_REFERENCE[t][p] = overlap_area(gt_[p][first],
                                                    gt_[p][second], edges)


def overlap_reference(tissue, p):
    """Truth halves overlap for one tissue, or the mean over tissues."""
    if tissue != "mean":
        return _OVERLAP_REFERENCE.get(tissue, {}).get(p, np.nan)
    values = [_OVERLAP_REFERENCE.get(t, {}).get(p, np.nan) for t in TISSUES]
    values = np.asarray(values, float)
    return float(np.nanmean(values)) if np.isfinite(values).any() else np.nan


def reconstruct(Dpar, Dint, Dmv, Fint, Fmv, S0, bvals):
    """Three compartment IVIM forward model."""
    b = np.asarray(bvals, dtype=np.float64).reshape(1, -1)
    Fpar = 1.0 - Fint - Fmv
    return (S0[:, None] * (Fpar[:, None] * np.exp(-b * Dpar[:, None])
                           + Fint[:, None] * np.exp(-b * Dint[:, None])
                           + Fmv[:, None] * np.exp(-b * Dmv[:, None])))


def cnr_magnotta(g1, g2):
    """|mu1 - mu2| / sqrt(var1 + var2), ddof=1, as the notebook defines it."""
    g1 = np.asarray(g1, np.float64); g1 = g1[np.isfinite(g1)]
    g2 = np.asarray(g2, np.float64); g2 = g2[np.isfinite(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan
    denom = np.sqrt(g1.var(ddof=1) + g2.var(ddof=1))
    if denom <= 0:
        return np.nan
    return float(abs(g1.mean() - g2.mean()) / denom)


def discover(results_root, trees, snr, models=None):
    found = []
    for tree in trees:
        base = results_root / tree
        if not base.is_dir():
            print(f"[warn] no such tree: {base}", file=sys.stderr)
            continue
        for d in sorted(base.iterdir()):
            m = DIR_RE.match(d.name) if d.is_dir() else None
            if not m or int(m.group("snr")) != snr:
                continue
            model = m.group("model")
            if models and model not in models:
                continue
            seeds = sorted(int(FILE_RE.match(f.name).group("seed"))
                           for f in d.iterdir() if FILE_RE.match(f.name))
            if seeds:
                found.append((tree, model, snr, seeds))
    return found


def valid_mask(r, n):
    """r.valid_mask where present, all true otherwise."""
    m = getattr(r, "valid_mask", None)
    if m is None:
        return np.ones(n, bool)
    return np.asarray(m).reshape(-1).astype(bool)


def one_seed(C, base, results_root, tree, model, snr, seed, paired):
    """Every metric for one inference file, tissue resolved where it
    applies.
    """
    paths = dataclasses.replace(base, results_dir=results_root / tree)
    r = C.load_result(model, snr, seed, paths=paths)
    lab = np.asarray(C.decode_tissue_array(r.tissue, r.lesion)).reshape(-1)
    n = len(lab)
    v = valid_mask(r, n)
    pr = paired if (paired is not None and len(paired) == n) else None

    pd_ = {p: np.asarray(r.params[p], np.float64).reshape(-1) for p in PARAMS}
    gt_ = {p: np.asarray(r.gt[p], np.float64).reshape(-1) for p in PARAMS}
    s0_p = np.asarray(r.params.get("S0", np.ones(n)), np.float64).reshape(-1)
    s0_g = np.asarray(r.gt.get("S0", np.ones(n)), np.float64).reshape(-1)
    bvals = np.asarray(getattr(r, "bvals", None)
                       if getattr(r, "bvals", None) is not None
                       else C.load_bvals(base), np.float64).reshape(-1)

    out = {"param_rmse": {}, "param_bias": {}, "param_esd": {},
           "signal_rmse": {}, "cnr_pred": {},
           "cnr_true": {}, "corr": {}, "sd_lin": {},
           "paired_rmse": {}, "overlap": {}}

    resid = []
    for t in TISSUES:
        m = v & (lab == t)
        ok = m.sum() >= 10
        out["param_rmse"][t] = {}
        out["param_bias"][t] = {}
        out["param_esd"][t] = {}
        for p in PARAMS:
            if not ok:
                out["param_rmse"][t][p] = np.nan
                out["param_bias"][t][p] = np.nan
                out["param_esd"][t][p] = np.nan
                continue
            e = pd_[p][m] - gt_[p][m]
            r = float(np.sqrt((e ** 2).mean()))
            b = float(e.mean())
            s = float(e.std())
            out["param_rmse"][t][p] = r
            out["param_bias"][t][p] = b
            out["param_esd"][t][p] = s
            if r > 0:
                resid.append(abs(float(np.hypot(b, s)) - r) / r)
    for k in ("param_rmse", "param_bias", "param_esd"):
        out[k]["mean"] = {
            p: float(np.nanmean([out[k][t][p] for t in TISSUES]))
            for p in PARAMS}
    out["_decomp"] = {"": max(resid) if resid else 0.0}

    prepare_overlap(lab, gt_)
    for t in TISSUES:
        m = v & (lab == t)
        out["overlap"][t] = {
            p: (overlap_area(gt_[p][m], pd_[p][m], _OVERLAP_EDGES[(t, p)])
                if m.sum() >= 10 and (t, p) in _OVERLAP_EDGES else np.nan)
            for p in PARAMS}
    out["overlap"]["mean"] = {}
    for p in PARAMS:
        values = np.asarray([out["overlap"][t][p] for t in TISSUES], float)
        out["overlap"]["mean"][p] = (float(np.nanmean(values))
                                     if np.isfinite(values).any() else np.nan)

    if pr is None:
        out["paired_rmse"]["all"] = {p: np.nan for p in PARAMS}
    else:
        m = v & pr
        out["paired_rmse"]["all"] = {
            p: (float(np.sqrt(((pd_[p][m] - gt_[p][m]) ** 2).mean()))
                if m.sum() >= 10 else np.nan) for p in PARAMS}

    sp = reconstruct(pd_["Dpar"], pd_["Dint"], pd_["Dmv"],
                     pd_["Fint"], pd_["Fmv"], s0_p, bvals)
    sg = reconstruct(gt_["Dpar"], gt_["Dint"], gt_["Dmv"],
                     gt_["Fint"], gt_["Fmv"], s0_g, bvals)
    per_voxel = np.sqrt(np.nanmean((sp - sg) ** 2, axis=1))
    for t in TISSUES:
        m = v & (lab == t)
        out["signal_rmse"][t] = (float(np.nanmean(per_voxel[m]))
                                 if m.sum() >= 10 else np.nan)
    out["signal_rmse"]["mean"] = float(np.nanmean(
        [out["signal_rmse"][t] for t in TISSUES]))
    out["signal_rmse"]["all"] = float(np.nanmean(per_voxel[v]))

    a, b_ = v & (lab == "WMH"), v & (lab == "NAWM")
    for p in PARAMS:
        out["cnr_pred"][p] = cnr_magnotta(pd_[p][a], pd_[p][b_])
        out["cnr_true"][p] = cnr_magnotta(gt_[p][a], gt_[p][b_])

        cs, vp, vt = [], [], []
        for m in (a, b_):
            if m.sum() < 50:
                continue
            if pd_[p][m].std() > 0 and gt_[p][m].std() > 0:
                cs.append(np.corrcoef(pd_[p][m], gt_[p][m])[0, 1])
            vp.append(pd_[p][m].var()); vt.append(gt_[p][m].var())
        out["corr"][p] = float(np.mean(cs)) if cs else np.nan
        out["sd_lin"][p] = (float(np.sqrt(np.mean(vp)) / np.sqrt(np.mean(vt)))
                            if vt and np.mean(vt) > 0 else np.nan)
    return out


def _mean_of(rows):
    """Average a list of per seed metric dicts into one, key by key."""
    def rec(vals):
        if isinstance(vals[0], dict):
            return {k: rec([v[k] for v in vals]) for k in vals[0]}
        a = np.asarray(vals, float)
        return float(np.nanmean(a)) if np.isfinite(a).any() else np.nan

    return rec(rows)


def cell(v, sd=None, col=20):
    if v is None or not np.isfinite(v):
        return f"{'-':>{col}}"
    if sd is None or not np.isfinite(sd) or sd == 0:
        return f"{v:>{col}.4g}"
    return f"{v:.4g} ±{sd:.1g}".rjust(col)


def block(title, labels, get, width, col=20, cols=PARAMS, note=None):
    print(title)
    if note:
        print(f"  {note}")
    print(f"{'':{width}}" + "".join(f"{c:>{col}}" for c in cols))
    for label in labels:
        line = f"{label:{width}}"
        for c in cols:
            v, sd = get(label, c)
            line += cell(v, sd, col)
        print(line)
    print()


DIRECTION = {"signal": "low", "param": "low", "paired": "low",
             "cnr": "low", "corr": "high", "sd": "one", "bias": "zero",
             "overlap": "high"}


def score(kind, value):
    if not np.isfinite(value):
        return np.inf
    if DIRECTION[kind] == "low":
        return value
    if DIRECTION[kind] == "high":
        return -value
    if DIRECTION[kind] == "zero":
        return abs(value)
    return abs(value - 1.0)


def ranked(labels, title, kind, table, cols=PARAMS, top=5):
    """The best few arms per parameter, in order, with their values."""
    print(title)
    if kind == "sd":
        print("  ranked by |ratio - 1|")
    if kind == "bias":
        print("  ranked by |bias|, sign shown")
    for c in cols:
        vals = [(l, table[l][c]) for l in labels
                if np.isfinite(table[l].get(c, np.nan))]
        if not vals:
            continue
        vals.sort(key=lambda kv: score(kind, kv[1]))
        head = f"  {c:>6}  " if c else "  "
        parts = []
        for i, (l, v) in enumerate(vals[:top], 1):
            parts.append(f"{i}. {l} {v:.4g}")
        print(head + "   ".join(parts))
        if len(vals) > 1:
            gap = score(kind, vals[1][1]) - score(kind, vals[0][1])
            rel = gap / abs(score(kind, vals[0][1])) if score(
                kind, vals[0][1]) else float("inf")
            if np.isfinite(rel) and rel < 0.03:
                print(f"  {'':>6}  first two within 3 percent, treat as a "
                      f"tie")
    print()


def verdict(name, kind, labels, table):
    """Best arm per parameter, then per group, counting wins."""
    per_param = []
    for p in PARAMS:
        cand = {l: table[l][p] for l in labels}
        best = min(cand, key=lambda l: score(kind, cand[l]))
        if np.isfinite(cand[best]):
            per_param.append(f"{p} {best} ({cand[best]:.4g})")
    if not per_param:
        return
    print(name)
    print("  " + "   ".join(per_param))
    for gname, group in GROUPS:
        wins = {l: 0 for l in labels}
        for p in group:
            cand = {l: table[l][p] for l in labels}
            best = min(cand, key=lambda l: score(kind, cand[l]))
            if np.isfinite(cand[best]):
                wins[best] += 1
        top = max(wins.values())
        if top:
            lead = [l for l, c in wins.items() if c == top]
            print(f"  {gname}: {', '.join(lead)} ({top} of {len(group)})")
    print()


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Compare result trees, following the inference "
                    "summary conventions.")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--trees", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--snr", type=int, default=25)
    ap.add_argument("--all-snr", nargs="*", type=int, default=None,
                    metavar="SNR",
                    help="Average over several SNRs instead of reading one. "
                         "With no values it uses 20, 25, 30 and 35. Each "
                         "SNR is averaged over its own seeds first and the "
                         "SNRs are then averaged equally, so five seeds at "
                         "25 do not outweigh one at 20, which is the "
                         "convention the summary workbook uses. An arm "
                         "missing an SNR is skipped for that SNR and the "
                         "count is reported, since an arm averaged over "
                         "two noise levels is not comparable to one "
                         "averaged over four.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--verbose", action="store_true",
                    help="Also print the CNR bias and cross seed spread, "
                         "and the within class correlation. All three are "
                         "computed either way; they are held back because "
                         "the predicted CNR beside its truth row already "
                         "shows the direction.")
    ap.add_argument("--by-tissue", action="store_true",
                    help="Also print parameter and signal RMSE separately "
                         "for GM, NAWM and WMH.")
    ap.add_argument("--test-h5", default=None,
                    help="Where the Paired flag is read from. Needed for a "
                         "tree inferred against a different phantom.")
    args = ap.parse_args(argv)

    repo = Path(args.repo).expanduser().resolve()
    sys.path.insert(0, str(repo))
    import h5py
    from pace import common as C

    base = C.load_paths()
    results_root = repo / "results"
    test_h5 = Path(args.test_h5).expanduser() if args.test_h5 else Path(
        base.synthetic_test)
    if not test_h5.is_absolute():
        test_h5 = repo / test_h5

    paired = None
    try:
        with h5py.File(test_h5, "r") as f:
            if "Paired" in f:
                paired = f["Paired"][:].astype(bool)
    except OSError as exc:
        print(f"[warn] could not read {test_h5}: {exc}", file=sys.stderr)

    snrs = ([20, 25, 30, 35] if args.all_snr == [] else
            args.all_snr or [args.snr])

    per_snr, labels, nseeds, coverage = {}, [], {}, {}
    for snr in snrs:
        for tree, model, _s, seeds in discover(results_root, args.trees,
                                               snr, args.models):
            use = [s for s in seeds if not args.seeds or s in args.seeds]
            rows = []
            for s in use:
                try:
                    rows.append(one_seed(C, base, results_root, tree,
                                         model, snr, s, paired))
                except Exception as exc:
                    print(f"[warn] {tree}/{model} snr {snr} seed {s}: "
                          f"{exc}", file=sys.stderr)
            if not rows:
                continue
            label = model if tree == "synthetic" else f"{tree}/{model}"
            per_snr.setdefault(label, []).append(rows)
            nseeds[label] = nseeds.get(label, 0) + len(rows)
            coverage.setdefault(label, []).append(snr)

    if not per_snr:
        print(f"[ERROR] nothing found at SNR "
              f"{', '.join(map(str, snrs))} in "
              f"{', '.join(args.trees)}", file=sys.stderr)
        return 1

    data, labels = {}, []
    for label, blocks in per_snr.items():
        if len(blocks) == 1:
            data[label] = blocks[0]
        else:
            data[label] = [_mean_of(b) for b in blocks]
        labels.append(label)
    labels.sort()

    if not labels:
        print("[ERROR] nothing loaded", file=sys.stderr)
        return 1

    width = max(len(l) for l in labels) + 2

    def agg(label, path):
        """Mean and spread across seeds for one nested key path."""
        vals = []
        for d in data[label]:
            x = d
            for k in path:
                x = x[k]
            vals.append(x)
        vals = np.asarray(vals, float)
        if not np.isfinite(vals).any():
            return np.nan, None
        m = float(np.nanmean(vals))
        sd = float(np.nanstd(vals)) if len(vals) > 1 else None
        return m, sd

    if len(snrs) > 1:
        print("SNR " + ", ".join(map(str, snrs))
              + "; each averaged over its seeds, then over SNRs equally")
        short = [l for l in labels if len(coverage[l]) < len(snrs)]
        for l in labels:
            miss = [s for s in snrs if s not in coverage[l]]
            print(f"  {l:34} {len(coverage[l])}/{len(snrs)} SNR, "
                  f"{nseeds[l]} file(s)"
                  + (f", missing {', '.join(map(str, miss))}" if miss else ""))
        if short:
            print("  Arms above with fewer SNRs are averaged over fewer "
                  "noise levels and are not")
            print("  directly comparable to the rest.")
    else:
        print(f"SNR {snrs[0]}, seeds per arm: "
              + ", ".join(f"{l} {nseeds[l]}" for l in labels))
    print()

    cnr_err, cnr_bias, cnr_prec = {}, {}, {}
    for l in labels:
        cnr_err[l], cnr_bias[l], cnr_prec[l] = {}, {}, {}
        for prm in PARAMS:
            pred = np.asarray([d["cnr_pred"][prm] for d in data[l]], float)
            true = np.asarray([d["cnr_true"][prm] for d in data[l]], float)
            if not np.isfinite(pred).any() or not np.isfinite(true).any():
                cnr_err[l][prm] = cnr_bias[l][prm] = np.nan
                cnr_prec[l][prm] = np.nan
                continue
            tv = float(np.nanmean(true))
            b = float(np.nanmean(pred) - tv)
            s = float(np.nanstd(pred)) if len(pred) > 1 else 0.0
            cnr_bias[l][prm] = b
            cnr_prec[l][prm] = s if len(pred) > 1 else np.nan
            cnr_err[l][prm] = (b * b + s * s) ** 0.5

    print("=" * 60)
    print("HEADLINE")
    print("=" * 60)
    print()

    block("Signal RMSE, reconstructed from parameters", labels,
          lambda l, c: agg(l, ("signal_rmse", c)), width,
          cols=("GM", "NAWM", "WMH", "mean"),
          note="Both signals noiseless.")

    block("Parameter RMSE, mean over GM, NAWM and WMH", labels,
          lambda l, prm: agg(l, ("param_rmse", "mean", prm)), width,
          note="Per tissue then averaged, as the summary notebook does. "
               "Pooling would weight GM eight times WMH.")
    block("Parameter signed bias", labels,
          lambda l, prm: agg(l, ("param_bias", "mean", prm)), width,
          note="Negative is underestimation. RMSE has no direction, so a "
               "systematic error and honest scatter read the same "
               "without this. The dispersion of the errors is "
               "sqrt(RMSE^2 - bias^2), decomposed within each tissue "
               "before averaging since that identity does not survive "
               "macro averaging.")

    block("Predicted CNR, WMH against NAWM (Magnotta)", labels,
          lambda l, prm: agg(l, ("cnr_pred", prm)), width)
    print(f"{'truth':{width}}" + "".join(
        cell(agg(labels[0], ("cnr_true", prm))[0]) for prm in PARAMS))
    print()
    block("Lesion CNR RMSE", labels,
          lambda l, prm: (cnr_err[l][prm], None), width,
          note="sqrt(bias^2 + cross seed variance), in CNR units. "
               "Not divided by the true contrast, so it matches "
               "what the figures report.")

    worst = max((agg(l, ("_decomp", ""))[0] or 0.0) for l in labels)
    if worst > 1e-4:
        print(f"  [warn] the per tissue decomposition disagrees by up to "
              f"{100 * worst:.2f}%, which should be numerical only")
        print()

    print("=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    print()

    block("Within class SD, predicted over ground truth", labels,
          lambda l, prm: agg(l, ("sd_lin", prm)), width,
          note="Pooled over WMH and NAWM, ideal 1.0. Below means the "
               "estimator shrank the within tissue variation, which "
               "lowers RMSE without measuring anything. IVIM3C on Dmv is "
               "the case to have in mind: the best RMSE in the table at "
               "half the true spread.")

    block("Overlap with truth, mean over GM, NAWM and WMH", labels,
          lambda l, prm: agg(l, ("overlap", "mean", prm)), width,
          note="Shared area under the predicted and true histograms, 1 "
               "for identical distributions. Penalizes a narrowed "
               "distribution and a shifted one alike. Bins are shared by "
               "every arm. The truth halves row is two random halves of "
               "the truth against each other, the level histogram noise "
               "alone allows.")
    print(f"{'truth halves':{width}}" + "".join(
        cell(overlap_reference("mean", prm)) for prm in PARAMS))
    print()

    if paired is not None:
        block("Parameter RMSE, planted degenerate voxels", labels,
              lambda l, prm: agg(l, ("paired_rmse", "all", prm)), width,
              note="Voxels whose clean signals match a cross class "
                   "partner to 0.2 percent while a parameter differs by "
                   "a sigma. Nothing in the signal separates them, so a "
                   "gain here came from the anatomy. Not a general "
                   "performance measure and not ranked.")

    if args.verbose:
        block("CNR bias, predicted minus true", labels,
              lambda l, prm: (cnr_bias[l][prm], None), width,
              note="Positive is over separation.")
        block("CNR cross seed SD", labels,
              lambda l, prm: (cnr_prec[l][prm], None), width,
              note="Blank where only one seed exists.")
        block("Within class correlation, predicted against true", labels,
              lambda l, prm: agg(l, ("corr", prm)), width,
              note="Whether the right voxels got the right values.")
        if args.by_tissue:
            for tis in TISSUES:
                block(f"Parameter RMSE, {tis}", labels,
                      lambda l, prm, tis=tis: agg(l,
                                                  ("param_rmse", tis, prm)),
                      width)
            for tis in TISSUES:
                block(f"Overlap with truth, {tis}", labels,
                      lambda l, prm, tis=tis: agg(l, ("overlap", tis, prm)),
                      width)
                print(f"{'truth halves':{width}}" + "".join(
                    cell(overlap_reference(tis, prm)) for prm in PARAMS))
                print()

    print("=" * 60)
    print("RANKINGS")
    print("=" * 60)
    print()

    ranked(labels, "Signal RMSE, mean over tissues", "signal",
           {l: {"": agg(l, ("signal_rmse", "mean"))[0]} for l in labels},
           cols=("",))
    ranked(labels, "Parameter RMSE", "param",
           {l: {prm: agg(l, ("param_rmse", "mean", prm))[0]
                for prm in PARAMS} for l in labels})
    ranked(labels, "Overlap with truth, mean over tissues", "overlap",
           {l: {prm: agg(l, ("overlap", "mean", prm))[0]
                for prm in PARAMS} for l in labels})
    ranked(labels, "Lesion CNR RMSE", "cnr", cnr_err)
    return 0


if __name__ == "__main__":
    sys.exit(main())
