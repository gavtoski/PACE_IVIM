"""
Conventional IVIM Fitting Baselines -- Two-Step LSQ + NNLS + Bayesian MAP
==========================================================
LSQ/NNLS adapted from: Paulien Voorter (Jan 2022)
  p.voorter@maastrichtuniversity.nl
  https://www.github.com/paulienvoorter

Bayesian MAP adapted from: Gurney-Champion / Barbieri (MRM 2021)
  o.j.gurney-champion@amsterdamumc.nl
  Originally 2-compartment; extended here to 3-compartment IVIM.
  Uses data-driven empirical priors (lognormal for diffusivities,
  beta for fractions) fitted from initial LSQ estimates, rather
  than fixed Gaussian priors centered on bounds midpoints.

Modifications (March 2026):
  - Standalone module (no global arg dependency)
  - Bounds loaded dynamically from hyperparams_deep2.net_pars()
  - Consistent output order: [Dpar, Fint, Dint, Fmv, Dmv, S0]
  - All methods normalize internally + return absolute S0
  - Diagnostic output: per-method fitting MSE + convergence stats
  - B-value averaging (Voorter Trace-averaging equivalent)
  - Multi-start LSQ: 3 random initializations, keep best MSE
  - Fast MAP prior: raw numpy math instead of scipy.stats.pdf (~40x speedup)
  - FIX (March 2026): When a compartment fraction is negligible (F<1e-4)
    or the NNLS spectrum has no weight in a compartment, the corresponding
    diffusivity is set to the midpoint of its bounds instead of NaN.
  - FIX (March 11 2026): NaN-safe normalization via _safe_normalize().
    Handles NaN/Inf/negative S0_init. Per-voxel NNLS try/except.
    Invalid voxels masked to finite sentinels at output.
  - FIX (March 12 2026): split_prior=True in MAP prevents double-dipping
    where the empirical prior learns the answer from the same voxels it
    regularizes. Prior is estimated from a random half; the other half is
    MAP-fitted. Critical for fair synthetic benchmarks.

requirements: numpy, scipy, tqdm, joblib, h5py
"""

import numpy as np
from scipy.optimize import curve_fit, nnls, minimize
from scipy import stats
from joblib import Parallel, delayed
import tqdm
import warnings


# ===================================================================
#  B-VALUE AVERAGING (mimics Voorter Trace averaging)
# ===================================================================
def _average_bval_repeats(bvalues, dw_data):
    unique_b = np.unique(bvalues)
    if len(unique_b) == len(bvalues):
        return bvalues, dw_data
    single = dw_data.ndim == 1
    if single:
        dw_data = dw_data[np.newaxis, :]
    dw_avg = np.zeros((dw_data.shape[0], len(unique_b)), dtype=dw_data.dtype)
    for j, b in enumerate(unique_b):
        mask = bvalues == b
        dw_avg[:, j] = np.mean(dw_data[:, mask], axis=1)
    if single:
        return unique_b.astype(bvalues.dtype), dw_avg[0]
    return unique_b.astype(bvalues.dtype), dw_avg


# ===================================================================
#  DEFAULT BOUNDS (dynamically loaded from pace.hyperparams)
# ===================================================================
try:
    from pace.hyperparams import net_pars
    _hp = net_pars()
    _c_min = _hp.cons_min
    _c_max = _hp.cons_max
    DEFAULT_BOUNDS = (
        [_c_min[5], _c_min[0], _c_min[1], _c_min[2], _c_min[3], _c_min[4]],
        [_c_max[5], _c_max[0], _c_max[1], _c_max[2], _c_max[3], _c_max[4]],
    )
except ImportError:
    warnings.warn("hyperparams_deep2 not found -- using hardcoded fallback bounds")
    DEFAULT_BOUNDS = (
        [0.7,    0.0001, 0.0,   0.0015, 0.0,  0.004],
        [1.3,    0.0015, 0.4,   0.004,  0.2,  0.2  ],
    )

if DEFAULT_BOUNDS[0][2] + DEFAULT_BOUNDS[0][4] > 1.0:
    warnings.warn("[BOUNDS] Lower bounds for Fint and Fmv already violate "
                  f"Fint+Fmv<=1: {DEFAULT_BOUNDS[0][2]}+{DEFAULT_BOUNDS[0][4]}"
                  f"={DEFAULT_BOUNDS[0][2]+DEFAULT_BOUNDS[0][4]:.4f}")
if DEFAULT_BOUNDS[1][2] + DEFAULT_BOUNDS[1][4] > 1.0:
    warnings.warn("[BOUNDS] Upper bounds allow Fint+Fmv>1: "
                  f"{DEFAULT_BOUNDS[1][2]}+{DEFAULT_BOUNDS[1][4]}"
                  f"={DEFAULT_BOUNDS[1][2]+DEFAULT_BOUNDS[1][4]:.4f} "
                  "— LSQ starts are simplex-clamped but curve_fit is not constrained")


# ===================================================================
#  BOUND MIDPOINT HELPERS
# ===================================================================
def _midpoint_Dpar(bounds=DEFAULT_BOUNDS):
    return (bounds[0][1] + bounds[1][1]) / 2.0

def _midpoint_Dint(bounds=DEFAULT_BOUNDS):
    return (bounds[0][3] + bounds[1][3]) / 2.0

def _midpoint_Dmv(bounds=DEFAULT_BOUNDS):
    return (bounds[0][5] + bounds[1][5]) / 2.0


# ===================================================================
#  NaN-SAFE NORMALIZATION
# ===================================================================
def _safe_normalize(dw_data, bvalues, eps=1e-8):
    """
    Normalize DWI signals by b=0 mean, handling all edge cases.

    Returns:
        dw_norm:    (N, nb) — normalized signal, NaN-free
        S0_init:    (N,) — cleaned b=0 mean (finite for valid voxels)
        valid_mask: (N,) bool — True for voxels safe to fit
    """
    dw_data = np.asarray(dw_data, dtype=np.float64)

    min_b = np.min(bvalues)

    finite_rows = np.all(np.isfinite(dw_data), axis=1)

    dw_clean = np.nan_to_num(dw_data, nan=0.0, posinf=0.0, neginf=0.0)
    dw_clean = np.maximum(dw_clean, 0.0)

    S0_init = np.mean(dw_clean[:, bvalues == min_b], axis=1)

    valid_s0 = np.isfinite(S0_init) & (S0_init > eps)
    valid_mask = finite_rows & valid_s0

    S0_safe = np.where(valid_mask, S0_init, 1.0)
    dw_norm = dw_clean / S0_safe[:, None]

    n_invalid = np.sum(~valid_mask)
    if n_invalid > 0:
        print(f"    [NORM] {n_invalid}/{len(valid_mask)} voxels invalid "
              f"(non-finite or zero b=0 after cleaning) — sentinel values applied")

    return dw_norm, S0_init, valid_mask


def _mask_invalid_outputs(Dpar, Fint, Dint, Fmv, Dmv, S0,
                          valid_mask, bounds=DEFAULT_BOUNDS):
    """Set invalid voxels to finite sentinel values."""
    inv = ~valid_mask
    Dpar[inv] = 0.0
    Fint[inv] = 0.0
    Dint[inv] = _midpoint_Dint(bounds)
    Fmv[inv]  = 0.0
    Dmv[inv]  = _midpoint_Dmv(bounds)
    S0[inv]   = 0.0
    return Dpar, Fint, Dint, Fmv, Dmv, S0


# ===================================================================
#  TRI-EXPONENTIAL MODEL FUNCTIONS
# ===================================================================
def tri_exp_noS0(bvalues, Dpar, Fint, Dint, Fmv, Dmv):
    Fpar = np.maximum(0.0, 1.0 - Fmv - Fint)
    return (Fmv * np.exp(-bvalues * Dmv)
            + Fint * np.exp(-bvalues * Dint)
            + Fpar * np.exp(-bvalues * Dpar))

def tri_exp(bvalues, S0, Dpar, Fint, Dint, Fmv, Dmv):
    Fpar = np.maximum(0.0, 1.0 - Fmv - Fint)
    return S0 * (Fmv * np.exp(-bvalues * Dmv)
                 + Fint * np.exp(-bvalues * Dint)
                 + Fpar * np.exp(-bvalues * Dpar))


# ===================================================================
#  DIAGNOSTICS HELPER
# ===================================================================
def _compute_fitting_mse(bvalues, dw_norm, Dpar, Fint, Dint, Fmv, Dmv, S0):
    N = len(dw_norm)
    mse_per_voxel = np.full(N, np.nan)
    for i in range(N):
        dp = Dpar[i]; fi = Fint[i]; di = Dint[i]; fv = Fmv[i]; dv = Dmv[i]; s0 = S0[i]
        if np.isnan(dp) or np.isnan(s0):
            continue
        if dp == 0 and fi == 0 and fv == 0 and s0 == 0:
            continue
        if np.isnan(di): di = 0.0
        if np.isnan(dv): dv = 0.0
        fp = max(0.0, 1.0 - fi - fv)
        pred = s0 * (fv * np.exp(-bvalues * dv)
                     + fi * np.exp(-bvalues * di)
                     + fp * np.exp(-bvalues * dp))
        mse_per_voxel[i] = np.mean((dw_norm[i] - pred) ** 2)
    return mse_per_voxel


def _print_diagnostics(method_name, bvalues, dw_norm, result, extra=None):
    Dpar = result["Dpar"]; Fint = result["Fint"]; Dint = result["Dint"]
    Fmv  = result["Fmv"];  Dmv  = result["Dmv"];  S0   = result["S0"]
    N = len(Dpar)

    mse = _compute_fitting_mse(bvalues, dw_norm, Dpar, Fint, Dint, Fmv, Dmv, S0)

    n_valid_mse = np.sum(np.isfinite(mse))
    print(f"\n  [{method_name}] Diagnostics ({N} voxels, {n_valid_mse} with valid MSE):")
    if n_valid_mse > 0:
        print(f"    Fitting MSE: mean={np.nanmean(mse):.2e}, median={np.nanmedian(mse):.2e}, "
              f"p95={np.nanpercentile(mse, 95):.2e}, max={np.nanmax(mse):.2e}, "
              f">1e-2: {np.nansum(mse > 1e-2):.0f}/{n_valid_mse}")

    for pname, arr in [("Dpar", Dpar), ("Fint", Fint), ("Dint", Dint),
                        ("Fmv", Fmv), ("Dmv", Dmv), ("S0", S0)]:
        finite = arr[np.isfinite(arr)]
        n_nan = np.sum(~np.isfinite(arr))
        if len(finite) > 0:
            print(f"    {pname:>4s}: mean={np.mean(finite):.6f}  "
                  f"std={np.std(finite):.6f}  "
                  f"[{np.min(finite):.6f}, {np.max(finite):.6f}]"
                  f"{'  ('+str(n_nan)+' NaN)' if n_nan > 0 else ''}")
        else:
            print(f"    {pname:>4s}: all NaN ({n_nan})")

    zero_mask = (Dpar == 0) & (Fint == 0) & (Fmv == 0)
    n_fail = np.sum(zero_mask)
    if n_fail > 0:
        print(f"    Failures (all-zero): {n_fail}/{N} ({100*n_fail/N:.1f}%)")

    n_simplex = np.sum(np.isfinite(Fint) & np.isfinite(Fmv) & ((Fint + Fmv) > 1.0))
    if n_simplex > 0:
        print(f"    Simplex violations (Fint+Fmv>1): {n_simplex}/{N} ({100*n_simplex/N:.1f}%)")

    if extra:
        for k, v in extra.items():
            print(f"    {k}: {v}")


# ===================================================================
#  STEP-1 HELPER (shared by LSQ and MAP)
# ===================================================================
def _step1_dpar(bvalues, dw_data, bounds, cutoff=200):
    """Segment high-b signal to estimate Dpar via A*exp(-b*Dpar)."""
    def monofit(b, A, Dpar):
        return A * np.exp(-b * Dpar)

    mask = bvalues >= cutoff
    high_b = bvalues[mask]
    high_dw = dw_data[mask]

    if high_b.size < 2:
        return float((bounds[0][1] + bounds[1][1]) / 2.0)

    p0 = [np.clip(high_dw[0], 0.1, 1.5),
          (bounds[0][1] + bounds[1][1]) / 2.0]
    fit_bounds = ([0.0, bounds[0][1]],
                  [1.5, bounds[1][1]])
    try:
        params, _ = curve_fit(monofit, high_b, high_dw,
                              p0=p0, bounds=fit_bounds)
        return float(np.clip(params[1], bounds[0][1], bounds[1][1]))
    except Exception:
        return float(p0[1])


# ===================================================================
#  METHOD 1: TWO-STEP LEAST SQUARES
# ===================================================================
def fit_least_squares_single(bvalues, dw_data, fitS0=True,
                              bounds=DEFAULT_BOUNDS, cutoff=200,
                              n_starts=3):
    try:
        Dpar1 = _step1_dpar(bvalues, dw_data, bounds, cutoff)
        ub_dpar = max(Dpar1, bounds[0][1] + 1e-9)

        rng = np.random.default_rng()
        best_result = None
        best_mse = np.inf

        for start_idx in range(n_starts):
            try:
                if start_idx == 0:
                    Fint_hi = min(bounds[1][2], 1.0 - bounds[0][4])
                    Fint_g = (bounds[0][2] + max(bounds[0][2], Fint_hi)) / 2
                    fmv_hi = min(bounds[1][4], 1.0 - Fint_g)
                    Fmv_g  = (bounds[0][4] + max(bounds[0][4], fmv_hi)) / 2
                    Dint_g = (bounds[0][3] + bounds[1][3]) / 2
                    Dmv_g  = (bounds[0][5] + bounds[1][5]) / 2
                else:
                    Fint_hi = min(bounds[1][2], 1.0 - bounds[0][4])
                    Fint_g = rng.uniform(bounds[0][2], max(bounds[0][2], Fint_hi))
                    fmv_hi = min(bounds[1][4], 1.0 - Fint_g)
                    Fmv_g  = rng.uniform(bounds[0][4], max(bounds[0][4], fmv_hi))
                    Dint_g = rng.uniform(bounds[0][3], bounds[1][3])
                    Dmv_g  = rng.uniform(bounds[0][5], bounds[1][5])

                if fitS0:
                    bounds2 = (
                        [bounds[0][0], bounds[0][1], bounds[0][2], bounds[0][3], bounds[0][4], bounds[0][5]],
                        [bounds[1][0], ub_dpar,      bounds[1][2], bounds[1][3], bounds[1][4], bounds[1][5]],
                    )
                    p0 = [1.0, Dpar1, Fint_g, Dint_g, Fmv_g, Dmv_g]
                    params, _ = curve_fit(tri_exp, bvalues, dw_data, p0=p0, bounds=bounds2)
                    pred = tri_exp(bvalues, *params)
                    mse = np.mean((dw_data - pred) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        S0 = params[0]
                        best_result = (params[1], params[2], params[3], params[4], params[5], S0)
                else:
                    bounds2 = (
                        [bounds[0][1], bounds[0][2], bounds[0][3], bounds[0][4], bounds[0][5]],
                        [ub_dpar,      bounds[1][2], bounds[1][3], bounds[1][4], bounds[1][5]],
                    )
                    p0 = [Dpar1, Fint_g, Dint_g, Fmv_g, Dmv_g]
                    params, _ = curve_fit(tri_exp_noS0, bvalues, dw_data, p0=p0, bounds=bounds2)
                    pred = tri_exp_noS0(bvalues, *params)
                    mse = np.mean((dw_data - pred) ** 2)
                    if mse < best_mse:
                        best_mse = mse
                        best_result = (params[0], params[1], params[2], params[3], params[4], 1.0)
            except Exception:
                continue

        if best_result is None:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        Dpar, Fint, Dint, Fmv, Dmv, S0 = best_result
        if Fint < 1e-4: Dint = _midpoint_Dint(bounds)
        if Fmv < 1e-4:  Dmv  = _midpoint_Dmv(bounds)
        return Dpar, Fint, Dint, Fmv, Dmv, S0
    except Exception:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def fit_least_squares_array(bvalues, dw_data, fitS0=True,
                             bounds=DEFAULT_BOUNDS, cutoff=200, njobs=4,
                             verbose=True, average_repeats=False):
    N = len(dw_data)
    if N == 0:
        return {"Dpar": np.array([]), "Fint": np.array([]), "Dint": np.array([]),
                "Fmv": np.array([]), "Dmv": np.array([]), "S0": np.array([])}

    dw_norm, S0_init, valid_mask = _safe_normalize(dw_data, bvalues)

    if average_repeats:
        bvals_avg, dw_avg = _average_bval_repeats(bvalues, dw_norm)
        if len(bvals_avg) < len(bvalues) and verbose:
            print(f"    [LSQ] Averaged {len(bvalues)} -> {len(bvals_avg)} unique b-values")
    else:
        bvals_avg, dw_avg = bvalues, dw_norm

    use_fallback = True
    if njobs > 1:
        try:
            output = Parallel(n_jobs=njobs)(
                delayed(fit_least_squares_single)(bvals_avg, dw_avg[i], fitS0, bounds, cutoff)
                for i in tqdm.tqdm(range(N), desc="LSQ (Parallel)", mininterval=20))
            Dpar, Fint, Dint, Fmv, Dmv, S0 = np.array(output).T
            use_fallback = False
        except Exception as e:
            print(f"Parallel LSQ failed ({e}). Falling back to single-threaded.")
            use_fallback = True

    if use_fallback:
        Dpar = np.zeros(N); Fint = np.zeros(N); Dint = np.zeros(N)
        Fmv  = np.zeros(N); Dmv  = np.zeros(N); S0   = np.zeros(N)
        for i in tqdm.tqdm(range(N), desc="LSQ (Single)", mininterval=20):
            Dpar[i], Fint[i], Dint[i], Fmv[i], Dmv[i], S0[i] = \
                fit_least_squares_single(bvals_avg, dw_avg[i], fitS0, bounds, cutoff)

    Dpar, Fint, Dint, Fmv, Dmv, S0 = _mask_invalid_outputs(
        Dpar, Fint, Dint, Fmv, Dmv, S0, valid_mask, bounds)

    if verbose:
        result_diag = {"Dpar": Dpar, "Fint": Fint, "Dint": Dint,
                       "Fmv": Fmv, "Dmv": Dmv, "S0": S0.copy()}
        _print_diagnostics("LSQ", bvals_avg, dw_avg, result_diag)

    S0_init_safe = np.where(valid_mask, S0_init, 1.0)
    S0 = S0 * S0_init_safe
    S0[~valid_mask] = 0.0

    return {"Dpar": Dpar, "Fint": Fint, "Dint": Dint, "Fmv": Fmv, "Dmv": Dmv, "S0": S0}


# ===================================================================
#  METHOD 2: NNLS (Non-Negative Least Squares Spectral)
# ===================================================================
def fit_NNLS_array(bvalues, dw_data, bounds=DEFAULT_BOUNDS, num_D=200,
                    verbose=True, average_repeats=False):
    N = len(dw_data)
    if N == 0:
        return {"Dpar": np.array([]), "Fint": np.array([]), "Dint": np.array([]),
                "Fmv": np.array([]), "Dmv": np.array([]), "S0": np.array([])}

    dw_norm, S0_init, valid_mask = _safe_normalize(dw_data, bvalues)

    if average_repeats:
        bvals_avg, dw_avg = _average_bval_repeats(bvalues, dw_norm)
        if len(bvals_avg) < len(bvalues) and verbose:
            print(f"    [NNLS] Averaged {len(bvalues)} -> {len(bvals_avg)} unique b-values")
    else:
        bvals_avg, dw_avg = bvalues, dw_norm

    Dspace = np.logspace(np.log10(bounds[0][1]), np.log10(bounds[1][5]), num=num_D)
    Dbasis = np.exp(-np.outer(bvals_avg, Dspace))

    idx_par_int = np.searchsorted(Dspace, bounds[1][1], side="right")
    idx_int_mv  = np.searchsorted(Dspace, bounds[1][3], side="right")

    mid_par = _midpoint_Dpar(bounds)
    mid_int = _midpoint_Dint(bounds)
    mid_mv  = _midpoint_Dmv(bounds)

    Dpar = np.zeros(N); Fint = np.zeros(N); Dint = np.zeros(N)
    Fmv  = np.zeros(N); Dmv  = np.zeros(N); S0   = np.zeros(N)

    n_skip = 0
    n_fail = 0

    for i in tqdm.tqdm(range(N), desc="NNLS", mininterval=20):
        if not valid_mask[i]:
            Dpar[i] = mid_par; Dint[i] = mid_int; Dmv[i] = mid_mv
            Fint[i] = 0.0; Fmv[i] = 0.0; S0[i] = 0.0
            n_skip += 1
            continue

        try:
            x, _ = nnls(Dbasis, dw_avg[i])
        except Exception:
            Dpar[i] = mid_par; Dint[i] = mid_int; Dmv[i] = mid_mv
            Fint[i] = 0.0; Fmv[i] = 0.0; S0[i] = 0.0
            n_fail += 1
            continue

        ampl_par = np.sum(x[:idx_par_int])
        ampl_int = np.sum(x[idx_par_int:idx_int_mv])
        ampl_mv  = np.sum(x[idx_int_mv:])

        nz_par = np.nonzero(x[:idx_par_int])[0]
        nz_int = np.nonzero(x[idx_par_int:idx_int_mv])[0]
        nz_mv  = np.nonzero(x[idx_int_mv:])[0]

        avg_par = (np.sum(Dspace[nz_par] * x[nz_par]) / ampl_par
                   if len(nz_par) > 0 and ampl_par > 0 else mid_par)
        avg_int = (np.sum(Dspace[idx_par_int + nz_int] * x[idx_par_int + nz_int]) / ampl_int
                   if len(nz_int) > 0 and ampl_int > 0 else mid_int)
        avg_mv  = (np.sum(Dspace[idx_int_mv + nz_mv] * x[idx_int_mv + nz_mv]) / ampl_mv
                   if len(nz_mv) > 0 and ampl_mv > 0 else mid_mv)

        total = ampl_par + ampl_int + ampl_mv
        if total > 0:
            Fint[i] = ampl_int / total
            Fmv[i]  = ampl_mv / total
        else:
            Fint[i] = 0.0; Fmv[i] = 0.0

        Dpar[i] = avg_par; Dint[i] = avg_int; Dmv[i] = avg_mv
        S0[i] = total

    if verbose:
        if n_skip > 0:
            print(f"    [NNLS] Skipped {n_skip} invalid voxels")
        if n_fail > 0:
            print(f"    [NNLS] Failed {n_fail} voxels (per-voxel nnls exception)")
        result_diag = {"Dpar": Dpar, "Fint": Fint, "Dint": Dint,
                       "Fmv": Fmv, "Dmv": Dmv, "S0": S0.copy()}
        _print_diagnostics("NNLS", bvals_avg, dw_avg, result_diag)

    S0_init_safe = np.where(valid_mask, S0_init, 1.0)
    S0 = S0 * S0_init_safe
    S0[~valid_mask] = 0.0

    return {"Dpar": Dpar, "Fint": Fint, "Dint": Dint, "Fmv": Fmv, "Dmv": Dmv, "S0": S0}


# ===================================================================
#  METHOD 3: BAYESIAN MAP (Data-Driven Prior, 3-Compartment)
# ===================================================================

def _build_empirical_prior_3c(Dpar0, Fint0, Dint0, Fmv0, Dmv0, S00=None):
    """
    Build data-driven prior from initial LSQ estimates.
    Following Gurney-Champion/Barbieri:
      - Diffusivities (Dpar, Dint, Dmv): lognormal
      - Fractions (Fint, Fmv): beta on (0, 1)
      - S0: beta on (0, 2)
    """
    eps = 1e-8

    valid = np.ones(len(Dpar0), dtype=bool)
    for arr in [Dpar0, Dint0, Dmv0]:
        valid &= np.isfinite(arr) & (arr > eps) & (arr < 1.0)
    for arr in [Fint0, Fmv0]:
        valid &= np.isfinite(arr) & (arr >= 0) & (arr < 1.0 - eps)
    valid &= np.isfinite(Fint0 + Fmv0) & ((Fint0 + Fmv0) <= 1.0 - eps)
    if S00 is not None:
        valid &= np.isfinite(S00) & (S00 > eps) & (S00 < 2.0 - eps)

    n_valid = np.sum(valid)
    if n_valid < 10:
        warnings.warn(f"[MAP PRIOR] Only {n_valid} valid voxels for prior — using flat prior")
        def flat_prior(p):
            return 0.0
        return flat_prior

    Dpar_v = Dpar0[valid]
    Fint_v = Fint0[valid]
    Dint_v = Dint0[valid]
    Fmv_v  = Fmv0[valid]
    Dmv_v  = Dmv0[valid]

    try:
        Dpar_shape, _, Dpar_scale = stats.lognorm.fit(Dpar_v, floc=0)
    except Exception:
        Dpar_shape, Dpar_scale = 0.5, np.median(Dpar_v)
    try:
        Dint_shape, _, Dint_scale = stats.lognorm.fit(Dint_v, floc=0)
    except Exception:
        Dint_shape, Dint_scale = 0.5, np.median(Dint_v)
    try:
        Dmv_shape, _, Dmv_scale = stats.lognorm.fit(Dmv_v, floc=0)
    except Exception:
        Dmv_shape, Dmv_scale = 0.5, np.median(Dmv_v)

    Dpar_shape = max(Dpar_shape, 1e-5)
    Dint_shape = max(Dint_shape, 1e-5)
    Dmv_shape  = max(Dmv_shape, 1e-5)

    Fint_c = np.clip(Fint_v, eps, 1.0 - eps)
    Fmv_c  = np.clip(Fmv_v,  eps, 1.0 - eps)
    try:
        Fint_a, Fint_b, _, _ = stats.beta.fit(Fint_c, floc=0, fscale=1)
    except Exception:
        Fint_a, Fint_b = 2.0, 5.0
    try:
        Fmv_a, Fmv_b, _, _ = stats.beta.fit(Fmv_c, floc=0, fscale=1)
    except Exception:
        Fmv_a, Fmv_b = 2.0, 10.0

    if S00 is not None:
        S0_v = S00[valid]
        try:
            S0_a, S0_b, _, _ = stats.beta.fit(S0_v / 2.0, floc=0, fscale=1)
        except Exception:
            S0_a, S0_b = 5.0, 5.0
        fit_s0 = True
    else:
        fit_s0 = False

    from scipy.special import betaln as _betaln
    _ln2pi = np.log(2.0 * np.pi)
    _Dp_logs = np.log(Dpar_scale)
    _Di_logs = np.log(Dint_scale)
    _Dv_logs = np.log(Dmv_scale)
    _Fi_betaln = _betaln(Fint_a, Fint_b)
    _Fv_betaln = _betaln(Fmv_a, Fmv_b)
    if fit_s0:
        _S0_betaln = _betaln(S0_a, S0_b)

    def neg_log_prior(p):
        if fit_s0:
            S0_, Dp_, Fi_, Di_, Fv_, Dv_ = p
        else:
            Dp_, Fi_, Di_, Fv_, Dv_ = p

        Dp_ = max(Dp_, eps); Di_ = max(Di_, eps); Dv_ = max(Dv_, eps)
        nlp  = (np.log(Dp_) + np.log(Dpar_shape) + 0.5 * _ln2pi
                + 0.5 * ((np.log(Dp_) - _Dp_logs) / Dpar_shape) ** 2)
        nlp += (np.log(Di_) + np.log(Dint_shape) + 0.5 * _ln2pi
                + 0.5 * ((np.log(Di_) - _Di_logs) / Dint_shape) ** 2)
        nlp += (np.log(Dv_) + np.log(Dmv_shape) + 0.5 * _ln2pi
                + 0.5 * ((np.log(Dv_) - _Dv_logs) / Dmv_shape) ** 2)

        Fi_ = min(max(Fi_, eps), 1.0 - eps)
        Fv_ = min(max(Fv_, eps), 1.0 - eps)
        nlp += -(Fint_a - 1) * np.log(Fi_) - (Fint_b - 1) * np.log(1.0 - Fi_) + _Fi_betaln
        nlp += -(Fmv_a - 1) * np.log(Fv_) - (Fmv_b - 1) * np.log(1.0 - Fv_) + _Fv_betaln

        if fit_s0:
            S0_c = min(max(S0_ / 2.0, eps), 1.0 - eps)
            nlp += -(S0_a - 1) * np.log(S0_c) - (S0_b - 1) * np.log(1.0 - S0_c) + _S0_betaln

        return nlp

    return neg_log_prior


def _neg_log_likelihood_3c(p, bvalues, dw_data, fitS0=True):
    if fitS0:
        S0_, Dp_, Fi_, Di_, Fv_, Dv_ = p
    else:
        Dp_, Fi_, Di_, Fv_, Dv_ = p
        S0_ = 1.0

    Fp_ = np.maximum(0.0, 1.0 - Fv_ - Fi_)
    pred = S0_ * (Fv_ * np.exp(-bvalues * Dv_)
                  + Fi_ * np.exp(-bvalues * Di_)
                  + Fp_ * np.exp(-bvalues * Dp_))
    ssr = np.sum((dw_data - pred) ** 2)
    return 0.5 * (len(bvalues) + 1) * np.log(max(ssr, 1e-30))


def _neg_log_posterior_3c(p, bvalues, dw_data, neg_log_prior, fitS0=True):
    if fitS0:
        _, _, Fi_, _, Fv_, _ = p
    else:
        _, Fi_, _, Fv_, _ = p
    simplex_pen = 1e6 * max(0.0, Fi_ + Fv_ - 1.0) ** 2
    return (_neg_log_likelihood_3c(p, bvalues, dw_data, fitS0)
            + neg_log_prior(p) + simplex_pen)


def fit_MAP_single_bayesian(bvalues, dw_data, neg_log_prior, x0,
                             bounds, fitS0=True, return_diagnostics=False):
    try:
        result = minimize(
            _neg_log_posterior_3c,
            x0=x0,
            args=(bvalues, dw_data, neg_log_prior, fitS0),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 15000},
        )

        if fitS0:
            S0, Dpar, Fint, Dint, Fmv, Dmv = result.x
        else:
            Dpar, Fint, Dint, Fmv, Dmv = result.x
            S0 = 1.0

        if isinstance(bounds[0], tuple):
            if fitS0:
                mid_Dint = (bounds[3][0] + bounds[3][1]) / 2.0
                mid_Dmv  = (bounds[5][0] + bounds[5][1]) / 2.0
            else:
                mid_Dint = (bounds[2][0] + bounds[2][1]) / 2.0
                mid_Dmv  = (bounds[4][0] + bounds[4][1]) / 2.0
        else:
            mid_Dint = _midpoint_Dint()
            mid_Dmv  = _midpoint_Dmv()

        if Fint < 1e-4: Dint = mid_Dint
        if Fmv < 1e-4:  Dmv  = mid_Dmv

        if return_diagnostics:
            return Dpar, Fint, Dint, Fmv, Dmv, S0, result.success, result.nit, result.fun
        return Dpar, Fint, Dint, Fmv, Dmv, S0

    except Exception:
        if return_diagnostics:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False, 0, float("inf")
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


def fit_MAP_array(bvalues, dw_data, fitS0=True, bounds=DEFAULT_BOUNDS,
                   cutoff=200, njobs=4, verbose=True, average_repeats=False,
                   prior_strength=None, lsq_precomputed=None,
                   split_prior=False, split_seed=42):
    """
    Bayesian MAP fitting with data-driven empirical prior.

    Args:
        split_prior: If True, estimate the prior from a random half of voxels
            and MAP-fit only the other half. The prior-half keeps its LSQ
            estimates. This prevents double-dipping where the prior learns
            the answer from the same voxels it regularizes — critical for
            fair synthetic benchmarks. Default False for backward compat.
        split_seed: Random seed for the prior/fit split (reproducibility).
    """
    N = len(dw_data)
    if N == 0:
        return {"Dpar": np.array([]), "Fint": np.array([]), "Dint": np.array([]),
                "Fmv": np.array([]), "Dmv": np.array([]), "S0": np.array([])}

    dw_norm, S0_init, valid_mask = _safe_normalize(dw_data, bvalues)

    if average_repeats:
        bvals_avg, dw_avg = _average_bval_repeats(bvalues, dw_norm)
        if len(bvals_avg) < len(bvalues) and verbose:
            print(f"    [MAP] Averaged {len(bvalues)} -> {len(bvals_avg)} unique b-values")
    else:
        bvals_avg, dw_avg = bvalues, dw_norm

    # Step 1: LSQ estimates (always on all voxels — needed for prior and as fallback)
    if lsq_precomputed is not None:
        if verbose:
            print(f"    [MAP] Step 1: Using precomputed LSQ results")
        lsq_result = lsq_precomputed
    else:
        if verbose:
            print(f"    [MAP] Step 1: Running LSQ on {N} voxels for prior estimation...")
        lsq_result = fit_least_squares_array(
            bvalues, dw_data, fitS0=fitS0, bounds=bounds,
            cutoff=cutoff, njobs=njobs, verbose=False, average_repeats=average_repeats,
        )

    Dpar0 = lsq_result["Dpar"]
    Fint0 = lsq_result["Fint"]
    Dint0 = lsq_result["Dint"]
    Fmv0  = lsq_result["Fmv"]
    Dmv0  = lsq_result["Dmv"]

    S0_init_safe = np.where(valid_mask, S0_init, 1.0)
    S00 = lsq_result["S0"] / S0_init_safe

    Dint0_clean = np.where(np.isfinite(Dint0), Dint0, _midpoint_Dint(bounds))
    Dmv0_clean  = np.where(np.isfinite(Dmv0),  Dmv0,  _midpoint_Dmv(bounds))
    Dpar0_clean = np.where(np.isfinite(Dpar0) & (Dpar0 > 0), Dpar0, _midpoint_Dpar(bounds))

    # Step 2: Build empirical prior
    # split_prior=True: estimate prior from random half, fit the other half.
    # Prevents double-dipping where the prior learns the answer from the
    # same voxels it regularizes — critical for fair synthetic benchmarks.
    if split_prior:
        rng_split = np.random.default_rng(split_seed)
        prior_idx = rng_split.choice(N, size=N // 2, replace=False)
        prior_mask = np.zeros(N, dtype=bool)
        prior_mask[prior_idx] = True
        if verbose:
            print(f"    [MAP] Step 2: Split prior — estimating from {prior_mask.sum()} voxels, "
                  f"MAP-fitting {(~prior_mask & valid_mask).sum()} voxels "
                  f"(prior half keeps LSQ estimates)")
        neg_log_prior = _build_empirical_prior_3c(
            Dpar0_clean[prior_mask], Fint0[prior_mask],
            Dint0_clean[prior_mask], Fmv0[prior_mask],
            Dmv0_clean[prior_mask],
            S00[prior_mask] if fitS0 else None
        )
        # Only MAP-fit the non-prior half; prior half keeps LSQ estimates
        fit_mask = ~prior_mask & valid_mask
    else:
        if verbose:
            print(f"    [MAP] Step 2: Fitting empirical prior distributions...")
        neg_log_prior = _build_empirical_prior_3c(
            Dpar0_clean, Fint0, Dint0_clean, Fmv0, Dmv0_clean,
            S00 if fitS0 else None
        )
        fit_mask = valid_mask

    # Step 3: MAP per voxel
    if verbose:
        n_to_fit = int(np.sum(fit_mask))
        print(f"    [MAP] Step 3: Bayesian MAP optimization on {n_to_fit} voxels...")

    if fitS0:
        fit_bounds = [
            (bounds[0][0], bounds[1][0]),
            (bounds[0][1], bounds[1][1]),
            (bounds[0][2], bounds[1][2]),
            (bounds[0][3], bounds[1][3]),
            (bounds[0][4], bounds[1][4]),
            (bounds[0][5], bounds[1][5]),
        ]
    else:
        fit_bounds = [
            (bounds[0][1], bounds[1][1]),
            (bounds[0][2], bounds[1][2]),
            (bounds[0][3], bounds[1][3]),
            (bounds[0][4], bounds[1][4]),
            (bounds[0][5], bounds[1][5]),
        ]

    # Initialize outputs from LSQ (prior-half voxels keep these;
    # fit-half voxels get overwritten by MAP results)
    Dpar = Dpar0_clean.copy()
    Fint = Fint0.copy()
    Dint = Dint0_clean.copy()
    Fmv  = Fmv0.copy()
    Dmv  = Dmv0_clean.copy()
    S0   = S00.copy()

    converged = np.zeros(N, dtype=bool)
    niter = np.zeros(N, dtype=int)
    fun_vals = np.full(N, np.inf)

    use_fallback = True
    if njobs > 1:
        try:
            def parfun(i):
                if not fit_mask[i]:
                    # Skip: either invalid or in the prior half (keeps LSQ)
                    return None
                if fitS0:
                    x0 = [S00[i], Dpar0_clean[i], Fint0[i],
                           Dint0_clean[i], Fmv0[i], Dmv0_clean[i]]
                else:
                    x0 = [Dpar0_clean[i], Fint0[i],
                           Dint0_clean[i], Fmv0[i], Dmv0_clean[i]]
                x0 = [np.clip(v, b[0] + 1e-10, b[1] - 1e-10)
                       for v, b in zip(x0, fit_bounds)]
                return fit_MAP_single_bayesian(
                    bvals_avg, dw_avg[i], neg_log_prior, x0,
                    fit_bounds, fitS0, return_diagnostics=True)

            output = Parallel(n_jobs=njobs)(
                delayed(parfun)(i)
                for i in tqdm.tqdm(range(N), desc="MAP Bayesian (Parallel)", mininterval=20))

            for i, res in enumerate(output):
                if res is None:
                    continue  # prior-half or invalid — keeps LSQ init
                dp, fi, di, fv, dv, s0, conv, nit, fval = res
                Dpar[i] = dp; Fint[i] = fi; Dint[i] = di
                Fmv[i] = fv; Dmv[i] = dv; S0[i] = s0
                converged[i] = conv; niter[i] = nit; fun_vals[i] = fval
            use_fallback = False
        except Exception as e:
            print(f"Parallel MAP failed ({e}). Falling back to single-threaded.")
            use_fallback = True

    if use_fallback:
        for i in tqdm.tqdm(range(N), desc="MAP Bayesian (Single)", mininterval=20):
            if not fit_mask[i]:
                continue  # prior-half or invalid — keeps LSQ init

            if fitS0:
                x0 = [S00[i], Dpar0_clean[i], Fint0[i],
                       Dint0_clean[i], Fmv0[i], Dmv0_clean[i]]
            else:
                x0 = [Dpar0_clean[i], Fint0[i],
                       Dint0_clean[i], Fmv0[i], Dmv0_clean[i]]
            x0 = [np.clip(v, b[0] + 1e-10, b[1] - 1e-10)
                   for v, b in zip(x0, fit_bounds)]
            Dpar[i], Fint[i], Dint[i], Fmv[i], Dmv[i], S0[i], \
                converged[i], niter[i], fun_vals[i] = \
                fit_MAP_single_bayesian(
                    bvals_avg, dw_avg[i], neg_log_prior, x0,
                    fit_bounds, fitS0, return_diagnostics=True)

    # Mask invalid voxels to sentinel values
    Dpar, Fint, Dint, Fmv, Dmv, S0 = _mask_invalid_outputs(
        Dpar, Fint, Dint, Fmv, Dmv, S0, valid_mask, bounds)

    if verbose:
        result_diag = {"Dpar": Dpar, "Fint": Fint, "Dint": Dint,
                       "Fmv": Fmv, "Dmv": Dmv, "S0": S0.copy()}

        n_valid = int(np.sum(valid_mask))
        n_fitted = int(np.sum(fit_mask))
        n_conv = np.sum(converged & fit_mask)
        n_maxiter = np.sum((niter >= 14999) & fit_mask)

        valid_fun = np.isfinite(fun_vals) & fit_mask
        valid_iter = fit_mask

        if np.any(valid_fun):
            cost_mean = np.mean(fun_vals[valid_fun])
            cost_median = np.median(fun_vals[valid_fun])
        else:
            cost_mean = np.nan
            cost_median = np.nan

        if np.any(valid_iter):
            iter_mean = np.mean(niter[valid_iter])
            iter_median = np.median(niter[valid_iter])
            iter_max = np.max(niter[valid_iter])
        else:
            iter_mean = np.nan
            iter_median = np.nan
            iter_max = 0

        n_skipped = N - n_valid
        extra = {
            "Valid voxels": f"{n_valid}/{N} ({n_skipped} skipped)",
            "MAP-fitted": f"{n_fitted}/{n_valid}"
                          + (f" (prior half: {n_valid - n_fitted} kept as LSQ)" if split_prior else ""),
            "Converged": f"{n_conv}/{n_fitted} ({100*n_conv/max(n_fitted,1):.1f}%)",
            "Iterations": f"mean={iter_mean:.1f}, median={iter_median:.0f}, "
                          f"max={iter_max}, hit_maxiter={n_maxiter}",
            "Cost (final)": f"mean={cost_mean:.4f}, median={cost_median:.4f}",
        }
        if split_prior:
            extra["Split prior"] = f"seed={split_seed}, prior_half={int(np.sum(prior_mask))}, fit_half={n_fitted}"
        _print_diagnostics("MAP (Bayesian)", bvals_avg, dw_avg, result_diag, extra)

    # Scale S0 back to absolute
    S0 = S0 * S0_init_safe
    S0[~valid_mask] = 0.0

    return {"Dpar": Dpar, "Fint": Fint, "Dint": Dint, "Fmv": Fmv, "Dmv": Dmv, "S0": S0}


# ===================================================================
#  HDF5 WRAPPERS & UTILS
# ===================================================================
def _load_signals_from_h5(h5_path, bvals):
    import h5py
    with h5py.File(h5_path, "r") as f:
        for sig_key in ["signal", "Signal", "IVIM_cube", "IVIM", "dw_data", "DWI"]:
            if sig_key in f:
                signals = f[sig_key][:]
                break
        else:
            raise KeyError(f"No signal key found in {h5_path}. Keys: {list(f.keys())}")
        if signals.ndim == 3:
            signals = signals[:, 0, :]
        tissue = f["tissue"][:] if "tissue" in f else (
            f["Tissue"][:] if "Tissue" in f else None)
        lesion = f["lesion"][:] if "lesion" in f else (
            f["Lesion"][:] if "Lesion" in f else None)
    return signals, tissue, lesion


def fit_h5(h5_path, bvals, method="LSQ", fitS0=True, bounds=DEFAULT_BOUNDS,
           njobs=4, **kwargs):
    signals, tissue, lesion = _load_signals_from_h5(h5_path, bvals)
    print(f"[{method}] {h5_path}: {signals.shape[0]} voxels, {signals.shape[1]} b-values")

    if method.upper() == "LSQ":
        result = fit_least_squares_array(bvals, signals, fitS0=fitS0,
                                          bounds=bounds, njobs=njobs, **kwargs)
    elif method.upper() == "NNLS":
        result = fit_NNLS_array(bvals, signals, bounds=bounds, **kwargs)
    elif method.upper() == "MAP":
        result = fit_MAP_array(bvals, signals, fitS0=fitS0,
                                bounds=bounds, njobs=njobs, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'LSQ', 'NNLS', or 'MAP'.")

    result["tissue"] = tissue
    result["lesion"] = lesion
    return result


def results_to_array(result_dict, param_order=None):
    if param_order is None:
        param_order = ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"]
    return np.stack([result_dict[p] for p in param_order], axis=-1)


# ===================================================================
#  QUICK TEST
# ===================================================================
if __name__ == "__main__":
    print("Testing fitting_algorithms with synthetic signal...")
    print(f"Bounds: {DEFAULT_BOUNDS}")

    bvals = np.array([0, 0, 0, 0, 5, 5, 5, 7, 7, 7, 10, 10, 10,
                      15, 15, 15, 20, 20, 20, 30, 30, 30, 40, 40, 40,
                      50, 50, 50, 60, 60, 60, 100, 100, 100,
                      200, 200, 200, 400, 400, 400,
                      700, 700, 700, 1000, 1000, 1000],
                     dtype=np.float32)

    true_signal = 1.0 * (0.05 * np.exp(-bvals * 0.05)
                         + 0.10 * np.exp(-bvals * 0.003)
                         + 0.85 * np.exp(-bvals * 0.001))

    rng = np.random.default_rng(42)
    N_test = 50
    dw_data = np.tile(true_signal, (N_test, 1))
    dw_data += rng.normal(0, 0.02, size=dw_data.shape)
    dw_data = np.clip(dw_data, 0.01, None)

    dw_data[0, :] = np.nan
    dw_data[1, :4] = np.nan
    dw_data[2, :] = -1.0
    dw_data[3, :] = 0.0

    print(f"  {len(bvals)} b-values, {len(np.unique(bvals))} unique, "
          f"{N_test} voxels (4 adversarial), SNR~50")

    for name, fn in [("LSQ", fit_least_squares_array),
                     ("NNLS", fit_NNLS_array)]:
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        r = fn(bvals, dw_data, verbose=True) if name == "NNLS" else fn(bvals, dw_data, njobs=1, verbose=True)
        n_nan = sum(np.sum(~np.isfinite(r[p])) for p in ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"])
        print(f"  Total NaN in output: {n_nan}")

    # Test MAP with and without split_prior
    for split in [False, True]:
        label = f"MAP (split_prior={split})"
        print(f"\n{'='*50}")
        print(f"  {label}")
        print(f"{'='*50}")
        r = fit_MAP_array(bvals, dw_data, njobs=1, verbose=True, split_prior=split)
        n_nan = sum(np.sum(~np.isfinite(r[p])) for p in ["Dpar", "Fint", "Dint", "Fmv", "Dmv", "S0"])
        print(f"  Total NaN in output: {n_nan}")

    print("\nDone.")