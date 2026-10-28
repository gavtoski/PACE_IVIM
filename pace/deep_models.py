#!/usr/bin/env python3

import math
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pace.hyperparams import net_pars


TISSUE_CLASS_GM = 0
TISSUE_CLASS_NAWM = 1
TISSUE_CLASS_WMH = 2
NUM_TISSUE_CLASSES = 3
TISSUE_NAMES = ["GM", "NAWM", "WMH"]


def tissue_lesion_to_class(tissue_tensor, lesion_tensor):
    """Convert tissue/lesion labels to 3-class labels."""
    tissue = tissue_tensor.long()
    lesion = lesion_tensor.long()
    labels = torch.full_like(tissue, -1)

    wmh_mask = lesion != 0
    labels[wmh_mask] = TISSUE_CLASS_WMH
    labels[(tissue == 2) & (~wmh_mask)] = TISSUE_CLASS_GM
    labels[(tissue == 3) & (~wmh_mask)] = TISSUE_CLASS_NAWM

    return labels, labels >= 0


def compute_static_ce_weights(h5_paths, num_classes=3, clamp_max=5.0):
    """Compute sqrt-inverse-frequency class weights from dataset(s)."""
    counts = np.zeros(num_classes, dtype=np.int64)
    for p in h5_paths:
        with h5py.File(p, "r") as f:
            tissue = f["Tissue"][...].astype(np.int64)
            lesion = f["Lesion"][...].astype(np.int64)

        labels = np.full_like(tissue, -1)
        wmh = lesion != 0
        labels[wmh] = 2
        labels[(tissue == 2) & (~wmh)] = 0
        labels[(tissue == 3) & (~wmh)] = 1

        valid = labels >= 0
        counts += np.bincount(labels[valid], minlength=num_classes)

    counts = np.maximum(counts, 1)
    w = 1.0 / np.sqrt(counts.astype(np.float64))
    w = w / w.sum() * num_classes
    w = np.minimum(w, clamp_max)
    return w.astype(np.float32), counts


def clean_ivim_signals(X_batch, P_batch, B0_batch, bvals, inject_noise=False, noise_type="rician", noise_level=25.0):
    """Clean/normalize IVIM signals with optional noise injection."""
    device, dtype = X_batch.device, X_batch.dtype
    B, Nb = X_batch.shape

    assert X_batch.ndim == 2, f"X_batch must be (B, Nb), got {X_batch.shape}"
    assert len(bvals) == Nb, f"b-values ({len(bvals)}) must match signal dim ({Nb})"

    patch_size = 3
    if P_batch is not None:
        patch_size = P_batch.shape[-1]
    elif B0_batch is not None:
        patch_size = B0_batch.shape[-1]
    if P_batch is None:
        P_batch = torch.zeros((B, 2, patch_size, patch_size), device=device, dtype=dtype)
    if B0_batch is None:
        B0_batch = torch.zeros((B, 1, patch_size, patch_size), device=device, dtype=dtype)

    bvals_tensor = torch.as_tensor(bvals, device=device, dtype=torch.float32)
    b0_mask = bvals_tensor.abs() < 1e-3
    if not torch.any(b0_mask): raise ValueError("No b0 volumes found.")

    X_float = X_batch.to(torch.float32)
    S0 = torch.clamp(torch.median(X_float[:, b0_mask], dim=1, keepdim=True).values, min=1e-6)
    Xn = torch.nan_to_num(X_float / S0, nan=0.0, posinf=0.0, neginf=0.0)

    _do_inject = (inject_noise
                  and noise_level is not None
                  and float(noise_level) > 0.0)

    if _do_inject and noise_type == "rician":
        snr_linear = 10.0 ** (float(noise_level) / 20.0)
        sigma = torch.full_like(Xn, 1.0 / max(snr_linear, 1e-8))
        n1, n2 = sigma * torch.randn_like(Xn), sigma * torch.randn_like(Xn)
        Xn = torch.nan_to_num(torch.sqrt((Xn + n1) ** 2 + n2 ** 2), nan=0.0)
    elif _do_inject and noise_type not in ("none", "rician"):
        raise ValueError("noise_type must be 'none' or 'rician'")

    valid = torch.isfinite(Xn).all(dim=1)
    def pct99(x): return torch.quantile(x, 0.99, dim=1)

    if torch.any(bvals_tensor < 50): valid &= pct99(Xn[:, bvals_tensor < 50]) < 1.3
    if torch.any((bvals_tensor > 50) & (bvals_tensor < 150)): valid &= pct99(Xn[:, (bvals_tensor > 50) & (bvals_tensor < 150)]) < 1.2
    if torch.any(bvals_tensor > 150): valid &= pct99(Xn[:, bvals_tensor > 150]) < 1.0

    if not torch.any(valid): raise RuntimeError("All samples rejected.")
    return Xn[valid].to(dtype), P_batch[valid], B0_batch[valid], valid


try:
    from numpy.lib.stride_tricks import sliding_window_view
    NP_SLIDING_EXISTS = True
except ImportError: NP_SLIDING_EXISTS = False

try:
    from skimage.util import view_as_windows
    SKIMAGE_EXISTS = True
except ImportError: SKIMAGE_EXISTS = False


def extract_3x3(vol2d, x, y):
    v = np.pad(vol2d, pad_width=1, mode="edge")
    x, y = int(x) + 1, int(y) + 1
    patch = v[x-1:x+2, y-1:y+2].astype(np.float32, copy=False)
    invalid = ~np.isfinite(patch)
    if invalid.any():
        valid_vals = patch[~invalid]
        patch[invalid] = float(np.median(valid_vals)) if valid_vals.size > 0 else 0.0
    return patch

def extract_all_patches_fast(vol2d):
    vol2d_f = vol2d.astype(np.float32, copy=False)
    v_padded = np.pad(vol2d_f, pad_width=1, mode="edge")
    if NP_SLIDING_EXISTS: return sliding_window_view(v_padded, window_shape=(3, 3))
    if SKIMAGE_EXISTS: return view_as_windows(v_padded, window_shape=(3, 3))
    return None

def robust_normalize(arr):
    p1, p99 = np.percentile(arr, [1, 99])
    return (np.clip(arr, p1, p99) - p1) / (p99 - p1 + 1e-6)


def anat_scale_from_h5(h5_path, sample=20000, seed=0):
    """Returns one entry for T1, FLAIR and b0 and scale them"""
    with h5py.File(h5_path, "r") as f:
        n = f["Struct_patch"].shape[0]
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(n, min(sample, n), replace=False))
        chans = [f["Struct_patch"][idx][:, 0], f["Struct_patch"][idx][:, 1]]
        chans.append(f["B0_patch"][idx][:, 0] if "B0_patch" in f else None)

    out = []
    for c in chans:
        if c is None:
            out.append(None)
            continue
        v = c[np.isfinite(c)]
        if v.size == 0 or float(v.max()) <= 1.0 + 1e-6:
            out.append(None)
            continue
        p1, p99 = np.percentile(v, [1, 99])
        out.append((float(p1), float(p99)) if p99 > p1 else None)
    return out


def apply_anat_scale(patch, scale):
    """Map a patch through percentiles, or return it unchanged."""
    if scale is None:
        return patch
    p1, p99 = scale
    return (np.clip(patch, p1, p99) - p1) / (p99 - p1 + 1e-6)

def prepare_inference_patches(ivim_4d, t1_3d, flair_3d, bvals, device="cuda", normalize_inputs=True):
    X, Y, Z, nb = ivim_4d.shape
    bvals_arr = np.asarray(bvals)
    b0_vol = ivim_4d[..., int(np.argmin(np.abs(bvals_arr)))]

    if normalize_inputs:
        t1_in, flair_in, b0_in = robust_normalize(t1_3d), robust_normalize(flair_3d), robust_normalize(b0_vol)
    else:
        t1_in, flair_in, b0_in = t1_3d, flair_3d, b0_vol

    valid_mask = np.isfinite(ivim_4d).all(axis=-1) & (ivim_4d.sum(axis=-1) > 0) & (ivim_4d.min(axis=-1) > 0.01)
    ijk = np.argwhere(valid_mask)
    num_voxels = len(ijk)
    print(f"[INFO] {num_voxels} valid voxels for inference")

    X_data = np.zeros((num_voxels, nb), np.float32)
    struct_patches = np.zeros((num_voxels, 2, 3, 3), np.float32)
    b0_patches = np.zeros((num_voxels, 1, 3, 3), np.float32)

    z_groups = {}
    for voxel_id, (x, y, z) in enumerate(ijk): z_groups.setdefault(int(z), []).append(voxel_id)

    for z, voxel_ids in z_groups.items():
        t1_p, fl_p, b0_p = extract_all_patches_fast(t1_in[..., z]), extract_all_patches_fast(flair_in[..., z]), extract_all_patches_fast(b0_in[..., z])
        fast_ok = all(p is not None for p in [t1_p, fl_p, b0_p])

        for i in voxel_ids:
            x, y, zz = ijk[i].astype(int)
            X_data[i] = ivim_4d[x, y, zz, :].astype(np.float32, copy=False)
            if fast_ok:
                struct_patches[i, 0] = t1_p[x, y]
                struct_patches[i, 1] = fl_p[x, y]
                b0_patches[i, 0] = b0_p[x, y]
            else:
                struct_patches[i, 0] = extract_3x3(t1_in[..., zz], x, y)
                struct_patches[i, 1] = extract_3x3(flair_in[..., zz], x, y)
                b0_patches[i, 0] = extract_3x3(b0_in[..., zz], x, y)

    return X_data, torch.from_numpy(struct_patches).to(device=device, dtype=torch.float32), torch.from_numpy(b0_patches).to(device=device, dtype=torch.float32), ijk


def calibrate_anat_k(h5_path, keep_center=False, quiet=False):
    """For testing effectiveness of passing the tissue ID directly."""
    with h5py.File(h5_path, "r") as f:
        if "Tissue_patch" not in f:
            return 6
        T = f["Tissue_patch"][:, 0]
    c = T.shape[-1] // 2
    same = (T == T[:, c, c][:, None, None])
    if not keep_center:
        same[:, c, c] = False
    mean = float(same.reshape(len(T), -1).sum(axis=1).mean())
    k = max(1, int(round(mean)))
    if not quiet:
        print(f"[DATA  ] anatomy label select: k={k}, calibrated against a mean "
              f"of {mean:.2f} same class neighbours")
    return k


def ivim_neighbourhood(patch, keep_center=False, cls_patch=None,
                       anat_patch=None, anat_k=6, eps=1e-6):
    """Prepare an IVIM_patch window for the spatial token."""
    output_patch = np.asarray(patch, np.float32).copy()
    center = output_patch.shape[-1] // 2
    center_s0 = float(output_patch[0, center, center])
    if abs(center_s0) > eps:
        output_patch /= center_s0

    if anat_patch is not None:
        anatomy = np.asarray(anat_patch, np.float32)
        anatomy = anatomy.reshape(anatomy.shape[0], -1)
        center_flat = (output_patch.shape[-1] // 2) * output_patch.shape[-1] + (output_patch.shape[-1] // 2)
        distance = np.sqrt(((anatomy - anatomy[:, center_flat:center_flat + 1]) ** 2).sum(axis=0))
        distance[center_flat] = np.inf if not keep_center else -np.inf
        neighbour_count = max(1, min(int(anat_k), distance.size - 1))
        selected = np.argpartition(distance, neighbour_count - 1)[:neighbour_count]
        flat_patch = output_patch.reshape(output_patch.shape[0], -1)
        mean_signal = flat_patch[:, selected].mean(axis=1)
        return np.broadcast_to(mean_signal[:, None, None], output_patch.shape).copy()

    if cls_patch is not None:
        class_patch = np.asarray(cls_patch).reshape(output_patch.shape[-2], output_patch.shape[-1])
        same_class = class_patch == class_patch[center, center]
        if not keep_center:
            same_class[center, center] = False
        neighbour_count = int(same_class.sum())
        mean_signal = (output_patch[:, same_class].mean(axis=1) if neighbour_count
                       else np.zeros(output_patch.shape[0], np.float32))
        return np.broadcast_to(mean_signal[:, None, None], output_patch.shape).copy()

    if not keep_center:
        output_patch[:, center, center] = 0.0
    return output_patch


class IVIMDataset(torch.utils.data.Dataset):
    """Reads the H5 once and slices it in memory."""

    def __init__(self, h5_path, use_struct=True, use_b0=True,
                 use_2d_struct=True, use_ivim_nb=False,
                 ivim_nb_keep_center=False, ivim_nb_oracle=False,
                 ivim_nb_anat_oracle=False, use_ansc=False):
        super().__init__()
        self.h5_path = h5_path
        self.use_struct, self.use_b0 = use_struct, use_b0
        self.use_2d_struct = use_2d_struct

        if use_ivim_nb and not use_b0:
            raise ValueError(
                "use_ivim_nb requires use_b0=True. The neighbouring IVIM "
                "curves travel in the b0 spatial token, so switching b0 "
                "off discards them and trains a plain no-b0 arm.")

        self.use_ivim_nb = use_ivim_nb
        self.ivim_nb_keep_center = ivim_nb_keep_center
        self.ivim_nb_oracle = ivim_nb_oracle
        self.ivim_nb_anat_oracle = ivim_nb_anat_oracle
        self.ivim_nb_anat_k = 6

        with h5py.File(h5_path, "r") as f:
            self.X = f["IVIM_cube"][:].astype(np.float32)
            self.SP = f["Struct_patch"][:].astype(np.float32)
            if use_ivim_nb:
                if "IVIM_patch" not in f:
                    raise KeyError(
                        f"{h5_path} has no IVIM_patch. Run "
                        f"add_neighbour_signals.py on it first.")
                self.B0 = f["IVIM_patch"][:].astype(np.float32)
                self.CLP = (f["Tissue_patch"][:] if ivim_nb_oracle
                            and "Tissue_patch" in f else None)
                if ivim_nb_oracle and self.CLP is None:
                    raise KeyError(f"{h5_path} has no Tissue_patch")
            else:
                self.CLP = None
                self.B0 = f["B0_patch"][:].astype(np.float32)
            self.T = f["Tissue"][:]
            self.L = f["Lesion"][:]
            self.P = (f["Paired"][:] if "Paired" in f
                      else np.zeros(len(self.X), dtype=np.int8))
            gt_keys = {"Dpar": "Dpar_gt", "Dint": "Dint_gt",
                       "Dmv": "Dmv_gt", "Fint": "fint_gt",
                       "Fmv": "fmv_gt", "S0": "S0_gt"}
            self.gt_params = {k: f[v][:].astype(np.float32).ravel()
                              for k, v in gt_keys.items() if v in f}

        self.use_ansc = use_ansc
        self.NBP = self.AMP = None
        if use_ansc:
            with h5py.File(h5_path, "r") as f:
                if "IVIM_patch" not in f:
                    raise KeyError(
                        f"{h5_path} has no IVIM_patch. Run "
                        f"add_neighbour_signals.py on it first.")
                self.NBP = f["IVIM_patch"][:].astype(np.float32)
                self.AMP = f["B0_patch"][:].astype(np.float32)

        self.anat_scale = anat_scale_from_h5(h5_path)
        if ivim_nb_anat_oracle:
            self.ivim_nb_anat_k = calibrate_anat_k(h5_path)

        self.nbvals = self.X.shape[-1]
        self.patch_size = int(self.SP.shape[-1])
        self._zero_sp = np.zeros((2, self.patch_size, self.patch_size),
                                 dtype=np.float32)
        self._zero_b0 = np.zeros((self.B0.shape[1], self.patch_size,
                                  self.patch_size), dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.use_struct:
            sp = self.SP[idx].copy()
            sp[0] = apply_anat_scale(sp[0], self.anat_scale[0])
            sp[1] = apply_anat_scale(sp[1], self.anat_scale[1])
        else:
            sp = self._zero_sp.copy()

        if not self.use_b0:
            b0 = self._zero_b0.copy()
        elif self.use_ivim_nb:
            b0 = ivim_neighbourhood(self.B0[idx], self.ivim_nb_keep_center,
                                    None if self.CLP is None else self.CLP[idx],
                                    sp if self.ivim_nb_anat_oracle else None,
                                    self.ivim_nb_anat_k)
        else:
            b0 = self.B0[idx].copy()
            b0[0] = apply_anat_scale(b0[0], self.anat_scale[2])

        return {
            "x_true": torch.from_numpy(self.X[idx]),
            "struct_patch": torch.from_numpy(sp),
            "b0_patch": torch.from_numpy(b0),
            "tissue": torch.tensor(int(self.T[idx]), dtype=torch.int16),
            "lesion": torch.tensor(int(self.L[idx]), dtype=torch.int16),
            "paired": torch.tensor(int(self.P[idx]), dtype=torch.int16),
            **{f"gt_{k}": torch.tensor(float(v[idx]), dtype=torch.float32)
               for k, v in self.gt_params.items()},
            **({"ansc_ivim_patch": torch.from_numpy(self.NBP[idx]),
                "ansc_b0_amp": torch.from_numpy(self.AMP[idx])}
               if self.use_ansc else {}),
        }

    def close(self):
        pass


PreloadedIVIMDataset = IVIMDataset


def recon_ramp(epoch, start=5, length=5):
    """Linear ramp from 0->1 starting at `start` over `length` epochs."""
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start) / max(1, length))


TISSUE_PRIOR_MU = {
    "GM":   {"Dpar": -7.372, "Dint": -5.943, "Dmv": -2.493,
             "Fint": -1.361, "Fmv": -3.645},
    "NAWM": {"Dpar": -7.470, "Dint": -6.088, "Dmv": -2.539,
             "Fint": -1.493, "Fmv": -4.187},
    "WMH":  {"Dpar": -7.138, "Dint": -6.030, "Dmv": -2.686,
             "Fint": -1.096, "Fmv": -4.443},
}
TISSUE_PRIOR_SD = {"Dpar": 0.271, "Dint": 0.258, "Dmv": 0.551,
                   "Fint": 0.431, "Fmv": 0.866}

TISSUE_PRIOR_CLASSES = ("GM", "NAWM", "WMH")
TISSUE_PRIOR_ORDER = ("Dpar", "Fint", "Dint", "Fmv", "Dmv")


def ansc_target(ivim_patch, b0_amp, anat, x_center, target="rel", tau=0.5,
               b0_floor=0.05, noise_level=None, eps=1e-6):
    """Build the anatomy-weighted neighbouring signal target."""
    batch_size, num_bvalues, patch_size, _ = ivim_patch.shape
    center_index = patch_size // 2
    dtype = ivim_patch.dtype

    anatomy_flat = anat.reshape(batch_size, anat.shape[1], -1)
    center_flat_index = center_index * patch_size + center_index
    anatomy_distance_squared = (
        (anatomy_flat - anatomy_flat[:, :, center_flat_index:center_flat_index + 1]) ** 2
    ).sum(dim=1)

    anatomy_weights = torch.exp(-anatomy_distance_squared / (tau * tau))
    anatomy_weights[:, center_flat_index] = 0.0
    weight_sum = anatomy_weights.sum(dim=1, keepdim=True).clamp(min=eps)

    b0_amplitude = b0_amp[:, 0].reshape(batch_size, -1)
    center_b0 = b0_amplitude[:, center_flat_index]
    valid_voxels = (center_b0 > b0_floor).to(dtype)
    relative_b0_amplitude = (b0_amplitude / center_b0.clamp(min=b0_floor)[:, None])

    if target == "b0":
        weighted_amplitude_ratio = ((anatomy_weights * relative_b0_amplitude).sum(dim=1)/ weight_sum[:, 0])
        return weighted_amplitude_ratio[:, None] * x_center, valid_voxels

    neighbour_signals = ivim_patch.reshape(batch_size, num_bvalues, -1)
    if target == "rel":
        neighbour_signals = neighbour_signals * relative_b0_amplitude[:, None, :]
    elif target == "abs":
        neighbour_signals = neighbour_signals * b0_amplitude[:, None, :]
    elif target != "decay":
        raise ValueError(f"unknown ansc target {target!r}")

    if noise_level is not None and float(noise_level) > 0:
        noise_sigma = 1.0 / (10.0 ** (float(noise_level) / 20.0))
        noise_real = torch.randn_like(neighbour_signals) * noise_sigma
        noise_imag = torch.randn_like(neighbour_signals) * noise_sigma
        neighbour_signals = torch.sqrt(
            (neighbour_signals + noise_real) ** 2 + noise_imag ** 2
        )

    spatial_target = (
        (neighbour_signals * anatomy_weights[:, None, :]).sum(dim=2)
        / weight_sum
    )
    return spatial_target, valid_voxels

def tissue_prior_loss(ivim_params, tissue_logits, sigma_scale=2.0,
                      prior_weights=None, eps=1e-8):
    """Compute the tissue-conditioned parameter prior loss."""
    if ivim_params is None or tissue_logits is None:
        zero = torch.tensor(0.0)
        return zero, zero

    Dpar, Fint, Dint, Fmv, Dmv, _S0 = ivim_params
    parameter_values = {
        "Dpar": Dpar,"Fint": Fint,"Dint": Dint,
        "Fmv": Fmv,"Dmv": Dmv,
        }
    if any(value is None for value in parameter_values.values()):
        zero = torch.tensor(0.0, device=tissue_logits.device)
        return zero, zero

    tissue_probabilities = torch.softmax(tissue_logits, dim=-1).detach()
    device = tissue_logits.device
    dtype = tissue_logits.dtype

    if prior_weights is None:
        normalized_prior_weights = None
    else:
        normalized_prior_weights = prior_weights.reshape(-1).to(device=device,dtype=dtype,)
        normalized_prior_weights = normalized_prior_weights / torch.clamp(normalized_prior_weights.sum(),min=1.0,)

    total_prior_loss = torch.zeros((), device=device, dtype=dtype)
    collapse_fraction_sum = torch.zeros((), device=device, dtype=dtype)
    num_parameters = 0

    for parameter_name in TISSUE_PRIOR_ORDER:
        parameter_flat = parameter_values[parameter_name].reshape(-1)
        log_parameter = torch.log(torch.clamp(parameter_flat, min=eps))
        prior_sd = TISSUE_PRIOR_SD[parameter_name] * sigma_scale
        class_log_means = torch.tensor(
            [TISSUE_PRIOR_MU[tissue_class][parameter_name] for tissue_class in TISSUE_PRIOR_CLASSES],
            device=device,dtype=dtype,
            )

        class_cost = ((log_parameter[:, None] - class_log_means[None, :]) ** 2/ (2.0 * prior_sd * prior_sd))
        class_probabilities = tissue_probabilities.reshape(-1,tissue_probabilities.shape[-1],)[:, :class_cost.shape[1]]
        voxel_prior_loss = (class_probabilities * class_cost).sum(dim=-1)

        if normalized_prior_weights is None:
            total_prior_loss = total_prior_loss + voxel_prior_loss.mean()
        else:
            total_prior_loss = total_prior_loss + (normalized_prior_weights * voxel_prior_loss).sum()

        with torch.no_grad():
            most_probable_class = class_probabilities.argmax(dim=-1)
            normalized_distance = ((log_parameter - class_log_means[most_probable_class]).abs()/ prior_sd)
            near_class_mean = (normalized_distance < 0.25).to(dtype)
            if normalized_prior_weights is None:
                collapse_fraction_sum = (collapse_fraction_sum + near_class_mean.mean())
            else:
                collapse_fraction_sum = (collapse_fraction_sum + (normalized_prior_weights * near_class_mean).sum())

        num_parameters += 1

    if num_parameters:
        total_prior_loss = total_prior_loss / num_parameters
        collapse_fraction_sum = collapse_fraction_sum / num_parameters

    return total_prior_loss, collapse_fraction_sum

def custom_loss_function(X_pred, X_true,
                         ivim_params=None,
                         recon_dict=None,
                         latent_dict=None,
                         struct_patch=None, b0_patch=None,
                         tissue_labels=None, tissue_valid=None,
                         use_parallel=True,
                         alpha_recon=0.05,
                         alpha_ce=0.2,
                         permute_ce_labels=False,
                         alpha_penalty=0.1,
                         ce_class_weights=None,
                         recon_warmup_start=5,
                         recon_warmup_length=5,
                         recon_cap=10.0,
                         alpha_latent=0.0, epoch=0, latent_start_epoch=0,
                         alpha_prior=0.0, prior_sigma_scale=2.0,
                         prior_weights=None,
                         alpha_ansc=0.0, ansc_target_mode="rel", ansc_tau=0.5,
                         ansc_ivim_patch=None, ansc_b0_amp=None,
                         ansc_noise_level=None, ansc_b0_floor=0.05,
                         debug=True):
    """Loss: IVIM MSE + alpha_recon_eff * Recon MSE + alpha_ce * CE +
    alpha_penalty * diffusion_bound_penalty.
    """
    mse_fn = nn.MSELoss(reduction="mean")
    device = X_pred.device

    if not hasattr(custom_loss_function, "batch_count"):
        custom_loss_function.batch_count = 0
    custom_loss_function.batch_count += 1

    alpha_recon_eff = alpha_recon * recon_ramp(epoch, recon_warmup_start, recon_warmup_length)

    ivim_mse = mse_fn(X_pred, X_true)

    recon_loss = torch.tensor(0.0, device=device)
    n_recon = 0

    def relative_mse(pred, target, eps=1e-6):
        """Scale reconstruction MSE by the target's mean square."""
        num = torch.mean((pred - target) ** 2)
        den = torch.mean(target ** 2) + eps
        return num / den

    if recon_dict is not None:
        if recon_dict.get("t1") is not None and struct_patch is not None:
            recon_loss = recon_loss + relative_mse(recon_dict["t1"], struct_patch[:, 0:1])
            n_recon += 1
        if recon_dict.get("flair") is not None and struct_patch is not None:
            recon_loss = recon_loss + relative_mse(recon_dict["flair"], struct_patch[:, 1:2])
            n_recon += 1
        if recon_dict.get("b0") is not None and b0_patch is not None:
            recon_loss = recon_loss + relative_mse(recon_dict["b0"], b0_patch)
            n_recon += 1

    if n_recon > 0:
        recon_loss = recon_loss / n_recon
        recon_loss = torch.clamp(recon_loss, max=recon_cap)

    ce_loss = torch.tensor(0.0, device=device)
    tissue_acc = torch.tensor(0.0, device=device)
    n_ce = torch.tensor(0.0, device=device)

    tissue_logits = latent_dict.get("tissue_logits") if latent_dict else None

    if tissue_logits is not None and tissue_labels is not None and tissue_valid is not None:
        valid_mask = tissue_valid.to(device)
        if valid_mask.any():
            valid_logits = tissue_logits[valid_mask]
            valid_labels = tissue_labels[valid_mask].to(device).long()
            n_ce = valid_mask.float().sum()

            if permute_ce_labels:
                perm = torch.randperm(valid_labels.shape[0],
                                      device=valid_labels.device)
                valid_labels = valid_labels[perm]

            if ce_class_weights is not None:
                ce_loss = F.cross_entropy(valid_logits, valid_labels,
                                          weight=ce_class_weights.to(device))
            else:
                class_counts = torch.bincount(valid_labels, minlength=NUM_TISSUE_CLASSES).float().clamp(min=1.0)
                class_weights = 1.0 / torch.sqrt(class_counts)
                class_weights = class_weights / class_weights.sum() * NUM_TISSUE_CLASSES
                ce_loss = F.cross_entropy(valid_logits, valid_labels, weight=class_weights.to(device))

            with torch.no_grad():
                preds = valid_logits.argmax(dim=1)
                tissue_acc = (preds == valid_labels).float().mean()

    diff_penalty = torch.tensor(0.0, device=device)
    if ivim_params is not None and alpha_penalty > 0:
        Dpar, Fint, Dint, Fmv, Dmv, S0 = ivim_params
        pen = torch.relu(Dpar - 0.0015) ** 2
        if Dint is not None:
            pen = pen + torch.relu(Dint - 0.0040) ** 2
        pen = pen + torch.relu(Dmv - 0.2000) ** 2
        diff_penalty = pen.mean()

    prior_loss = torch.tensor(0.0, device=device)
    prior_near = torch.tensor(0.0, device=device)
    if alpha_prior > 0:
        prior_loss, prior_near = tissue_prior_loss(
            ivim_params, tissue_logits, sigma_scale=prior_sigma_scale,
            prior_weights=prior_weights)


    ansc_loss = torch.tensor(0.0, device=device)
    ansc_active = torch.tensor(0.0, device=device)
    ansc_shift = torch.tensor(0.0, device=device)
    if (alpha_ansc > 0 and ansc_ivim_patch is not None
            and ansc_b0_amp is not None and struct_patch is not None):
        with torch.no_grad():
            sbar, c_ansc = ansc_target(
                ansc_ivim_patch, ansc_b0_amp, struct_patch, X_true,
                target=ansc_target_mode, tau=ansc_tau,
                b0_floor=ansc_b0_floor, noise_level=ansc_noise_level)

            ansc_active = c_ansc.mean()
            ansc_shift = (((sbar - X_true).abs().mean(dim=1)* c_ansc).sum()/ c_ansc.sum().clamp(min=1.0))

        ansc_loss = (((X_pred - sbar) ** 2).mean(dim=1)
                     * c_ansc).sum() / X_pred.shape[0]

    if debug and alpha_ansc > 0 and (custom_loss_function.batch_count % 200 == 0):

        beta_eff = alpha_ansc / (1.0 + alpha_ansc)
        print(f"  ANSC:    {ansc_loss.item():.5f}   beta_eff {beta_eff:.3f}   "
              f"active {100 * ansc_active.item():.0f}%   "
              f"shift {ansc_shift.item():.4f}   "
              f"[{ansc_target_mode}, tau {ansc_tau}]")

    total = (ivim_mse
             + alpha_recon_eff * recon_loss
             + alpha_ce * ce_loss
             + alpha_penalty * diff_penalty
             + alpha_prior * prior_loss
             + alpha_ansc * ansc_loss)

    if debug and alpha_prior > 0 and (custom_loss_function.batch_count % 200 == 0):
        cov = (1.0 if prior_weights is None
               else (prior_weights.reshape(-1) > 0).float().mean().item())

        if not getattr(custom_loss_function, "prior_legend", False):
            print("  [Prior] value | collapse = affected voxels within "
                  "0.25 sd of class mean | coverage = batch share")
            custom_loss_function.prior_legend = True

        if cov == 0.0:
            print("  Prior:  -   collapse -   coverage 0%")
        else:
            print(f"  Prior:  {prior_loss.item():.4f}   "
                  f"collapse {100 * prior_near.item():.0f}%   "
                  f"coverage {100 * cov:.0f}%")

    if debug and (custom_loss_function.batch_count % 200 == 0):
        attn_w = latent_dict.get("attn_weights") if latent_dict else None
        mod_names = latent_dict.get("modality_names") if latent_dict else None
        mod_mask = latent_dict.get("modality_mask") if latent_dict else None

        attn_str = "N/A"
        mask_str = ""
        per_class_attn_str = ""
        if attn_w is not None and mod_names is not None:
            with torch.no_grad():
                n_heads = attn_w.size(1)
                avg_all = attn_w.mean(dim=0).squeeze(-2)
                avg_over_heads = avg_all.mean(dim=0)

                head_strs = []
                for h in range(n_heads):
                    vals = ", ".join(f"{n}={avg_all[h, j].item():.3f}" for j, n in enumerate(mod_names))
                    head_strs.append(f"H{h}=[{vals}]")
                avg_str = ", ".join(f"{n}={avg_over_heads[j].item():.3f}" for j, n in enumerate(mod_names))
                attn_str = f"Avg=[{avg_str}] | " + " | ".join(head_strs)

                if mod_mask is not None:
                    mask_rates = mod_mask.float().mean(dim=0)
                    mr_str = ", ".join(f"{n}={mask_rates[j].item():.2f}" for j, n in enumerate(mod_names))
                    mask_str = f"  Mask rates:  [{mr_str}]"

                if tissue_labels is not None and tissue_valid is not None:
                    valid_mask_t = tissue_valid.to(device)
                    if valid_mask_t.any():
                        valid_labels = tissue_labels[valid_mask_t].to(device).long()
                        attn_avg = attn_w.mean(dim=1).squeeze(-2)
                        valid_attn = attn_avg[valid_mask_t]

                        if mod_mask is None:
                            valid_unmasked = torch.ones_like(valid_attn, dtype=torch.bool)
                        else:
                            valid_unmasked = ~mod_mask.to(device)[valid_mask_t]

                        class_strs = []
                        for c, cname in enumerate(TISSUE_NAMES):
                            cmask = valid_labels == c
                            if cmask.sum() < 5:
                                continue
                            vals_per_mod = []
                            for j, n in enumerate(mod_names):
                                keep_j = cmask & valid_unmasked[:, j]
                                if keep_j.sum() < 5:
                                    vals_per_mod.append(f"{n}=n/a")
                                else:
                                    vals_per_mod.append(f"{n}={valid_attn[keep_j, j].mean().item():.3f}")
                            class_strs.append(f"{cname}=[{', '.join(vals_per_mod)}]")
                        if class_strs:
                            per_class_attn_str = "  Attn/class:  " + " | ".join(class_strs)

        param_stats_str = ""
        if ivim_params is not None:
            with torch.no_grad():
                Dpar, Fint, Dint, Fmv, Dmv, S0 = ivim_params
                def format_diffusion_stat(name, t, upper):
                    pct_over = (t > upper).float().mean().item() * 100
                    return f"{name}: {t.mean().item():.5f} (max={t.max().item():.5f}, >{upper}:{pct_over:.1f}%)"
                parts = [format_diffusion_stat("Dpar", Dpar, 0.0015)]
                if Dint is not None:
                    parts.append(format_diffusion_stat("Dint", Dint, 0.0040))
                parts.append(format_diffusion_stat("Dmv", Dmv, 0.2000))
                param_stats_str = "  Diff params: " + " | ".join(parts)

        spatial_gates = latent_dict.get("spatial_gates") if latent_dict else None

        print(f"\n[Loss] Epoch {epoch} | Batch {custom_loss_function.batch_count}")
        print(f"  IVIM MSE:    {ivim_mse.item():.6f}")
        print(f"  Recon Loss:  {recon_loss.item():.6f} (x{alpha_recon_eff:.4f}, warmup from ep{recon_warmup_start})")
        print(f"  CE Loss:     {ce_loss.item():.4f} (x{alpha_ce})"
              f"{'  [static wt]' if ce_class_weights is not None else '  [batch wt]'}")
        print(f"  Diff Penalty:{diff_penalty.item():.6f} (x{alpha_penalty})")
        print(f"  Tissue Acc:  {tissue_acc.item():.3f}")
        print(f"  Attn:        {attn_str}")
        attn_scale_val = latent_dict.get("attn_scale") if latent_dict else None
        if attn_scale_val is not None:
            print(f"  Attn Scale:  {attn_scale_val:.4f}")
        if spatial_gates:
            gate_str = "  ".join(f"{k}={v:.4f}" for k, v in spatial_gates.items())
            detach_flag = latent_dict.get("detach_spatial_delta", False) if latent_dict else False
            print(f"  Spat Gates:  {gate_str}  {'[z_ivim_q detached]' if detach_flag else '[z_ivim_q attached]'}")
        if param_stats_str:
            print(param_stats_str)
        if mask_str:
            print(mask_str)
        if per_class_attn_str:
            print(per_class_attn_str)
        print(f"  Total:       {total.item():.6f}")

    return {
        "total": total,
        "ivim_mse": ivim_mse,
        "recon_loss": recon_loss,
        "ce_loss": ce_loss,
        "diff_penalty": diff_penalty,
        "ansc_loss": ansc_loss,
        "ansc_active": ansc_active,
        "tissue_acc": tissue_acc,
        "n_ce": n_ce,
    }


class ResBlock2D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.act = nn.ELU(inplace=True)

    def forward(self, x):
        return self.act(x + self.net(x))


class MiniDecoder(nn.Module):
    """Reconstruct a 3x3 patch from a latent vector."""
    def __init__(self, in_dim, out_channels=1, patch_size=3, hidden_dim=32, n_res=2):
        super().__init__()
        self.patch_size = patch_size
        self.fc = nn.Linear(in_dim, hidden_dim * patch_size * patch_size)
        self.res = nn.Sequential(*[ResBlock2D(hidden_dim) for _ in range(n_res)])
        self.out = nn.Conv2d(hidden_dim, out_channels, 3, padding=1)

    def forward(self, z):
        x = self.fc(z).view(z.size(0), -1, self.patch_size, self.patch_size)
        return self.out(self.res(x))


class ConcatFusion(nn.Module):
    """Drop-in replacement for nn.MultiheadAttention: concat + MLP."""

    def __init__(self, embed_dim, n_spatial_tokens=2, dropout=0.1):
        super().__init__()
        self.n_spatial = n_spatial_tokens
        input_dim = embed_dim * n_spatial_tokens
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, query, key, value, key_padding_mask=None,
                average_attn_weights=False):
        B, _, D = query.shape

        N = value.shape[1]
        if N < self.n_spatial:
            pad = value.new_zeros(B, self.n_spatial - N, D)
            v = torch.cat([value, pad], dim=1)
        else:
            v = value[:, :self.n_spatial]

        if key_padding_mask is not None:
            M = key_padding_mask.shape[1]
            if M < self.n_spatial:
                pad_mask = key_padding_mask.new_ones(B, self.n_spatial - M)
                key_padding_mask = torch.cat([key_padding_mask, pad_mask], dim=1)
            mask = key_padding_mask[:, :self.n_spatial].unsqueeze(-1)
            v = v.masked_fill(mask, 0.0)

        v_flat = v.reshape(B, -1)

        out = self.proj(v_flat).unsqueeze(1)
        return out, None


class Net(nn.Module):
    """PACE: Per-parameter Anatomically Conditioned Estimation for IVIM."""

    def __init__(self, bvals, net_pars, patch_size=3, cnn_channels=2,
                 spatial_on=True, parallel_heads=True,
                 recon_struct=True, recon_b0=True,
                 use_struct=True, use_b0=True,
                 ivim_latent_dim=32,
                 latent_dropout_p=0.3,
                 modality_dropout_p=0.15,
                 n_attn_heads=2,
                 sigmoid_temperature=1.0,
                 detach_recon=False,
                 detach_spatial_delta=True,
                 use_anat_token=True,
                 gate_inits=None,
                 use_ordered_diffusion=True,
                 use_softmax_fractions=True,
                 fusion_mode="attention",
                 b0_channels=1,
                 recon_grad_clip=10.0):
        super().__init__()

        self.register_buffer('bvals', torch.tensor(bvals, dtype=torch.float32))
        self.nbvals = len(bvals)
        self.net_pars = net_pars
        self.patch_size = patch_size
        self.spatial_on = spatial_on
        self.parallel_heads = parallel_heads
        self.recon_struct = recon_struct
        self.recon_b0 = recon_b0
        self.use_struct = use_struct
        self.use_b0 = use_b0
        self.b0_channels = int(b0_channels)
        self.ivim_latent_dim = ivim_latent_dim
        self.latent_dropout_p = latent_dropout_p
        self.modality_dropout_p = modality_dropout_p
        self.sigmoid_temperature = sigmoid_temperature
        self.detach_recon = detach_recon
        self.detach_spatial_delta = detach_spatial_delta
        self.use_anat_token = use_anat_token
        self.use_ordered_diffusion = use_ordered_diffusion
        self.fusion_mode = fusion_mode
        if fusion_mode not in ("attention", "concat"):
            raise ValueError(
                f"fusion_mode must be 'attention' or 'concat', got '{fusion_mode}'")
        self.recon_grad_clip = recon_grad_clip

        if not spatial_on and use_softmax_fractions:
            print("[Net] softmax fractions with spatial_on=False: the "
                  "fraction heads share one input and differ only in "
                  "their weights")
        self.use_softmax_fractions = use_softmax_fractions

        self.use_three_compartment = net_pars.use_three_compartment
        self.fitS0 = net_pars.fitS0

        if spatial_on and not parallel_heads:
            raise ValueError("Head-wise fusion requires parallel_heads=True.")

        if net_pars.width is None:
            self.net_pars.width = self.nbvals

        if self.use_three_compartment:
            self.ivim_dim = 6 if self.fitS0 else 5
        else:
            self.ivim_dim = 4 if self.fitS0 else 3

        if self.use_three_compartment:
            bound_indices = [0, 1, 2, 3, 4, 5] if self.fitS0 else [0, 1, 2, 3, 4]
        else:
            bound_indices = [0, 3, 4, 5] if self.fitS0 else [0, 3, 4]

        cons_min = [net_pars.cons_min[i] for i in bound_indices]
        cons_max = [net_pars.cons_max[i] for i in bound_indices]
        self.register_buffer('cons_min_vec', torch.tensor(cons_min, dtype=torch.float32))
        self.register_buffer('cons_max_vec', torch.tensor(cons_max, dtype=torch.float32))

        D = ivim_latent_dim

        self.ivim_encoder = self.make_encoder(self.nbvals, D)
        self.norm_ivim = nn.LayerNorm(D)

        if spatial_on and use_struct:
            self.t1_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ELU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ELU()
            )
            self.t1_fc = nn.Linear(16 * patch_size * patch_size, D)
            self.norm_t1 = nn.LayerNorm(D)

            self.flair_cnn = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ELU(),
                nn.Conv2d(16, 16, 3, padding=1), nn.ELU()
            )
            self.flair_fc = nn.Linear(16 * patch_size * patch_size, D)
            self.norm_flair = nn.LayerNorm(D)
        else:
            self.t1_cnn = self.t1_fc = self.norm_t1 = None
            self.flair_cnn = self.flair_fc = self.norm_flair = None

        if spatial_on and use_b0:
            self.b0_cnn = nn.Sequential(
                nn.Conv2d(int(b0_channels), 32, 3, padding=1), nn.ELU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.ELU()
            )
            self.b0_fc = nn.Linear(32 * patch_size * patch_size, D)
            self.norm_b0 = nn.LayerNorm(D)
        else:
            self.b0_cnn = self.b0_fc = self.norm_b0 = None

        if spatial_on and use_struct and use_anat_token:
            self.anat_fusion = nn.Sequential(
                nn.Linear(2 * D, D), nn.ELU(), nn.LayerNorm(D)
            )
        else:
            self.anat_fusion = None

        if spatial_on:
            self.tissue_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            self.query_proj = nn.Linear(D, D)
            n_spatial_tokens = 0
            if use_struct:
                n_spatial_tokens += 1 if use_anat_token else 2
            if use_b0:
                n_spatial_tokens += 1
            n_spatial_tokens = max(n_spatial_tokens, 1)

            if fusion_mode == "concat":
                self.cross_attn = ConcatFusion(
                    D, n_spatial_tokens=n_spatial_tokens,
                    dropout=latent_dropout_p)
            else:
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=D, num_heads=n_attn_heads,
                    dropout=latent_dropout_p, batch_first=True
                )
            self.n_attn_heads = n_attn_heads
            self.attn_norm = nn.LayerNorm(D)
            self.attn_scale = nn.Parameter(torch.tensor(0.1))

        
            inits = {"Dpar": -2.2, "Fmv": -2.2, "Dmv": -2.2,
                     "S0": -2.2, "Fint": -2.2, "Dint": -2.2,
                     "Fpar": -2.2}
            if gate_inits is not None:
                inits.update(gate_inits)

            self.spatial_gate_dpar_raw = nn.Parameter(torch.tensor(float(inits["Dpar"])))
            self.spatial_gate_fmv_raw  = nn.Parameter(torch.tensor(float(inits["Fmv"])))
            self.spatial_gate_dmv_raw  = nn.Parameter(torch.tensor(float(inits["Dmv"])))
            self.spatial_gate_s0_raw   = nn.Parameter(torch.tensor(float(inits["S0"])))

            if self.net_pars.use_three_compartment:
                self.spatial_gate_fint_raw = nn.Parameter(torch.tensor(float(inits["Fint"])))
                self.spatial_gate_dint_raw = nn.Parameter(torch.tensor(float(inits["Dint"])))
                self.spatial_gate_fpar_raw = nn.Parameter(torch.tensor(float(inits["Fpar"])))
            else:
                self.spatial_gate_fint_raw = None
                self.spatial_gate_dint_raw = None
                self.spatial_gate_fpar_raw = None

            self.max_spatial_mods = 3
            self.spatial_concat_dim = D * self.max_spatial_mods

            self.spatial_recon_proj = nn.Sequential(
                nn.Linear(self.spatial_concat_dim, D), nn.ELU(), nn.LayerNorm(D)
            )
            self.spatial_ce_proj = nn.Sequential(
                nn.Linear(self.spatial_concat_dim, D), nn.ELU(), nn.LayerNorm(D)
            )
        else:
            self.tissue_token = None
            self.query_proj = None
            self.cross_attn = None
            self.n_attn_heads = 0
            self.attn_norm = None
            self.attn_scale = None
            self.spatial_gate_dpar_raw = None
            self.spatial_gate_fint_raw = None
            self.spatial_gate_dint_raw = None
            self.spatial_gate_fpar_raw = None
            self.spatial_gate_fmv_raw = None
            self.spatial_gate_dmv_raw = None
            self.spatial_gate_s0_raw = None
            self.max_spatial_mods = 0
            self.spatial_concat_dim = 0
            self.spatial_recon_proj = None
            self.spatial_ce_proj = None

        if parallel_heads:
            self.Dpar_head = nn.Linear(D, 1)
            self.Dmv_head = nn.Linear(D, 1)
            self.S0_head = nn.Linear(D, 1) if self.fitS0 else None
            if self.use_three_compartment:
                self.Dint_head = nn.Linear(D, 1)
                self.Fint_head = nn.Linear(D, 1)
                self.Fmv_head = nn.Linear(D, 1)
                self.Fpar_head = nn.Linear(D, 1) if use_softmax_fractions else None
            else:
                self.Fmv_head = nn.Linear(D, 1)
                self.Fpar_head = None
        else:
            self.param_head = nn.Sequential(
                nn.Linear(D, D), nn.ELU(),
                nn.Linear(D, self.ivim_dim)
            )
            self.Fpar_head = None

        if spatial_on and use_struct and recon_struct:
            self.t1_decoder = MiniDecoder(D, 1, patch_size)
            self.flair_decoder = MiniDecoder(D, 1, patch_size)
        else:
            self.t1_decoder = None
            self.flair_decoder = None

        if spatial_on and use_b0 and recon_b0:
            self.b0_decoder = MiniDecoder(D, int(b0_channels), patch_size)
        else:
            self.b0_decoder = None

        self.tissue_classifier = nn.Sequential(
            nn.Linear(D, D), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(D, NUM_TISSUE_CLASSES)
            )


    def make_encoder(self, in_dim, out_dim):
        """Build the fully connected IVIM encoder."""
        w = self.net_pars.width
        d = self.net_pars.depth
        layers = []
        for i in range(d):
            layers.extend([
                nn.Linear(in_dim if i == 0 else w, w),
                nn.ELU()
            ])
            if self.net_pars.dropout > 0 and i < (d - 1):
                layers.append(nn.Dropout(self.net_pars.dropout))
        layers.append(nn.Linear(w, out_dim))
        return nn.Sequential(*layers)

    def freeze_spatial_gates(self, value=0.0, subset=None):
        """Freeze spatial gates at a target sigmoid value for ablation."""
        eps = 1e-7
        value_clamped = max(eps, min(1.0 - eps, value))
        raw_value = math.log(value_clamped / (1.0 - value_clamped))

        gate_map = {
            "Dpar": "spatial_gate_dpar_raw",
            "Fmv":  "spatial_gate_fmv_raw",
            "Dmv":  "spatial_gate_dmv_raw",
            "S0":   "spatial_gate_s0_raw",
            "Fint": "spatial_gate_fint_raw",
            "Dint": "spatial_gate_dint_raw",
            "Fpar": "spatial_gate_fpar_raw",
        }

        targets = subset if subset is not None else list(gate_map.keys())
        frozen = []

        for name in targets:
            attr = gate_map.get(name)
            if attr is None:
                raise ValueError(f"Unknown gate name '{name}'. Choose from {list(gate_map.keys())}")
            param = getattr(self, attr, None)
            if param is None:
                continue
            with torch.no_grad():
                param.fill_(raw_value)
            param.requires_grad_(False)
            frozen.append(f"{name}={value:.3f} (raw={raw_value:.1f})")

        print(f"[freeze_spatial_gates] Frozen: {', '.join(frozen)}")

    def predict_signal_standard(self, params, bvals):
        """Reconstruct the IVIM signal with the standard forward model."""
        Dpar, Fint, Dint, Fmv, Dmv, S0 = params
        if self.use_three_compartment:
            return S0 * (
                (1 - Fmv - Fint) * torch.exp(-bvals * Dpar) +
                Fint * torch.exp(-bvals * Dint) +
                Fmv * torch.exp(-bvals * Dmv)
            )
        return S0 * ((1 - Fmv) * torch.exp(-bvals * Dpar) + Fmv * torch.exp(-bvals * Dmv))

    def predict_signal_ir(self, params, bvals, device):
        """Reconstruct the IVIM signal with inversion-recovery weighting."""
        Dpar, Fint, Dint, Fmv, Dmv, S0 = params
        rt = self.net_pars.rel_times
        dtype = Dpar.dtype

        TE = torch.as_tensor(rt.echotime, device=device, dtype=dtype)
        TR = torch.as_tensor(rt.repetitiontime, device=device, dtype=dtype)
        TI = torch.as_tensor(rt.inversiontime, device=device, dtype=dtype)
        T1t = torch.as_tensor(rt.tissueT1, device=device, dtype=dtype)
        T2t = torch.as_tensor(rt.tissueT2, device=device, dtype=dtype)
        T1i = torch.as_tensor(rt.isfT1, device=device, dtype=dtype)
        T2i = torch.as_tensor(rt.isfT2, device=device, dtype=dtype)
        T1b = torch.as_tensor(rt.bloodT1, device=device, dtype=dtype)
        T2b = torch.as_tensor(rt.bloodT2, device=device, dtype=dtype)

        tissue_ir = 1 - 2 * torch.exp(-TI / T1t) + torch.exp(-TR / T1t)
        tissue_t2 = torch.exp(-TE / T2t)
        blood_ir = 1 - torch.exp(-TR / T1b)
        blood_t2 = torch.exp(-TE / T2b)

        if self.use_three_compartment:
            isf_ir = 1 - 2 * torch.exp(-TI / T1i) + torch.exp(-TR / T1i)
            isf_t2 = torch.exp(-TE / T2i)
            num = (
                (1 - Fmv - Fint) * tissue_ir * tissue_t2 * torch.exp(-bvals * Dpar) +
                Fint * isf_ir * isf_t2 * torch.exp(-bvals * Dint) +
                Fmv * blood_ir * blood_t2 * torch.exp(-bvals * Dmv)
                )
            denom = (
                (1 - Fmv - Fint) * tissue_ir * tissue_t2 +
                Fint * isf_ir * isf_t2 +
                Fmv * blood_ir * blood_t2
                )
        else:
            num = (
                (1 - Fmv) * tissue_ir * tissue_t2 * torch.exp(-bvals * Dpar) +
                Fmv * blood_ir * blood_t2 * torch.exp(-bvals * Dmv)
                 )
            denom = (1 - Fmv) * tissue_ir * tissue_t2 + Fmv * blood_ir * blood_t2

        return S0 * (num / denom)


    def forward(self, ivim_signal=None, struct_patch=None, b0_patch=None,
                ivim_true_signal=None, structural_patch=None,
                fusion_active=True):

        if ivim_signal is None: ivim_signal = ivim_true_signal
        if struct_patch is None: struct_patch = structural_patch

        device = ivim_signal.device
        dtype = ivim_signal.dtype
        B = ivim_signal.size(0)

        struct_given = struct_patch is not None
        b0_given = b0_patch is not None
        has_struct = self.spatial_on and self.use_struct and struct_given and self.t1_cnn is not None
        has_b0 = self.spatial_on and self.use_b0 and b0_given and self.b0_cnn is not None

        if has_struct: struct_patch = struct_patch.to(device=device, dtype=dtype)
        if has_b0: b0_patch = b0_patch.to(device=device, dtype=dtype)

        z_ivim = self.norm_ivim(self.ivim_encoder(ivim_signal))

        attn_weights = None
        key_padding_mask = None
        modality_names = []
        token_recon = None
        token_ce = None

        if self.spatial_on and self.cross_attn is not None:
            z_ivim_q = z_ivim.detach() if self.detach_spatial_delta else z_ivim

            sequence = []
            modality_names = []

            z_t1 = z_flair = z_anat = z_b0 = None

            if has_struct:
                t1_feat = self.t1_cnn(struct_patch[:, 0:1]).view(B, -1)
                z_t1 = self.norm_t1(self.t1_fc(t1_feat))

                flair_feat = self.flair_cnn(struct_patch[:, 1:2]).view(B, -1)
                z_flair = self.norm_flair(self.flair_fc(flair_feat))

                if self.anat_fusion is not None:
                    z_anat = self.anat_fusion(torch.cat([z_t1, z_flair], dim=-1))
                    sequence.append(z_anat)
                    modality_names.append("Anat")
                else:
                    sequence.append(z_t1)
                    modality_names.append("T1")
                    sequence.append(z_flair)
                    modality_names.append("FLAIR")

            if has_b0:
                b0_feat = self.b0_cnn(b0_patch).view(B, -1)
                z_b0 = self.norm_b0(self.b0_fc(b0_feat))
                sequence.append(z_b0)
                modality_names.append("B0")

            if len(sequence) == 0:
                token_dpar = token_fmv = token_dmv = token_s0 = z_ivim
                token_fint = token_dint = z_ivim if self.use_three_compartment else None
                z_fused = z_ivim
                gate_vals = {}

            else:
                kv_sequence = torch.stack(sequence, dim=1)
                N_mod = kv_sequence.size(1)
                D = self.ivim_latent_dim

                if self.training and self.modality_dropout_p > 0 and N_mod > 1:
                    mask = torch.rand(B, N_mod, device=device) < self.modality_dropout_p

                    all_masked = mask.all(dim=1)
                    if all_masked.any():
                        pick = torch.randint(0, N_mod, (B,), device=device)
                        mask[all_masked, pick[all_masked]] = False

                    key_padding_mask = mask
                else:
                    key_padding_mask = None

                has_any_spatial = has_struct or has_b0

                if has_any_spatial:
                    z_t1_cat = z_ivim.new_zeros(B, D)
                    z_fl_cat = z_ivim.new_zeros(B, D)
                    z_b0_cat = z_ivim.new_zeros(B, D)

                    col = 0
                    if has_struct:
                        if self.anat_fusion is not None:
                            keep_anat = (~key_padding_mask[:, col]).bool() if key_padding_mask is not None else torch.ones(B, device=device, dtype=torch.bool)
                            z_t1_cat[keep_anat] = z_t1[keep_anat]
                            z_fl_cat[keep_anat] = z_flair[keep_anat]
                            col += 1
                        else:
                            keep_t1 = (~key_padding_mask[:, col]).bool() if key_padding_mask is not None else torch.ones(B, device=device, dtype=torch.bool)
                            z_t1_cat[keep_t1] = z_t1[keep_t1]
                            col += 1
                            keep_fl = (~key_padding_mask[:, col]).bool() if key_padding_mask is not None else torch.ones(B, device=device, dtype=torch.bool)
                            z_fl_cat[keep_fl] = z_flair[keep_fl]
                            col += 1

                    if has_b0:
                        keep_b0 = (~key_padding_mask[:, col]).bool() if key_padding_mask is not None else torch.ones(B, device=device, dtype=torch.bool)
                        z_b0_cat[keep_b0] = z_b0[keep_b0]

                    spatial_concat = torch.cat([z_t1_cat, z_fl_cat, z_b0_cat], dim=1)

                    token_recon = self.spatial_recon_proj(spatial_concat)
                    if self.recon_grad_clip is not None and token_recon.requires_grad:
                        _clip = float(self.recon_grad_clip)
                        token_recon.register_hook(lambda g, c=_clip: g.clamp(-c, c))
                    token_ce = self.spatial_ce_proj(spatial_concat)

                query = self.query_proj(z_ivim_q).unsqueeze(1) + self.tissue_token
                token_out, attn_weights = self.cross_attn(
                    query=query, key=kv_sequence, value=kv_sequence,
                    key_padding_mask=key_padding_mask,
                    average_attn_weights=False,
                )

                delta = self.attn_norm(self.attn_scale * token_out.squeeze(1))
                z_fused = z_ivim + delta

                gate_vals = {}
                def apply_spatial_gate(name, raw_param):
                    """Add the gated spatial residual for one parameter
                    head.
                    """
                    g = torch.sigmoid(raw_param)
                    gate_vals[name] = g.item()
                    return z_ivim + g * delta

                token_dpar = apply_spatial_gate("Dpar", self.spatial_gate_dpar_raw)
                token_fmv  = apply_spatial_gate("Fmv",  self.spatial_gate_fmv_raw)
                token_dmv  = apply_spatial_gate("Dmv",  self.spatial_gate_dmv_raw)
                token_s0   = apply_spatial_gate("S0",   self.spatial_gate_s0_raw)

                if self.use_three_compartment:
                    token_fint = apply_spatial_gate("Fint", self.spatial_gate_fint_raw)
                    token_dint = apply_spatial_gate("Dint", self.spatial_gate_dint_raw)
                    token_fpar = apply_spatial_gate("Fpar", self.spatial_gate_fpar_raw)
                else:
                    token_fint = token_dint = token_fpar = None

        else:
            token_dpar = token_fmv = token_dmv = token_s0 = z_ivim
            token_fint = token_dint = z_ivim if self.use_three_compartment else None
            token_fpar = z_ivim if self.use_three_compartment else None
            z_fused = z_ivim
            gate_vals = {}

        temp = self.sigmoid_temperature
        gmin = self.cons_min_vec
        gmax = self.cons_max_vec

        def constrain_fraction(logit, idx):
            return gmin[idx] + torch.sigmoid(torch.nan_to_num(logit) / temp) * (gmax[idx] - gmin[idx])

        def constrain_diffusion(logit, idx):
            return gmin[idx] + torch.sigmoid(torch.nan_to_num(logit) / temp) * (gmax[idx] - gmin[idx])

        if self.parallel_heads:
            if self.use_three_compartment:

                if self.use_ordered_diffusion:
                    Dpar = gmin[0] + F.softplus(torch.nan_to_num(self.Dpar_head(token_dpar)))
                    Dint = Dpar    + F.softplus(torch.nan_to_num(self.Dint_head(token_dint)))
                    Dmv  = Dint    + F.softplus(torch.nan_to_num(self.Dmv_head(token_dmv)))
                else:
                    Dpar = constrain_diffusion(self.Dpar_head(token_dpar), 0)
                    Dint = constrain_diffusion(self.Dint_head(token_dint), 2)
                    Dmv  = constrain_diffusion(self.Dmv_head(token_dmv), 4)

                if self.use_softmax_fractions:
                 
                    logit_par = self.Fpar_head(token_fpar)
                    logit_int = self.Fint_head(token_fint)
                    logit_mv  = self.Fmv_head(token_fmv)
                    logits_cat = torch.cat([logit_par, logit_int, logit_mv], dim=1)
                    fracs = F.softmax(logits_cat / temp, dim=1)
                    Fint = fracs[:, 1:2]
                    Fmv  = fracs[:, 2:3]
                else:
                    f_int_raw = constrain_fraction(self.Fint_head(token_fint), 1)
                    f_mv_raw  = constrain_fraction(self.Fmv_head(token_fmv), 3)
                    sf = torch.clamp(f_mv_raw + f_int_raw, min=1.0)
                    Fint = f_int_raw / sf
                    Fmv  = f_mv_raw / sf

                S0 = constrain_fraction(self.S0_head(token_s0), 5) if self.fitS0 else torch.ones_like(Dpar)

            else:
                Dpar = constrain_diffusion(self.Dpar_head(token_dpar), 0)
                Fmv  = constrain_fraction(self.Fmv_head(token_fmv), 1)
                Dmv  = constrain_diffusion(self.Dmv_head(token_dmv), 2)
                S0   = constrain_fraction(self.S0_head(token_s0), 3) if self.fitS0 else torch.ones_like(Dpar)
                Fint, Dint = None, None

        else:
            raw = self.param_head(z_fused)
            splits = torch.split(raw, 1, dim=1)
            if self.use_three_compartment:
                Dpar = constrain_diffusion(splits[0], 0)
                f_int_raw = constrain_fraction(splits[1], 1)
                Dint = constrain_diffusion(splits[2], 2)
                f_mv_raw = constrain_fraction(splits[3], 3)
                Dmv = constrain_diffusion(splits[4], 4)
                S0 = constrain_fraction(splits[5], 5) if self.fitS0 else torch.ones_like(Dpar)
                sf = torch.clamp(f_mv_raw + f_int_raw, min=1.0)
                Fint = f_int_raw / sf
                Fmv = f_mv_raw / sf
            else:
                Dpar = constrain_diffusion(splits[0], 0)
                Fmv = constrain_fraction(splits[1], 1)
                Dmv = constrain_diffusion(splits[2], 2)
                S0 = constrain_fraction(splits[3], 3) if self.fitS0 else torch.ones_like(Dpar)
                Fint, Dint = None, None

        if self.use_three_compartment:
            params = (Dpar, Fint, Dint, Fmv, Dmv, S0)
            params_vec = torch.cat([Dpar, Fint, Dint, Fmv, Dmv] + ([S0] if self.fitS0 else []), dim=1)
        else:
            params = (Dpar, None, None, Fmv, Dmv, S0)
            params_vec = torch.cat([Dpar, Fmv, Dmv] + ([S0] if self.fitS0 else []), dim=1)

        recons = {}
        if self.spatial_on:
            if token_recon is not None:
                recon_input = token_recon.detach() if self.detach_recon else token_recon
            elif token_ce is not None:
                recon_input = token_ce.detach()
            else:
                recon_input = None

            if recon_input is not None:
                if self.t1_decoder is not None:
                    recons["t1"] = self.t1_decoder(recon_input)
                if self.flair_decoder is not None:
                    recons["flair"] = self.flair_decoder(recon_input)
                if self.b0_decoder is not None:
                    recons["b0"] = self.b0_decoder(recon_input)

        if self.spatial_on and token_ce is not None:
            tissue_logits = self.tissue_classifier(token_ce)
        else:
            tissue_logits = None

        bvals_t = self.bvals.view(1, -1)
        if self.net_pars.IR:
            X_pred = self.predict_signal_ir(params, bvals_t, device)
        else:
            X_pred = self.predict_signal_standard(params, bvals_t)
        X_pred = torch.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

        return {
            "X_pred": X_pred,
            "ivim_params": params,
            "ivim_params_vec": params_vec,
            "recon": recons if recons else None,
            "latent": {
                "tissue_logits": tissue_logits,
                "attn_weights": attn_weights,
                "modality_names": modality_names,
                "modality_mask": key_padding_mask,
                "z_fused": z_fused,
                "token_recon": token_recon,
                "token_ce": token_ce,
                "attn_scale": self.attn_scale.item() if self.attn_scale is not None else None,
                "spatial_gates": gate_vals,
                "detach_spatial_delta": self.detach_spatial_delta,
            },
        }
