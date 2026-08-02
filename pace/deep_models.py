#!/usr/bin/env python3
# -----------------------------------------------------------
# pace.deep_models.py
# Bin Hoang, University of Rochester, Jan/Feb/March 2026
# Tissue Token IVIM — PACE (Per-parameter Anatomically Conditioned Estimation)
# -----------------------------------------------------------
#
# Architecture notes:
#
#   Modality encoders produce [z_ivim, z_T1, z_FLAIR, z_b0].
#
#   ANATOMY TOKEN:
#     z_anat = fusion_proj(cat[z_T1, z_FLAIR])
#     T1 and FLAIR together define tissue state (GM/NAWM/WMH).
#     Fusing them into one token prevents attention from wasting
#     capacity weighing image contrasts against each other.
# 
#
#   CROSS-ATTENTION — spatial-only K,V (April 2026):
#     K,V sequence: [z_anat, z_b0] (spatial tokens only).
#     z_ivim is NOT in K,V — it enters only through the query:
#       Q = query_proj(z_ivim_q) + tissue_token τ
#     This forces attention output to be purely spatial context.
#     Previous design had z_ivim in K,V which let attention "hide"
#     by self-attending, duplicating the per-parameter gates' job.
#
#   Y-BRANCH (gradient isolation):
#     token_recon = recon_proj(spatial_concat)  → recon decoders
#     token_ce    = ce_proj(spatial_concat)     → tissue classifier
#     Separate projections so recon and CE learn different features
#     from the same spatial input without competing. 
#
#   PER-PARAMETER SPATIAL GATING — eq (5) in methodology:
#     z_fused = LayerNorm(z_ivim_q + α · CrossAttn(Q, K, V))
#
#     Each IVIM parameter head gets its own learned gate g_X:
#       z_X = (1 - σ(g_X)) · z_ivim + σ(g_X) · z_fused
#
#     When σ(g_X) ≈ 0: z_X ≈ z_ivim       (pure signal, no spatial)
#     When σ(g_X) ≈ 1: z_X ≈ z_fused       (full spatial context)
#
#     Gate inits:
#       Dpar, Dint, Fint, S0: -5.0  → σ ≈ 0.007 (nearly closed)
#       Fmv, Dmv:             -2.0  → σ ≈ 0.12  (open to spatial)
#       Configurable via gate_inits argument to Net()
#
#   GRADIENT FLOW (fully separated):
#     IVIM MSE  → physics heads → z_X → (1-g)*z_ivim → IVIM encoder
#                                      → g*z_fused   → attn, spatial encoders
#     (detach_spatial_delta=True isolates IVIM encoder from attention path)
#     Recon MSE → recon decoders → token_recon → spatial_recon_proj → spatial encoders ONLY
#     CE        → classifier → token_ce → spatial_ce_proj → spatial encoders ONLY
#
# Loss: IVIM MSE + alpha_recon_eff * Recon MSE + alpha_ce * CE
#       + alpha_penalty * soft_upper_bound_penalty
# -----------------------------------------------------------

import math
import numpy as np
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pace.hyperparams import net_pars


# ===========================================================
#  TISSUE LABEL UTILITIES
# ===========================================================

TISSUE_CLASS_GM = 0
TISSUE_CLASS_NAWM = 1
TISSUE_CLASS_WMH = 2
NUM_TISSUE_CLASSES = 3
TISSUE_NAMES = ["GM", "NAWM", "WMH"]


def tissue_lesion_to_class(tissue_tensor, lesion_tensor):
    """Convert tissue/lesion labels to 3-class labels. Returns labels, valid mask."""
    tissue = tissue_tensor.long()
    lesion = lesion_tensor.long()
    labels = torch.full_like(tissue, -1)

    wmh_mask = lesion != 0
    labels[wmh_mask] = TISSUE_CLASS_WMH
    labels[(tissue == 2) & (~wmh_mask)] = TISSUE_CLASS_GM
    labels[(tissue == 3) & (~wmh_mask)] = TISSUE_CLASS_NAWM

    return labels, labels >= 0


# ===========================================================
#  STATIC CE WEIGHTS (dataset-level, sqrt inverse-frequency)
# ===========================================================

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


# ===========================================================
#  DATA PREPROCESSING
# ===========================================================

def clean_ivim_signals(X_batch, P_batch, B0_batch, bvals, inject_noise=False, noise_type="rician", noise_level=25.0):
    """Clean/normalize IVIM signals with optional noise injection."""
    device, dtype = X_batch.device, X_batch.dtype
    B, Nb = X_batch.shape

    assert X_batch.ndim == 2, f"X_batch must be (B, Nb), got {X_batch.shape}"
    assert len(bvals) == Nb, f"b-values ({len(bvals)}) must match signal dim ({Nb})"

    if P_batch is None: P_batch = torch.zeros((B, 2, 3, 3), device=device, dtype=dtype)
    if B0_batch is None: B0_batch = torch.zeros((B, 1, 3, 3), device=device, dtype=dtype)

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


# ===========================================================
#  INFERENCE HELPERS
# ===========================================================

try:
    from numpy.lib.stride_tricks import sliding_window_view
    NP_SLIDING_EXISTS = True
except ImportError: NP_SLIDING_EXISTS = False

try:
    from skimage.util import view_as_windows
    SKIMAGE_EXISTS = True
except ImportError: SKIMAGE_EXISTS = False

def zscore_patch_np(patch, sd_floor=1e-2):
    med = np.median(patch)
    sd = 1.4826 * np.median(np.abs(patch - med))
    return (patch - med) if sd < sd_floor else (patch - med) / sd

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

def prepare_inference_patches(ivim_4d, t1_3d, flair_3d, bvals, device="cuda", normalize_inputs=False):
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
                struct_patches[i, 0] = zscore_patch_np(t1_p[x, y])
                struct_patches[i, 1] = zscore_patch_np(fl_p[x, y])
                b0_patches[i, 0] = zscore_patch_np(b0_p[x, y])
            else:
                struct_patches[i, 0] = zscore_patch_np(extract_3x3(t1_in[..., zz], x, y))
                struct_patches[i, 1] = zscore_patch_np(extract_3x3(flair_in[..., zz], x, y))
                b0_patches[i, 0] = zscore_patch_np(extract_3x3(b0_in[..., zz], x, y))

    return X_data, torch.from_numpy(struct_patches).to(device=device, dtype=torch.float32), torch.from_numpy(b0_patches).to(device=device, dtype=torch.float32), ijk


# ===========================================================
#  DATASET
# ===========================================================

class IVIMDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, use_struct=True, use_b0=True, use_2d_struct=True):
        super().__init__()
        self.h5_path = h5_path
        self.use_struct, self.use_b0, self.use_2d_struct = use_struct, use_b0, use_2d_struct
        self.file = None
        with h5py.File(self.h5_path, "r") as f:
            self.length = f["IVIM_cube"].shape[0]
            self.nbvals = f["IVIM_cube"].shape[-1]

    def __len__(self): return self.length

    def __getitem__(self, idx):
        if self.file is None: self.file = h5py.File(self.h5_path, "r")

        ivim_signal = self.file["IVIM_cube"][idx].astype(np.float32)
        struct_patch = self.file["Struct_patch"][idx].astype(np.float32)
        b0_patch = self.file["B0_patch"][idx].astype(np.float32)
        tissue = int(self.file["Tissue"][idx])
        lesion = int(self.file["Lesion"][idx])

        if self.use_struct:
            struct_patch[0] = zscore_patch_np(struct_patch[0])
            struct_patch[1] = zscore_patch_np(struct_patch[1])
        else: struct_patch = np.zeros((2, 3, 3), dtype=np.float32)

        if self.use_b0: b0_patch[0] = zscore_patch_np(b0_patch[0])
        else: b0_patch = np.zeros((1, 3, 3), dtype=np.float32)

        return {"x_true": torch.from_numpy(ivim_signal), "struct_patch": torch.from_numpy(struct_patch),
                "b0_patch": torch.from_numpy(b0_patch), "tissue": torch.tensor(tissue, dtype=torch.int16),
                "lesion": torch.tensor(lesion, dtype=torch.int16)}

    def close(self):
        if self.file is not None: self.file.close(); self.file = None


# ===========================================================
#  LOSS FUNCTION
# ===========================================================

def _recon_ramp(epoch, start=5, length=5):
    """Linear ramp from 0→1 starting at `start` over `length` epochs."""
    if epoch < start:
        return 0.0
    return min(1.0, (epoch - start) / max(1, length))


def custom_loss_function(X_pred, X_true,
                         ivim_params=None,
                         recon_dict=None,
                         latent_dict=None,
                         struct_patch=None, b0_patch=None,
                         tissue_labels=None, tissue_valid=None,
                         use_parallel=True,
                         alpha_recon=0.05,
                         alpha_ce=0.2,
                         alpha_penalty=0.1,
                         ce_class_weights=None,
                         recon_warmup_start=5,
                         recon_warmup_length=5,
                         recon_cap=10.0,
                         alpha_latent=0.0, epoch=0, latent_start_epoch=0,
                         debug=True):
    """
    Loss: IVIM MSE + alpha_recon_eff * Recon MSE + alpha_ce * CE
          + alpha_penalty * diffusion_bound_penalty.
    """
    mse_fn = nn.MSELoss(reduction="mean")
    device = X_pred.device

    if not hasattr(custom_loss_function, "batch_count"):
        custom_loss_function.batch_count = 0
    custom_loss_function.batch_count += 1

    alpha_recon_eff = alpha_recon * _recon_ramp(epoch, recon_warmup_start, recon_warmup_length)

    # 1) IVIM signal fidelity
    ivim_mse = mse_fn(X_pred, X_true)

    # 2) Spatial reconstruction (with warmup)
    recon_loss = torch.tensor(0.0, device=device)
    n_recon = 0

    if recon_dict is not None:
        if recon_dict.get("t1") is not None and struct_patch is not None:
            recon_loss = recon_loss + mse_fn(recon_dict["t1"], struct_patch[:, 0:1])
            n_recon += 1
        if recon_dict.get("flair") is not None and struct_patch is not None:
            recon_loss = recon_loss + mse_fn(recon_dict["flair"], struct_patch[:, 1:2])
            n_recon += 1
        if recon_dict.get("b0") is not None and b0_patch is not None:
            recon_loss = recon_loss + mse_fn(recon_dict["b0"], b0_patch)
            n_recon += 1

    if n_recon > 0:
        recon_loss = recon_loss / n_recon
        # Cap recon loss to prevent explosion when spatial encoders shift
        # under dominant CE gradients (normal recon MSE << 1.0 on z-scored patches)
        recon_loss = torch.clamp(recon_loss, max=recon_cap)

    # 3) Tissue classification CE
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

    # 4) Soft upper-bound penalty for diffusion parameters
    # Note: when use_ordered_diffusion=True, Dpar < Dint < Dmv is guaranteed
    # by architecture, so no ordering penalty is needed — only upper bounds.
    diff_penalty = torch.tensor(0.0, device=device)
    if ivim_params is not None and alpha_penalty > 0:
        Dpar, Fint, Dint, Fmv, Dmv, S0 = ivim_params
        pen = torch.relu(Dpar - 0.0015) ** 2
        if Dint is not None:
            pen = pen + torch.relu(Dint - 0.0040) ** 2
        pen = pen + torch.relu(Dmv - 0.2000) ** 2
        diff_penalty = pen.mean()

    # Total
    total = (ivim_mse
             + alpha_recon_eff * recon_loss
             + alpha_ce * ce_loss
             + alpha_penalty * diff_penalty)

    # Debug
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
                def _stat(name, t, upper):
                    pct_over = (t > upper).float().mean().item() * 100
                    return f"{name}: {t.mean().item():.5f} (max={t.max().item():.5f}, >{upper}:{pct_over:.1f}%)"
                parts = [_stat("Dpar", Dpar, 0.0015)]
                if Dint is not None:
                    parts.append(_stat("Dint", Dint, 0.0040))
                parts.append(_stat("Dmv", Dmv, 0.2000))
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
        "tissue_acc": tissue_acc,
        "n_ce": n_ce,
    }


# ===========================================================
#  NETWORK COMPONENTS
# ===========================================================

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


# ===========================================================
#  CONCAT FUSION (drop-in alternative to cross-attention)
# ===========================================================

class ConcatFusion(nn.Module):
    """
    Drop-in replacement for nn.MultiheadAttention: concat + MLP.

    Instead of QKV attention, concatenates the query (z_ivim) with
    spatial tokens (z_anat, z_b0) and projects through a 2-layer MLP.
    Matches nn.MultiheadAttention's call signature so Net.forward()
    requires no changes.

    Input:  query (B, 1, D), key/value (B, N_mod, D)
    Output: (B, 1, D), None
    """

    def __init__(self, embed_dim, n_spatial_tokens=2, dropout=0.1):
        super().__init__()
        self.n_spatial = n_spatial_tokens
        input_dim = embed_dim * (1 + n_spatial_tokens)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
        )

    def forward(self, query, key, value, key_padding_mask=None,
                average_attn_weights=False):
        B, _, D = query.shape
        q = query.squeeze(1)                        # (B, D)

        # Pad or truncate spatial tokens to fixed size
        N = value.shape[1]
        if N < self.n_spatial:
            pad = value.new_zeros(B, self.n_spatial - N, D)
            v = torch.cat([value, pad], dim=1)
        else:
            v = value[:, :self.n_spatial]

        # Zero out masked modalities (modality dropout)
        if key_padding_mask is not None:
            M = key_padding_mask.shape[1]
            if M < self.n_spatial:
                pad_mask = key_padding_mask.new_ones(B, self.n_spatial - M)
                key_padding_mask = torch.cat([key_padding_mask, pad_mask], dim=1)
            mask = key_padding_mask[:, :self.n_spatial].unsqueeze(-1)
            v = v.masked_fill(mask, 0.0)

        v_flat = v.reshape(B, -1)                   # (B, n_spatial * D)
        combined = torch.cat([q, v_flat], dim=1)     # (B, (1+n_spatial) * D)

        out = self.proj(combined).unsqueeze(1)       # (B, 1, D)
        return out, None


# ===========================================================
#  MAIN NETWORK
# ===========================================================

class Net(nn.Module):
    """
    PACE: Per-parameter Anatomically Conditioned Estimation for IVIM.

    Cross-attention K,V contains spatial tokens only [z_anat, z_b0].
    Query is built from z_ivim + learned tissue token τ.
    Per-parameter gating (eq 5): z_X = (1-σ(g_X))·z_ivim + σ(g_X)·z_fused.
    Y-branch: separate projections for recon and tissue classification.
    """

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
                 use_ordered_diffusion=True,     # R21: cumulative softplus ordering
                 use_softmax_fractions=True,     # R21: softmax simplex (auto-off when spatial_on=False)
                 fusion_mode="attention",        # R33: "attention" (default) or "concat" (ConcatFusion)
                 recon_grad_clip=10.0):          # clamp recon gradient to [-val, +val] at token_recon (None disables)
        super().__init__()

        # ===========================================
        # Configuration
        # ===========================================
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

        # Softmax fractions require per-parameter gating (spatial_on=True)
        # to produce differentiated token inputs for Fpar/Fint/Fmv heads.
        # Without spatial gating all three heads see identical z_ivim,
        # so softmax degenerates — fall back to sigmoid+clamp.
        if not spatial_on and use_softmax_fractions:
            print("[Net] use_softmax_fractions forced OFF (spatial_on=False)")
            use_softmax_fractions = False
        self.use_softmax_fractions = use_softmax_fractions

        self.use_three_compartment = net_pars.use_three_compartment
        self.fitS0 = net_pars.fitS0

        if spatial_on and not parallel_heads:
            raise ValueError("Head-wise fusion requires parallel_heads=True.")

        if net_pars.width is None:
            self.net_pars.width = self.nbvals

        # IVIM parameter dimensionality
        if self.use_three_compartment:
            self.ivim_dim = 6 if self.fitS0 else 5
        else:
            self.ivim_dim = 4 if self.fitS0 else 3

        # Constraint buffers
        if self.use_three_compartment:
            bound_indices = [0, 1, 2, 3, 4, 5] if self.fitS0 else [0, 1, 2, 3, 4]
        else:
            bound_indices = [0, 3, 4, 5] if self.fitS0 else [0, 3, 4]

        cons_min = [net_pars.cons_min[i] for i in bound_indices]
        cons_max = [net_pars.cons_max[i] for i in bound_indices]
        self.register_buffer('cons_min_vec', torch.tensor(cons_min, dtype=torch.float32))
        self.register_buffer('cons_max_vec', torch.tensor(cons_max, dtype=torch.float32))

        D = ivim_latent_dim

        # ===========================================
        # Modality Encoders (all project to D)
        # ===========================================
        self.ivim_encoder = self._make_encoder(self.nbvals, D)
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
                nn.Conv2d(1, 32, 3, padding=1), nn.ELU(),
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

        # ===========================================
        # Fusion Module (spatial-only K,V) + Y-Branch
        #   fusion_mode="attention" → nn.MultiheadAttention (default)
        #   fusion_mode="concat"   → ConcatFusion (R33 ablation)
        # ===========================================
        if spatial_on:
            self.tissue_token = nn.Parameter(torch.randn(1, 1, D) * 0.02)
            self.query_proj = nn.Linear(D, D)
            if fusion_mode == "concat":
                self.cross_attn = ConcatFusion(
                    D, n_spatial_tokens=2, dropout=latent_dropout_p)
            else:
                self.cross_attn = nn.MultiheadAttention(
                    embed_dim=D, num_heads=n_attn_heads,
                    dropout=latent_dropout_p, batch_first=True
                )
            self.n_attn_heads = n_attn_heads
            self.attn_norm = nn.LayerNorm(D)
            self.attn_scale = nn.Parameter(torch.tensor(0.1))

            # Per-parameter spatial gates
            inits = {"Dpar": -5.0, "Fmv": -2.0, "Dmv": -2.0,
                     "S0": -5.0, "Fint": -5.0, "Dint": -5.0}
            if gate_inits is not None:
                inits.update(gate_inits)

            self.spatial_gate_dpar_raw = nn.Parameter(torch.tensor(float(inits["Dpar"])))
            self.spatial_gate_fmv_raw  = nn.Parameter(torch.tensor(float(inits["Fmv"])))
            self.spatial_gate_dmv_raw  = nn.Parameter(torch.tensor(float(inits["Dmv"])))
            self.spatial_gate_s0_raw   = nn.Parameter(torch.tensor(float(inits["S0"])))

            if self.net_pars.use_three_compartment:
                self.spatial_gate_fint_raw = nn.Parameter(torch.tensor(float(inits["Fint"])))
                self.spatial_gate_dint_raw = nn.Parameter(torch.tensor(float(inits["Dint"])))
            else:
                self.spatial_gate_fint_raw = None
                self.spatial_gate_dint_raw = None

            # Y-branch
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
            self.spatial_gate_fmv_raw = None
            self.spatial_gate_dmv_raw = None
            self.spatial_gate_s0_raw = None
            self.max_spatial_mods = 0
            self.spatial_concat_dim = 0
            self.spatial_recon_proj = None
            self.spatial_ce_proj = None

        # ===========================================
        # Downstream Decoders — Physics Heads
        # ===========================================
        if parallel_heads:
            self.Dpar_head = nn.Linear(D, 1)
            self.Dmv_head = nn.Linear(D, 1)
            self.S0_head = nn.Linear(D, 1) if self.fitS0 else None
            if self.use_three_compartment:
                self.Dint_head = nn.Linear(D, 1)
                self.Fint_head = nn.Linear(D, 1)
                self.Fmv_head = nn.Linear(D, 1)
                # R21: Fpar head for softmax simplex
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

        # Recon decoders
        if spatial_on and use_struct and recon_struct:
            self.t1_decoder = MiniDecoder(D, 1, patch_size)
            self.flair_decoder = MiniDecoder(D, 1, patch_size)
        else:
            self.t1_decoder = None
            self.flair_decoder = None

        if spatial_on and use_b0 and recon_b0:
            self.b0_decoder = MiniDecoder(D, 1, patch_size)
        else:
            self.b0_decoder = None

        # Tissue classifier
        self.tissue_classifier = nn.Sequential(
            nn.Linear(D, D), nn.ELU(), nn.Dropout(0.1),
            nn.Linear(D, NUM_TISSUE_CLASSES)
        )

    # ===========================================
    # Helpers
    # ===========================================

    def _make_encoder(self, in_dim, out_dim):
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

    def _predict_signal_standard(self, params, bvals):
        Dpar, Fint, Dint, Fmv, Dmv, S0 = params
        if self.use_three_compartment:
            return S0 * (
                (1 - Fmv - Fint) * torch.exp(-bvals * Dpar) +
                Fint * torch.exp(-bvals * Dint) +
                Fmv * torch.exp(-bvals * Dmv)
            )
        return S0 * ((1 - Fmv) * torch.exp(-bvals * Dpar) + Fmv * torch.exp(-bvals * Dmv))

    def _predict_signal_ir(self, params, bvals, device):
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

    # ===========================================
    # Forward
    # ===========================================

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

        # =============================================
        # IVIM encoding
        # =============================================
        z_ivim = self.norm_ivim(self.ivim_encoder(ivim_signal))

        # =============================================
        # Fusion path (spatial) or DNN bypass
        # =============================================
        attn_weights = None
        key_padding_mask = None
        modality_names = []
        token_recon = None
        token_ce = None

        if self.spatial_on and self.cross_attn is not None:
            z_ivim_q = z_ivim.detach() if self.detach_spatial_delta else z_ivim

            # K,V sequence: spatial tokens only (z_ivim enters via query)
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

            # Guard: if no spatial tokens available, fall through to DNN bypass
            if len(sequence) == 0:
                token_dpar = token_fmv = token_dmv = token_s0 = z_ivim
                token_fint = token_dint = z_ivim if self.use_three_compartment else None
                z_fused = z_ivim
                gate_vals = {}

            else:
                kv_sequence = torch.stack(sequence, dim=1)
                N_mod = kv_sequence.size(1)
                D = self.ivim_latent_dim

                # Modality dropout (spatial tokens only — always keep at least one)
                if self.training and self.modality_dropout_p > 0 and N_mod > 1:
                    mask = torch.rand(B, N_mod, device=device) < self.modality_dropout_p

                    all_masked = mask.all(dim=1)
                    if all_masked.any():
                        pick = torch.randint(0, N_mod, (B,), device=device)
                        mask[all_masked, pick[all_masked]] = False

                    key_padding_mask = mask
                else:
                    key_padding_mask = None

                # =============================================
                # Y-BRANCH: Token split
                # =============================================
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
                    # Bound recon gradient flowing back into spatial encoders to
                    # [-recon_grad_clip, +recon_grad_clip] (value-clip, not norm).
                    # This is a hook on the single entry point of the recon path,
                    # so it catches all recon-loss gradient before it touches
                    # spatial_recon_proj and the T1/FLAIR/B0 encoders, while
                    # leaving IVIM MSE, CE, and penalty gradients untouched.
                    if self.recon_grad_clip is not None and token_recon.requires_grad:
                        _clip = float(self.recon_grad_clip)
                        token_recon.register_hook(lambda g, c=_clip: g.clamp(-c, c))
                    token_ce = self.spatial_ce_proj(spatial_concat)

                # =============================================
                # Fusion: Q from z_ivim, K/V from spatial tokens
                #   attention mode → nn.MultiheadAttention (QKV)
                #   concat mode   → ConcatFusion (concat + MLP)
                # =============================================
                query = self.query_proj(z_ivim_q).unsqueeze(1) + self.tissue_token
                token_out, attn_weights = self.cross_attn(
                    query=query, key=kv_sequence, value=kv_sequence,
                    key_padding_mask=key_padding_mask,
                    average_attn_weights=False,
                )

                # =============================================
                # PER-PARAMETER SPATIAL GATING — eq (5)
                #   z_X = (1 - σ(g_X)) · z_ivim + σ(g_X) · z_fused
                # =============================================
                z_fused = self.attn_norm(z_ivim_q + self.attn_scale * token_out.squeeze(1))

                gate_vals = {}
                def _gate(name, raw_param):
                    g = torch.sigmoid(raw_param)
                    gate_vals[name] = g.item()
                    return (1.0 - g) * z_ivim + g * z_fused

                token_dpar = _gate("Dpar", self.spatial_gate_dpar_raw)
                token_fmv  = _gate("Fmv",  self.spatial_gate_fmv_raw)
                token_dmv  = _gate("Dmv",  self.spatial_gate_dmv_raw)
                token_s0   = _gate("S0",   self.spatial_gate_s0_raw)

                if self.use_three_compartment:
                    token_fint = _gate("Fint", self.spatial_gate_fint_raw)
                    token_dint = _gate("Dint", self.spatial_gate_dint_raw)
                else:
                    token_fint = token_dint = None

        else:
            token_dpar = token_fmv = token_dmv = token_s0 = z_ivim
            token_fint = token_dint = z_ivim if self.use_three_compartment else None
            z_fused = z_ivim
            gate_vals = {}

        # =============================================
        # Physics heads
        # =============================================
        temp = self.sigmoid_temperature
        gmin = self.cons_min_vec
        gmax = self.cons_max_vec

        def constrain_fraction(logit, idx):
            return gmin[idx] + torch.sigmoid(torch.nan_to_num(logit) / temp) * (gmax[idx] - gmin[idx])

        def constrain_diffusion(logit, idx):
            return gmin[idx] + F.softplus(torch.nan_to_num(logit))

        if self.parallel_heads:
            if self.use_three_compartment:

                # ── Diffusivities ──
                if self.use_ordered_diffusion:
                    # R21: Hard cumulative ordering
                    # Dpar = floor + softplus(raw)
                    # Dint = Dpar  + softplus(raw)  → strictly > Dpar
                    # Dmv  = Dint  + softplus(raw)  → strictly > Dint
                    Dpar = gmin[0] + F.softplus(torch.nan_to_num(self.Dpar_head(token_dpar)))
                    Dint = Dpar    + F.softplus(torch.nan_to_num(self.Dint_head(token_dint)))
                    Dmv  = Dint    + F.softplus(torch.nan_to_num(self.Dmv_head(token_dmv)))
                else:
                    # Legacy: independent softplus (R0-R20)
                    Dpar = constrain_diffusion(self.Dpar_head(token_dpar), 0)
                    Dint = constrain_diffusion(self.Dint_head(token_dint), 2)
                    Dmv  = constrain_diffusion(self.Dmv_head(token_dmv), 4)

                # ── Fractions ──
                if self.use_softmax_fractions:
                    # R21: Softmax simplex — Fpar + Fint + Fmv = 1 exactly
                    # Each head outputs an unconstrained logit.
                    # Fpar_head uses the signal token (token_dpar) since Fpar
                    # is dominated by signal, not spatial context.
                    logit_par = self.Fpar_head(token_dpar)
                    logit_int = self.Fint_head(token_fint)
                    logit_mv  = self.Fmv_head(token_fmv)
                    logits_cat = torch.cat([logit_par, logit_int, logit_mv], dim=1)
                    fracs = F.softmax(logits_cat / temp, dim=1)
                    # fracs[:, 0] = Fpar (implicit, not stored)
                    Fint = fracs[:, 1:2]
                    Fmv  = fracs[:, 2:3]
                else:
                    # Legacy: independent sigmoid + clamp (R0-R20)
                    f_int_raw = constrain_fraction(self.Fint_head(token_fint), 1)
                    f_mv_raw  = constrain_fraction(self.Fmv_head(token_fmv), 3)
                    sf = torch.clamp(f_mv_raw + f_int_raw, min=1.0)
                    Fint = f_int_raw / sf
                    Fmv  = f_mv_raw / sf

                S0 = constrain_fraction(self.S0_head(token_s0), 5) if self.fitS0 else torch.ones_like(Dpar)

            else:
                # 2-compartment: unchanged
                Dpar = constrain_diffusion(self.Dpar_head(token_dpar), 0)
                Fmv  = constrain_fraction(self.Fmv_head(token_fmv), 1)
                Dmv  = constrain_diffusion(self.Dmv_head(token_dmv), 2)
                S0   = constrain_fraction(self.S0_head(token_s0), 3) if self.fitS0 else torch.ones_like(Dpar)
                Fint, Dint = None, None

        else:
            # Non-parallel fallback
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

        # =============================================
        # Recon decoders
        # =============================================
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

        # =============================================
        # Tissue classifier
        # =============================================
        if self.spatial_on and token_ce is not None:
            tissue_logits = self.tissue_classifier(token_ce)
        else:
            tissue_logits = None

        # =============================================
        # Signal prediction
        # =============================================
        bvals_t = self.bvals.view(1, -1)
        if self.net_pars.IR:
            X_pred = self._predict_signal_ir(params, bvals_t, device)
        else:
            X_pred = self._predict_signal_standard(params, bvals_t)
        X_pred = torch.nan_to_num(X_pred, nan=0.0, posinf=0.0, neginf=0.0)

        # =============================================
        # Return
        # =============================================
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
