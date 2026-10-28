#!/usr/bin/env python3

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_param_groups(model,
                        lr_ivim=1e-4,
                        lr_spatial=5e-4,
                        lr_gates=5e-4,
                        lr_heads=1e-4,
                        verbose=True):
    """Split model parameters into groups with different learning rates."""
    ivim_encoder_params = []
    physics_head_params = []
    spatial_encoder_params = []
    attention_params = []
    ybranch_params = []
    decoder_classifier_params = []
    gate_params = []
    other_params = []

    head_names = {"Dpar_head", "Fmv_head", "Dmv_head", "S0_head",
                  "Fint_head", "Dint_head", "param_head"}
    spatial_enc_prefixes = ("t1_cnn", "t1_fc", "norm_t1",
                            "flair_cnn", "flair_fc", "norm_flair",
                            "b0_cnn", "b0_fc", "norm_b0",
                            "anat_fusion")
    attn_names = ("cross_attn", "query_proj", "attn_norm")
    ybranch_names = ("spatial_recon_proj", "spatial_ce_proj")
    decoder_names = ("t1_decoder", "flair_decoder", "b0_decoder",
                     "tissue_classifier")

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        top_level = name.split(".")[0]

        if "spatial_gate" in name:
            gate_params.append(param)
        elif top_level in ("ivim_encoder", "norm_ivim"):
            ivim_encoder_params.append(param)
        elif top_level in head_names:
            physics_head_params.append(param)
        elif top_level.startswith(spatial_enc_prefixes):
            spatial_encoder_params.append(param)
        elif top_level.startswith(attn_names):
            attention_params.append(param)
        elif top_level.startswith(ybranch_names):
            ybranch_params.append(param)
        elif top_level.startswith(decoder_names):
            decoder_classifier_params.append(param)
        else:
            other_params.append(param)

    groups = []

    def _add(params, lr, label):
        if params:
            groups.append({"params": params, "lr": lr, "label": label})

    _add(ivim_encoder_params,       lr_ivim,    "ivim_encoder")
    _add(physics_head_params,       lr_heads,   "physics_heads")
    _add(spatial_encoder_params,    lr_spatial,  "spatial_encoders")
    _add(attention_params,          lr_spatial,  "cross_attention")
    _add(ybranch_params,            lr_spatial,  "y_branch_projs")
    _add(decoder_classifier_params, lr_spatial,  "decoders+classifier")
    _add(gate_params,               lr_gates,    "spatial_gates")
    _add(other_params,              lr_spatial,  "other")

    if verbose:
        total = 0
        print("\n[Param Groups]")
        for g in groups:
            n = sum(p.numel() for p in g["params"])
            total += n
            print(f"  {g['label']:25s}  lr={g['lr']:.1e}  params={n:,}")
        print(f"  {'TOTAL':25s}  params={total:,}\n")

    return groups


def build_scheduler(optimizer,
                    total_epochs=100,
                    warmup_epochs=5,
                    delay_epochs=20,
                    min_lr_fraction=0.01,
                    verbose=True):
    """Delayed cosine scheduler with per-group warmup."""
    import math

    assert delay_epochs >= warmup_epochs, \
        f"delay_epochs ({delay_epochs}) must be >= warmup_epochs ({warmup_epochs})"

    decay_epochs = total_epochs - delay_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return min_lr_fraction + (1.0 - min_lr_fraction) * (epoch / max(1, warmup_epochs))

        if epoch < delay_epochs:
            return 1.0

        progress = (epoch - delay_epochs) / max(1, decay_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_fraction + (1.0 - min_lr_fraction) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    if verbose:
        print(f"[Scheduler] warmup={warmup_epochs}ep → hold until ep{delay_epochs} "
              f"→ cosine decay over {decay_epochs}ep → min_lr={min_lr_fraction:.0%}")
        samples = [0, warmup_epochs, delay_epochs,
                   (delay_epochs + total_epochs) // 2, total_epochs - 1]
        for e in samples:
            if e < total_epochs:
                frac = lr_lambda(e)
                print(f"    ep{e:3d}: {frac:.4f}x")
        print()

    return scheduler


if __name__ == "__main__":
    print("=" * 60)
    print("  Training Utils — Integration Example")
    print("=" * 60)
