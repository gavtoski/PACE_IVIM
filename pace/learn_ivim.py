#!/usr/bin/env python3
# pace.learn_ivim.py
# Bin Hoang | University of Rochester | Feb 2026
# Training for Tissue Token IVIM — Per-Parameter Gating + Y-Branch.
#
# Changes (Mar 12 2026):
#   - Added use_ordered_diffusion and use_softmax_fractions pass-through
#     to Net() for R21 identifiability fix. Default True.
#     use_softmax_fractions auto-disabled when spatial_on=False (DNN mode).
#
# Changes (Mar 1 2026):
#   - CE warmup: ce_warmup_start/ce_warmup_length for delayed CE activation
#     (R9_CEWarmup uses ce_warmup_start=10 to let signal converge first)
#
# Changes (Feb 27 2026):
#   - Per-parameter spatial gating (6 gates: Dpar, Fint, Dint, Fmv, Dmv, S0)
#   - detach_spatial_delta: isolates IVIM encoder from attention gradients
#   - Separate param groups (IVIM slow, spatial fast) via pace.training_utils
#   - Delayed cosine scheduler (warmup → hold → decay)
#   - freeze_spatial_gates() replaces old freeze_perf_gate hack
#   - Tracks all 6 gate values in history
#
# Previous changes (Feb 25 2026):
#   - Static sqrt-inverse-freq CE weights (dataset-level, no batch thrash)
#   - Recon warmup: CE active from epoch 0, recon ramps in after warmup_start
#   - Updated default alphas: alpha_ce=0.2, alpha_recon=0.05

import sys, os, copy, time, argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from pace.deep_models import (
    Net, IVIMDataset, clean_ivim_signals,
    custom_loss_function, tissue_lesion_to_class,
    compute_static_ce_weights,
)
from pace.hyperparams import net_pars
from pace.training_utils import build_param_groups, build_scheduler


def learn_IVIM(
    train_h5, val_h5, bvals,
    seed=0, n_epochs=10, batch_size=128, lr=1e-4,
    out_dir="./trained_nets",
    spatial_on=True, use_struct=True, use_b0=True,
    use_three_compartment=True, fitS0=False,
    parallel_heads=True,
    recon_struct=True, recon_b0=True, recon_all=True,
    ivim_latent_dim=32, latent_dropout_p=0.3,
    alpha_ce=0.2, alpha_recon=0.05, alpha_penalty=0.1,
    modality_dropout_p=0.15,
    noise_level=25.0,
    train_dataset=None, val_dataset=None,
    pretrained_path=None,
    run_tag=None,
    test_mode=False, max_iter=None,
    detach_recon=False,
    detach_spatial_delta=True,
    use_anat_token=True,
    recon_warmup_start=5,
    recon_warmup_length=5,
    ce_warmup_start=None,
    ce_warmup_length=5,
    ce_class_weights=None,
    gate_inits=None,
    # --- R21 identifiability fix (March 12 2026) ---
    use_ordered_diffusion=True,     # cumulative softplus: Dpar < Dint < Dmv
    use_softmax_fractions=True,     # softmax simplex (auto-off when spatial_on=False)
    # --- R33 fusion ablation ---
    fusion_mode="attention",        # "attention" (default) or "concat" (ConcatFusion)
    # --- Gate freezing (replaces old freeze_perf_gate) ---
    freeze_gates_value=None,
    freeze_gates_subset=None,
    # --- Training dynamics ---
    lr_ivim=None,
    lr_spatial=None,
    scheduler_delay=20,
    scheduler_warmup=5,
    early_stop_start=None,
    # --- Legacy compat ---
    freeze_perf_gate=False,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    inject_noise = noise_level is not None and float(noise_level) > 0.0

    # Default LRs: spatial 5x faster than IVIM
    if lr_ivim is None:
        lr_ivim = lr
    if lr_spatial is None:
        lr_spatial = lr * 5

    # Don't early-stop before aux losses have time to converge
    if early_stop_start is None:
        early_stop_start = scheduler_delay

    print(f"[INIT] seed={seed} | {device} | spatial={spatial_on} | "
          f"ce={alpha_ce} recon={alpha_recon} penalty={alpha_penalty} | "
          f"noise={'SNR=' + str(int(noise_level)) + 'dB' if inject_noise else 'OFF'} | "
          f"attn_drop={latent_dropout_p} | "
          f"recon_warmup=ep{recon_warmup_start}+{recon_warmup_length} | "
          f"ce_warmup={'ep'+str(ce_warmup_start)+'+'+str(ce_warmup_length) if ce_warmup_start is not None else 'OFF'} | "
          f"{'DNN bypass (no attention)' if not spatial_on else 'per-param gating + Y-branch'} | "
          f"detach_recon={detach_recon} | detach_delta={detach_spatial_delta} | "
          f"anat_token={use_anat_token} | "
          f"ordered_diff={use_ordered_diffusion} | softmax_frac={use_softmax_fractions} | "
          f"lr_ivim={lr_ivim:.1e} lr_spatial={lr_spatial:.1e} | "
          f"sched_delay={scheduler_delay} | early_stop@{early_stop_start}")

    # Mode tag for filenames
    if run_tag is not None:
        mode = run_tag
    else:
        mode = "DNN" if not spatial_on else "perparam_ybranch"
        if recon_all: mode += "_reconall"
        if parallel_heads: mode += "_parallel"

    if test_mode:
        max_iter = 3000
    elif max_iter is None:
        max_iter = 10000

    if recon_all:
        recon_struct = recon_b0 = True

    # --- Datasets ---
    owns_datasets = False
    if train_dataset is not None and val_dataset is not None:
        train_set, val_set = train_dataset, val_dataset
    else:
        train_set = IVIMDataset(train_h5, use_struct=use_struct, use_b0=use_b0, use_2d_struct=True)
        val_set = IVIMDataset(val_h5, use_struct=use_struct, use_b0=use_b0, use_2d_struct=True)
        owns_datasets = True

    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    valloader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

    # --- Static CE class weights (dataset-level) ---
    if ce_class_weights is not None:
        # Pre-computed weights passed in (e.g. from pooled in-vivo subjects)
        if not isinstance(ce_class_weights, torch.Tensor):
            ce_class_weights = torch.tensor(ce_class_weights, dtype=torch.float32)
        ce_class_weights = ce_class_weights.to(device)
        print(f"[CE WEIGHTS] Pre-computed: {ce_class_weights.tolist()}")
    elif spatial_on:
        h5_paths = [train_h5]
        if val_h5 is not None:
            h5_paths.append(val_h5)
        try:
            w_np, counts = compute_static_ce_weights(h5_paths)
            ce_class_weights = torch.tensor(w_np, dtype=torch.float32, device=device)
            print(f"[CE WEIGHTS] Static sqrt-inv-freq: "
                  f"GM={w_np[0]:.3f} NAWM={w_np[1]:.3f} WMH={w_np[2]:.3f} | "
                  f"counts: GM={counts[0]} NAWM={counts[1]} WMH={counts[2]}")
        except Exception as e:
            print(f"[CE WEIGHTS] Failed to compute static weights: {e}")
            print(f"[CE WEIGHTS] Falling back to batch-level weights")

    # --- Model ---
    net_params = net_pars()
    net_params.use_three_compartment = use_three_compartment
    net_params.fitS0 = fitS0
    net_params.device = device

    net = Net(
        bvals=bvals, net_pars=net_params, patch_size=3,
        spatial_on=spatial_on, parallel_heads=parallel_heads,
        recon_struct=recon_struct, recon_b0=recon_b0,
        use_struct=use_struct, use_b0=use_b0,
        ivim_latent_dim=ivim_latent_dim,
        latent_dropout_p=latent_dropout_p,
        modality_dropout_p=modality_dropout_p,
        detach_recon=detach_recon,
        detach_spatial_delta=detach_spatial_delta,
        use_anat_token=use_anat_token,
        gate_inits=gate_inits,
        use_ordered_diffusion=use_ordered_diffusion,
        use_softmax_fractions=use_softmax_fractions,
        fusion_mode=fusion_mode,
    ).to(device)

    # Debug: verify constraint bounds are correctly indexed
    print(f"[BOUNDS] cons_min_vec={net.cons_min_vec.tolist()}")
    print(f"[BOUNDS] cons_max_vec={net.cons_max_vec.tolist()}")
    print(f"[BOUNDS] source len: cons_min={len(net_params.cons_min)}, cons_max={len(net_params.cons_max)}")
    if use_ordered_diffusion or use_softmax_fractions:
        print(f"[R21] ordered_diffusion={use_ordered_diffusion} | softmax_fractions={use_softmax_fractions}")
    if fusion_mode != "attention":
        print(f"[R33] fusion_mode={fusion_mode}")

    # --- Gate freezing (new API) ---
    # Legacy compat: old freeze_perf_gate → freeze all gates at 0
    if freeze_perf_gate and freeze_gates_value is None:
        print("[COMPAT] freeze_perf_gate=True → freeze_gates_value=0.0 (all gates)")
        freeze_gates_value = 0.0

    if pretrained_path is not None:
        print(f"[TRANSFER] Loading {pretrained_path}")
        try:
            state = torch.load(pretrained_path, map_location=device, weights_only=True)
        except TypeError:
            state = torch.load(pretrained_path, map_location=device)
        missing, unexpected = net.load_state_dict(state, strict=False)
        if missing: print(f"[TRANSFER] Missing: {len(missing)} keys")
        if unexpected: print(f"[TRANSFER] Unexpected: {len(unexpected)} keys")

    # Gate freezing AFTER pretrained load — otherwise loaded weights overwrite frozen values
    if freeze_gates_value is not None and spatial_on:
        net.freeze_spatial_gates(
            value=freeze_gates_value,
            subset=freeze_gates_subset,
        )

    # --- Optimizer: separate param groups ---
    optimizer = optim.AdamW(
        build_param_groups(net, lr_ivim=lr_ivim, lr_spatial=lr_spatial),
        weight_decay=1e-5,
    )

    # --- Scheduler: delayed cosine ---
    scheduler = build_scheduler(
        optimizer,
        total_epochs=n_epochs,
        warmup_epochs=scheduler_warmup,
        delay_epochs=scheduler_delay,
        min_lr_fraction=0.01,
    )

    print(f"[INIT] {sum(p.numel() for p in net.parameters())/1e6:.2f}M params | "
          f"{n_epochs} ep | max_iter {max_iter}")

    best_val = float("inf")
    best_model = copy.deepcopy(net.state_dict())
    patience_counter = 0
    start = time.time()

    # --- CE warmup ramp (mirrors recon warmup logic) ---
    def _ce_ramp(epoch):
        """Returns multiplier [0,1] for alpha_ce based on CE warmup schedule."""
        if ce_warmup_start is None:
            return 1.0
        if epoch < ce_warmup_start:
            return 0.0
        return min(1.0, (epoch - ce_warmup_start) / max(1, ce_warmup_length))

    # --- History dict ---
    gate_names = ["Dpar", "Fmv", "Dmv", "S0"]
    if use_three_compartment:
        gate_names.extend(["Fint", "Dint"])

    H = {k: [] for k in [
        "train_mse", "val_mse", "lr_ivim", "lr_spatial",
        "train_recon", "train_ce", "train_tissue_acc", "train_diff_penalty",
        "val_recon", "val_ce", "val_tissue_acc", "val_diff_penalty",
        "attn_scale",
    ]}
    # Add per-gate tracking
    for gn in gate_names:
        H[f"gate_{gn}"] = []

    # ======================== TRAINING ========================
    for ep in range(n_epochs):
        net.train()
        run_mse, run_recon, run_ce, run_acc, run_pen = 0., 0., 0., 0., 0.
        n_samples, n_ce_samples = 0, 0

        for i, batch in enumerate(tqdm(
                trainloader, total=min(len(trainloader), max_iter),
                desc=f"Ep {ep+1}", leave=False)):
            if i >= max_iter:
                break

            SP_in = batch["struct_patch"].to(device) if (spatial_on and use_struct) else None
            B0_in = batch["b0_patch"].to(device) if (spatial_on and use_b0) else None

            X, SP_clean, B0_clean, vm = clean_ivim_signals(
                batch["x_true"].to(device), SP_in, B0_in, bvals,
                inject_noise=inject_noise,
                noise_level=(noise_level if inject_noise else None))

            SP = SP_clean if spatial_on else None
            B0 = B0_clean if spatial_on else None
            if X.shape[0] < 2:
                continue

            tissue_labels, tissue_valid = tissue_lesion_to_class(
                batch["tissue"].to(device)[vm], batch["lesion"].to(device)[vm])

            optimizer.zero_grad()
            out = net(ivim_true_signal=X, structural_patch=SP, b0_patch=B0)
            alpha_ce_eff = alpha_ce * _ce_ramp(ep)
            ld = custom_loss_function(
                X_pred=out["X_pred"], X_true=X,
                ivim_params=out["ivim_params"],
                recon_dict=out["recon"], latent_dict=out["latent"],
                struct_patch=SP, b0_patch=B0,
                tissue_labels=tissue_labels, tissue_valid=tissue_valid,
                alpha_recon=alpha_recon, alpha_ce=alpha_ce_eff,
                alpha_penalty=alpha_penalty,
                ce_class_weights=ce_class_weights,
                recon_warmup_start=recon_warmup_start,
                recon_warmup_length=recon_warmup_length,
                epoch=ep)

            loss = ld["total"]
            if torch.isnan(loss):
                print("[ERROR] NaN loss")
                sys.exit(1)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            optimizer.step()

            bs, nce = X.size(0), ld["n_ce"].item()
            run_mse += ld["ivim_mse"].item() * bs
            run_recon += ld["recon_loss"].item() * bs
            run_ce += ld["ce_loss"].item() * nce
            run_acc += ld["tissue_acc"].item() * nce
            run_pen += ld["diff_penalty"].item() * bs
            n_samples += bs
            n_ce_samples += nce

        ns = max(1, n_samples)
        nc = max(1, n_ce_samples)
        H["train_mse"].append(run_mse / ns)
        H["train_recon"].append(run_recon / ns)
        H["train_ce"].append(run_ce / nc)
        H["train_tissue_acc"].append(run_acc / nc)
        H["train_diff_penalty"].append(run_pen / ns)

        # ======================== VALIDATION ========================
        net.eval()
        v_mse, v_rec, v_ce, v_acc, v_pen = 0., 0., 0., 0., 0.
        v_n, v_nc = 0, 0

        with torch.no_grad():
            for j, batch in enumerate(valloader):
                if j >= max_iter // 2:
                    break

                SP_in = batch["struct_patch"].to(device) if (spatial_on and use_struct) else None
                B0_in = batch["b0_patch"].to(device) if (spatial_on and use_b0) else None

                X, SP_clean, B0_clean, vm = clean_ivim_signals(
                    batch["x_true"].to(device), SP_in, B0_in, bvals,
                    inject_noise=False)

                SP = SP_clean if spatial_on else None
                B0 = B0_clean if spatial_on else None
                if X.shape[0] < 2:
                    continue

                tissue_labels, tissue_valid = tissue_lesion_to_class(
                    batch["tissue"].to(device)[vm], batch["lesion"].to(device)[vm])

                out = net(ivim_true_signal=X, structural_patch=SP, b0_patch=B0)
                alpha_ce_eff = alpha_ce * _ce_ramp(ep)
                ld = custom_loss_function(
                    X_pred=out["X_pred"], X_true=X,
                    ivim_params=out["ivim_params"],
                    recon_dict=out["recon"], latent_dict=out["latent"],
                    struct_patch=SP, b0_patch=B0,
                    tissue_labels=tissue_labels, tissue_valid=tissue_valid,
                    alpha_recon=alpha_recon, alpha_ce=alpha_ce_eff,
                    alpha_penalty=alpha_penalty,
                    ce_class_weights=ce_class_weights,
                    recon_warmup_start=recon_warmup_start,
                    recon_warmup_length=recon_warmup_length,
                    epoch=ep)

                bs, nce = X.size(0), ld["n_ce"].item()
                v_mse += ld["ivim_mse"].item() * bs
                v_rec += ld["recon_loss"].item() * bs
                v_ce += ld["ce_loss"].item() * nce
                v_acc += ld["tissue_acc"].item() * nce
                v_pen += ld["diff_penalty"].item() * bs
                v_n += bs
                v_nc += nce

        val_ivim = float("inf") if v_n == 0 else v_mse / v_n
        vns = max(1, v_n)
        vnc = max(1, v_nc)
        H["val_mse"].append(val_ivim)
        H["val_recon"].append(v_rec / vns)
        H["val_ce"].append(v_ce / vnc)
        H["val_tissue_acc"].append(v_acc / vnc)
        H["val_diff_penalty"].append(v_pen / vns)

        # Track per-group LRs
        lr_dict = {g.get("label", "?"): g["lr"] for g in optimizer.param_groups}
        H["lr_ivim"].append(lr_dict.get("ivim_encoder", lr_ivim))
        H["lr_spatial"].append(lr_dict.get("spatial_encoders", lr_spatial))

        # Track attn_scale and per-parameter gates
        cur_attn_scale = net.attn_scale.item() if net.attn_scale is not None else 0.0
        H["attn_scale"].append(cur_attn_scale)

        cur_gates = {}
        for gn in gate_names:
            attr = getattr(net, f"spatial_gate_{gn.lower()}_raw", None)
            if attr is not None:
                gval = torch.sigmoid(attr).item()
            else:
                gval = 0.0
            cur_gates[gn] = gval
            H[f"gate_{gn}"].append(gval)

        # --- Scheduler step (cosine, epoch-based) ---
        scheduler.step()

        # --- Early stopping on val_mse ---
        star = ""
        if val_ivim < best_val:
            best_val = val_ivim
            best_model = copy.deepcopy(net.state_dict())
            patience_counter = 0
            star = " *"
        elif ep >= early_stop_start:
            patience_counter += 1

        # --- Epoch summary ---
        gate_str = " ".join(f"{k}={v:.3f}" for k, v in cur_gates.items())
        print(f"  Ep {ep+1:3d} | MSE {H['train_mse'][-1]:.6f}/{val_ivim:.6f} | "
              f"Recon {H['train_recon'][-1]:.5f}/{H['val_recon'][-1]:.5f} | "
              f"CE {H['train_ce'][-1]:.4f}/{H['val_ce'][-1]:.4f} | "
              f"Acc {H['train_tissue_acc'][-1]:.3f}/{H['val_tissue_acc'][-1]:.3f} | "
              f"LR {H['lr_ivim'][-1]:.1e}/{H['lr_spatial'][-1]:.1e} | "
              f"aS {cur_attn_scale:.3f} | G[{gate_str}]{star}")

        if patience_counter >= net_params.patience:
            print(f"[STOP] No improvement for {net_params.patience} epochs.")
            break

    # ======================== SAVE ========================
    print(f"[DONE] {(time.time() - start)/60:.1f} min")

    if owns_datasets:
        train_set.close()
        val_set.close()

    net.load_state_dict(best_model)

    noise_tag = int(noise_level) if inject_noise else 0
    prefix = "test_" if test_mode else ""
    base = f"{prefix}{mode}_{seed}_noise{noise_tag}dB"

    torch.save(net.state_dict(), os.path.join(out_dir, f"trained_net_{base}.pt"))
    np.savez(os.path.join(out_dir, f"log_{base}.npz"),
             **{k: np.array(v, dtype=np.float32) for k, v in H.items()})

    # --- Summary row ---
    best_ep = int(np.argmin(H["val_mse"]))
    summary = {
        "run_tag": mode, "seed": seed, "noise": noise_tag,
        "best_epoch": best_ep,
        "best_val_mse": H["val_mse"][best_ep],
        "best_val_recon": H["val_recon"][best_ep],
        "best_val_ce": H["val_ce"][best_ep],
        "best_val_acc": H["val_tissue_acc"][best_ep],
        "best_val_penalty": H["val_diff_penalty"][best_ep],
        "final_attn_scale": H["attn_scale"][-1],
        "alpha_recon": alpha_recon, "alpha_ce": alpha_ce,
        "alpha_penalty": alpha_penalty,
        "spatial_on": spatial_on, "detach_recon": detach_recon,
        "detach_spatial_delta": detach_spatial_delta,
        "use_anat_token": use_anat_token,
        "use_ordered_diffusion": use_ordered_diffusion,
        "use_softmax_fractions": use_softmax_fractions,
        "recon_warmup_start": recon_warmup_start,
        "recon_warmup_length": recon_warmup_length,
        "ce_warmup_start": ce_warmup_start,
        "ce_warmup_length": ce_warmup_length,
    }
    # Add final gate values to summary
    for gn in gate_names:
        summary[f"final_gate_{gn}"] = H[f"gate_{gn}"][-1] if H[f"gate_{gn}"] else 0.0

    summary_path = os.path.join(out_dir, f"summary_{base}.npz")
    np.savez(summary_path, **{k: np.array(v) for k, v in summary.items()})

    gate_summary = " ".join(f"{gn}={summary[f'final_gate_{gn}']:.3f}" for gn in gate_names)
    print(f"\n[SUMMARY] {mode} | seed={seed} | best_ep={best_ep}")
    print(f"  val_mse={summary['best_val_mse']:.6f} | "
          f"val_recon={summary['best_val_recon']:.5f} | "
          f"val_ce={summary['best_val_ce']:.4f} | "
          f"val_acc={summary['best_val_acc']:.3f} | "
          f"attn_scale={summary['final_attn_scale']:.3f}")
    print(f"  Gates: {gate_summary}")

    # --- Loss curves ---
    ep_range = np.arange(len(H["train_mse"]))
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(f"{mode} | seed={seed} | noise={noise_tag}dB | penalty={alpha_penalty} | "
                 f"ce={alpha_ce} recon={alpha_recon}(warmup@{recon_warmup_start}) | "
                 f"ce_warmup={ce_warmup_start if ce_warmup_start is not None else 'OFF'} | "
                 f"detach_delta={detach_spatial_delta} | "
                 f"ord_diff={use_ordered_diffusion} sm_frac={use_softmax_fractions}", fontsize=10)

    for ax, kt, kv, title in [
        (axes[0, 0], "train_mse", "val_mse", "IVIM MSE"),
        (axes[0, 1], "train_recon", "val_recon", "Recon Loss"),
        (axes[0, 2], "train_ce", "val_ce", "CE Loss"),
        (axes[1, 0], "train_tissue_acc", "val_tissue_acc", "Tissue Acc"),
    ]:
        ax.plot(ep_range, H[kt], label="Train")
        ax.plot(ep_range, H[kv], label="Val", ls="--")
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.legend()

    # Mark recon warmup region
    axes[0, 1].axvspan(0, recon_warmup_start, alpha=0.1, color='gray', label='recon off')
    if recon_warmup_start + recon_warmup_length < len(ep_range):
        axes[0, 1].axvspan(recon_warmup_start, recon_warmup_start + recon_warmup_length,
                           alpha=0.1, color='orange', label='ramp')
    axes[0, 1].legend(fontsize=8)

    # CE warmup shading (if applicable)
    if ce_warmup_start is not None:
        axes[0, 2].axvspan(0, ce_warmup_start, alpha=0.1, color='gray', label='CE off')
        if ce_warmup_start + ce_warmup_length < len(ep_range):
            axes[0, 2].axvspan(ce_warmup_start, ce_warmup_start + ce_warmup_length,
                               alpha=0.1, color='orange', label='CE ramp')
        axes[0, 2].legend(fontsize=8)

    axes[1, 0].axhline(1/3, color='gray', ls=':', alpha=.5)

    # LR + Attn Scale
    axes[1, 1].plot(ep_range, H["lr_ivim"], color='teal', label="LR ivim")
    axes[1, 1].plot(ep_range, H["lr_spatial"], color='steelblue', ls='--', label="LR spatial")
    axes[1, 1].set_title("Learning Rates"); axes[1, 1].set_yscale('log')
    axes[1, 1].set_ylabel("LR"); axes[1, 1].set_xlabel("Epoch")
    ax_as = axes[1, 1].twinx()
    ax_as.plot(ep_range, H["attn_scale"], color='crimson', ls=':', label="attn_scale")
    ax_as.set_ylabel("attn_scale", color='crimson')
    ax_as.tick_params(axis='y', labelcolor='crimson')
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = ax_as.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=7)

    # Per-parameter gates
    gate_colors = {"Dpar": "navy", "Fint": "darkgreen", "Dint": "olive",
                   "Fmv": "red", "Dmv": "orange", "S0": "purple"}
    for gn in gate_names:
        c = gate_colors.get(gn, "gray")
        ls = "--" if gn in ("Dpar", "Fint", "Dint", "S0") else "-"
        axes[1, 2].plot(ep_range, H[f"gate_{gn}"], color=c, ls=ls, label=gn)
    axes[1, 2].set_title("Spatial Gates (sigmoid)")
    axes[1, 2].set_xlabel("Epoch"); axes[1, 2].set_ylabel("gate value")
    axes[1, 2].set_ylim(-0.05, 1.05)
    axes[1, 2].axhline(0.5, color='gray', ls=':', alpha=.3)
    axes[1, 2].legend(fontsize=7, ncol=2)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"curves_{base}.png"), dpi=150)
    plt.close()

    return net, H


# ======================== CLI ========================
if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train Tissue Token IVIM (Per-Param Gating + Y-Branch)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test_mode", action="store_true")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Base LR for IVIM encoder (spatial gets 5x)")
    p.add_argument("--lr_ivim", type=float, default=None,
                   help="Override IVIM encoder LR (default: same as --lr)")
    p.add_argument("--lr_spatial", type=float, default=None,
                   help="Override spatial/CE/gate LR (default: 5x --lr)")
    p.add_argument("--out_dir", default="./trained_nets")

    p.add_argument("--spatial_on", action="store_true")
    p.add_argument("--use_struct", action="store_true")
    p.add_argument("--use_b0", action="store_true")
    p.add_argument("--parallel_heads", action="store_true")
    p.add_argument("--reconall", action="store_true")
    p.add_argument("--latent_dim", type=int, default=32)

    p.add_argument("--alpha_ce", type=float, default=0.2)
    p.add_argument("--alpha_recon", type=float, default=0.05)
    p.add_argument("--alpha_penalty", type=float, default=0.1)
    p.add_argument("--modality_dropout", type=float, default=0.15)
    p.add_argument("--noise_level", type=float, default=25.0)
    p.add_argument("--pretrained_path", default=None)
    p.add_argument("--detach_recon", action="store_true",
                   help="Detach token before recon decoders (sanity check)")
    p.add_argument("--detach_spatial_delta", action="store_true", default=True,
                   help="Detach z_ivim in attention query/delta (default: True)")
    p.add_argument("--no_detach_spatial_delta", dest="detach_spatial_delta",
                   action="store_false",
                   help="Allow attention gradients to flow into IVIM encoder")
    p.add_argument("--use_anat_token", action="store_true", default=True,
                   help="Fuse T1+FLAIR into single anatomy token (default: True)")
    p.add_argument("--no_anat_token", dest="use_anat_token",
                   action="store_false",
                   help="Keep T1 and FLAIR as separate cross-attn tokens (ablation)")

    # R21 identifiability fix
    p.add_argument("--use_ordered_diffusion", action="store_true", default=True,
                   help="Cumulative softplus: Dpar < Dint < Dmv (R21, default ON)")
    p.add_argument("--no_ordered_diffusion", dest="use_ordered_diffusion",
                   action="store_false",
                   help="Disable ordered diffusion (legacy R0-R20 behavior)")
    p.add_argument("--use_softmax_fractions", action="store_true", default=True,
                   help="Softmax simplex: Fpar + Fint + Fmv = 1 (R21, default ON, auto-off for DNN)")
    p.add_argument("--no_softmax_fractions", dest="use_softmax_fractions",
                   action="store_false",
                   help="Disable softmax fractions (legacy R0-R20 behavior)")

    # R33 fusion ablation
    p.add_argument("--fusion_mode", default="attention",
                   choices=["attention", "concat"],
                   help="Fusion module: 'attention' (default, MultiheadAttention) "
                        "or 'concat' (ConcatFusion, R33 ablation)")

    p.add_argument("--run_tag", default=None,
                   help="Ablation name for filenames (e.g. FULL, NoCE, DNN)")
    p.add_argument("--recon_warmup_start", type=int, default=5,
                   help="Epoch at which recon loss begins ramping in")
    p.add_argument("--recon_warmup_length", type=int, default=5,
                   help="Number of epochs over which recon ramps from 0 to full")

    # CE warmup
    p.add_argument("--ce_warmup_start", type=int, default=None,
                   help="Epoch at which CE loss begins ramping in (None=active from ep 0)")
    p.add_argument("--ce_warmup_length", type=int, default=5,
                   help="Number of epochs over which CE ramps from 0 to full")

    # Gate initialization
    p.add_argument("--gate_inits", nargs="+", default=None,
                   help="Initialize specific gates, e.g., --gate_inits Fmv=-5.0 Dmv=-5.0")
    # Gate freezing
    p.add_argument("--freeze_gates_value", type=float, default=None,
                   help="Freeze spatial gates at this sigmoid value (0.0=closed, 1.0=open)")
    p.add_argument("--freeze_gates_subset", nargs="+", default=None,
                   help="Which gates to freeze (e.g. Dpar S0 Dint Fint). Default: all.")
    # Legacy compat
    p.add_argument("--freeze_perf_gate", action="store_true",
                   help="[LEGACY] Freeze ALL gates at 0 (same as --freeze_gates_value 0.0)")

    # Scheduler
    p.add_argument("--scheduler_delay", type=int, default=20,
                   help="Epoch when cosine decay begins (hold full LR until then)")
    p.add_argument("--scheduler_warmup", type=int, default=5,
                   help="Linear warmup epochs at start of training")
    p.add_argument("--early_stop_start", type=int, default=None,
                   help="Don't count early-stop patience before this epoch (default: scheduler_delay)")

    p.add_argument("--bval_path",
        default="/scratch/gschifit_lab/NeuroCovid/IVIM/IVIM3brain-NET-main/bval46.bval")
    p.add_argument("--train_h5",
        default="/scratch/gschifit_lab/NeuroCovid/IVIM/Training_Data_Sets/train_data_IVIM_CNN.h5")
    p.add_argument("--val_h5",
        default="/scratch/gschifit_lab/NeuroCovid/IVIM/Training_Data_Sets/test_data_IVIM_CNN.h5")

    a = p.parse_args()
    bvals = np.loadtxt(a.bval_path).astype(np.float32)

    # --- Parse gate_inits into a dictionary ---
    parsed_gate_inits = None
    if a.gate_inits is not None:
        parsed_gate_inits = {}
        for item in a.gate_inits:
            try:
                k, v = item.split('=')
                parsed_gate_inits[k] = float(v)
            except ValueError:
                print(f"[ERROR] --gate_inits must be in key=value format (e.g., Fmv=-5.0). Got: {item}")
                sys.exit(1)

    learn_IVIM(
        train_h5=a.train_h5, val_h5=a.val_h5, bvals=bvals,
        seed=a.seed, n_epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
        lr_ivim=a.lr_ivim, lr_spatial=a.lr_spatial,
        out_dir=a.out_dir, test_mode=a.test_mode,
        spatial_on=a.spatial_on, use_struct=a.use_struct, use_b0=a.use_b0,
        parallel_heads=a.parallel_heads,
        recon_all=a.reconall,
        ivim_latent_dim=a.latent_dim,
        modality_dropout_p=a.modality_dropout,
        alpha_ce=a.alpha_ce, alpha_recon=a.alpha_recon,
        alpha_penalty=a.alpha_penalty,
        noise_level=a.noise_level, pretrained_path=a.pretrained_path,
        run_tag=a.run_tag,
        detach_recon=a.detach_recon,
        detach_spatial_delta=a.detach_spatial_delta,
        use_anat_token=a.use_anat_token,
        use_ordered_diffusion=a.use_ordered_diffusion,
        use_softmax_fractions=a.use_softmax_fractions,
        fusion_mode=a.fusion_mode,
        recon_warmup_start=a.recon_warmup_start,
        recon_warmup_length=a.recon_warmup_length,
        ce_warmup_start=a.ce_warmup_start,
        ce_warmup_length=a.ce_warmup_length,
        gate_inits=parsed_gate_inits,
        freeze_gates_value=a.freeze_gates_value,
        freeze_gates_subset=a.freeze_gates_subset,
        freeze_perf_gate=a.freeze_perf_gate,
        scheduler_delay=a.scheduler_delay,
        scheduler_warmup=a.scheduler_warmup,
        early_stop_start=a.early_stop_start,
    )