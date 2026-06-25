import os
import math
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import config
import dataset
from monai.metrics import SSIMMetric
from torch.amp import autocast
from helpers.eval_helpers import load_diffusion_model, build_ddpm_scheduler


def evaluate_numerical(num_inference_steps=1000, max_batches=None):
    """
    This function runs full reverse diffusion on the test set and saves MAE/bias/SSIM/PSNR 
    results
    """
    if config.CURRENT_EVAL == "residual_learning" and config.CURRENT_TRAINING != "residual_learning":
        raise ValueError(
            f"CURRENT_EVAL='residual_learning' but CURRENT_TRAINING='{config.CURRENT_TRAINING}'. "
            "Set CURRENT_TRAINING='residual_learning' in config.py before running."
        )

    device = config.DEVICE
    print("Loading data...")
    _, _, test_loader = dataset.get_dataloaders(include_test=True)
    print("Loading diffusion model...")
    dif_model, _ = load_diffusion_model()

    scheduler = build_ddpm_scheduler()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    ssim_metric = SSIMMetric(spatial_dims=2, data_range=1.0)
    all_mae, all_ssim, all_psnr, all_bias = [], [], [], []
    resume_from = 0

    # resume from a previous run's progress, if available
    ckpt_path = os.path.join(config.RESULTS_DIR, f"models_{config.CURRENT_EVAL}", "numerical_results_ckpt.json")
    if max_batches is None and os.path.exists(ckpt_path):
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        all_mae  = ckpt["all_mae"]
        all_ssim = ckpt["all_ssim"]
        all_psnr = ckpt["all_psnr"]
        all_bias = ckpt["all_bias"]
        resume_from = ckpt["next_batch"]
        print(f"Resuming from batch {resume_from} ({len(all_mae)} slices already processed)")

    n_batches = len(test_loader) if max_batches is None else min(max_batches, len(test_loader))

    print(f"\n--- evaluate_numerical: {n_batches} batches x {num_inference_steps} DDPM steps ---")

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if batch_idx < resume_from:
                continue
            if max_batches is not None and batch_idx >= max_batches:
                break
            if config.CURRENT_EVAL == "residual_learning":
                # use residual-map as a target, and get center middle slice for reconstruction
                inputs, _, pre_center, post = batch
                pre_center = pre_center.to(device)
                post = post.to(device)
            else:
                # use direct post-contrast as a target
                inputs, _, _, post = batch
                pre_center = inputs[:, 1:2].to(device)
                post = post.to(device)

            inputs = inputs.to(device)
            B, _, H, W = inputs.shape

            torch.manual_seed(batch_idx)
            sample = torch.randn(B, 1, H, W, device=device) # start from random noise

            print(f"  batch {batch_idx+1}/{n_batches}", flush=True)
            for step_idx, t in enumerate(scheduler.timesteps):
                if step_idx % 200 == 0:
                    print(f"    step {step_idx}/{num_inference_steps}", flush=True)
                ts = torch.full((B,), t, device=device, dtype=torch.long)
                with autocast("cuda", enabled=(device.type == "cuda")):
                    noise_pred = dif_model(torch.cat([sample, inputs], dim=1), ts) # predict noise
                sample = scheduler.step(noise_pred.float(), t, sample)[0] # reverse diffusion step

            if config.CURRENT_EVAL == "residual_learning":
                # reconstruct post-contrast image from generated residual
                pred = pre_center + sample * config.RESIDUAL_SCALE
            else:
                pred = sample

            # calculate per-slice numerical evaluation metrics
            mae_per  = torch.mean(torch.abs(pred - post), dim=[1, 2, 3]).tolist()
            bias_per = torch.mean(pred - post, dim=[1, 2, 3]).tolist()
            mse_per  = torch.mean((pred - post) ** 2, dim=[1, 2, 3]).tolist()
            psnr_per = [20 * math.log10(1.0 / math.sqrt(m)) if m > 0 else 100.0 for m in mse_per]
            for b in range(pred.shape[0]):
                ssim_val = ssim_metric(pred[b:b+1], post[b:b+1]).mean().item()
                ssim_metric.reset()
                all_ssim.append(ssim_val)
            all_mae.extend(mae_per)
            all_bias.extend(bias_per)
            all_psnr.extend(psnr_per)

            if batch_idx % 5 == 0:
                print(f"  batch {batch_idx+1}/{n_batches} | MAE {float(np.mean(mae_per)):.4f} | SSIM {float(np.mean(all_ssim[-pred.shape[0]:])):.4f} | PSNR {float(np.mean(psnr_per)):.2f}")

            # save progress after each batch so an interrupted run can resume
            if max_batches is None:
                tmp_path = ckpt_path + ".tmp"
                with open(tmp_path, "w") as f:
                    json.dump({"all_mae": all_mae, "all_ssim": all_ssim, "all_psnr": all_psnr, "all_bias": all_bias, "next_batch": batch_idx + 1}, f)
                os.replace(tmp_path, ckpt_path)

    results = {
        "mae":  {"mean": float(np.mean(all_mae)),  "std": float(np.std(all_mae)),  "per_slice": all_mae},
        "bias": {"mean": float(np.mean(all_bias)), "std": float(np.std(all_bias)), "per_slice": all_bias},
        "ssim": {"mean": float(np.mean(all_ssim)), "std": float(np.std(all_ssim)), "per_slice": all_ssim},
        "psnr": {"mean": float(np.mean(all_psnr)), "std": float(np.std(all_psnr)), "per_slice": all_psnr},
        "num_inference_steps": num_inference_steps,
        "scheduler": "ddpm",
        "n_batches": n_batches,
    }

    print(f"  Mean | MAE {results['mae']['mean']:.4f} +/- {results['mae']['std']:.4f} | "
          f"Bias {results['bias']['mean']:+.4f} | "
          f"SSIM {results['ssim']['mean']:.4f} +/- {results['ssim']['std']:.4f} | "
          f"PSNR {results['psnr']['mean']:.2f} +/- {results['psnr']['std']:.2f} dB")

    if max_batches is not None:
        out_path = os.path.join(config.RESULTS_DIR, f"models_{config.CURRENT_EVAL}/numerical_results_debug.json")
        all_results = {}
    else:
        out_path = os.path.join(config.RESULTS_DIR, f"models_{config.CURRENT_EVAL}/numerical_results.json")
        if os.path.exists(out_path):
            with open(out_path) as f:
                all_results = json.load(f)
        else:
            all_results = {}

    all_results[str(num_inference_steps)] = results
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved to {out_path}")

    if max_batches is None and os.path.exists(ckpt_path):
        os.remove(ckpt_path)


if __name__ == "__main__":
    evaluate_numerical(num_inference_steps=1000)