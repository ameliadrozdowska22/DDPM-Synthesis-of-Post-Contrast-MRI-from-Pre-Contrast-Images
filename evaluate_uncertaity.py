
import os
import math
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import config
from torch.amp import autocast
from helpers.eval_helpers import load_diffusion_model, build_ddpm_scheduler, get_one_slice_per_subject_loader


def evaluate_uncertainty(num_inference_steps=1000, n_samples=10, sample_chunk_size=2):
    """
    This function runs full reverse diffusion on one slice per test subject, generating
    multiple samples per input, and saves MAE/variance/correlation results
    """

    if config.CURRENT_EVAL == "residual_learning" and config.CURRENT_TRAINING != "residual_learning":
        raise ValueError(
            f"CURRENT_EVAL='residual_learning' but CURRENT_TRAINING='{config.CURRENT_TRAINING}'. "
            "Set CURRENT_TRAINING='residual_learning' in config.py before running."
        )

    device = config.DEVICE
    print("Loading data...")
    test_loader, subject_ids = get_one_slice_per_subject_loader()

    print("Loading diffusion model...")
    dif_model, _ = load_diffusion_model()

    scheduler = build_ddpm_scheduler()
    scheduler.set_timesteps(num_inference_steps=num_inference_steps)

    all_mae, all_mean_var = [], []
    n_batches = len(test_loader)

    n_chunks = math.ceil(n_samples / sample_chunk_size)
    print(
        f"Evaluating {n_batches} batches | {num_inference_steps} DDPM steps | "
        f"{n_samples} samples per input ({n_chunks} chunks of {sample_chunk_size})\n"
    )

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
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

            print(f"  Batch {batch_idx+1}/{n_batches}")
            chunk_results = []
            for chunk_idx, chunk_start in enumerate(range(0, n_samples, sample_chunk_size)):
                chunk_size = min(sample_chunk_size, n_samples - chunk_start)
                inputs_exp = inputs.repeat_interleave(chunk_size, dim=0)
                torch.manual_seed(batch_idx * 1000 + chunk_idx)
                sample = torch.randn(B * chunk_size, 1, H, W, device=device)  # start from random noise

                print(f"    chunk {chunk_idx+1}/{n_chunks}", flush=True)
                for step_idx, t in enumerate(scheduler.timesteps):
                    if step_idx % 200 == 0:
                        print(f"      step {step_idx}/{num_inference_steps}", flush=True)
                    ts = torch.full((B * chunk_size,), t, device=device, dtype=torch.long)
                    with autocast("cuda", enabled=(device.type == "cuda")):
                        noise_pred = dif_model(torch.cat([sample, inputs_exp], dim=1), ts)  # predict noise
                    sample = scheduler.step(noise_pred.float(), t, sample)[0]  # reverse diffusion step

                chunk_results.append(sample.view(B, chunk_size, 1, H, W))

            # [B, N, 1, H, W]
            samples = torch.cat(chunk_results, dim=1)

            if config.CURRENT_EVAL == "residual_learning":
                # reconstruct post-contrast image from generated residual
                samples = pre_center.unsqueeze(1) + samples * config.RESIDUAL_SCALE

            mean_pred = samples.mean(dim=1)   # [B,1,H,W]
            uncertainty = samples.var(dim=1)  # [B,1,H,W]  pixel-wise variance

            # calculate per-slice numerical evaluation metrics
            mae_per = torch.mean(torch.abs(mean_pred - post), dim=[1, 2, 3]).tolist()
            var_per = uncertainty.mean(dim=[1, 2, 3]).tolist()
            all_mae.extend(mae_per)
            all_mean_var.extend(var_per)

            if batch_idx % 50 == 0 or batch_idx == n_batches - 1:
                print(
                    f"  Batch {batch_idx+1}/{n_batches} | "
                    f"MAE: {float(np.mean(mae_per)):.4f} | Mean Var: {float(np.mean(var_per)):.6f}"
                )

    # correlation between predictive uncertainty and actual error
    corr_var_mae = float(np.corrcoef(np.array(all_mean_var), np.array(all_mae))[0, 1])

    results = {
        "mae":             {"mean": float(np.mean(all_mae)),      "std": float(np.std(all_mae)),      "per_slice": all_mae},
        "mean_pixel_var":  {"mean": float(np.mean(all_mean_var)), "std": float(np.std(all_mean_var)), "per_slice": all_mean_var},
        "corr_var_vs_mae": corr_var_mae,
        "subject_ids": subject_ids,
        "num_inference_steps": num_inference_steps,
        "n_samples": n_samples,
        "scheduler": "ddpm",
        "n_batches": n_batches,
    }

    print(f"\n--- Uncertainty Results ({num_inference_steps} DDPM steps, N={n_samples}) ---")
    print(f"MAE:            {results['mae']['mean']:.4f} +/- {results['mae']['std']:.4f}")
    print(f"Mean Pixel Var: {results['mean_pixel_var']['mean']:.6f} +/- {results['mean_pixel_var']['std']:.6f}")
    print(f"Corr(var, MAE): {corr_var_mae:+.3f}")

    out_path = os.path.join(config.RESULTS_DIR, f"models_{config.CURRENT_EVAL}/uncertainty_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")

    return results



if __name__ == "__main__":
    evaluate_uncertainty(num_inference_steps=1000, n_samples=20, sample_chunk_size=1)
