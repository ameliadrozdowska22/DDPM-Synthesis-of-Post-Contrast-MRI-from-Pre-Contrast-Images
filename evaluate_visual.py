
import os
import json
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
import dataset
from torch.amp import autocast
from helpers.eval_helpers import orient, load_diffusion_model, build_ddpm_scheduler


def save_images_diff_figure(case_dir, pre_center, post, target, all_samples, num_samples, mid):
    """
    This function produces and saves figure with visual comparison of multiple noise samples
    """
    path = os.path.join(case_dir, "images_diff.png")
    ground_truth = post if config.CURRENT_EVAL == "residual_learning" else target

    plt.style.use("default")
    fig, axes = plt.subplots(3, num_samples, figsize=(3 * num_samples, 12))

    # 1. condition
    for k in range(num_samples):
        axes[0, k].axis("off")
    axes[0, mid].imshow(orient(pre_center[0, 0]), cmap="gray", vmin=0, vmax=1)
    axes[0, mid].set_title("Condition input")
    axes[0, mid].axis("off")

    # 2. generated outputs
    for k in range(num_samples):
        if config.CURRENT_EVAL == "residual_learning":
            # reconstruct the full image from residaul-map and pre-contrast
            generated = pre_center[0, 0] + all_samples[f"sample_{k}"]["sample"][0, 0] * config.RESIDUAL_SCALE
            axes[1, k].imshow(orient(generated), cmap="gray", vmin=0, vmax=1)
        else:
            axes[1, k].imshow(orient(all_samples[f"sample_{k}"]["sample"][0, 0]), cmap="gray", vmin=0, vmax=1)
        axes[1, k].set_title(f"Sample {k}")
        axes[1, k].axis("off")

    # 3. ground truth
    for k in range(num_samples):
        axes[2, k].axis("off")
    axes[2, mid].imshow(orient(ground_truth[0, 0]), cmap="gray", vmin=0, vmax=1)
    axes[2, mid].set_title("Ground truth")
    axes[2, mid].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_denoising_chain_figure(case_dir, pre_center, all_samples, num_samples, mid):
    """
    This function produces and saves denoising chain visualization
    """
    path = os.path.join(case_dir, "denoising_chain.png")
    plt.style.use("default")
    fig_denoise, axes_denoise = plt.subplots(num_samples)
    for k in range(num_samples):
        axes_denoise[k].axis("off")

    chain_key = f"sample_{mid}"
    chain_steps = [orient(img[0, 0]) for img in all_samples[chain_key]["decoded_images"]]

    if config.CURRENT_EVAL == "residual_learning":
        pre_oriented = orient(pre_center[0, 0])
        chain_steps = [pre_oriented + step * config.RESIDUAL_SCALE for step in chain_steps]

    chain = np.concatenate(chain_steps, axis=1)

    axes_denoise[mid].imshow(chain, cmap="gray", vmin=0, vmax=1)
    axes_denoise[mid].set_title("Denoising chain")
    axes_denoise[mid].axis("off")

    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def save_error_figure(case_dir, pre_center, post, target, all_samples):
    """
    This function produces and saves an error map
    """
    path = os.path.join(case_dir, "error.png")
    ground_truth = post if config.CURRENT_EVAL == "residual_learning" else target

    if config.CURRENT_EVAL == "residual_learning":
        # reconstruct the full image from residaul-map and pre-contrast
        generated = pre_center[0:1] + all_samples["sample_0"]["sample"][0:1] * config.RESIDUAL_SCALE
    else:
        generated = all_samples["sample_0"]["sample"][0:1]

    error = generated - ground_truth[0:1] # calculate error

    fig_err, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(orient(error[0, 0]), cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax)
    ax.set_title("Error")
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_samples(dif_model, scheduler, inputs, targets, num_samples):
    """
    This function generates samples using the diffusion model
    """
    all_samples = {}

    for k in range(num_samples):
        sample_dict = {}
        decoded_images = []

        print(f"\nSample {k+1}/{num_samples}")

        # start from random noise shaped like the target
        torch.manual_seed(k)
        sample = torch.randn_like(targets)

        scheduler.set_timesteps(num_inference_steps=1000)

        decoded_images.append(sample[0:1].detach().cpu())  # pure noise

        with torch.no_grad():
            for step_idx, t in enumerate(scheduler.timesteps):

                if step_idx % 100 == 0:
                    print(f"  Step {step_idx}/1000")

                timesteps = torch.full(
                    (inputs.shape[0],),
                    t,
                    device=config.DEVICE,
                    dtype=torch.long,
                )

                model_input = torch.cat([sample, inputs], dim=1)
                with autocast("cuda", enabled=(config.DEVICE.type == "cuda")):
                    noise_pred = dif_model(model_input, timesteps)

                sample = scheduler.step(noise_pred.float(), t, sample)[0]  # reverse diffusion step

                # save state for denoising chain visualization
                if (step_idx + 1) % 100 == 0:
                    decoded_images.append(sample[0:1].detach().cpu())

        sample_dict["sample"] = sample.detach().cpu()
        sample_dict["decoded_images"] = decoded_images

        all_samples[f"sample_{k}"] = sample_dict

    return all_samples


def load_case_batch(test_loader, case):
    """
    This function looks up the requested slice in the test set
    and returns its batch
    """
    target_subject, target_slice = case["case"], case["slice"]
    test_ds = test_loader.dataset
    with open(dataset.get_split_path()) as f:
        test_subjects = json.load(f)["test"]
    if target_subject not in test_subjects:
        raise ValueError(f"{target_subject} is not part of the test set")
    vol_idx = test_subjects.index(target_subject)
    sample_to_idx = {sample: idx for idx, sample in enumerate(test_ds.samples)}
    mid_dataset_idx = sample_to_idx.get((vol_idx, target_slice))
    if mid_dataset_idx is None:
        raise ValueError(
            f"{target_subject} slice {target_slice} is not in the test set"
        )
    batch = [t.unsqueeze(0) for t in test_ds[mid_dataset_idx]]
    return target_subject, target_slice, batch


def evaluate_visual():
    """
    This function serves as a main skeleton for visual evaluation. It loads the test data, the model
    and contains the evaluation loop accross the selected visual cases
    """
    print("Loading data...")
    _, _, test_loader = dataset.get_dataloaders()

    print("Loading diffusion model...")
    dif_model, diff_ckpt = load_diffusion_model()

    scheduler = build_ddpm_scheduler()
    dif_model.eval()

    for case in config.SELECTED_SLICES:
        target_subject, target_slice, batch = load_case_batch(test_loader, case)

        if config.CURRENT_EVAL == "residual_learning":
            # use residual-map as a target, and get center middle slice for reconstruction
            inputs, targets, pre_center, post = batch
            post = post.to(config.DEVICE)
        else:
            # use direct post-contrast as a target
            inputs, _, _, targets = batch
            pre_center = inputs[:, 1:2]
            post = None

        inputs = inputs.to(config.DEVICE)
        targets = targets.to(config.DEVICE)
        pre_center = pre_center.to(config.DEVICE)
        
        print(f"Evaluating: {case}")
        num_samples = 5
        all_samples = generate_samples(dif_model, scheduler, inputs, targets, num_samples)

        target = targets.detach().cpu()
        pre_center = pre_center.detach().cpu()

        if config.CURRENT_EVAL == "residual_learning":
            post = post.detach().cpu()

        mid = num_samples // 2
        case_dir = os.path.join(
            config.RESULTS_DIR, "figures", "visual_eval",
            config.CURRENT_EVAL, f"{target_subject}_slice{target_slice}"
        )
        os.makedirs(case_dir, exist_ok=True)

        save_images_diff_figure(case_dir, pre_center, post, target, all_samples, num_samples, mid)
        save_denoising_chain_figure(case_dir, pre_center, all_samples, num_samples, mid)
        save_error_figure(case_dir, pre_center, post, target, all_samples)
        print(f"Figures saved to {case_dir}")


if __name__ == "__main__":
    if config.CURRENT_EVAL == "residual_learning" and config.CURRENT_TRAINING != "residual_learning":
        raise ValueError(
            f"CURRENT_EVAL='residual_learning' but CURRENT_TRAINING='{config.CURRENT_TRAINING}'. "
            "The dataset will return zeros for pre_center, corrupting residual model evaluation. "
            "Set CURRENT_TRAINING='residual_learning' in config.py before running."
        )
    
    print(f"Evaluating: {config.CURRENT_EVAL}")
    evaluate_visual()