import os
import torch
import numpy as np
import matplotlib
import random
from torch.utils.data import DataLoader, Subset
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
import model
import dataset
from monai.networks.schedulers import DDPMScheduler


def load_diffusion_checkpoint():
    """
    This function gets the current best performing checkpoint
    """
    ckpt_path = os.path.join(config.RESULTS_DIR, f"models_{config.CURRENT_EVAL}/best_diffusion.pth")
    checkpoint = torch.load(ckpt_path, map_location=config.DEVICE)
    print(
        f"Loaded diffusion checkpoint from epoch {checkpoint['epoch'] + 1} "
        f"with val loss {checkpoint['val_noise_loss']:.4f}"
    )
    return checkpoint


def load_diffusion_model():
    """
    This function builds and populates the network with current best weights
    """
    dif_model = model.get_model_diff().to(config.DEVICE)
    checkpoint = load_diffusion_checkpoint()
    dif_model.load_state_dict(checkpoint["model_state_dict"])
    dif_model.eval()
    return dif_model, checkpoint



def build_ddpm_scheduler():
    """
    This function constructs the MONAI DDPMScheduler that performs the DDPM sampling process
    """
    if config.CURRENT_EVAL == "residual_learning":
        # model predicts (post - pre) / RESIDUAL_SCALE
        # post and pre are each in [0, 1] so their raw difference is in [-1, 1]
        clip_min = -1.0 / config.RESIDUAL_SCALE
        clip_max =  1.0 / config.RESIDUAL_SCALE
    else:
        # model predicts the post-contrast image directly so we just 
        # normalize to [0, 1]
        clip_min = 0.0
        clip_max = 1.0
    return DDPMScheduler(
        num_train_timesteps=1000,
        schedule="linear_beta",
        beta_start=0.0015,
        beta_end=0.0195,
        clip_sample=True,
        clip_sample_min=clip_min,
        clip_sample_max=clip_max,
    )


def orient(array):
    """
    This function reorients the slice for display so it matches the typical 
    way radiologist view slices
    """
    return np.flipud(np.asarray(array).T)



def get_one_slice_per_subject_loader():
    """
    This function builds a DataLoader containing exactly one random slice 
    per test subject for uncertaity evaluation
    """
    _, _, full_loader = dataset.get_dataloaders(include_test=True)
    test_ds = full_loader.dataset

    vol_to_indices = {}
    for idx, (vol_idx, _) in enumerate(test_ds.samples):
        if vol_idx not in vol_to_indices:
            vol_to_indices[vol_idx] = []
        vol_to_indices[vol_idx].append(idx)

    vol_indices = sorted(vol_to_indices.keys())
    rng = random.Random(42)
    selected = [rng.choice(vol_to_indices[v]) for v in vol_indices]
    print(f"  1 random slice per subject: {len(selected)} slices total")

    loader = DataLoader(
        Subset(test_ds, selected),
        batch_size=config.EVAL_BATCH_SIZE,
        shuffle=False,
    )
    return loader, vol_indices