import os
import json
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import config
import random

def get_subject_list(max_subjects=None, seed=42):
    """
    This function returns the list of patient IDs from the dataset
    """
    subjects = sorted([
        s for s in os.listdir(config.DATA_DIR)
        if os.path.isdir(os.path.join(config.DATA_DIR, s))
    ])

    if max_subjects is not None:
        random.seed(seed)
        subjects = random.sample(subjects, min(max_subjects, len(subjects)))

    return subjects

def split_subjects(subjects):
    """
    This function splits the patient cases into train/val/test subsets
    """
    # first split off test set
    train_val, test = train_test_split(
        subjects,
        test_size=config.TEST_RATIO,
        random_state=42
    )
    # then split train and validation sets
    val_size = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)
    train, val = train_test_split(
        train_val,
        test_size=val_size,
        random_state=42
    )
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)} subjects")
    return train, val, test

def load_volume(path):
    """
    This function loads a NIfTI file and returns its voxel data as a float32 array
    """
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    return data

def normalize_pair(pre, post):
    """
    This function clips and normalizes the intensities of pre- and post- 
    contrast volumes 
    """
    combined = np.concatenate([pre.flatten(), post.flatten()])
    low = np.percentile(combined, 1)
    high = np.percentile(combined, 99)

    pre  = np.clip(pre, low, high)
    post = np.clip(post, low, high)

    if high - low == 0:
        # degenerate case (e.g. constant-intensity volume): clipped but NOT
        # scaled to [0, 1] like the normal path below
        return pre, post

    pre  = (pre - low) / (high - low)
    post = (post - low) / (high - low)

    return pre, post


class ContrastDataset(Dataset):
    """
    This class indexes and loads 2D slices from paired pre/post-contrast MRI volumes
    """
    def __init__(self, subjects):
        """
        This function builds the slice index, loads and normalizes all volumes
        """
        self.samples = []
        self.volumes = []

        for subject in subjects:
            subj_dir = os.path.join(config.DATA_DIR, subject)

            for pre_fname, post_fname in config.PAIRS:
                pre_path  = os.path.join(subj_dir, pre_fname)
                post_path = os.path.join(subj_dir, post_fname)

                if not os.path.exists(pre_path) or not os.path.exists(post_path):
                    print(f"  [SKIP] Missing files for {subject} - {pre_fname}/{post_fname}")
                    continue

                pre_vol  = load_volume(pre_path)
                post_vol = load_volume(post_path)
                pre_vol, post_vol = normalize_pair(pre_vol, post_vol)

                vol_idx = len(self.volumes)
                self.volumes.append((pre_vol, post_vol))

                n_slices = pre_vol.shape[config.SLICE_AXIS]

                for i in range(1, n_slices - 1): # skip first/last slice to not make them the center slice
                    if config.SLICE_AXIS == 2:  # axial
                        pre_curr = pre_vol[:, :, i]
                    elif config.SLICE_AXIS == 1:
                        pre_curr = pre_vol[:, i, :]
                    else:
                        pre_curr = pre_vol[i, :, :]

                    foreground = np.mean(pre_curr > 0)
                    if foreground < config.MIN_FOREGROUND:
                        continue

                    self.samples.append((vol_idx, i))

        print(f"  Total slices: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        This function slices the volumes and returns (pre, residual, pre_center, post) for that slice
        residual and pre_center are zeros unless CURRENT_TRAINING is residual_learning
        """
        vol_idx, i = self.samples[idx]
        pre_vol, post_vol = self.volumes[vol_idx]

        # slicing the volume
        if config.SLICE_AXIS == 2:  # axial
            pre_prev = pre_vol[:, :, i - 1]
            pre_curr = pre_vol[:, :, i]
            pre_next = pre_vol[:, :, i + 1]
            pre_slice = np.stack([pre_prev, pre_curr, pre_next], axis=0)
            post_slice = post_vol[:, :, i]
        elif config.SLICE_AXIS == 1:
            pre_prev = pre_vol[:, i - 1, :]
            pre_curr = pre_vol[:, i, :]
            pre_next = pre_vol[:, i + 1, :]
            pre_slice = np.stack([pre_prev, pre_curr, pre_next], axis=0)
            post_slice = post_vol[:, i, :]
        else:
            pre_prev = pre_vol[i - 1, :, :]
            pre_curr = pre_vol[i, :, :]
            pre_next = pre_vol[i + 1, :, :]
            pre_slice = np.stack([pre_prev, pre_curr, pre_next], axis=0)
            post_slice = post_vol[i, :, :]

        pre = torch.from_numpy(pre_slice).float()
        pre_center = torch.from_numpy(pre_curr).float().unsqueeze(0)
        post = torch.from_numpy(post_slice).float().unsqueeze(0)

        if config.CURRENT_TRAINING == "residual_learning":
            # compute the actual residual target, scaled to match the control target's std
            residual_slice = (post_slice - pre_curr) / config.RESIDUAL_SCALE
            residual = torch.from_numpy(residual_slice).float().unsqueeze(0)
            return pre, residual, pre_center, post

        # control mode doesn't need the residual/pre_center fields, so return zeros
        return pre, torch.zeros_like(post), torch.zeros_like(post), post

def get_dataloaders(include_test=True):
    """
    This function builds the train/val/test DataLoaders
    """
    split_path = get_split_path()

    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        train_subjects = split["train"]
        val_subjects   = split["val"]
        test_subjects  = split["test"]
        print(f"Loaded split from {split_path}")
    else:
        train_subjects, val_subjects, test_subjects = create_and_save_split(
            max_subjects=None, seed=42
        )

    print("Loading train set...")
    train_ds = ContrastDataset(train_subjects)
    print("Loading val set...")
    val_ds   = ContrastDataset(val_subjects)
    if include_test:
        print("Loading test set...")
        test_ds = ContrastDataset(test_subjects)
    else:
        test_ds = None

    use_cuda = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_cuda,
        persistent_workers=(config.NUM_WORKERS > 0),
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=use_cuda,
        persistent_workers=(config.NUM_WORKERS > 0),
        prefetch_factor=2 if config.NUM_WORKERS > 0 else None,
    )

    if include_test:
        test_loader = DataLoader(
            test_ds,
            batch_size=config.EVAL_BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=use_cuda,
            persistent_workers=(config.NUM_WORKERS > 0),
            prefetch_factor=2 if config.NUM_WORKERS > 0 else None,
        )
    else:
        test_loader = None

    return train_loader, val_loader, test_loader


def get_split_path():
    return os.path.join(config.RESULTS_DIR, "subject_split/subject_split.json")


def create_and_save_split(max_subjects=None, seed=42):
    """
    This function creates and saves the file that stores train/val/test split
    """
    subjects = get_subject_list(max_subjects=max_subjects, seed=seed)
    train_subjects, val_subjects, test_subjects = split_subjects(subjects)

    split_data = {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }

    os.makedirs(os.path.dirname(get_split_path()), exist_ok=True)
    with open(get_split_path(), "w") as f:
        json.dump(split_data, f, indent=2)

    print(f"Saved split to {get_split_path()}")
    return train_subjects, val_subjects, test_subjects