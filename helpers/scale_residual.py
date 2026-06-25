import numpy as np
import json
import nibabel as nib
import config
import os

with open(os.path.join(config.RESULTS_DIR, "subject_split/subject_split.json")) as f:
    split = json.load(f)

residuals = []
for subject in split["train"]:
    subj_dir = os.path.join(config.DATA_DIR, subject)
    for pre_fname, post_fname in config.PAIRS:
        pre_path  = os.path.join(subj_dir, pre_fname)
        post_path = os.path.join(subj_dir, post_fname)
        if not os.path.exists(pre_path) or not os.path.exists(post_path):
            continue

        pre  = nib.load(pre_path).get_fdata(dtype=np.float32)
        post = nib.load(post_path).get_fdata(dtype=np.float32)

        # normalize the same way as dataset.py does
        combined = np.concatenate([pre.flatten(), post.flatten()])
        low  = np.percentile(combined, 1)
        high = np.percentile(combined, 99)
        if high - low == 0:
            continue
        pre  = (pre  - low) / (high - low)
        post = (post - low) / (high - low)

        residuals.append((post - pre).flatten())

all_r = np.concatenate(residuals)

#compare residual std compare to control after scaling
RESIDUAL_SCALE = np.std(all_r) / 0.2427  # match control std exactly
residual_std_after_scaling = np.std(all_r) / RESIDUAL_SCALE
control_std = 0.2427  # from slurm output targets std

print("control std:                    ", control_std)
print("residual std after scaling:     ", residual_std_after_scaling)
print(RESIDUAL_SCALE)