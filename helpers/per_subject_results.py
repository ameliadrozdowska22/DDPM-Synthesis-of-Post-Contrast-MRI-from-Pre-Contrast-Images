import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
import dataset

MODELS = ["control", "residual_learning"]
METRICS = ["mae", "ssim", "bias"]


def load_per_slice(model_name, num_inference_steps=1000):
    """
    This function loads the per-slice MAE/SSIM/bias values from saved numerical results
    """
    path = os.path.join(config.RESULTS_DIR, f"models_{model_name}/numerical_results.json")
    with open(path) as f:
        data = json.load(f)[str(num_inference_steps)]
    return {metric: np.array(data[metric]["per_slice"]) for metric in METRICS}


def per_subject_means(per_slice, vol_idx_per_slice, n_subjects):
    """
    This function calculates an average value of all slices that belong to one patient case
    """
    sums = [[] for _ in range(n_subjects)]
    for value, vol_idx in zip(per_slice, vol_idx_per_slice):
        sums[vol_idx].append(value)
    return np.array([np.mean(s) for s in sums])


def main():
    """
    This function takes the per-slice numerical results (MAE/SSIM/bias) 
    and uses them to get per-subject average values. (Used for statistical significance)
    """

    with open(dataset.get_split_path()) as f:
        test_subjects = json.load(f)["test"]

    print("Loading test set...")
    test_ds = dataset.ContrastDataset(test_subjects)
    vol_idx_per_slice = [vol_idx for vol_idx, _ in test_ds.samples]
    n_subjects = len(test_ds.volumes)

    for model_name in MODELS:
        per_slice = load_per_slice(model_name)

        result = {"n_subjects": n_subjects}
        for metric in METRICS:
            values = per_subject_means(per_slice[metric], vol_idx_per_slice, n_subjects)
            result[metric] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "per_subject": values.tolist(),
            }

        out_path = os.path.join(config.RESULTS_DIR, f"models_{model_name}/per_subject_results.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
