# DDPM Synthesis of Post-Contrast MRI from Pre-Contrast Images

This is the codebase used for the bachelor thesis "Generating Contrast-Enhanced MRI from Pre-Contrast MRI for Head and Neck Cancer: Assessing Residual Map Prediction as an Alternative to Full-Image Synthesis in Conditional Diffusion Models".
The codebase is used to train and evaluate DDPM U-Net based models based on two different approaches:
* full-image-based approach, in which the model predicts the post-contrast image directly
* residual-map-based approach, in which the model predicts the contrast enhancement residual map

## Description

**Scripts**
- `config.py` - central configuration file with paths, slicing parameters, training hyperparameters, and the `CURRENT_TRAINING`/`CURRENT_EVAL` mode switches.
- `dataset.py` - loads, normalizes, and slices the paired pre/post-contrast MRI volumes into the PyTorch dataloaders.
- `model.py` - defines the diffusion U-Net architecture.
- `train.py` - trains the diffusion model in "control" mode (full-image-based approach).
- `train_residual_l.py` - trains the diffusion model in "residual_learning" mode (residual-map-based approach).
- `evaluate_visual.py` - generates qualitative comparison figures such as samples, denoising chain and error maps for a fixed set of manually selected test slices.
- `evaluate_numerical.py` - runs full sampling diffusion process over the entire test set and computes MAE/bias/SSIM/PSNR.
- `evaluate_uncertaity.py` - estimates per-pixel variance across multiple samples per input and calculates its correlation with MAE.
- `requirements.txt` - contains package dependencies.
- `helpers/` - shared helper files:
  - `eval_helpers.py` - helper functions used by all evaluation scripts.
  - `per_subject_results.py` - turns per-slice numerical results into per-subject results.
  - `scale_residual.py` - computes `RESIDUAL_SCALE`, the constant that rescales the residual target to match the control target's variance.

**Folders**
- `subject_split/` - the saved train/val/test subject split, generated once and reused across runs.
- `models_control/` - checkpoints, numerical results, and debug snapshots from the full-image-based model.
- `models_residual_learning/` - checkpoints, numerical results, and debug snapshots from the residual-map-based model.
- `figures/` - generated qualitative-evaluation figures, organized per case and model.

## Getting Started

### Dependencies

* Python 3.11
* PyTorch
* MONAI, NiBabel, NumPy, scikit-learn, Matplotlib (see `requirements.txt`)
* A GPU is strongly recommended for full 1000 step DDPM reverse-diffusion per sample

### Installing

* Install all dependencies:
  ```bash
  pip install -r requirements.txt
  ```
* This repo does not include the original dataset due to patient privacy reasons and hospital guidelines. For training and evaluation use your own data. Your dataset is expected in the `dataset/` folder, with one subfolder per subject named as `DIST_{ID}`, e.g. `dataset/DIST_0005/`, `dataset/DIST_0433/`
* Each subject's subfolder must contain exactly two NIfTI files, named `pre.nii.gz` and `post.nii.gz`
* Outputs are written to the repo root by default. Override the dataset and output locations with the `DIFFUSION_DATA_DIR` and `DIFFUSION_RESULTS_DIR` environment variables if needed

### Executing program

**1. Train** - set `config.CURRENT_TRAINING` to one of:
  * `"control"` - for the full-image-based approach (predicting the post-contrast image directly)
  * `"residual_learning"` - for the residual-map-based approach (predicting the contrast enhancement residual map)

  then run the matching script:
  ```bash
  python train.py              # control mode
  python train_residual_l.py   # residual-learning mode
  ```
  If `config.RESUME_TRAINING = True` and a checkpoint already exists for that model, running the train command resumes training automatically from the last saved epoch.
  Set `RESUME_TRAINING = False` to start over the training.

**2. Evaluate** - set `config.CURRENT_EVAL` to one of:
  * `"control"`
  * `"residual_learning"`

  then run any of:
  ```bash
  python evaluate_visual.py       # qualitative figures for selected slices
  python evaluate_numerical.py    # MAE/bias/SSIM/PSNR over the full test set (slow, resumable)
  python evaluate_uncertaity.py   # per-pixel uncertainty via repeated sampling
  ```

## License

This project is licensed under the MIT License

## Built With

- [MONAI](https://project-monai.github.io/)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [NiBabel](https://nipy.org/nibabel/)
