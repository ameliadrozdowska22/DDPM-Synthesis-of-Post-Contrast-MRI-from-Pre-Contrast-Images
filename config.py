import os

# paths (override via env vars; defaults are relative to this repo)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("DIFFUSION_DATA_DIR", os.path.join(PROJECT_ROOT, "dataset"))
RESULTS_DIR = os.environ.get("DIFFUSION_RESULTS_DIR", PROJECT_ROOT)

# data
PRE_FRAMES  = ["pre.nii.gz"]
POST_FRAMES = ["post.nii.gz"]

PAIRS = [
    ("pre.nii.gz", "post.nii.gz"),
]

# slicing
SLICE_AXIS = 2 # axial
MIN_FOREGROUND = 0.05 # skip slices with less than 5% non-zero voxels

# split subjects
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

CURRENT_TRAINING = "residual_learning" # "control" or "residual_learning"
CURRENT_EVAL = "residual_learning" # "control" or "residual_learning"

# training
BATCH_SIZE = 2
EVAL_BATCH_SIZE = 16
NUM_EPOCHS = 150
LEARNING_RATE = 5e-5 
NUM_WORKERS = 4 
VAL_INTERVAL = 1 
MIN_DELTA = 1e-5
EARLY_STOPPING_PATIENCE = 25
RESUME_TRAINING = True

RESIDUAL_SCALE = 0.6350 

# device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# slices selected for visual evaluation:
# patient cases were selected randomly 
# slices were selected manualy  to make sure they contain pathological tissues
SELECTED_SLICES = [
    {
        "case": "DIST_0725",
        "slice": 16
    },
    {
        "case": "DIST_0596",
        "slice": 15
    },
    {
        "case": "DIST_0433",
        "slice": 23
    },
    {
        "case": "DIST_0165",
        "slice": 16
    }
]