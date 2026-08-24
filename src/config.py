"""Shared configuration: paths, feature groups, cost matrix."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "app" / "artifacts"

# --- Raw column groups (from the dataset documentation) ---
COUNTER_COLS = ["171_0", "666_0", "427_0", "837_0", "309_0", "835_0", "370_0", "100_0"]
HISTOGRAMS = {"167": 10, "272": 10, "291": 11, "158": 10, "459": 20, "397": 36}

RATE_COLS  = [f"{c}_rate"  for c in COUNTER_COLS]
RESET_COLS = [f"{c}_reset" for c in COUNTER_COLS]
SHARE_COLS = [f"{v}_{i}_share" for v, n in HISTOGRAMS.items() for i in range(n)]

# --- Class definition: time-to-failure windows (time_step units) ---
CLASS_WINDOWS = [48, 24, 12, 6]   # >48 -> 0, 48-24 -> 1, 24-12 -> 2, 12-6 -> 3, <=6 -> 4

# --- Official misclassification cost matrix (rows=actual, cols=predicted) ---
COST = np.array([
    [  0,   7,   8,   9,  10],
    [200,   0,   7,   8,   9],
    [300, 200,   0,   7,   8],
    [400, 300, 200,   0,   7],
    [500, 400, 300, 200,   0],
])

RANDOM_SEED = 42
VAL_FRACTION = 0.2