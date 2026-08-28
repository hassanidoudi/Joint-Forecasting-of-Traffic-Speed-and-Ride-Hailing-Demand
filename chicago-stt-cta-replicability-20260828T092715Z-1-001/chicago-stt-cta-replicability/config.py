"""
config.py — Central configuration for the Joint Speed-Demand Forecaster.
All hyperparameters, paths, and constants in one place.
"""

import os
import math
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
TENSOR_DIR = os.path.join(OUTPUT_DIR, "tensors")

# Input CSV files
BOUNDARIES_CSV = os.path.join(DATA_DIR, "Boundaries_-_Community_Areas_20260209.csv")
TRAFFIC_CSV = os.path.join(DATA_DIR, "Chicago_Traffic_Tracker.csv")
TNP_CSV = os.path.join(DATA_DIR, "Transportation_Network_Providers_-_Trips_(2025-)_20260209.csv")
WEATHER_CSV = os.path.join(DATA_DIR, "weather_loop_2025.csv")

# Create output directories
for d in [OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, FIGURE_DIR, TENSOR_DIR]:
    os.makedirs(d, exist_ok=True)

# ─── Spatial Configuration ───────────────────────────────────────────────────
# Grid cell size in meters (configurable)
GRID_CELL_SIZE_M = 500

# Loop bounding box — will be overwritten by data_preprocessing if shapefile
# is parsed, but these are good defaults extracted from the WKT polygon.
LOOP_BBOX = {
    "min_lat": 41.8673,   # South
    "max_lat": 41.8906,   # North
    "min_lon": -87.6383,  # West
    "max_lon": -87.6024,  # East
}


def compute_grid_dims(bbox: dict = None, cell_size_m: float = None):
    """Compute grid dimensions (rows, cols) from bounding box and cell size.

    Uses a simple equirectangular approximation:
      1 degree latitude  ≈ 111,320 m
      1 degree longitude ≈ 111,320 * cos(lat) m
    """
    if bbox is None:
        bbox = LOOP_BBOX
    if cell_size_m is None:
        cell_size_m = GRID_CELL_SIZE_M

    lat_range_m = (bbox["max_lat"] - bbox["min_lat"]) * 111_320
    mid_lat_rad = math.radians((bbox["max_lat"] + bbox["min_lat"]) / 2)
    lon_range_m = (bbox["max_lon"] - bbox["min_lon"]) * 111_320 * math.cos(mid_lat_rad)

    n_rows = max(1, int(math.ceil(lat_range_m / cell_size_m)))
    n_cols = max(1, int(math.ceil(lon_range_m / cell_size_m)))
    return n_rows, n_cols


GRID_ROWS, GRID_COLS = compute_grid_dims()

# Patch size for the transformer (1 = each cell is a token)
PATCH_SIZE = 1

# ─── Temporal Configuration ─────────────────────────────────────────────────
T_IN = 24          # Input window (hours) — full diurnal cycle
T_OUT = 6          # Prediction horizon (hours)
TIME_BIN = "1h"    # Aggregation bin size

# ─── Data Split Dates (temporal, no shuffling) ──────────────────────────────
TRAIN_END = "2025-09-10 23:00:00"
VAL_END = "2025-10-16 23:00:00"
# Test goes until 2025-12-31 23:00:00

# ─── Feature Configuration ──────────────────────────────────────────────────
# Data features (per grid cell per hour)
SPEED_FEATURES = ["avg_speed", "min_speed", "num_segments", "bus_count"]
DEMAND_FEATURES = ["pickup_count", "dropoff_count", "avg_trip_duration", "avg_trip_distance"]
TEMPORAL_FEATURES = ["hour_of_day", "day_of_week", "is_weekend", "month", "is_holiday"]
WEATHER_FEATURES = ["temperature_2m", "precipitation"]

ALL_FEATURES = SPEED_FEATURES + DEMAND_FEATURES + TEMPORAL_FEATURES + WEATHER_FEATURES
NUM_FEATURES = len(ALL_FEATURES)

# Indices for target features
SPEED_TARGET_IDX = ALL_FEATURES.index("avg_speed")
PICKUP_TARGET_IDX = ALL_FEATURES.index("pickup_count")
DROPOFF_TARGET_IDX = ALL_FEATURES.index("dropoff_count")

# ─── Model Hyperparameters ──────────────────────────────────────────────────
D_MODEL = 64
NUM_HEADS = 4
N_SPATIAL_LAYERS = 2
N_TEMPORAL_LAYERS = 2
N_CROSS_LAYERS = 2
DROPOUT = 0.1
ACTIVATION = "gelu"
FF_DIM = D_MODEL * 4  # Feed-forward hidden dimension

# ─── Training Hyperparameters ───────────────────────────────────────────────
LEARNING_RATE = 3e-4
BATCH_SIZE = 64
MAX_EPOCHS = 60
PATIENCE = 5          # Early stopping patience
SEED = 42
MAX_GRAD_NORM = 1.0    # Gradient clipping
WEIGHT_DECAY = 1e-5    # AdamW weight decay
# ─── Task Loss Weights ───────────────────────────────────────────────────
SPEED_LOSS_WEIGHT = 6.0    # Give speed prediction 3x priority
DEMAND_LOSS_WEIGHT = 1.0
# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── CSV Chunk Size (memory management for large files) ─────────────────────
CHUNK_SIZE = 500_000

# ─── Model Names ─────────────────────────────────────────────────────────────
MODEL_NAMES = [
    "speed_only",
    "demand_only",
    "independent_dual",
    "joint",
]


def print_config():
    """Print current configuration for logging."""
    print("=" * 60)
    print("  Joint Speed-Demand Forecaster — Configuration")
    print("=" * 60)
    print(f"  Grid cell size   : {GRID_CELL_SIZE_M} m")
    print(f"  Grid dimensions  : {GRID_ROWS} rows × {GRID_COLS} cols = {GRID_ROWS * GRID_COLS} cells")
    print(f"  Input window     : {T_IN} hours")
    print(f"  Prediction horizon: {T_OUT} hours")
    print(f"  Features         : {NUM_FEATURES}")
    print(f"  d_model          : {D_MODEL}")
    print(f"  Attention heads  : {NUM_HEADS}")
    print(f"  Spatial layers   : {N_SPATIAL_LAYERS}")
    print(f"  Temporal layers  : {N_TEMPORAL_LAYERS}")
    print(f"  Cross-task layers: {N_CROSS_LAYERS}")
    print(f"  Batch size       : {BATCH_SIZE}")
    print(f"  Learning rate    : {LEARNING_RATE}")
    print(f"  Device           : {DEVICE}")
    print(f"  Train end        : {TRAIN_END}")
    print(f"  Val end          : {VAL_END}")
    print("=" * 60)


if __name__ == "__main__":
    print_config()
