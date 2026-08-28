"""
dataset.py — PyTorch Dataset with sliding-window generation.

Loads the preprocessed grid tensor and creates (input, target) pairs
using a sliding window approach with proper temporal splits.
"""

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config


class TrafficDemandDataset(Dataset):
    """Sliding-window dataset for the joint speed-demand forecaster.

    Parameters
    ----------
    tensor : np.ndarray, shape (T, H, W, F)
        The full fused grid tensor.
    time_index : pd.DatetimeIndex
        Timestamps for each time step.
    scaler : dict
        {"mean": np.ndarray (F,), "std": np.ndarray (F,)} from training set.
    mode : str
        One of "train", "val", "test".
    t_in : int
        Number of input time steps (default: config.T_IN).
    t_out : int
        Number of output/prediction time steps (default: config.T_OUT).
    normalize : bool
        Whether to normalize features using the scaler.
    """

    def __init__(self, tensor, time_index, scaler, mode="train",
                 t_in=None, t_out=None, normalize=True):
        super().__init__()
        self.t_in = t_in or config.T_IN
        self.t_out = t_out or config.T_OUT
        self.scaler = scaler
        self.normalize = normalize
        self.mode = mode

        # Determine temporal bounds for this split
        train_end = np.datetime64(config.TRAIN_END)
        val_end = np.datetime64(config.VAL_END)

        time_np = time_index.values

        if mode == "train":
            mask = time_np <= train_end
        elif mode == "val":
            mask = (time_np > train_end) & (time_np <= val_end)
        elif mode == "test":
            mask = time_np > val_end
        else:
            raise ValueError(f"Unknown mode: {mode}")

        indices = np.where(mask)[0]
        self.start_idx = indices[0]
        self.end_idx = indices[-1]

        # For val/test, we need T_IN steps *before* the split start for context
        if mode in ("val", "test"):
            self.context_start = max(0, self.start_idx - self.t_in)
        else:
            self.context_start = self.start_idx

        # Store the tensor slice we need
        self.data = tensor.copy()

        # Normalize if requested
        if self.normalize:
            mean = scaler["mean"].reshape(1, 1, 1, -1)
            std = scaler["std"].reshape(1, 1, 1, -1)
            self.data = (self.data - mean) / std

        # Compute valid window start positions
        # A window starting at t uses data[t:t+t_in] as input
        # and data[t+t_in:t+t_in+t_out] as target
        self.window_starts = []

        if mode == "train":
            # Windows fully within training period
            for t in range(self.start_idx,
                           self.end_idx - self.t_in - self.t_out + 2):
                self.window_starts.append(t)
        elif mode == "val":
            # Input can reach into training period, but targets must be in val
            for t in range(self.context_start,
                           self.end_idx - self.t_in - self.t_out + 2):
                target_start = t + self.t_in
                target_end = t + self.t_in + self.t_out - 1
                if target_start >= self.start_idx and target_end <= self.end_idx:
                    self.window_starts.append(t)
        elif mode == "test":
            for t in range(self.context_start,
                           self.end_idx - self.t_in - self.t_out + 2):
                target_start = t + self.t_in
                target_end = t + self.t_in + self.t_out - 1
                if target_start >= self.start_idx and target_end <= self.end_idx:
                    self.window_starts.append(t)

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        t = self.window_starts[idx]

        # Input: (T_in, H, W, F)
        x = self.data[t: t + self.t_in].copy()

        # Speed target: (T_out, H, W)
        y_speed = self.data[
            t + self.t_in: t + self.t_in + self.t_out,
            :, :,
            config.SPEED_TARGET_IDX
        ].copy()

        # Demand targets: (T_out, H, W, 2) — pickup + dropoff
        y_demand = self.data[
            t + self.t_in: t + self.t_in + self.t_out,
            :, :,
            [config.PICKUP_TARGET_IDX, config.DROPOFF_TARGET_IDX]
        ].copy()

        return (
            torch.from_numpy(x).float(),
            torch.from_numpy(y_speed).float(),
            torch.from_numpy(y_demand).float(),
        )


def load_preprocessed_data():
    """Load the preprocessed tensor, time index, and scaler from disk."""
    tensor = np.load(os.path.join(config.TENSOR_DIR, "grid_tensor.npy"))
    with open(os.path.join(config.TENSOR_DIR, "time_index.pkl"), "rb") as f:
        time_index = pickle.load(f)
    with open(os.path.join(config.TENSOR_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(config.TENSOR_DIR, "grid_meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    return tensor, time_index, scaler, meta


def create_dataloaders(tensor, time_index, scaler, batch_size=None):
    """Create train, val, and test DataLoaders.

    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_ds = TrafficDemandDataset(tensor, time_index, scaler, mode="train")
    val_ds = TrafficDemandDataset(tensor, time_index, scaler, mode="val")
    test_ds = TrafficDemandDataset(tensor, time_index, scaler, mode="test")

    print(f"Dataset sizes — Train: {len(train_ds)}, Val: {len(val_ds)}, "
          f"Test: {len(test_ds)}")

    use_pin = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=use_pin, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=use_pin,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Quick test
    tensor, time_index, scaler, meta = load_preprocessed_data()
    print(f"Tensor shape: {tensor.shape}")
    print(f"Time range: {time_index[0]} — {time_index[-1]}")

    train_loader, val_loader, test_loader = create_dataloaders(
        tensor, time_index, scaler
    )

    # Check one batch
    for x, y_speed, y_demand in train_loader:
        print(f"Input:  {x.shape}")          # (B, T_in, H, W, F)
        print(f"Speed:  {y_speed.shape}")     # (B, T_out, H, W)
        print(f"Demand: {y_demand.shape}")    # (B, T_out, H, W, 2)
        break
