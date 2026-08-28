"""
dataset_v2.py — LagAwareDataset: same sliding windows as TrafficDemandDataset,
plus per-target-step demand lags for the model_v4 mixture head.

For each target time step (window start t, horizon h, target index t+T_IN+h):
  lag168 = normalized demand observed at t+T_IN+h-168   (same hour, last week)
  lag24  = normalized demand observed at t+T_IN+h-24    (same hour, yesterday)

Both are strictly in the past relative to the target — no leakage. For val and
test windows whose lags reach back into the training period, that is exactly
what a deployed forecaster would do (use observed history), so it is
methodologically clean. lag24 is always available (index t+h >= 0). lag168 is
unavailable only for the first 144 training windows (first ~6 days of data);
there we fall back to lag24 so the mixture always receives valid values.

Returned sample: (x, y_speed, y_demand, demand_lag)
  demand_lag shape: (T_OUT, H, W, 2, 2) with [..., 0]=lag168, [..., 1]=lag24,
  in the same normalized space as y_demand and the model's demand output.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader

import config
from dataset import TrafficDemandDataset, load_preprocessed_data


DEMAND_IDX = None  # resolved lazily so config edits are always respected


def _demand_idx():
    global DEMAND_IDX
    if DEMAND_IDX is None:
        DEMAND_IDX = [config.PICKUP_TARGET_IDX, config.DROPOFF_TARGET_IDX]
    return DEMAND_IDX


class LagAwareDataset(TrafficDemandDataset):
    LAG_WEEK = 168
    LAG_DAY = 24

    def __getitem__(self, idx):
        x, y_speed, y_demand = super().__getitem__(idx)

        t = self.window_starts[idx]
        dem_idx = _demand_idx()
        t_out = self.t_out

        h_dim, w_dim = self.data.shape[1], self.data.shape[2]
        lags = np.empty((t_out, h_dim, w_dim, 2, 2), dtype=np.float32)

        for h in range(t_out):
            tgt = t + self.t_in + h
            lag24 = self.data[tgt - self.LAG_DAY][:, :, dem_idx]
            t168 = tgt - self.LAG_WEEK
            lag168 = self.data[t168][:, :, dem_idx] if t168 >= 0 else lag24
            lags[h, :, :, :, 0] = lag168
            lags[h, :, :, :, 1] = lag24

        return x, y_speed, y_demand, torch.from_numpy(lags)


def create_lag_dataloaders(tensor, time_index, scaler, batch_size=None):
    """Train/val/test loaders yielding 4-tuples (x, y_speed, y_demand, lag)."""
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    train_ds = LagAwareDataset(tensor, time_index, scaler, mode="train")
    val_ds = LagAwareDataset(tensor, time_index, scaler, mode="val")
    test_ds = LagAwareDataset(tensor, time_index, scaler, mode="test")

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
    tensor, time_index, scaler, meta = load_preprocessed_data()
    config.GRID_ROWS = meta["n_rows"]
    config.GRID_COLS = meta["n_cols"]
    train_loader, val_loader, test_loader = create_lag_dataloaders(
        tensor, time_index, scaler
    )
    for x, y_speed, y_demand, lag in train_loader:
        print(f"x: {tuple(x.shape)}  y_speed: {tuple(y_speed.shape)}  "
              f"y_demand: {tuple(y_demand.shape)}  lag: {tuple(lag.shape)}")
        break
