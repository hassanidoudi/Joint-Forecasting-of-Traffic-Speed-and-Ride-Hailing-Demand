"""
baselines.py — Baseline models for comparison.

1. HistoricalAverage — mean speed/demand per (cell, hour, day_of_week)
2. SpeedOnlyTransformer — Transformer predicting only speed (no demand features)
3. DemandOnlyTransformer — Transformer predicting only demand (no speed features)
4. IndependentDualTransformer — Two separate transformers, no cross-task attention
5. JointForecaster (in model.py) — Full model with cross-task attention
"""

import torch
import torch.nn as nn
import numpy as np

import config
from model import (
    PatchEmbedding,
    SpatialPositionEncoding,
    TemporalPositionEncoding,
    PreNormTransformerLayer,
    SpeedDecoder,
    DemandDecoder,
    count_parameters,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Historical Average
# ═══════════════════════════════════════════════════════════════════════════════

class HistoricalAverage:
    """Non-parametric baseline: predict the training-set mean for the same
    (hour_of_day, day_of_week, grid_cell) combination.

    This is not a nn.Module — it's a lookup table.
    """

    def __init__(self):
        self.speed_table = None   # (24, 7, H, W)
        self.pickup_table = None  # (24, 7, H, W)
        self.dropoff_table = None # (24, 7, H, W)

    def fit(self, tensor, time_index, train_end):
        """Build lookup tables from training data.

        Parameters
        ----------
        tensor : np.ndarray, shape (T, H, W, F)
        time_index : pd.DatetimeIndex
        train_end : str or pd.Timestamp
        """
        import pandas as pd
        train_mask = time_index <= pd.Timestamp(train_end)
        train_idx = np.where(train_mask)[0]

        T_train = len(train_idx)
        _, H, W, F = tensor.shape

        speed_idx = config.ALL_FEATURES.index("avg_speed")
        pickup_idx = config.ALL_FEATURES.index("pickup_count")
        dropoff_idx = config.ALL_FEATURES.index("dropoff_count")

        # Accumulators
        speed_sum = np.zeros((24, 7, H, W), dtype=np.float64)
        pickup_sum = np.zeros((24, 7, H, W), dtype=np.float64)
        dropoff_sum = np.zeros((24, 7, H, W), dtype=np.float64)
        count = np.zeros((24, 7, 1, 1), dtype=np.float64)

        for t in train_idx:
            ts = time_index[t]
            h = ts.hour
            d = ts.dayofweek
            speed_sum[h, d] += tensor[t, :, :, speed_idx]
            pickup_sum[h, d] += tensor[t, :, :, pickup_idx]
            dropoff_sum[h, d] += tensor[t, :, :, dropoff_idx]
            count[h, d] += 1

        count = np.maximum(count, 1)
        self.speed_table = (speed_sum / count).astype(np.float32)
        self.pickup_table = (pickup_sum / count).astype(np.float32)
        self.dropoff_table = (dropoff_sum / count).astype(np.float32)

    def predict(self, time_index_future):
        """Predict for a sequence of future timestamps.

        Parameters
        ----------
        time_index_future : pd.DatetimeIndex
            Timestamps to predict for.

        Returns
        -------
        speed_pred : np.ndarray, shape (T, H, W)
        demand_pred : np.ndarray, shape (T, H, W, 2)
        """
        T_fut = len(time_index_future)
        H, W = self.speed_table.shape[2], self.speed_table.shape[3]

        speed_pred = np.zeros((T_fut, H, W), dtype=np.float32)
        demand_pred = np.zeros((T_fut, H, W, 2), dtype=np.float32)

        for i, ts in enumerate(time_index_future):
            h = ts.hour
            d = ts.dayofweek
            speed_pred[i] = self.speed_table[h, d]
            demand_pred[i, :, :, 0] = self.pickup_table[h, d]
            demand_pred[i, :, :, 1] = self.dropoff_table[h, d]

        return speed_pred, demand_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Single-Task Transformer (base class for Speed-Only and Demand-Only)
# ═══════════════════════════════════════════════════════════════════════════════

class SingleTaskTransformer(nn.Module):
    """Transformer that predicts only one task (speed or demand).

    Uses spatial + temporal attention but NO cross-task attention.
    Can optionally restrict input features to only task-relevant ones.
    """

    def __init__(
        self,
        task="speed",
        input_feature_indices=None,
        d_model=None,
        num_heads=None,
        n_spatial=None,
        n_temporal=None,
        t_in=None,
        t_out=None,
        grid_h=None,
        grid_w=None,
        patch_size=None,
        dropout=None,
    ):
        super().__init__()
        self.task = task
        d_model = d_model if d_model is not None else config.D_MODEL
        num_heads = num_heads if num_heads is not None else config.NUM_HEADS
        n_spatial = n_spatial if n_spatial is not None else config.N_SPATIAL_LAYERS
        n_temporal = n_temporal if n_temporal is not None else config.N_TEMPORAL_LAYERS
        t_in = t_in if t_in is not None else config.T_IN
        t_out = t_out if t_out is not None else config.T_OUT
        grid_h = grid_h if grid_h is not None else config.GRID_ROWS
        grid_w = grid_w if grid_w is not None else config.GRID_COLS
        patch_size = patch_size if patch_size is not None else config.PATCH_SIZE
        dropout = dropout if dropout is not None else config.DROPOUT
        ff_dim = d_model * 4

        self.d_model = d_model
        self.t_in = t_in
        self.t_out = t_out
        self.grid_h = grid_h
        self.grid_w = grid_w

        # Determine input features
        if input_feature_indices is not None:
            self.feature_indices = input_feature_indices
            num_features = len(input_feature_indices)
        else:
            self.feature_indices = None
            num_features = config.NUM_FEATURES

        self.patch_embed = PatchEmbedding(num_features, d_model, patch_size)

        h_p = grid_h // patch_size
        w_p = grid_w // patch_size
        self.spatial_pos = SpatialPositionEncoding(h_p, w_p, d_model)
        self.temporal_pos = TemporalPositionEncoding(t_in, d_model)

        self.spatial_layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(n_spatial)
        ])
        self.temporal_layers = nn.ModuleList([
            PreNormTransformerLayer(d_model, num_heads, ff_dim, dropout)
            for _ in range(n_temporal)
        ])

        self.temporal_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        # Learnable recency bias toward recent time steps
        self.recency_bias = nn.Parameter(torch.linspace(-1.0, 1.0, t_in))

        if task == "speed":
            self.decoder = SpeedDecoder(d_model, t_out, dropout)
        else:
            self.decoder = DemandDecoder(d_model, t_out, dropout)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        x: (B, T, H, W, F)
        """
        B, T, H, W, F_ = x.shape
        device = x.device
        x_orig = x  # keep full-feature input for residual connection

        # Select features if restricted
        if self.feature_indices is not None:
            x = x[:, :, :, :, self.feature_indices]

        tokens, H_p, W_p = self.patch_embed(x)
        S = H_p * W_p

        spatial_pe = self.spatial_pos(H_p, W_p, device)
        temporal_pe = self.temporal_pos(T, device)
        tokens = tokens + spatial_pe.unsqueeze(0).unsqueeze(0)
        tokens = tokens + temporal_pe.unsqueeze(0).unsqueeze(2)

        # Spatial attention
        tokens = tokens.reshape(B * T, S, self.d_model)
        for layer in self.spatial_layers:
            tokens = layer(tokens)
        tokens = tokens.reshape(B, T, S, self.d_model)

        # Temporal attention
        tokens = tokens.permute(0, 2, 1, 3).reshape(B * S, T, self.d_model)
        for layer in self.temporal_layers:
            tokens = layer(tokens)
        tokens = tokens.reshape(B, S, T, self.d_model)

        # Temporal pooling with recency bias
        time_weights = self.temporal_pool(tokens)
        recency = self.recency_bias[:T].reshape(1, 1, T, 1)
        time_weights = time_weights + recency
        time_weights = torch.softmax(time_weights, dim=2)
        tokens_pooled = (tokens * time_weights).sum(dim=2)

        # Decode
        if self.task == "speed":
            out = self.decoder(tokens_pooled)
        else:
            out = self.decoder(tokens_pooled)

        if self.task == "speed":
            out = out.reshape(B, H_p, W_p, self.t_out)
            # Decaying residual: strong at 1h, weaker at 6h
            last_speed = x_orig[:, -1, :, :, config.SPEED_TARGET_IDX]  # (B, H, W)
            mean_speed = x_orig[:, :, :, :, config.SPEED_TARGET_IDX].mean(dim=1)  # (B, H, W)
            decay = torch.exp(-0.25 * torch.arange(self.t_out, device=x_orig.device, dtype=torch.float32))
            # anchor: (B, T_out, H, W)
            anchor = (decay.reshape(1, -1, 1, 1) * last_speed.unsqueeze(1) +
                      (1 - decay).reshape(1, -1, 1, 1) * mean_speed.unsqueeze(1))
            out = out.permute(0, 3, 1, 2) + anchor  # (B, T_out, H, W)
            out = out.unsqueeze(-1)  # (B, T_out, H, W, 1)
        else:
            out = out.reshape(B, H_p, W_p, self.t_out, 2)
            out = out.permute(0, 3, 1, 2, 4)  # (B, T_out, H, W, 2)

        return out


def build_speed_only_transformer():
    """Speed-only model: uses only speed + temporal + weather features."""
    speed_feat_idx = [config.ALL_FEATURES.index(f) for f in
                      config.SPEED_FEATURES + config.TEMPORAL_FEATURES + config.WEATHER_FEATURES]
    return SingleTaskTransformer(task="speed", input_feature_indices=speed_feat_idx)


def build_demand_only_transformer():
    """Demand-only model: uses only demand + temporal + weather features."""
    demand_feat_idx = [config.ALL_FEATURES.index(f) for f in
                       config.DEMAND_FEATURES + config.TEMPORAL_FEATURES + config.WEATHER_FEATURES]
    return SingleTaskTransformer(task="demand", input_feature_indices=demand_feat_idx)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Independent Dual Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class IndependentDualTransformer(nn.Module):
    """Two separate transformers: one for speed, one for demand.

    No cross-task attention — this is the key ablation baseline.
    Uses ALL features as input to both branches (unlike the single-task models).
    """

    def __init__(self):
        super().__init__()
        self.speed_model = SingleTaskTransformer(
            task="speed", input_feature_indices=None
        )
        self.demand_model = SingleTaskTransformer(
            task="demand", input_feature_indices=None
        )

    def forward(self, x):
        """
        x: (B, T, H, W, F)
        Returns: speed_pred (B, T_out, H, W, 1), demand_pred (B, T_out, H, W, 2)
        """
        speed_pred = self.speed_model(x)
        demand_pred = self.demand_model(x)
        return speed_pred, demand_pred


# ═══════════════════════════════════════════════════════════════════════════════
# Single-Task Loss Wrappers
# ═══════════════════════════════════════════════════════════════════════════════

class SpeedOnlyLoss(nn.Module):
    """Loss for speed-only model with horizon weighting."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
        # Horizon weights: emphasise longer horizons (3h/6h)
        raw = torch.exp(0.04 * torch.arange(config.T_OUT, dtype=torch.float32))
        self.register_buffer("horizon_weights", raw / raw.mean())

    def forward(self, speed_pred, speed_target, demand_pred=None, demand_target=None):
        per_element = self.loss_fn(speed_pred.squeeze(-1), speed_target)  # (B, T_out, H, W)
        hw = self.horizon_weights.reshape(1, -1, 1, 1)
        l_speed = (per_element * hw).mean()
        return l_speed, l_speed, torch.tensor(0.0, device=speed_pred.device)


class DemandOnlyLoss(nn.Module):
    """Loss for demand-only model."""
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.MSELoss(reduction="mean")

    def forward(self, speed_pred=None, speed_target=None,
                demand_pred=None, demand_target=None):
        l_demand = self.loss_fn(demand_pred, demand_target)
        return l_demand, torch.tensor(0.0, device=demand_pred.device), l_demand


# ═══════════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════════

def build_model(model_name):
    """Build a model by name.

    Returns (model, criterion) tuple.
    """
    from model import JointForecaster, MultiTaskLoss

    if model_name == "joint":
        model = JointForecaster()
        criterion = MultiTaskLoss()
    elif model_name == "speed_only":
        model = build_speed_only_transformer()
        criterion = SpeedOnlyLoss()
    elif model_name == "demand_only":
        model = build_demand_only_transformer()
        criterion = DemandOnlyLoss()
    elif model_name == "independent_dual":
        model = IndependentDualTransformer()
        criterion = MultiTaskLoss()
    elif model_name == "historical_average":
        return HistoricalAverage(), None
    else:
        raise ValueError(f"Unknown model: {model_name}")

    print(f"Model '{model_name}' — {count_parameters(model):,} parameters")
    return model, criterion


if __name__ == "__main__":
    config.print_config()

    for name in ["joint", "speed_only", "demand_only", "independent_dual"]:
        model, criterion = build_model(name)
        B = 2
        x = torch.randn(B, config.T_IN, config.GRID_ROWS, config.GRID_COLS,
                         config.NUM_FEATURES)

        if name in ("joint", "independent_dual"):
            speed_pred, demand_pred = model(x)
            print(f"  {name}: speed={speed_pred.shape}, demand={demand_pred.shape}")
        elif name == "speed_only":
            speed_pred = model(x)
            print(f"  {name}: speed={speed_pred.shape}")
        elif name == "demand_only":
            demand_pred = model(x)
            print(f"  {name}: demand={demand_pred.shape}")
        print()
