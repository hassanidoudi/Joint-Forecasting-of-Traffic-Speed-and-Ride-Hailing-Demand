"""
model.py — Spatial-Temporal Transformer with Cross-Task Attention.

Architecture:
  Input (T_in, H, W, F)
   → Patch Embedding + Positional Encodings
   → Spatial Self-Attention (N_s layers)
   → Temporal Self-Attention (N_t layers)
   → Cross-Task Attention (N_c layers)
   → Speed Decoder (MLP) + Demand Decoder (MLP)
   → Speed predictions (T_out, H, W, 1)
   → Demand predictions (T_out, H, W, 2)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import config


# ═══════════════════════════════════════════════════════════════════════════════
# Building Blocks
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Project each grid cell's feature vector into d_model dimensions.

    For small grids (e.g. 6×6), patch_size=1 means each cell is a token.
    For larger grids, patches group adjacent cells.
    """

    def __init__(self, num_features, d_model, patch_size=1):
        super().__init__()
        self.patch_size = patch_size
        self.input_dim = num_features * patch_size * patch_size
        self.proj = nn.Linear(self.input_dim, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        x: (B, T, H, W, F)
        Returns: (B, T, S, d_model) where S = (H/p)*(W/p)
        """
        B, T, H, W, F_ = x.shape
        p = self.patch_size

        if p > 1:
            # Reshape into patches
            H_p, W_p = H // p, W // p
            x = x.reshape(B, T, H_p, p, W_p, p, F_)
            x = x.permute(0, 1, 2, 4, 3, 5, 6)  # (B, T, H_p, W_p, p, p, F)
            x = x.reshape(B, T, H_p * W_p, p * p * F_)
        else:
            H_p, W_p = H, W
            x = x.reshape(B, T, H * W, F_)

        x = self.proj(x)
        x = self.norm(x)
        return x, H_p, W_p


class SpatialPositionEncoding(nn.Module):
    """Learnable 2D spatial position encoding (row + col embeddings)."""

    def __init__(self, max_h, max_w, d_model):
        super().__init__()
        self.row_embed = nn.Embedding(max_h, d_model // 2)
        self.col_embed = nn.Embedding(max_w, d_model // 2)
        self.max_h = max_h
        self.max_w = max_w

    def forward(self, H_p, W_p, device):
        """Returns: (S, d_model) where S = H_p * W_p."""
        rows = torch.arange(H_p, device=device)
        cols = torch.arange(W_p, device=device)
        row_emb = self.row_embed(rows)   # (H_p, d/2)
        col_emb = self.col_embed(cols)   # (W_p, d/2)
        # Outer product combination
        pos = torch.cat([
            row_emb.unsqueeze(1).expand(-1, W_p, -1),
            col_emb.unsqueeze(0).expand(H_p, -1, -1),
        ], dim=-1)  # (H_p, W_p, d_model)
        return pos.reshape(H_p * W_p, -1)  # (S, d_model)


class TemporalPositionEncoding(nn.Module):
    """Learnable temporal position encoding."""

    def __init__(self, max_t, d_model):
        super().__init__()
        self.embed = nn.Embedding(max_t, d_model)

    def forward(self, T, device):
        """Returns: (T, d_model)."""
        positions = torch.arange(T, device=device)
        return self.embed(positions)


class PreNormTransformerLayer(nn.Module):
    """Standard pre-norm Transformer encoder layer.

    Pre-LayerNorm → MultiHeadAttention → Residual
    Pre-LayerNorm → FFN → Residual
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, need_weights=False):
        """x: (B, seq_len, d_model)"""
        # Self-attention with pre-norm
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(
            x_norm, x_norm, x_norm, need_weights=need_weights
        )
        x = x + self.dropout(attn_out)

        # FFN with pre-norm
        x_norm = self.norm2(x)
        x = x + self.ffn(x_norm)

        if need_weights:
            return x, attn_weights
        return x


class CrossAttentionLayer(nn.Module):
    """Cross-attention between speed and demand token streams.

    This is the key innovation: explicitly models the bidirectional
    feedback loop between speed and demand.
    """

    def __init__(self, d_model, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # Speed attends to demand
        self.norm_s_q = nn.LayerNorm(d_model)
        self.norm_d_kv = nn.LayerNorm(d_model)
        self.cross_attn_s2d = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # Demand attends to speed
        self.norm_d_q = nn.LayerNorm(d_model)
        self.norm_s_kv = nn.LayerNorm(d_model)
        self.cross_attn_d2s = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )

        # FFNs for each stream
        self.norm_s_ff = nn.LayerNorm(d_model)
        self.ffn_s = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        self.norm_d_ff = nn.LayerNorm(d_model)
        self.ffn_d = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

        # Learnable gates — speed gate nearly shut to prevent
        # demand noise from degrading speed predictions
        # sigmoid(-5) ≈ 0.007 for speed, sigmoid(-2) ≈ 0.12 for demand
        self.gate_s = nn.Parameter(torch.tensor(-5.0))
        self.gate_d = nn.Parameter(torch.tensor(-2.0))

        self.dropout = nn.Dropout(dropout)

    def forward(self, speed_tokens, demand_tokens, need_weights=False):
        """
        speed_tokens:  (B, seq_len, d_model)
        demand_tokens: (B, seq_len, d_model)
        """
        # Speed attends to demand (speed = Q, demand = K,V)
        s_norm = self.norm_s_q(speed_tokens)
        d_norm_kv = self.norm_d_kv(demand_tokens)
        cross_s, attn_s2d = self.cross_attn_s2d(
            s_norm, d_norm_kv, d_norm_kv, need_weights=need_weights
        )
        speed_out = speed_tokens + torch.sigmoid(self.gate_s) * self.dropout(cross_s)

        # Demand attends to speed (demand = Q, speed = K,V)
        d_norm = self.norm_d_q(demand_tokens)
        s_norm_kv = self.norm_s_kv(speed_tokens)
        cross_d, attn_d2s = self.cross_attn_d2s(
            d_norm, s_norm_kv, s_norm_kv, need_weights=need_weights
        )
        demand_out = demand_tokens + torch.sigmoid(self.gate_d) * self.dropout(cross_d)

        # FFN for speed stream
        speed_out = speed_out + self.ffn_s(self.norm_s_ff(speed_out))
        # FFN for demand stream
        demand_out = demand_out + self.ffn_d(self.norm_d_ff(demand_out))

        if need_weights:
            return speed_out, demand_out, attn_s2d, attn_d2s
        return speed_out, demand_out


# ═══════════════════════════════════════════════════════════════════════════════
# Decoder Heads
# ═══════════════════════════════════════════════════════════════════════════════

class SpeedDecoder(nn.Module):
    """GRU-based autoregressive decoder for speed prediction.

    Each future step receives:
    - Spatial context (from encoder)
    - Horizon embedding (learnable per step)
    - Previous step's predicted delta (autoregressive feedback)
    """

    def __init__(self, d_model, t_out, dropout=0.1):
        super().__init__()
        self.t_out = t_out
        self.d_model = d_model
        # Horizon embedding: one vector per future step
        self.horizon_embed = nn.Embedding(t_out, d_model)
        self.norm = nn.LayerNorm(d_model)
        # GRU cell: input = context + previous delta
        self.gru = nn.GRUCell(d_model + 1, d_model)
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        # Horizon-adaptive output scaling: dampen GRU deltas at longer horizons
        # to reduce error accumulation. Learnable per-step scale, init ~1→0.5
        self.horizon_scale = nn.Parameter(
            torch.linspace(0.0, -0.7, t_out)  # sigmoid gives ~[0.50, 0.47, ..., 0.33]
        )

    def forward(self, x):
        """
        x: (B, S, d_model) where S = spatial tokens
        Returns: (B, S, T_out)
        """
        x = self.norm(x)  # (B, S, d_model)
        B, S, D = x.shape

        # Flatten batch and spatial dims for GRU processing
        x_flat = x.reshape(B * S, D)          # (B*S, D)
        h = x_flat                              # initial hidden state
        prev_delta = torch.zeros(B * S, 1, device=x.device)
        deltas = []

        for t in range(self.t_out):
            h_emb = self.horizon_embed.weight[t].unsqueeze(0)  # (1, D)
            ctx = x_flat + h_emb  # (B*S, D)
            gru_input = torch.cat([ctx, prev_delta], dim=-1)  # (B*S, D+1)
            h = self.gru(gru_input, h)          # (B*S, D)
            delta = self.output_proj(h)          # (B*S, 1)
            # Scale delta by learnable horizon-dependent factor
            scale = torch.sigmoid(self.horizon_scale[t])  # scalar
            delta = delta * scale
            deltas.append(delta)
            prev_delta = delta                   # feed back

        out = torch.cat(deltas, dim=-1)          # (B*S, T_out)
        return out.reshape(B, S, self.t_out)


class DemandDecoder(nn.Module):
    """MLP decoder for demand prediction (pickup + dropoff)."""

    def __init__(self, d_model, t_out, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, t_out * 2),  # 2 outputs: pickup + dropoff
        )
        self.t_out = t_out

    def forward(self, x):
        """
        x: (B, S, d_model) where S = spatial tokens
        Returns: (B, S, T_out, 2)
        """
        out = self.mlp(x)  # (B, S, T_out * 2)
        B, S, _ = out.shape
        return out.reshape(B, S, self.t_out, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# Main Model
# ═══════════════════════════════════════════════════════════════════════════════

class JointForecaster(nn.Module):
    """Spatial-Temporal Transformer with Cross-Task Attention for
    joint speed and demand forecasting.

    Parameters
    ----------
    num_features : int
        Number of input features per cell per time step.
    d_model : int
        Transformer hidden dimension.
    num_heads : int
        Number of attention heads.
    n_spatial : int
        Number of spatial self-attention layers.
    n_temporal : int
        Number of temporal self-attention layers.
    n_cross : int
        Number of cross-task attention layers.
    t_in : int
        Input window size (time steps).
    t_out : int
        Prediction horizon (time steps).
    grid_h, grid_w : int
        Grid dimensions.
    patch_size : int
        Spatial patch size (1 = each cell is a token).
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        num_features=None,
        d_model=None,
        num_heads=None,
        n_spatial=None,
        n_temporal=None,
        n_cross=None,
        t_in=None,
        t_out=None,
        grid_h=None,
        grid_w=None,
        patch_size=None,
        dropout=None,
    ):
        super().__init__()

        # Fill defaults from config (use `is not None` to allow 0 values)
        self.num_features = num_features if num_features is not None else config.NUM_FEATURES
        self.d_model = d_model if d_model is not None else config.D_MODEL
        self.num_heads = num_heads if num_heads is not None else config.NUM_HEADS
        self.n_spatial = n_spatial if n_spatial is not None else config.N_SPATIAL_LAYERS
        self.n_temporal = n_temporal if n_temporal is not None else config.N_TEMPORAL_LAYERS
        self.n_cross = n_cross if n_cross is not None else config.N_CROSS_LAYERS
        self.t_in = t_in if t_in is not None else config.T_IN
        self.t_out = t_out if t_out is not None else config.T_OUT
        self.grid_h = grid_h if grid_h is not None else config.GRID_ROWS
        self.grid_w = grid_w if grid_w is not None else config.GRID_COLS
        self.patch_size = patch_size if patch_size is not None else config.PATCH_SIZE
        self.dropout_rate = dropout if dropout is not None else config.DROPOUT
        ff_dim = self.d_model * 4

        # ── Embedding ──
        self.patch_embed = PatchEmbedding(
            self.num_features, self.d_model, self.patch_size
        )

        # ── Position Encodings ──
        h_p = self.grid_h // self.patch_size
        w_p = self.grid_w // self.patch_size
        self.spatial_pos = SpatialPositionEncoding(h_p, w_p, self.d_model)
        self.temporal_pos = TemporalPositionEncoding(self.t_in, self.d_model)

        # ── Spatial Self-Attention ──
        self.spatial_layers = nn.ModuleList([
            PreNormTransformerLayer(self.d_model, self.num_heads, ff_dim, self.dropout_rate)
            for _ in range(self.n_spatial)
        ])

        # ── Temporal Self-Attention ──
        self.temporal_layers = nn.ModuleList([
            PreNormTransformerLayer(self.d_model, self.num_heads, ff_dim, self.dropout_rate)
            for _ in range(self.n_temporal)
        ])

        # ── Task-Specific Projections ──
        self.speed_proj = nn.Linear(self.d_model, self.d_model)
        self.demand_proj = nn.Linear(self.d_model, self.d_model)

        # ── Cross-Task Attention ──
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(self.d_model, self.num_heads, ff_dim, self.dropout_rate)
            for _ in range(self.n_cross)
        ])

        # ── Temporal Aggregation (pool over time steps to produce one repr per spatial token) ──
        self.temporal_pool = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, 1),
        )

        # ── Recency gate: learnable bias toward recent time steps ──
        self.recency_bias = nn.Parameter(torch.linspace(-1.0, 1.0, self.t_in))  # (T_in,)

        # ── Decoder Heads ──
        self.speed_decoder = SpeedDecoder(self.d_model, self.t_out, self.dropout_rate)
        self.demand_decoder = DemandDecoder(self.d_model, self.t_out, self.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, need_weights=False):
        """
        x: (B, T_in, H, W, F)

        Returns
        -------
        speed_pred: (B, T_out, H, W, 1)
        demand_pred: (B, T_out, H, W, 2)
        attn_dict: dict (optional, only if need_weights=True)
        """
        B, T, H, W, F_ = x.shape
        device = x.device
        attn_dict = {}

        # ── Patch Embedding ──
        tokens, H_p, W_p = self.patch_embed(x)  # (B, T, S, d_model)
        S = H_p * W_p

        # ── Add Position Encodings ──
        spatial_pe = self.spatial_pos(H_p, W_p, device)    # (S, d_model)
        temporal_pe = self.temporal_pos(T, device)          # (T, d_model)

        tokens = tokens + spatial_pe.unsqueeze(0).unsqueeze(0)    # broadcast over B, T
        tokens = tokens + temporal_pe.unsqueeze(0).unsqueeze(2)   # broadcast over B, S

        # ── Spatial Self-Attention ──
        # Reshape: (B*T, S, d_model) — attention across spatial positions per time step
        tokens = tokens.reshape(B * T, S, self.d_model)
        for i, layer in enumerate(self.spatial_layers):
            if need_weights and i == len(self.spatial_layers) - 1:
                tokens, w = layer(tokens, need_weights=True)
                attn_dict["spatial_attn"] = w.detach()
            else:
                tokens = layer(tokens)
        tokens = tokens.reshape(B, T, S, self.d_model)

        # ── Temporal Self-Attention ──
        # Reshape: (B*S, T, d_model) — attention across time steps per spatial position
        tokens = tokens.permute(0, 2, 1, 3).reshape(B * S, T, self.d_model)
        for i, layer in enumerate(self.temporal_layers):
            if need_weights and i == len(self.temporal_layers) - 1:
                tokens, w = layer(tokens, need_weights=True)
                attn_dict["temporal_attn"] = w.detach()
            else:
                tokens = layer(tokens)
        tokens = tokens.reshape(B, S, T, self.d_model)
        # tokens: (B, S, T, d_model)

        # ── Temporal Aggregation ──
        # Compute attention-weighted pool with recency bias
        time_weights = self.temporal_pool(tokens)  # (B, S, T, 1)
        # Add learnable recency bias (favours recent time steps)
        recency = self.recency_bias[:T].reshape(1, 1, T, 1)  # (1, 1, T, 1)
        time_weights = time_weights + recency
        time_weights = torch.softmax(time_weights, dim=2)
        tokens_pooled = (tokens * time_weights).sum(dim=2)  # (B, S, d_model)

        # ── Task-Specific Projections ──
        speed_tokens = self.speed_proj(tokens_pooled)    # (B, S, d_model)
        demand_tokens = self.demand_proj(tokens_pooled)  # (B, S, d_model)

        # ── Cross-Task Attention ──
        for i, layer in enumerate(self.cross_layers):
            if need_weights and i == len(self.cross_layers) - 1:
                speed_tokens, demand_tokens, attn_s2d, attn_d2s = layer(
                    speed_tokens, demand_tokens, need_weights=True
                )
                attn_dict["cross_s2d"] = attn_s2d.detach()
                attn_dict["cross_d2s"] = attn_d2s.detach()
            else:
                speed_tokens, demand_tokens = layer(speed_tokens, demand_tokens)

        # ── Decode ──
        speed_delta = self.speed_decoder(speed_tokens)  # (B, S, T_out)
        demand_pred = self.demand_decoder(demand_tokens)  # (B, S, T_out, 2)

        # ── Decaying residual speed prediction ──
        # Residual strength decays with horizon: strong at 1h, weaker at 6h
        last_speed = x[:, -1, :, :, config.SPEED_TARGET_IDX]  # (B, H, W)
        last_speed = last_speed.reshape(B, S, 1)  # (B, S, 1)
        # Mean speed over the input window as a fallback anchor
        mean_speed = x[:, :, :, :, config.SPEED_TARGET_IDX].mean(dim=1)  # (B, H, W)
        mean_speed = mean_speed.reshape(B, S, 1)  # (B, S, 1)
        # Decay factor: 1.0 at h=0 → ~0.29 at h=T_out-1 (faster decay → lean toward mean at long horizon)
        decay = torch.exp(-0.25 * torch.arange(self.t_out, device=device, dtype=torch.float32))
        decay = decay.reshape(1, 1, self.t_out)  # (1, 1, T_out)
        # Blended anchor: recent speed at short horizon, mean speed at long horizon
        anchor = decay * last_speed + (1 - decay) * mean_speed  # (B, S, T_out)
        speed_pred = speed_delta + anchor

        # Reshape to grid
        speed_pred = speed_pred.reshape(B, H_p, W_p, self.t_out)
        speed_pred = speed_pred.permute(0, 3, 1, 2).unsqueeze(-1)  # (B, T_out, H, W, 1)

        demand_pred = demand_pred.reshape(B, H_p, W_p, self.t_out, 2)
        demand_pred = demand_pred.permute(0, 3, 1, 2, 4)  # (B, T_out, H, W, 2)

        if need_weights:
            return speed_pred, demand_pred, attn_dict
        return speed_pred, demand_pred


# ═══════════════════════════════════════════════════════════════════════════════
# Multi-Task Loss with Uncertainty Weighting (Kendall et al., 2018)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiTaskLoss(nn.Module):
    """Uncertainty-weighted multi-task loss.

    L = (1 / (2 * exp(log_var_s))) * L_speed + log_var_s / 2
      + (1 / (2 * exp(log_var_d))) * L_demand + log_var_d / 2

    Uses Huber loss for speed, MSE for demand.
    """

    def __init__(self):
        super().__init__()
        # Learnable log-variance parameters
        self.log_var_speed = nn.Parameter(torch.zeros(1))
        self.log_var_demand = nn.Parameter(torch.zeros(1))

        # Huber loss for speed (robust to noisy outliers), MSE for demand
        self.speed_loss_fn = nn.HuberLoss(reduction="none", delta=1.0)
        self.demand_loss_fn = nn.MSELoss(reduction="mean")

        # Horizon weights: emphasise LONGER horizons (3h/6h) which are harder
        # exp(+0.04*h) gives [1.0, 1.04, 1.08, 1.13, 1.17, 1.22]
        raw = torch.exp(0.04 * torch.arange(config.T_OUT, dtype=torch.float32))
        self.register_buffer(
            "horizon_weights", raw / raw.mean()
        )  # shape (T_out,)

    def forward(self, speed_pred, speed_target, demand_pred, demand_target):
        """
        speed_pred: (B, T_out, H, W, 1)
        speed_target: (B, T_out, H, W)
        demand_pred: (B, T_out, H, W, 2)
        demand_target: (B, T_out, H, W, 2)
        """
        # Speed loss with horizon weighting
        per_element = self.speed_loss_fn(speed_pred.squeeze(-1), speed_target)  # (B, T_out, H, W)
        hw = self.horizon_weights.reshape(1, -1, 1, 1)  # (1, T_out, 1, 1)
        l_speed = (per_element * hw).mean()

        # Demand loss (pickup + dropoff)
        l_demand = self.demand_loss_fn(demand_pred, demand_target)

        # Uncertainty weighting (Kendall et al. 2018) with explicit speed boost
        precision_s = torch.exp(-self.log_var_speed)
        precision_d = torch.exp(-self.log_var_demand)

        total = 0.5 * (config.SPEED_LOSS_WEIGHT * precision_s * l_speed + self.log_var_speed +
                       config.DEMAND_LOSS_WEIGHT * precision_d * l_demand + self.log_var_demand)

        return total, l_speed, l_demand

    def get_weights(self):
        """Return current effective task weights."""
        w_s = torch.exp(-self.log_var_speed).item()
        w_d = torch.exp(-self.log_var_demand).item()
        return w_s, w_d


# ═══════════════════════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config.print_config()
    model = JointForecaster()
    print(f"\nModel parameters: {count_parameters(model):,}")

    # Test forward pass
    B = 2
    x = torch.randn(B, config.T_IN, config.GRID_ROWS, config.GRID_COLS,
                     config.NUM_FEATURES)
    speed_pred, demand_pred = model(x)
    print(f"Speed pred shape:  {speed_pred.shape}")   # (2, 6, H, W, 1)
    print(f"Demand pred shape: {demand_pred.shape}")   # (2, 6, H, W, 2)

    # Test loss
    criterion = MultiTaskLoss()
    y_speed = torch.randn(B, config.T_OUT, config.GRID_ROWS, config.GRID_COLS)
    y_demand = torch.randn(B, config.T_OUT, config.GRID_ROWS, config.GRID_COLS, 2)
    loss, l_s, l_d = criterion(speed_pred, y_speed, demand_pred, y_demand)
    print(f"Total loss: {loss.item():.4f}, Speed: {l_s.item():.4f}, "
          f"Demand: {l_d.item():.4f}")
