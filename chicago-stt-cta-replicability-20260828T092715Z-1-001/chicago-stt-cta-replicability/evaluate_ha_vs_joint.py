"""
evaluate_ha_vs_joint.py — Focused evaluation: Historical Average vs Joint model.

Generates side-by-side comparison plots and a summary table for the paper.

Usage:
    python evaluate_ha_vs_joint.py
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import config
from dataset import load_preprocessed_data, TrafficDemandDataset
from model import JointForecaster, MultiTaskLoss
from baselines import build_model, HistoricalAverage

sns.set_theme(style="whitegrid", font_scale=1.1)

MODELS_TO_COMPARE = ["historical_average", "joint"]
COLORS = {"historical_average": "#7B8794", "joint": "#E63946"}
LABELS = {"historical_average": "Historical Average", "joint": "Joint ST-Transformer"}


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mae(pred, target):
    mask = ~(np.isnan(pred) | np.isnan(target))
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(pred[mask] - target[mask]))


def compute_rmse(pred, target):
    mask = ~(np.isnan(pred) | np.isnan(target))
    if mask.sum() == 0:
        return 0.0
    return np.sqrt(np.mean((pred[mask] - target[mask]) ** 2))


def compute_mape(pred, target, eps=1e-8):
    mask = np.abs(target) > eps
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs((pred[mask] - target[mask]) / target[mask])) * 100


def compute_all_metrics(pred, target, label=""):
    return {
        "label": label,
        "MAE": compute_mae(pred, target),
        "RMSE": compute_rmse(pred, target),
        "MAPE": compute_mape(pred, target),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_trained_model(model_name):
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] No checkpoint for '{model_name}'")
        return None, None

    if model_name == "historical_average":
        with open(ckpt_path, "rb") as f:
            model = pickle.load(f)
        return model, None

    model, criterion = build_model(model_name)
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if criterion is not None and "criterion_state_dict" in ckpt:
        criterion.load_state_dict(ckpt["criterion_state_dict"])
    model = model.to(config.DEVICE).eval()
    return model, criterion


@torch.no_grad()
def run_joint_inference(model, test_loader, scaler):
    device = config.DEVICE
    all_sp, all_st, all_dp, all_dt = [], [], [], []
    mean, std = scaler["mean"], scaler["std"]

    for x, y_speed, y_demand in test_loader:
        x = x.to(device)
        speed_pred, demand_pred = model(x)
        all_sp.append(speed_pred.squeeze(-1).cpu().numpy())
        all_st.append(y_speed.numpy())
        all_dp.append(demand_pred.cpu().numpy())
        all_dt.append(y_demand.numpy())

    sp = np.concatenate(all_sp); st = np.concatenate(all_st)
    dp = np.concatenate(all_dp); dt = np.concatenate(all_dt)

    # Denormalize
    s_m, s_s = mean[config.SPEED_TARGET_IDX], std[config.SPEED_TARGET_IDX]
    sp = sp * s_s + s_m;  st = st * s_s + s_m

    pu_m, pu_s = mean[config.PICKUP_TARGET_IDX], std[config.PICKUP_TARGET_IDX]
    do_m, do_s = mean[config.DROPOFF_TARGET_IDX], std[config.DROPOFF_TARGET_IDX]
    dp[:, :, :, :, 0] = dp[:, :, :, :, 0] * pu_s + pu_m
    dp[:, :, :, :, 1] = dp[:, :, :, :, 1] * do_s + do_m
    dt[:, :, :, :, 0] = dt[:, :, :, :, 0] * pu_s + pu_m
    dt[:, :, :, :, 1] = dt[:, :, :, :, 1] * do_s + do_m

    return sp, st, dp, dt


def run_ha_inference(model, tensor, time_index):
    val_end = pd.Timestamp(config.VAL_END)
    test_mask = time_index > val_end
    test_indices = np.where(test_mask)[0]

    speed_idx = config.ALL_FEATURES.index("avg_speed")
    pickup_idx = config.ALL_FEATURES.index("pickup_count")
    dropoff_idx = config.ALL_FEATURES.index("dropoff_count")

    all_sp, all_st, all_dp, all_dt = [], [], [], []
    T_out, T_in = config.T_OUT, config.T_IN

    for i in range(0, len(test_indices) - T_in - T_out + 1):
        t_start = test_indices[i] + T_in
        t_end = t_start + T_out
        if t_end > len(time_index):
            break

        future_times = time_index[t_start:t_end]
        sp, dp = model.predict(future_times)

        all_sp.append(sp)
        all_st.append(tensor[t_start:t_end, :, :, speed_idx])
        all_dp.append(np.stack([dp[:, :, :, 0], dp[:, :, :, 1]], axis=-1))
        all_dt.append(np.stack([
            tensor[t_start:t_end, :, :, pickup_idx],
            tensor[t_start:t_end, :, :, dropoff_idx],
        ], axis=-1))

    return (np.array(all_sp), np.array(all_st),
            np.array(all_dp), np.array(all_dt))


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_all_horizons(sp, st, dp, dt):
    """Return per-horizon + overall metrics."""
    results = []
    results.append(compute_all_metrics(sp.flatten(), st.flatten(), "Speed_overall"))
    results.append(compute_all_metrics(dp.flatten(), dt.flatten(), "Demand_overall"))

    horizons = {"1h": 0, "3h": 2, "6h": 5}
    for name, idx in horizons.items():
        if idx < sp.shape[1]:
            results.append(compute_all_metrics(sp[:, idx].flatten(),
                                               st[:, idx].flatten(), f"Speed_{name}"))
            results.append(compute_all_metrics(dp[:, idx].flatten(),
                                               dt[:, idx].flatten(), f"Demand_{name}"))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Plots
# ═══════════════════════════════════════════════════════════════════════════════

def plot_horizon_comparison(all_results):
    """Side-by-side bar chart: Speed MAE and Demand MAPE by horizon."""
    horizons = ["1h", "3h", "6h", "overall"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Speed MAE ──
    x = np.arange(len(horizons))
    width = 0.3
    for i, mname in enumerate(MODELS_TO_COMPARE):
        vals = []
        for h in horizons:
            r = next((r for r in all_results[mname]
                       if r["label"] == f"Speed_{h}"), None)
            vals.append(r["MAE"] if r else 0)
        bars = axes[0].bar(x + i * width, vals, width,
                           label=LABELS[mname], color=COLORS[mname])
        for bar in bars:
            axes[0].annotate(f"{bar.get_height():.3f}",
                             xy=(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=9)

    axes[0].set_title("Speed MAE by Forecast Horizon")
    axes[0].set_xticks(x + width / 2)
    axes[0].set_xticklabels(horizons)
    axes[0].set_ylabel("MAE (mph)")
    axes[0].legend()

    # ── Demand MAPE ──
    for i, mname in enumerate(MODELS_TO_COMPARE):
        vals = []
        for h in horizons:
            r = next((r for r in all_results[mname]
                       if r["label"] == f"Demand_{h}"), None)
            vals.append(r["MAPE"] if r else 0)
        bars = axes[1].bar(x + i * width, vals, width,
                           label=LABELS[mname], color=COLORS[mname])
        for bar in bars:
            axes[1].annotate(f"{bar.get_height():.1f}%",
                             xy=(bar.get_x() + bar.get_width() / 2,
                                 bar.get_height()),
                             xytext=(0, 3), textcoords="offset points",
                             ha="center", fontsize=9)

    axes[1].set_title("Demand MAPE by Forecast Horizon")
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels(horizons)
    axes[1].set_ylabel("MAPE (%)")
    axes[1].legend()

    fig.tight_layout()
    path = os.path.join(config.FIGURE_DIR, "ha_vs_joint_horizons.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_spatial_comparison(predictions):
    """Side-by-side spatial MAE heatmaps for both models."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    for col, mname in enumerate(MODELS_TO_COMPARE):
        sp, st, dp, dt = predictions[mname]

        # Speed MAE per cell
        speed_mae = np.nanmean(np.abs(sp - st), axis=(0, 1))
        im1 = axes[0, col].imshow(speed_mae, cmap="YlOrRd",
                                   aspect="auto", origin="lower")
        axes[0, col].set_title(f"Speed MAE — {LABELS[mname]}")
        axes[0, col].set_xlabel("Column")
        axes[0, col].set_ylabel("Row")
        plt.colorbar(im1, ax=axes[0, col], label="MAE (mph)")

        # Demand MAE per cell (total pickup + dropoff)
        demand_total_p = np.nansum(dp, axis=-1)
        demand_total_t = np.nansum(dt, axis=-1)
        demand_mae = np.nanmean(np.abs(demand_total_p - demand_total_t), axis=(0, 1))
        im2 = axes[1, col].imshow(demand_mae, cmap="YlOrRd",
                                   aspect="auto", origin="lower")
        axes[1, col].set_title(f"Demand MAE — {LABELS[mname]}")
        axes[1, col].set_xlabel("Column")
        axes[1, col].set_ylabel("Row")
        plt.colorbar(im2, ax=axes[1, col], label="MAE (trips)")

    fig.suptitle("Spatial Error Distribution: Historical Average vs Joint",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(config.FIGURE_DIR, "ha_vs_joint_spatial.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_improvement_map(predictions):
    """Heatmap of MAE improvement (HA − Joint). Positive = Joint is better."""
    ha_sp, ha_st, ha_dp, ha_dt = predictions["historical_average"]
    jt_sp, jt_st, jt_dp, jt_dt = predictions["joint"]

    # Align sample counts (HA may have slightly different count)
    n = min(ha_sp.shape[0], jt_sp.shape[0])

    ha_speed_mae = np.nanmean(np.abs(ha_sp[:n] - ha_st[:n]), axis=(0, 1))
    jt_speed_mae = np.nanmean(np.abs(jt_sp[:n] - jt_st[:n]), axis=(0, 1))
    speed_improve = ha_speed_mae - jt_speed_mae  # positive = joint better

    ha_demand_mae = np.nanmean(np.abs(np.nansum(ha_dp[:n], axis=-1) -
                                       np.nansum(ha_dt[:n], axis=-1)), axis=(0, 1))
    jt_demand_mae = np.nanmean(np.abs(np.nansum(jt_dp[:n], axis=-1) -
                                       np.nansum(jt_dt[:n], axis=-1)), axis=(0, 1))
    demand_improve = ha_demand_mae - jt_demand_mae

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    vmax_s = max(abs(speed_improve.min()), abs(speed_improve.max()))
    im1 = axes[0].imshow(speed_improve, cmap="RdBu", aspect="auto",
                          origin="lower", vmin=-vmax_s, vmax=vmax_s)
    axes[0].set_title("Speed MAE Improvement\n(HA − Joint; blue = Joint better)")
    axes[0].set_xlabel("Column"); axes[0].set_ylabel("Row")
    plt.colorbar(im1, ax=axes[0], label="ΔMAE (mph)")

    vmax_d = max(abs(demand_improve.min()), abs(demand_improve.max()))
    im2 = axes[1].imshow(demand_improve, cmap="RdBu", aspect="auto",
                          origin="lower", vmin=-vmax_d, vmax=vmax_d)
    axes[1].set_title("Demand MAE Improvement\n(HA − Joint; blue = Joint better)")
    axes[1].set_xlabel("Column"); axes[1].set_ylabel("Row")
    plt.colorbar(im2, ax=axes[1], label="ΔMAE (trips)")

    fig.suptitle("Per-Cell Improvement: Joint over Historical Average", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(config.FIGURE_DIR, "ha_vs_joint_improvement.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_timeseries_overlay(predictions, time_index, tensor):
    """Overlay HA and Joint predictions on the same time-series plot."""
    ha_sp, ha_st, _, _ = predictions["historical_average"]
    jt_sp, jt_st, _, _ = predictions["joint"]

    n = min(ha_sp.shape[0], jt_sp.shape[0])
    H, W = ha_st.shape[2], ha_st.shape[3]

    # Pick 3 highest-variance cells
    var_map = np.nanvar(jt_st[:n, 0], axis=0)  # horizon 0 variance
    flat_idx = np.argsort(var_map.flatten())[::-1]
    cells = [(idx // W, idx % W) for idx in flat_idx[:3]]

    steps = min(72, n)  # 3 days
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    for i, (r, c) in enumerate(cells):
        actual = jt_st[:steps, 0, r, c]
        ha_pred = ha_sp[:steps, 0, r, c]
        jt_pred = jt_sp[:steps, 0, r, c]

        axes[i].plot(actual, label="Actual", color="#1B2838", linewidth=1.5)
        axes[i].plot(ha_pred, label="Historical Avg", color=COLORS["historical_average"],
                     linewidth=1.2, alpha=0.8, linestyle="--")
        axes[i].plot(jt_pred, label="Joint Model", color=COLORS["joint"],
                     linewidth=1.2, alpha=0.8)
        axes[i].set_ylabel("Speed (mph)")
        axes[i].set_title(f"Cell ({r}, {c})")
        axes[i].legend(loc="upper right", fontsize=9)

    axes[-1].set_xlabel("Test Sample (hours)")
    fig.suptitle("Speed Predictions: Historical Average vs Joint (1h horizon)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(config.FIGURE_DIR, "ha_vs_joint_timeseries.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_error_distribution(predictions):
    """Histogram of prediction errors for both models."""
    ha_sp, ha_st, _, _ = predictions["historical_average"]
    jt_sp, jt_st, _, _ = predictions["joint"]

    n = min(ha_sp.shape[0], jt_sp.shape[0])

    ha_err = (ha_sp[:n] - ha_st[:n]).flatten()
    jt_err = (jt_sp[:n] - jt_st[:n]).flatten()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Speed error distribution
    axes[0].hist(ha_err, bins=80, alpha=0.6, color=COLORS["historical_average"],
                 label="Historical Avg", density=True)
    axes[0].hist(jt_err, bins=80, alpha=0.6, color=COLORS["joint"],
                 label="Joint Model", density=True)
    axes[0].axvline(0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_title("Speed Prediction Error Distribution")
    axes[0].set_xlabel("Error (mph)")
    axes[0].set_ylabel("Density")
    axes[0].legend()

    # Error by horizon
    horizons = {"1h": 0, "3h": 2, "6h": 5}
    ha_mae_h = [compute_mae(ha_sp[:n, idx].flatten(), ha_st[:n, idx].flatten())
                for idx in horizons.values()]
    jt_mae_h = [compute_mae(jt_sp[:n, idx].flatten(), jt_st[:n, idx].flatten())
                for idx in horizons.values()]

    x = np.arange(len(horizons))
    width = 0.3
    axes[1].bar(x, ha_mae_h, width, label="Historical Avg",
                color=COLORS["historical_average"])
    axes[1].bar(x + width, jt_mae_h, width, label="Joint Model",
                color=COLORS["joint"])
    axes[1].set_xticks(x + width / 2)
    axes[1].set_xticklabels(list(horizons.keys()))
    axes[1].set_ylabel("MAE (mph)")
    axes[1].set_title("Speed MAE: Horizon Breakdown")
    axes[1].legend()

    # Add improvement % annotations
    for j, (ha_v, jt_v) in enumerate(zip(ha_mae_h, jt_mae_h)):
        pct = (ha_v - jt_v) / ha_v * 100
        color = "green" if pct > 0 else "red"
        axes[1].annotate(f"{pct:+.1f}%",
                         xy=(x[j] + width, jt_v),
                         xytext=(0, 8), textcoords="offset points",
                         ha="center", fontsize=10, fontweight="bold", color=color)

    fig.tight_layout()
    path = os.path.join(config.FIGURE_DIR, "ha_vs_joint_errors.png")
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  Saved: {path}")


def print_summary_table(all_results):
    """Print a nicely formatted comparison table."""
    print("\n" + "=" * 72)
    print("  HISTORICAL AVERAGE vs JOINT MODEL — Summary")
    print("=" * 72)

    header = f"  {'Metric':<20s}"
    for mname in MODELS_TO_COMPARE:
        header += f"  {LABELS[mname]:>20s}"
    header += f"  {'Δ Improvement':>14s}"
    print(header)
    print("  " + "─" * 68)

    for label in ["Speed_overall", "Speed_1h", "Speed_3h", "Speed_6h",
                   "Demand_overall", "Demand_1h", "Demand_3h", "Demand_6h"]:
        row = f"  {label:<20s}"
        vals = []
        for mname in MODELS_TO_COMPARE:
            r = next((r for r in all_results[mname] if r["label"] == label), None)
            if r:
                if "Speed" in label:
                    row += f"  {r['MAE']:>20.3f}"
                    vals.append(r["MAE"])
                else:
                    row += f"  {r['MAPE']:>19.1f}%"
                    vals.append(r["MAPE"])
            else:
                row += f"  {'N/A':>20s}"
                vals.append(None)

        if len(vals) == 2 and all(v is not None for v in vals):
            delta = vals[0] - vals[1]
            pct = delta / vals[0] * 100 if vals[0] != 0 else 0
            sign = "+" if delta > 0 else ""
            row += f"  {sign}{pct:>12.1f}%"
        print(row)

    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  Evaluation: Historical Average vs Joint Model")
    print("=" * 60)
    print(f"  Config: T_IN={config.T_IN}, T_OUT={config.T_OUT}, "
          f"D_MODEL={config.D_MODEL}")
    print(f"  Spatial={config.N_SPATIAL_LAYERS}, Temporal={config.N_TEMPORAL_LAYERS}, "
          f"Cross={config.N_CROSS_LAYERS}")
    print(f"  MAX_EPOCHS={config.MAX_EPOCHS}, LR={config.LEARNING_RATE}")
    print()

    print("Loading preprocessed data …")
    tensor, time_index, scaler, meta = load_preprocessed_data()
    config.GRID_ROWS = meta["n_rows"]
    config.GRID_COLS = meta["n_cols"]
    print(f"  Tensor: {tensor.shape}")

    predictions = {}
    all_results = {}

    # ── Historical Average ──
    print("\n[1/2] Evaluating Historical Average …")
    ha_model, _ = load_trained_model("historical_average")
    if ha_model is None:
        print("  ERROR: historical_average checkpoint not found!")
        return
    sp, st, dp, dt = run_ha_inference(ha_model, tensor, time_index)
    predictions["historical_average"] = (sp, st, dp, dt)
    all_results["historical_average"] = evaluate_all_horizons(sp, st, dp, dt)

    # ── Joint Model ──
    print("[2/2] Evaluating Joint Model …")
    joint_model, _ = load_trained_model("joint")
    if joint_model is None:
        print("  ERROR: joint checkpoint not found!")
        return

    test_ds = TrafficDemandDataset(tensor, time_index, scaler, mode="test")
    from torch.utils.data import DataLoader
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                             shuffle=False, num_workers=0)

    sp, st, dp, dt = run_joint_inference(joint_model, test_loader, scaler)
    predictions["joint"] = (sp, st, dp, dt)
    all_results["joint"] = evaluate_all_horizons(sp, st, dp, dt)

    # ── Summary Table ──
    print_summary_table(all_results)

    # ── Generate All Comparison Plots ──
    print("\nGenerating comparison plots …")
    plot_horizon_comparison(all_results)
    plot_spatial_comparison(predictions)
    plot_improvement_map(predictions)
    plot_timeseries_overlay(predictions, time_index, tensor)
    plot_error_distribution(predictions)

    # ── Save results CSV ──
    rows = []
    for mname, results in all_results.items():
        for r in results:
            rows.append({"model": mname, **r})
    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.OUTPUT_DIR, "ha_vs_joint_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    print("\n✓ HA vs Joint evaluation complete!")
    print(f"  Figures saved to: {config.FIGURE_DIR}")


if __name__ == "__main__":
    main()
