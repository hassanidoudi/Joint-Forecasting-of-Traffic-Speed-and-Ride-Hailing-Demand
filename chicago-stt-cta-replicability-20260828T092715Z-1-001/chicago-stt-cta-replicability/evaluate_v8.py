"""
evaluate_v8.py — Evaluation entry point for the four trained paper models,
with the EXACT metric definitions, denormalization, and horizon/peak/weekend
breakdown logic from evaluate_v2.py (which itself matches the original
evaluate.py), generalized to dispatch across all four model types so every
model's numbers land in one directly-comparable table and CSV — plus the
same figure set the original evaluate.py produced (loss curves, spatial
error heatmaps, time-series comparisons, MAE comparison bar charts).

`historical_average` is deliberately excluded — it is a non-trained lookup
baseline, not one of the paper's actual models.

  - speed_only / demand_only / independent_dual
                         → plain (non-lag) TrafficDemandDataset, unchanged
  - joint                → LagAwareDataset + model_v4.JointForecaster ensemble
                            (same lag-mixture inference as evaluate_v2.py)

Usage (mirrors evaluate.py / evaluate_v2.py):
  python evaluate_v8.py                    # evaluate all 4 models + figures
  python evaluate_v8.py --model joint       # evaluate a single model
  python evaluate_v8.py --model all
  python evaluate_v8.py --no_figures        # skip figure generation
"""

import os
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for Colab/servers
import matplotlib.pyplot as plt
import seaborn as sns

import config
from dataset import load_preprocessed_data, TrafficDemandDataset
from dataset_v2 import LagAwareDataset
from baselines import build_model
import model_v4

sns.set_theme(style="whitegrid", font_scale=1.1)


ALL_MODEL_NAMES = ["speed_only", "demand_only", "independent_dual", "joint"]


# ── Metrics (identical to evaluate.py / evaluate_v2.py) ────────────────────

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


def compute_metrics(pred, target, label=""):
    return {"label": label, "MAE": compute_mae(pred, target),
            "RMSE": compute_rmse(pred, target),
            "MAPE": compute_mape(pred, target)}


def evaluate_by_horizon(sp, st, dp, dt):
    results = []
    horizons = {"1h": 0, "3h": 2, "6h": 5}
    for name, idx in horizons.items():
        if idx < sp.shape[1]:
            if not np.all(np.isnan(sp[:, idx])):
                results.append(compute_metrics(sp[:, idx], st[:, idx], f"Speed_{name}"))
            if not np.all(np.isnan(dp[:, idx])):
                results.append(compute_metrics(dp[:, idx], dt[:, idx], f"Demand_{name}"))
    return results


def evaluate_by_time_of_day(sp, st, dp, dt, time_index, window_starts, t_in):
    results = []
    n_samples = min(len(window_starts), sp.shape[0])
    peak_mask = np.zeros(n_samples, dtype=bool)
    weekend_mask = np.zeros(n_samples, dtype=bool)
    for i in range(n_samples):
        t = window_starts[i] + t_in
        if t < len(time_index):
            ts = time_index[t]
            h = ts.hour
            peak_mask[i] = (7 <= h <= 9) or (16 <= h <= 19)
            weekend_mask[i] = ts.dayofweek >= 5

    has_speed = not np.all(np.isnan(sp))
    has_demand = not np.all(np.isnan(dp))
    pairs = [("peak", peak_mask), ("offpeak", ~peak_mask),
             ("weekend", weekend_mask), ("weekday", ~weekend_mask)]
    for name, mask in pairs:
        if mask.sum() > 0:
            if has_speed:
                results.append(compute_metrics(sp[mask].flatten(), st[mask].flatten(),
                                               f"Speed_{name}"))
            if has_demand:
                results.append(compute_metrics(dp[mask].flatten(), dt[mask].flatten(),
                                               f"Demand_{name}"))
    return results


def full_report(tag, sp, st, dp, dt, time_index, window_starts):
    results = []
    has_speed = not np.all(np.isnan(sp))
    has_demand = not np.all(np.isnan(dp))
    if has_speed:
        results.append(compute_metrics(sp.flatten(), st.flatten(), "Speed_overall"))
    if has_demand:
        results.append(compute_metrics(dp.flatten(), dt.flatten(), "Demand_overall"))
    results.extend(evaluate_by_horizon(sp, st, dp, dt))
    results.extend(evaluate_by_time_of_day(sp, st, dp, dt, time_index,
                                           window_starts, config.T_IN))

    print(f"\n  [{tag}]")
    print(f"  {'Metric':<25s} {'MAE':>8s} {'RMSE':>8s} {'MAPE':>8s}")
    print(f"  {'─' * 50}")
    for r in results:
        print(f"  {r['label']:<25s} {r['MAE']:8.3f} {r['RMSE']:8.3f} {r['MAPE']:7.1f}%")
    return results


def denormalize_speed(arr, scaler):
    mean = scaler["mean"][config.SPEED_TARGET_IDX]
    std = scaler["std"][config.SPEED_TARGET_IDX]
    return arr * std + mean


def denormalize_demand(arr, scaler):
    out = arr.copy()
    pu_mean = scaler["mean"][config.PICKUP_TARGET_IDX]
    pu_std = scaler["std"][config.PICKUP_TARGET_IDX]
    do_mean = scaler["mean"][config.DROPOFF_TARGET_IDX]
    do_std = scaler["std"][config.DROPOFF_TARGET_IDX]
    out[..., 0] = out[..., 0] * pu_std + pu_mean
    out[..., 1] = out[..., 1] * do_std + do_mean
    return out


# ── Inference ───────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference_baseline(model, test_loader, model_name, scaler):
    device = config.DEVICE
    model.eval()
    all_sp, all_st, all_dp, all_dt = [], [], [], []
    for x, y_speed, y_demand in test_loader:
        x = x.to(device)
        if model_name == "independent_dual":
            speed_pred, demand_pred = model(x)
            speed_pred = speed_pred.squeeze(-1).cpu().numpy()
            demand_pred = demand_pred.cpu().numpy()
        elif model_name == "speed_only":
            speed_pred = model(x).squeeze(-1).cpu().numpy()
            demand_pred = np.full_like(y_demand.numpy(), np.nan)
        elif model_name == "demand_only":
            demand_pred = model(x).cpu().numpy()
            speed_pred = np.full_like(y_speed.numpy(), np.nan)
        else:
            raise ValueError(model_name)
        all_sp.append(speed_pred); all_st.append(y_speed.numpy())
        all_dp.append(demand_pred); all_dt.append(y_demand.numpy())

    sp = np.concatenate(all_sp, axis=0)
    st = np.concatenate(all_st, axis=0)
    dp = np.concatenate(all_dp, axis=0)
    dt = np.concatenate(all_dt, axis=0)

    if not np.all(np.isnan(sp)):
        sp = denormalize_speed(sp, scaler)
    st = denormalize_speed(st, scaler)
    if not np.all(np.isnan(dp)):
        dp = denormalize_demand(dp, scaler)
    dt = denormalize_demand(dt, scaler)
    return sp, st, dp, dt


@torch.no_grad()
def run_inference_joint(model, test_loader, scaler):
    device = config.DEVICE
    model.eval()
    all_sp, all_st, all_dp, all_dt = [], [], [], []
    for x, y_speed, y_demand, lag in test_loader:
        x = x.to(device); lag = lag.to(device)
        speed_pred, demand_pred = model(x, lag)
        all_sp.append(speed_pred.squeeze(-1).cpu().numpy())
        all_st.append(y_speed.numpy())
        all_dp.append(demand_pred.cpu().numpy())
        all_dt.append(y_demand.numpy())

    sp = np.concatenate(all_sp, axis=0); st = np.concatenate(all_st, axis=0)
    dp = np.concatenate(all_dp, axis=0); dt = np.concatenate(all_dt, axis=0)
    sp = denormalize_speed(sp, scaler); st = denormalize_speed(st, scaler)
    dp = denormalize_demand(dp, scaler); dt = denormalize_demand(dt, scaler)
    return sp, st, dp, dt


def evaluate_model(model_name, tensor, time_index, scaler):
    print(f"\n{'─' * 50}\n  Evaluating: {model_name}\n{'─' * 50}")
    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"  [SKIP] No checkpoint at {ckpt_path}")
        return [], None, None, None, None, None

    if model_name == "joint":
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
        assert ckpt.get("architecture") == "model_v4", (
            f"joint_best.pt architecture={ckpt.get('architecture')!r}, expected "
            f"model_v4. Run train_v8.py --model joint first."
        )
        model = model_v4.JointForecaster(n_members=ckpt.get("n_members", 3))
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(config.DEVICE)
        test_ds = LagAwareDataset(tensor, time_index, scaler, mode="test")
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        sp, st, dp, dt = run_inference_joint(model, test_loader, scaler)
        window_starts = test_ds.window_starts
    else:
        model, _ = build_model(model_name)
        ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(config.DEVICE)
        test_ds = TrafficDemandDataset(tensor, time_index, scaler, mode="test")
        test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
        sp, st, dp, dt = run_inference_baseline(model, test_loader, model_name, scaler)
        window_starts = test_ds.window_starts

    results = full_report(model_name, sp, st, dp, dt, time_index, window_starts)
    return results, sp, st, dp, dt


# ══════════════════════════════════════════════════════════════════════════
# Figures — same set as the original evaluate.py: loss curves, spatial error
# heatmaps, time-series comparisons, and an MAE comparison bar chart.
# ══════════════════════════════════════════════════════════════════════════

def plot_loss_curves(model_names):
    """Plot training/validation loss curves for all models with a history."""
    import pickle
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    plotted = False

    for name in model_names:
        hist_path = os.path.join(config.CHECKPOINT_DIR, f"{name}_history.pkl")
        if not os.path.exists(hist_path):
            continue
        with open(hist_path, "rb") as f:
            hist = pickle.load(f)
        if not hist.get("train_loss"):
            continue
        plotted = True
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=name)
        axes[1].plot(epochs, hist["val_loss"], label=name)

    if not plotted:
        plt.close(fig)
        return

    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].set_title("Validation Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(config.FIGURE_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_spatial_error_heatmap(sp, st, dp, dt, model_name):
    """Spatial heatmap of MAE per grid cell."""
    has_speed = not np.all(np.isnan(sp))
    has_demand = not np.all(np.isnan(dp))
    n_plots = int(has_speed) + int(has_demand)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5), squeeze=False)
    col = 0

    if has_speed:
        speed_mae = np.nanmean(np.abs(sp - st), axis=(0, 1))
        im1 = axes[0, col].imshow(speed_mae, cmap="YlOrRd", aspect="auto", origin="lower")
        axes[0, col].set_title(f"Speed MAE — {model_name}")
        axes[0, col].set_xlabel("Column")
        axes[0, col].set_ylabel("Row")
        plt.colorbar(im1, ax=axes[0, col], label="MAE (mph)")
        col += 1

    if has_demand:
        demand_total_pred = np.nansum(dp, axis=-1)
        demand_total_target = np.nansum(dt, axis=-1)
        demand_mae = np.nanmean(np.abs(demand_total_pred - demand_total_target), axis=(0, 1))
        im2 = axes[0, col].imshow(demand_mae, cmap="YlOrRd", aspect="auto", origin="lower")
        axes[0, col].set_title(f"Demand MAE — {model_name}")
        axes[0, col].set_xlabel("Column")
        axes[0, col].set_ylabel("Row")
        plt.colorbar(im2, ax=axes[0, col], label="MAE (trips)")

    fig.tight_layout()
    path = os.path.join(config.FIGURE_DIR, f"spatial_error_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_time_series_comparison(sp, st, dp, dt, model_name, n_cells=3, n_steps=48):
    """Time series of predicted vs. actual for sample grid cells."""
    H = st.shape[2]
    W = st.shape[3]

    has_speed = not np.all(np.isnan(sp))
    has_demand = not np.all(np.isnan(dp))

    if has_speed:
        activity = np.nanmean(np.abs(st), axis=(0, 1))
    else:
        activity = np.nanmean(np.abs(dt.sum(axis=-1)), axis=(0, 1))
    flat_idx = np.argsort(activity.flatten())[::-1]
    cells = [(idx // W, idx % W) for idx in flat_idx[:n_cells]]

    n_cols = int(has_speed) + int(has_demand)
    if n_cols == 0:
        return
    fig, axes = plt.subplots(n_cells, n_cols, figsize=(8 * n_cols, 4 * n_cells),
                             squeeze=False)

    steps = min(n_steps, sp.shape[0])

    for i, (r, c) in enumerate(cells):
        col = 0
        if has_speed:
            spv = sp[:steps, 0, r, c]
            stv = st[:steps, 0, r, c]
            axes[i, col].plot(stv, label="Actual", alpha=0.8)
            axes[i, col].plot(spv, label="Predicted", alpha=0.8)
            axes[i, col].set_title(f"Speed — Cell ({r},{c})")
            axes[i, col].set_ylabel("mph")
            axes[i, col].legend()
            col += 1

        if has_demand:
            dpv = dp[:steps, 0, r, c, 0]
            dtv = dt[:steps, 0, r, c, 0]
            axes[i, col].plot(dtv, label="Actual", alpha=0.8)
            axes[i, col].plot(dpv, label="Predicted", alpha=0.8)
            axes[i, col].set_title(f"Pickup Demand — Cell ({r},{c})")
            axes[i, col].set_ylabel("count")
            axes[i, col].legend()

    for c_idx in range(n_cols):
        axes[-1, c_idx].set_xlabel("Sample")
    fig.suptitle(f"Time Series — {model_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(config.FIGURE_DIR, f"timeseries_{model_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_mae_comparison(df):
    """Bar chart: overall MAE across models, plus a horizon breakdown."""
    overall = df[df["label"].isin(["Speed_overall", "Demand_overall"])]
    if overall.empty:
        return

    models, speed_maes, demand_maes = [], [], []
    for name in df["model"].unique():
        sub = overall[overall["model"] == name]
        s = sub[sub["label"] == "Speed_overall"]["MAE"]
        d = sub[sub["label"] == "Demand_overall"]["MAE"]
        if len(s) and len(d):
            models.append(name)
            speed_maes.append(s.iloc[0])
            demand_maes.append(d.iloc[0])

    if not models:
        return

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, speed_maes, width, label="Speed MAE", color="#2196F3")
    bars2 = ax.bar(x + width / 2, demand_maes, width, label="Demand MAE", color="#FF9800")

    ax.set_ylabel("MAE")
    ax.set_title("MAE Comparison Across Models")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()

    for bar in bars1:
        ax.annotate(f"{bar.get_height():.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        ax.annotate(f"{bar.get_height():.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(config.FIGURE_DIR, "mae_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")

    # ── Horizon breakdown ──
    horizons = ["1h", "3h", "6h"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for task_idx, task_name in enumerate(["Speed", "Demand"]):
        model_data = {}
        for name in df["model"].unique():
            sub = df[df["model"] == name]
            vals = []
            for h in horizons:
                r = sub[sub["label"] == f"{task_name}_{h}"]["MAE"]
                vals.append(r.iloc[0] if len(r) else 0)
            model_data[name] = vals

        x = np.arange(len(horizons))
        width = 0.8 / max(len(model_data), 1)
        for j, (mname, vals) in enumerate(model_data.items()):
            axes[task_idx].bar(x + j * width, vals, width, label=mname)

        axes[task_idx].set_title(f"{task_name} MAE by Horizon")
        axes[task_idx].set_xticks(x + width * (len(model_data) - 1) / 2)
        axes[task_idx].set_xticklabels(horizons)
        axes[task_idx].set_ylabel("MAE")
        axes[task_idx].legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(config.FIGURE_DIR, "mae_by_horizon.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ══════════════════════════════════════════════════════════════════════════
# Main — mirrors evaluate.py / evaluate_v2.py's --model CLI shape
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate paper models (v8: joint = model_v4 ensemble, "
                    "others unchanged)"
    )
    parser.add_argument("--model", type=str, default="all",
                        choices=ALL_MODEL_NAMES + ["all"],
                        help="Model to evaluate (default: all)")
    parser.add_argument("--no_figures", action="store_true",
                        help="Skip figure generation")
    args = parser.parse_args()

    print("Loading preprocessed data …")
    tensor, time_index, scaler, meta = load_preprocessed_data()
    print(f"Tensor: {tensor.shape}")
    config.GRID_ROWS = meta["n_rows"]
    config.GRID_COLS = meta["n_cols"]

    models_to_eval = ALL_MODEL_NAMES if args.model == "all" else [args.model]

    rows = []
    evaluated_names = []
    for name in models_to_eval:
        results, sp, st, dp, dt = evaluate_model(name, tensor, time_index, scaler)
        if not results:
            continue
        evaluated_names.append(name)
        for r in results:
            rows.append({"model": name, **r})

        if not args.no_figures:
            plot_spatial_error_heatmap(sp, st, dp, dt, name)
            plot_time_series_comparison(sp, st, dp, dt, name)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(config.OUTPUT_DIR, "evaluation_results_v8.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to {csv_path}")

    if not args.no_figures and len(evaluated_names) > 0:
        print("\nGenerating comparison figures …")
        plot_loss_curves(evaluated_names)
        plot_mae_comparison(df)

    if not df.empty:
        print(f"\n{'=' * 60}\n  SUMMARY — Overall MAE by model\n{'=' * 60}")
        print(f"  {'Model':<20s} {'Speed MAE':>10s} {'Demand MAE':>12s}")
        print(f"  {'─' * 44}")
        for name in models_to_eval:
            sub = df[df["model"] == name]
            s = sub[sub["label"] == "Speed_overall"]["MAE"]
            d = sub[sub["label"] == "Demand_overall"]["MAE"]
            s_str = f"{s.iloc[0]:.3f}" if len(s) else "  n/a"
            d_str = f"{d.iloc[0]:.3f}" if len(d) else "  n/a"
            print(f"  {name:<20s} {s_str:>10s} {d_str:>12s}")

    print("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
