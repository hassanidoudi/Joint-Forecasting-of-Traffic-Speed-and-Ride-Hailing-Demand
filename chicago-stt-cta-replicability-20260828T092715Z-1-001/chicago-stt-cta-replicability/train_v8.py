"""
train_v8.py — Training-only entry point for the four trained paper models,
with the same `--model` CLI shape as train_v3.py/train_v7.py so it drops
into the 06_standalone_regenerate_results.ipynb loop unchanged.

  - speed_only          → baselines.build_model (unchanged)
  - demand_only         → baselines.build_model (unchanged)
  - independent_dual    → baselines.build_model (unchanged)
  - joint                → model_v4 lag-mixture + seed ensemble — the EXACT
                            architecture/recipe from train_v7.py

`historical_average` is deliberately excluded — it is a non-trained lookup
baseline, not a trainable model, and has no place in a *training* entry point.

This file does NOT evaluate anything — see evaluate_v8.py for that, which
carries the identical metric definitions to evaluate_v2.py, generalized to
all four models, and also generates figures.

Usage (mirrors train_v3.py / train_v7.py):
  python train_v8.py --model speed_only
  python train_v8.py --model demand_only
  python train_v8.py --model independent_dual
  python train_v8.py --model joint                     # 3-member ensemble
  python train_v8.py --model joint --members 3 --seeds 42 1337 2026
  python train_v8.py --model all                        # all four, in order
"""

import os
import time
import shutil
import random
import pickle
import argparse

import numpy as np
import torch

import config
from dataset import load_preprocessed_data, create_dataloaders
from dataset_v2 import create_lag_dataloaders
from baselines import build_model
import model_v4


ALL_MODEL_NAMES = ["speed_only", "demand_only", "independent_dual", "joint"]
DEFAULT_SEEDS = [42, 1337, 2026]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ══════════════════════════════════════════════════════════════════════════
# Baseline models (speed_only / demand_only / independent_dual)
# Identical logic to train_v3.py's train_one_epoch / validate / training loop.
# ══════════════════════════════════════════════════════════════════════════

def run_epoch_baseline(model, loader, criterion, optimizer, device,
                       model_name, train):
    model.train(train)
    if hasattr(criterion, "train"):
        criterion.train(train)

    total_loss = total_speed = total_demand = 0.0
    n_batches = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for x, y_speed, y_demand in loader:
            x = x.to(device)
            y_speed = y_speed.to(device)
            y_demand = y_demand.to(device)

            if train:
                optimizer.zero_grad()

            if model_name == "independent_dual":
                speed_pred, demand_pred = model(x)
                loss, l_s, l_d = criterion(speed_pred, y_speed,
                                           demand_pred, y_demand)
            elif model_name == "speed_only":
                speed_pred = model(x)
                loss, l_s, l_d = criterion(speed_pred, y_speed)
            elif model_name == "demand_only":
                demand_pred = model(x)
                loss, l_s, l_d = criterion(demand_pred=demand_pred,
                                           demand_target=y_demand)
            else:
                raise ValueError(f"Unknown baseline model: {model_name}")

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()),
                    config.MAX_GRAD_NORM,
                )
                optimizer.step()

            total_loss += loss.item()
            total_speed += l_s.item()
            total_demand += l_d.item()
            n_batches += 1

    return (total_loss / n_batches,
            total_speed / n_batches,
            total_demand / n_batches)


def train_baseline_model(model_name, tensor, time_index, scaler,
                         select_on="val_loss"):
    """Full training pipeline for speed_only / demand_only / independent_dual.
    Unchanged from train_v3.py's train_neural_model, inlined here.
    """
    device = config.DEVICE
    set_seed(config.SEED)

    print(f"\n{'=' * 60}")
    print(f"  Training: {model_name}  (baselines.py, unchanged)")
    print(f"  Checkpoint selection metric: {select_on}")
    print(f"{'=' * 60}\n")

    model, criterion = build_model(model_name)
    model = model.to(device)
    criterion = criterion.to(device)

    train_loader, val_loader, _ = create_dataloaders(tensor, time_index, scaler)

    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.MAX_EPOCHS, eta_min=1e-6
    )

    ckpt_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.pt")
    best_metric = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [],
               "train_speed": [], "val_speed": [],
               "train_demand": [], "val_demand": []}

    start = time.time()
    for epoch in range(1, config.MAX_EPOCHS + 1):
        ep_start = time.time()
        tr_loss, tr_s, tr_d = run_epoch_baseline(
            model, train_loader, criterion, optimizer, device,
            model_name, train=True
        )
        va_loss, va_s, va_d = run_epoch_baseline(
            model, val_loader, criterion, None, device,
            model_name, train=False
        )
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        for key, val in zip(
            ["train_loss", "val_loss", "train_speed", "val_speed",
             "train_demand", "val_demand"],
            [tr_loss, va_loss, tr_s, va_s, tr_d, va_d],
        ):
            history[key].append(val)

        print(f"Epoch {epoch:3d}/{config.MAX_EPOCHS} | "
              f"Train: {tr_loss:.4f} (S:{tr_s:.4f} D:{tr_d:.4f}) | "
              f"Val: {va_loss:.4f} (S:{va_s:.4f} D:{va_d:.4f}) | "
              f"LR: {lr:.6f} | {time.time() - ep_start:.1f}s")

        current = {"val_demand": va_d, "val_speed": va_s,
                  "val_loss": va_loss}[select_on]
        if current < best_metric:
            best_metric = current
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "val_loss": va_loss, "val_speed": va_s, "val_demand": va_d,
                "select_on": select_on,
                "model_name": model_name,
                "architecture": "baseline",
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. "
                      f"Best epoch: {best_epoch} ({select_on}: {best_metric:.4f})")
                break

    print(f"{model_name} done in {(time.time() - start) / 60:.1f} min — "
          f"best {select_on} {best_metric:.4f} at epoch {best_epoch}")

    hist_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    return ckpt_path


# ══════════════════════════════════════════════════════════════════════════
# joint (model_v4 lag-mixture + seed ensemble)
# Identical logic to train_v7.py's run_epoch / train_member / assembly.
# ══════════════════════════════════════════════════════════════════════════

def run_epoch_joint(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    if hasattr(criterion, "train"):
        criterion.train(train)

    total_loss = total_speed = total_demand = 0.0
    n_batches = 0

    grad_ctx = torch.enable_grad() if train else torch.no_grad()
    with grad_ctx:
        for x, y_speed, y_demand, lag in loader:
            x = x.to(device)
            y_speed = y_speed.to(device)
            y_demand = y_demand.to(device)
            lag = lag.to(device)

            if train:
                optimizer.zero_grad()

            speed_pred, demand_pred = model(x, lag)
            loss, l_s, l_d = criterion(speed_pred, y_speed,
                                       demand_pred, y_demand)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(criterion.parameters()),
                    config.MAX_GRAD_NORM,
                )
                optimizer.step()

            total_loss += loss.item()
            total_speed += l_s.item()
            total_demand += l_d.item()
            n_batches += 1

    return (total_loss / n_batches,
            total_speed / n_batches,
            total_demand / n_batches)


def train_joint_member(member_idx, seed, tensor, time_index, scaler):
    device = config.DEVICE
    set_seed(seed)

    print(f"\n{'=' * 60}")
    print(f"  Training joint member {member_idx} (seed {seed}, model_v4)")
    print(f"  MAX_EPOCHS={config.MAX_EPOCHS}  PATIENCE={config.PATIENCE}  "
          f"SPEED_W={config.SPEED_LOSS_WEIGHT}  "
          f"DEMAND_W={config.DEMAND_LOSS_WEIGHT}")
    print(f"{'=' * 60}\n")

    model = model_v4.JointForecasterMember().to(device)
    criterion = model_v4.MultiTaskLoss().to(device)

    train_loader, val_loader, _ = create_lag_dataloaders(tensor, time_index, scaler)

    all_params = list(model.parameters()) + list(criterion.parameters())
    optimizer = torch.optim.AdamW(
        all_params, lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.MAX_EPOCHS, eta_min=1e-6
    )

    ckpt_path = os.path.join(config.CHECKPOINT_DIR,
                             f"joint_member{member_idx}_best.pt")
    best_val = float("inf")
    best_epoch = 0
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [],
               "train_speed": [], "val_speed": [],
               "train_demand": [], "val_demand": []}

    start = time.time()
    for epoch in range(1, config.MAX_EPOCHS + 1):
        ep_start = time.time()
        tr_loss, tr_s, tr_d = run_epoch_joint(model, train_loader, criterion,
                                              optimizer, device, train=True)
        va_loss, va_s, va_d = run_epoch_joint(model, val_loader, criterion,
                                              None, device, train=False)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        for key, val in zip(
            ["train_loss", "val_loss", "train_speed", "val_speed",
             "train_demand", "val_demand"],
            [tr_loss, va_loss, tr_s, va_s, tr_d, va_d],
        ):
            history[key].append(val)

        print(f"Epoch {epoch:3d}/{config.MAX_EPOCHS} | "
              f"Train: {tr_loss:.4f} (S:{tr_s:.4f} D:{tr_d:.4f}) | "
              f"Val: {va_loss:.4f} (S:{va_s:.4f} D:{va_d:.4f}) | "
              f"LR: {lr:.6f} | {time.time() - ep_start:.1f}s")

        if va_loss < best_val:
            best_val = va_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                "epoch": epoch, "seed": seed,
                "model_state_dict": model.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "val_loss": va_loss, "val_speed": va_s, "val_demand": va_d,
                "model_name": "joint",
                "architecture": "model_v4_member",
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch}. "
                      f"Best epoch: {best_epoch} (val_loss {best_val:.4f})")
                break

    print(f"Member {member_idx} done in {(time.time() - start) / 60:.1f} min "
          f"— best val_loss {best_val:.4f} at epoch {best_epoch}")

    hist_path = os.path.join(config.CHECKPOINT_DIR,
                             f"joint_member{member_idx}_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump(history, f)

    return ckpt_path


def train_joint_ensemble(tensor, time_index, scaler, n_members, seeds):
    joint_ckpt = os.path.join(config.CHECKPOINT_DIR, "joint_best.pt")
    backup = os.path.join(config.CHECKPOINT_DIR, "joint_best_model_v2_backup.pt")
    if os.path.exists(joint_ckpt) and not os.path.exists(backup):
        shutil.copy(joint_ckpt, backup)
        print(f"Backed up existing joint_best.pt -> {backup}")

    seeds = (seeds * ((n_members // len(seeds)) + 1))[:n_members]
    member_paths = [
        train_joint_member(i, seed, tensor, time_index, scaler)
        for i, seed in enumerate(seeds, start=1)
    ]

    device = config.DEVICE
    ensemble = model_v4.JointForecaster(n_members=len(member_paths)).to(device)
    criterion_state = None
    member_stats = []
    for member, path in zip(ensemble.members, member_paths):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        member.load_state_dict(ckpt["model_state_dict"])
        if criterion_state is None:
            criterion_state = ckpt["criterion_state_dict"]
        member_stats.append({
            "seed": ckpt["seed"], "epoch": ckpt["epoch"],
            "val_loss": ckpt["val_loss"], "val_speed": ckpt["val_speed"],
            "val_demand": ckpt["val_demand"],
        })
    ensemble.eval()

    criterion = model_v4.MultiTaskLoss().to(device)
    criterion.load_state_dict(criterion_state)
    _, val_loader, _ = create_lag_dataloaders(tensor, time_index, scaler)
    va_loss, va_s, va_d = run_epoch_joint(ensemble, val_loader, criterion,
                                          None, device, train=False)
    print(f"\nJoint ensemble validation — total: {va_loss:.4f}  "
          f"speed: {va_s:.4f}  demand: {va_d:.4f}")
    for i, st in enumerate(member_stats, start=1):
        print(f"  member {i} (seed {st['seed']}): val_loss {st['val_loss']:.4f}"
              f"  val_speed {st['val_speed']:.4f}"
              f"  val_demand {st['val_demand']:.4f}")

    torch.save({
        "model_state_dict": ensemble.state_dict(),
        "criterion_state_dict": criterion_state,
        "n_members": len(member_paths),
        "member_stats": member_stats,
        "val_loss": va_loss, "val_speed": va_s, "val_demand": va_d,
        "model_name": "joint",
        "architecture": "model_v4",
    }, joint_ckpt)
    print(f"✓ Joint ensemble saved as {joint_ckpt} (architecture=model_v4)")

    hist_path = os.path.join(config.CHECKPOINT_DIR, "joint_history.pkl")
    with open(hist_path, "wb") as f:
        pickle.dump({"train_loss": [], "val_loss": []}, f)

    return joint_ckpt


# ══════════════════════════════════════════════════════════════════════════
# Main — mirrors train_v3.py / train_v7.py's --model CLI shape
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Train the four trained paper models (v8: joint = "
                    "model_v4 ensemble, others unchanged from baselines.py)"
    )
    parser.add_argument("--model", type=str, default="all",
                        choices=ALL_MODEL_NAMES + ["all"],
                        help="Model to train (default: all)")
    parser.add_argument("--select_on", type=str, default="val_loss",
                        choices=["val_demand", "val_speed", "val_loss"],
                        help="Checkpoint selection metric for the 3 baseline models")
    parser.add_argument("--members", type=int, default=3,
                        help="Number of joint ensemble members (default 3)")
    parser.add_argument("--seeds", type=int, nargs="*", default=None,
                        help="Seeds for joint members (default 42 1337 2026)")
    args = parser.parse_args()
    seeds = args.seeds if args.seeds else DEFAULT_SEEDS

    print("Loading preprocessed data …")
    tensor, time_index, scaler, meta = load_preprocessed_data()
    print(f"Tensor: {tensor.shape}, Time range: {time_index[0]} — {time_index[-1]}")
    config.GRID_ROWS = meta["n_rows"]
    config.GRID_COLS = meta["n_cols"]
    config.print_config()

    models_to_train = ALL_MODEL_NAMES if args.model == "all" else [args.model]

    for name in models_to_train:
        if name == "joint":
            train_joint_ensemble(tensor, time_index, scaler,
                                 n_members=args.members, seeds=seeds)
        else:
            train_baseline_model(name, tensor, time_index, scaler,
                                 select_on=args.select_on)

    print("\n✓ Training complete!")


if __name__ == "__main__":
    main()
