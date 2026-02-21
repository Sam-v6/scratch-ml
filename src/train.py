"""
train.py -- train all four time-series models and produce comparison artifacts

Usage (from repo root):
    uv run python src/train.py

Outputs written to SCRATCH_HOME/artifacts/:
    {ModelName}_loss.png  -- per-model train/val RMSE curves
    comparison.png        -- 3x2 summary figure across all models

A summary table is also printed to stdout.
"""

import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models import TCN, ImprovedLSTM, NaiveLSTM, TimeSeriesTransformer
from path import SCRATCH_HOME

# =============================================================================
# Configuration
# =============================================================================

CONFIG = {
    "data_path": SCRATCH_HOME / "data" / "input" / "data_signals.csv",
    "artifacts": SCRATCH_HOME / "artifacts",  # directory for output PNGs
    "lookback": 256,  # number of past timesteps fed to the model
    "horizon": 1,  # number of future steps to predict
    "train_frac": 0.6,  # fraction of data used for training (temporal split)
    "batch_size": 64,
    "epochs": 100,
    "lr": 1e-3,
    "seed": 42,
}


# =============================================================================
# Data helpers
# =============================================================================


def make_windows(
    x: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build sliding windows over a time series.

    For a sequence of length N:
        X[i] = x[i : i+lookback]                  shape (lookback, F)
        Y[i] = y[i+lookback : i+lookback+horizon]  shape (horizon,)

    The windows step by 1 each time (maximum overlap).  Temporal order is
    preserved -- no shuffling here.

    Returns torch Tensors (float32).
    """
    xs, ys = [], []
    for i in range(len(x) - lookback - horizon + 1):
        xs.append(x[i : i + lookback])
        ys.append(y[i + lookback : i + lookback + horizon])
    X = torch.tensor(np.stack(xs), dtype=torch.float32)  # (N, lookback, F)
    Y = torch.tensor(np.stack(ys), dtype=torch.float32)  # (N, horizon)
    return X, Y


def load_data(config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Load the CSV, scale features, build sliding windows, and return DataLoaders.

    Data pipeline:
        1. Read CSV -- columns: time, sine, square, triangle, target
        2. Temporal split at train_frac (no shuffling; preserves time ordering)
        3. Fit StandardScaler on training features only -> transform both splits
           (prevents leakage: validation statistics must not influence scaling)
        4. Build overlapping windows of length `lookback`
        5. Wrap in DataLoaders (train shuffled, val not shuffled)
    """
    df = pd.read_csv(config["data_path"])

    features = ["sine", "square", "triangle"]
    target = ["target"]

    x = df[features].to_numpy(dtype=np.float32)
    y = df[target].to_numpy(dtype=np.float32).squeeze(-1)  # (N,) not (N, 1)

    # Temporal split
    split = int(config["train_frac"] * len(df))
    x_train, x_val = x[:split], x[split:]
    y_train, y_val = y[:split], y[split:]

    # Scale inputs on train statistics only
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    # Sliding windows
    X_train, Y_train = make_windows(x_train, y_train, config["lookback"], config["horizon"])
    X_val, Y_val = make_windows(x_val, y_val, config["lookback"], config["horizon"])

    print(f"Train windows: {X_train.shape}  |  Val windows: {X_val.shape}")

    # Fix the random generator so window shuffle order is reproducible
    g = torch.Generator()
    g.manual_seed(config["seed"])

    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=config["batch_size"],
        shuffle=True,
        generator=g,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_val, Y_val),
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    return train_loader, val_loader


# =============================================================================
# Training loop
# =============================================================================


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 300,
    lr: float = 1e-3,
    clip_grad: float | None = None,
    use_scheduler: bool = False,
) -> tuple[list, list, float]:
    """
    Train a model and return performance histories.

    Args:
        model:          any model with forward(x: (B,T,F)) -> (B,1)
        train_loader:   training DataLoader
        val_loader:     validation DataLoader
        device:         torch.device
        epochs:         number of training epochs
        lr:             initial Adam learning rate
        clip_grad:      if set, apply gradient clipping with this max_norm
        use_scheduler:  if True, halve lr when val RMSE stops improving

    Returns:
        train_rmse_history: list of per-epoch training RMSE values
        val_rmse_history:   list of per-epoch validation RMSE values
        wall_clock_seconds: total training time
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    scheduler = None
    if use_scheduler:
        # Halve the LR when val RMSE hasn't improved for 20 epochs
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=20, min_lr=1e-5)

    best_val_rmse = float("inf")
    best_state = None
    train_hist: list[float] = []
    val_hist: list[float] = []

    t_start = time.time()

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------------
        # Training phase
        # ------------------------------------------------------------------
        model.train()
        total_loss = 0.0
        total_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()

            if clip_grad is not None:
                # Clip the L2 norm of all parameter gradients to prevent
                # exploding gradients (common with deep/recurrent models)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)

            optimizer.step()

            # Accumulate loss weighted by batch size (batches may differ in size)
            total_loss += loss.item() * xb.size(0)
            total_n += xb.size(0)

        # RMSE = sqrt(mean squared error over all training samples)
        train_rmse = math.sqrt(total_loss / total_n)
        train_hist.append(train_rmse)

        # ------------------------------------------------------------------
        # Validation phase
        # ------------------------------------------------------------------
        model.eval()
        total_loss = 0.0
        total_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_loss += loss.item() * xb.size(0)
                total_n += xb.size(0)

        val_rmse = math.sqrt(total_loss / total_n)
        val_hist.append(val_rmse)

        if scheduler is not None:
            scheduler.step(val_rmse)

        # Save the model weights whenever validation RMSE improves
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0:
            print(f"  Epoch {epoch:3d} | train RMSE: {train_rmse:.4f} | val RMSE: {val_rmse:.4f}")

    wall_clock = time.time() - t_start

    # Restore the best checkpoint found during training
    model.load_state_dict(best_state)

    return train_hist, val_hist, wall_clock


# =============================================================================
# Utilities
# =============================================================================


def count_params(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Plotting
# =============================================================================


def plot_losses(
    name: str,
    train_rmse: list[float],
    val_rmse: list[float],
    out_dir: str | os.PathLike[str],
) -> None:
    """Save a train/val RMSE curve for a single model."""
    epochs = range(1, len(train_rmse) + 1)
    best = min(val_rmse)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, train_rmse, label="Train RMSE", linewidth=1.2)
    ax.plot(epochs, val_rmse, label="Val RMSE", linewidth=1.2)
    ax.set_title(f"{name} -- best val RMSE: {best:.4f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(out_dir, f"{name}_loss.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_summary(results: dict, out_dir: str | os.PathLike[str]) -> None:
    """
    Save a 3x2 comparison figure across all models.

    Rows:
        0 -- Training RMSE curves (full run + zoomed to last 100 epochs)
        1 -- Validation RMSE curves (full run + zoomed to last 100 epochs)
        2 -- Bar charts: best val RMSE and training time
    """
    model_names = list(results.keys())
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    n_epochs = len(next(iter(results.values()))["train_losses"])
    epochs_full = range(1, n_epochs + 1)
    zoom_start = max(1, n_epochs - 99)  # last 100 epochs
    epochs_zoom = range(zoom_start, n_epochs + 1)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("Model Comparison -- Time Series Forecasting", fontsize=13, y=1.01)

    # Row 0: Training RMSE
    for name, color in zip(model_names, colors, strict=False):
        axes[0, 0].plot(epochs_full, results[name]["train_losses"], label=name, color=color, linewidth=1.0)
        axes[0, 1].plot(
            epochs_zoom,
            results[name]["train_losses"][zoom_start - 1 :],
            label=name,
            color=color,
            linewidth=1.0,
        )

    for ax, title in zip(axes[0], ["Training RMSE", "Training RMSE (last 100 epochs)"], strict=False):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 1: Validation RMSE
    for name, color in zip(model_names, colors, strict=False):
        axes[1, 0].plot(epochs_full, results[name]["val_losses"], label=name, color=color, linewidth=1.0)
        axes[1, 1].plot(
            epochs_zoom,
            results[name]["val_losses"][zoom_start - 1 :],
            label=name,
            color=color,
            linewidth=1.0,
        )

    for ax, title in zip(axes[1], ["Validation RMSE", "Validation RMSE (last 100 epochs)"], strict=False):
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Row 2: Bar charts
    best_rmses = [results[n]["best_val_rmse"] for n in model_names]
    train_times = [results[n]["train_time"] for n in model_names]

    bars = axes[2, 0].bar(model_names, best_rmses, color=colors)
    axes[2, 0].set_title("Best Validation RMSE  (lower is better)")
    axes[2, 0].set_ylabel("RMSE")
    axes[2, 0].bar_label(bars, fmt="%.4f", padding=3)
    axes[2, 0].grid(True, alpha=0.3, axis="y")

    bars = axes[2, 1].bar(model_names, train_times, color=colors)
    axes[2, 1].set_title("Training Time  (seconds)")
    axes[2, 1].set_ylabel("Seconds")
    axes[2, 1].bar_label(bars, fmt="%.1f", padding=3)
    axes[2, 1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = os.path.join(out_dir, "comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved {path}")


def print_summary_table(results: dict) -> None:
    """Print a formatted summary table to stdout."""
    header = f"{'Model':<18} {'Params':>10} {'Best Val RMSE':>14} {'Train Time (s)':>15} {'Epochs':>7}"
    print(f"\n{header}")
    print("-" * len(header))
    for name, r in results.items():
        print(f"{name:<18} {r['param_count']:>10,} {r['best_val_rmse']:>14.4f} {r['train_time']:>15.1f} {r['epochs']:>7}")


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    # Reproducibility
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    os.makedirs(CONFIG["artifacts"], exist_ok=True)

    # Data
    train_loader, val_loader = load_data(CONFIG)

    # Model registry -- each entry specifies the model and its training recipe.
    # clip_grad and use_scheduler are only applied where the architecture needs them.
    models_cfg = [
        {
            "name": "NaiveLSTM",
            "model": NaiveLSTM(input_size=3),
            "clip_grad": None,
            "use_scheduler": False,
        },
        {
            "name": "ImprovedLSTM",
            "model": ImprovedLSTM(input_size=3),
            "clip_grad": 1.0,  # prevent exploding gradients in deep LSTM
            "use_scheduler": True,  # halve LR when val RMSE plateaus
        },
        {
            "name": "Transformer",
            "model": TimeSeriesTransformer(input_size=3),
            "clip_grad": None,
            "use_scheduler": False,
        },
        {
            "name": "TCN",
            "model": TCN(input_size=3),
            "clip_grad": None,
            "use_scheduler": False,
        },
    ]

    results: dict = {}
    for cfg in models_cfg:
        name = cfg["name"]
        model = cfg["model"]
        print(f"\n{'=' * 52}\nTraining: {name}  ({count_params(model):,} parameters)")

        train_hist, val_hist, elapsed = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=CONFIG["epochs"],
            lr=CONFIG["lr"],
            clip_grad=cfg["clip_grad"],
            use_scheduler=cfg["use_scheduler"],
        )

        results[name] = {
            "train_losses": train_hist,
            "val_losses": val_hist,
            "train_time": elapsed,
            "param_count": count_params(model),
            "best_val_rmse": min(val_hist),
            "epochs": CONFIG["epochs"],
        }

        print(f"  -> Best val RMSE: {min(val_hist):.4f} | Time: {elapsed:.1f}s")
        plot_losses(name, train_hist, val_hist, CONFIG["artifacts"])

    plot_summary(results, CONFIG["artifacts"])
    print_summary_table(results)


if __name__ == "__main__":
    main()
