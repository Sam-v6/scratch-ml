"""Reproducible rolling one-step sensor forecast comparison, independent of Ray/ONNX."""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models import TCN, ImprovedLSTM, NaiveLSTM, TimeSeriesTransformer
from models.timestral import forecast_timestral

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ["sine", "square", "triangle", "target"]
# Official model snapshots inspected on 2026-10-16.
MODEL_REVISIONS = {
    "5M": "bdd20cfc15a3fea0e4cb5ccef37f8d3ca7aa335f",
    "28M": "180698bbf5a50dda3516aa6419798437fb03a430",
    "123M": "5a147287addda85f6acd8da3fb8858fe9af7e8f7",
    "209M": "9874d252ab0aae16f7f7c999f98f98d6ad7d8365",
}


def make_windows(data: np.ndarray, start: int, stop: int, lookback: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Targets occupy [start,stop); every input ends strictly before its target."""
    if not 0 < lookback <= start < stop <= len(data):
        raise ValueError("Require 0 < lookback <= start < stop <= data length")
    indices = np.arange(start, stop)
    return np.stack([data[i - lookback : i] for i in indices]), data[indices, 3].copy(), indices


@torch.inference_mode()
def predict(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    return torch.cat([model(batch.to(device)).flatten() for batch in x.split(64)]).cpu().numpy()


def train_baseline(model: nn.Module, train_x: torch.Tensor, train_y: torch.Tensor, val_x: torch.Tensor, val_y: np.ndarray, epochs: int, seed: int) -> tuple[nn.Module, list[float], int]:
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=64, shuffle=True, generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=20, factor=0.5)
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            device = next(model.parameters()).device
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(x).flatten(), y)
            loss.backward()
            if isinstance(model, ImprovedLSTM):
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        val_rmse = float(np.sqrt(np.mean((predict(model, val_x) - val_y) ** 2)))
        history.append(val_rmse)
        if isinstance(model, ImprovedLSTM):
            scheduler.step(val_rmse)
        if val_rmse < best_loss:
            best_loss, best_epoch, best_state = val_rmse, epoch + 1, copy.deepcopy(model.state_dict())
        if (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1}: validation RMSE {val_rmse:.4f}", flush=True)
    model.load_state_dict(best_state)
    return model, history, best_epoch


def plot_results(predictions: pd.DataFrame, metrics: pd.DataFrame, histories: dict, out: Path) -> None:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
    ranked = metrics.sort_values("rmse")
    colors = ["#df7331" if n.startswith("Timestral") else "#337e9c" for n in ranked.model]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for ax, metric, label in zip(axes, ["rmse", "mae"], ["RMSE (V)", "MAE (V)"], strict=True):
        ax.barh(ranked.model, ranked[metric], color=colors)
        ax.invert_yaxis()
        ax.set_xlabel(label + " — lower is better")
        ax.bar_label(ax.containers[0], fmt="%.3f", padding=4)
        ax.set_xlim(0, ranked[metric].max() * 1.2)
    fig.suptitle("Next-step sensor forecasting • held-out final 4 seconds\nSame 256-step history; Timestral zero-shot, neural baselines trained locally")
    fig.savefig(out / "errors.png", dpi=160)
    plt.close(fig)
    nrows = (len(metrics) + 2) // 3
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 3.4 * nrows), sharex=True, sharey=True, layout="constrained")
    for ax, row in zip(axes.flat, metrics.itertuples(), strict=False):
        ax.plot(predictions.time, predictions.actual, color="#263645", lw=1, label="Actual")
        ax.plot(predictions.time, predictions[row.model], color="#df7331", lw=1, label="Prediction")
        if row.model + "_p10" in predictions:
            ax.fill_between(predictions.time, predictions[row.model + "_p10"], predictions[row.model + "_p90"], alpha=0.2, color="#df7331", label="p10–p90")
        ax.set_title(f"{row.model}  |  RMSE {row.rmse:.3f} V")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Target (V)")
    axes.flat[0].legend(fontsize=8)
    axes.flat[len(metrics) - 1].legend(fontsize=8)
    for ax in list(axes.flat)[len(metrics) :]:
        ax.set_visible(False)
    fig.suptitle("Rolling one-step predictions — earlier test observations become available over time")
    fig.savefig(out / "forecasts.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    for name, history in histories.items():
        ax.plot(np.arange(1, len(history) + 1), history, label=name)
    ax.set(xlabel="Epoch", ylabel="Validation RMSE (V)", title="Baseline checkpoint selection uses validation only")
    ax.legend()
    fig.savefig(out / "learning_curves.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--sizes", nargs="+", choices=["5M", "28M", "123M", "209M"], default=["5M", "28M", "209M"])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "timestral")
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("epochs and threads must be positive")
    from timestral import TimestralForecaster

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu explicitly for CPU runs")
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data_path = ROOT / "data/input/data_signals.csv"
    df = pd.read_csv(data_path)
    data = df[CHANNELS].to_numpy(dtype=np.float32)
    lookback, train_end, val_end = 256, int(0.6 * len(df)), int(0.8 * len(df))
    scaler = StandardScaler().fit(data[:train_end])
    scaled = scaler.transform(data).astype(np.float32)
    train_x, _, _ = make_windows(scaled, lookback, train_end, lookback)
    val_x, val_y, _ = make_windows(data, train_end, val_end, lookback)
    test_x, test_y, indices = make_windows(data, val_end, len(df), lookback)
    train_y = torch.from_numpy(data[lookback:train_end, 3].copy())
    val_scaled = torch.from_numpy(scaler.transform(val_x.reshape(-1, 4)).reshape(val_x.shape).astype(np.float32))
    test_scaled = torch.from_numpy(scaler.transform(test_x.reshape(-1, 4)).reshape(test_x.shape).astype(np.float32))
    predictions = pd.DataFrame({"index": indices, "time": df.time.iloc[indices].to_numpy(), "actual": test_y})
    rows, histories, revisions = [], {}, {}

    def record(name: str, pred: np.ndarray, train_s: float = 0, infer_s: float | None = None, params: int = 0, best_epoch: int = 0) -> None:
        if pred.shape != test_y.shape or not np.isfinite(pred).all():
            raise ValueError(f"Invalid predictions for {name}")
        predictions[name] = pred
        error = pred - test_y
        rows.append(
            dict(model=name, rmse=float(np.sqrt(np.mean(error**2))), mae=float(np.mean(np.abs(error))), train_seconds=train_s, inference_seconds=infer_s, parameters=params, best_epoch=best_epoch)
        )
        predictions.to_csv(out / "predictions.csv", index=False)
        pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
        print(f"{name}: test RMSE={rows[-1]['rmse']:.4f}, MAE={rows[-1]['mae']:.4f}", flush=True)

    record("Persistence", test_x[:, -1, 3])
    record("Seasonal-100", test_x[:, -100, 3])
    for name, factory in [("NaiveLSTM", NaiveLSTM), ("ImprovedLSTM", ImprovedLSTM), ("Transformer", TimeSeriesTransformer), ("TCN", TCN)]:
        print(f"Training {name}", flush=True)
        torch.manual_seed(args.seed)
        start = time.perf_counter()
        model, history, best_epoch = train_baseline(factory(input_size=4).to(args.device), torch.from_numpy(train_x), train_y, val_scaled, val_y, args.epochs, args.seed)
        train_s = time.perf_counter() - start
        histories[name] = history
        predict(model, test_scaled[:1])
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        pred = predict(model, test_scaled)
        infer_s = time.perf_counter() - start
        torch.save(model.state_dict(), out / f"{name}.pth")
        record(name, pred, train_s, infer_s, sum(p.numel() for p in model.parameters()), best_epoch)
    for size in args.sizes:
        name = f"Timestral-{size}"
        repo = f"mistralai/{name}"
        revision = MODEL_REVISIONS[size]
        revisions[repo] = revision
        print(f"Loading {repo}@{revision}", flush=True)
        forecaster = TimestralForecaster.from_pretrained(repo, revision=revision, device=args.device)
        forecast_timestral(forecaster, test_x[:1])
        if args.device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        quantiles = np.concatenate([forecast_timestral(forecaster, batch) for batch in np.array_split(test_x, (len(test_x) + 15) // 16)])
        infer_s = time.perf_counter() - start
        if not np.isfinite(quantiles).all() or not np.all(np.diff(quantiles, axis=1) >= 0):
            raise ValueError(f"Invalid quantiles for {name}")
        predictions[name + "_p10"], predictions[name + "_p90"] = quantiles[:, 0], quantiles[:, 2]
        record(name, quantiles[:, 1], infer_s=infer_s, params=sum(p.numel() for p in forecaster.model.parameters()))
        rows[-1]["coverage_80"] = float(np.mean((test_y >= quantiles[:, 0]) & (test_y <= quantiles[:, 2])))
        rows[-1]["interval_width"] = float(np.mean(quantiles[:, 2] - quantiles[:, 0]))
        del forecaster
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    metadata = dict(
        seed=args.seed,
        epochs=args.epochs,
        threads=args.threads,
        device=args.device,
        lookback=lookback,
        horizon=1,
        train_end=train_end,
        validation_end=val_end,
        test_end=len(df),
        channels=CHANNELS,
        data_sha256=hashlib.sha256(data_path.read_bytes()).hexdigest(),
        model_revisions=revisions,
        python=platform.python_version(),
        timestral_source_revision="a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff",
        neural_batch_size=64,
        timestral_batch_size=16,
        hardware=torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor(),
        versions={p: importlib.metadata.version(p) for p in ["torch", "numpy", "pandas", "scikit-learn", "timestral", "huggingface_hub", "safetensors"]},
        scaler_mean=scaler.mean_.tolist(),
        scaler_scale=scaler.scale_.tolist(),
        validation_rmse=histories,
    )
    (out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    plot_results(predictions, metrics, histories, out)
    print(metrics.to_string(index=False))
    print(f"Artifacts: {out}")


if __name__ == "__main__":
    main()
