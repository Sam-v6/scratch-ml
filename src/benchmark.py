"""
benchmark.py -- unified inference timing comparison across all runtimes

Measures single-window latency (batch_size=1) for four model architectures across
four inference backends, each on CPU and GPU, producing an 8-row comparison table.

Runtimes benchmarked:
  Python .pth  -- state-dict load + model class, PyTorch eager inference
  Python .pt   -- TorchScript traced model, PyTorch JIT inference
  ONNX C++     -- ONNX Runtime via cpp/build/inference  (CPU + CUDA EP)
  TensorRT C++ -- TensorRT engine via cpp/build/trt_inference  (GPU only)

Usage (from repo root):
    uv run python src/benchmark.py
    uv run python src/benchmark.py --onnx-dir artifacts/onnx
    uv run python src/benchmark.py --inference-bin cpp/build/inference
    uv run python src/benchmark.py --trt-bin cpp/build/trt_inference

Prerequisites:
  1. Run train.py first to produce .pth, .pt, .onnx files and holdout binaries.
  2. Build cpp/build/inference for ONNX rows (see README).
  3. Build cpp/build/trt_inference for TensorRT rows (see README, requires TRT install).

Outputs:
  artifacts/onnx/python_metrics.json  -- Python .pth and .pt timing results
  stdout                               -- 8-row summary table + p99 table
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
from contextlib import AbstractContextManager, nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from models import TCN, ImprovedLSTM, NaiveLSTM, TimeSeriesTransformer
from path import SCRATCH_HOME

# =============================================================================
# Constants
# =============================================================================

LOOKBACK: int = 256
N_FEATURES: int = 3
N_WARMUP: int = 10

MODEL_NAMES: list[str] = ["NaiveLSTM", "ImprovedLSTM", "Transformer", "TCN"]

# =============================================================================
# Defaults
# =============================================================================

DEFAULT_ONNX_DIR = SCRATCH_HOME / "artifacts" / "onnx"
DEFAULT_INFERENCE_BIN = SCRATCH_HOME / "cpp" / "build" / "inference"
DEFAULT_TRT_BIN = SCRATCH_HOME / "cpp" / "build" / "trt_inference"


# =============================================================================
# Data helpers
# =============================================================================


def load_holdout(onnx_dir: Path) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Load the holdout windows from binary files written by train.py.

    Returns:
        inputs  -- float32 array, shape (N, LOOKBACK, N_FEATURES)
        targets -- float32 array, shape (N,)
        n_windows -- number of windows
    """
    meta_path = onnx_dir / "holdout_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"holdout_meta.json not found at {meta_path}. Run train.py first.")

    with open(meta_path) as f:
        meta = json.load(f)

    n_windows: int = meta["n_windows"]

    inputs = np.fromfile(onnx_dir / "holdout_input.bin", dtype=np.float32)
    inputs = inputs.reshape(n_windows, LOOKBACK, N_FEATURES)

    targets = np.fromfile(onnx_dir / "holdout_target.bin", dtype=np.float32)

    return inputs, targets, n_windows


# =============================================================================
# Latency statistics
# =============================================================================


def compute_stats(latencies: list[float]) -> dict[str, float]:
    """Return mean/min/max/std/p50/p95/p99 from a list of latency values (ms)."""
    n = len(latencies)
    if n == 0:
        return {}
    arr = sorted(latencies)
    mean = sum(arr) / n
    variance = sum((v - mean) ** 2 for v in arr) / n

    def pct(p: float) -> float:
        idx = p * (n - 1)
        lo = int(idx)
        hi = lo + 1
        if hi >= n:
            return arr[-1]
        frac = idx - lo
        return arr[lo] * (1 - frac) + arr[hi] * frac

    return {
        "mean": mean,
        "min": arr[0],
        "max": arr[-1],
        "std": math.sqrt(variance),
        "p50": pct(0.50),
        "p95": pct(0.95),
        "p99": pct(0.99),
    }


# =============================================================================
# Python benchmark
# =============================================================================


def _make_model(name: str) -> nn.Module:
    """Instantiate the correct model class by name."""
    mapping: dict[str, type[nn.Module]] = {
        "NaiveLSTM": NaiveLSTM,
        "ImprovedLSTM": ImprovedLSTM,
        "Transformer": TimeSeriesTransformer,
        "TCN": TCN,
    }
    cls = mapping[name]
    return cls(input_size=N_FEATURES)


def _flatten_recurrent_weights(model: nn.Module) -> None:
    """
    Flatten recurrent weights (LSTM/GRU/RNN) after device placement.

    TorchScript traces can omit the runtime flatten_parameters() call because it
    has no tensor outputs; calling it once at load time prevents cuDNN
    contiguity warnings during GPU inference.
    """
    for module in model.modules():
        flatten = getattr(module, "flatten_parameters", None)
        if callable(flatten):
            try:
                flatten()
            except Exception:
                # Non-RNN modules may expose incompatible call signatures.
                pass


def _cudnn_context(disable_cudnn: bool) -> AbstractContextManager[object]:
    """Return a context manager that optionally disables cuDNN for one forward pass."""
    return torch.backends.cudnn.flags(enabled=False) if disable_cudnn else nullcontext()


def benchmark_python_format(
    model_name: str,
    fmt: str,  # "pth" or "pt"
    onnx_dir: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
    devices: list[torch.device],
) -> list[dict[str, Any]]:
    """
    Benchmark one model in one format (.pth or .pt) on each of the given devices.

    Returns a list of metric dicts (one per device), compatible with the
    cpp_metrics.json schema so they can be merged into a single comparison table.
    """
    results: list[dict[str, Any]] = []
    n_windows = inputs.shape[0]

    for device in devices:
        provider_label = "GPU" if device.type == "cuda" else "CPU"

        # --- Load model ---
        if fmt == "pth":
            model = _make_model(model_name)
            pth_path = onnx_dir / f"{model_name}.pth"
            if not pth_path.exists():
                print(f"  [SKIP] {model_name} .pth not found ({pth_path}). Run train.py first.")
                continue
            model.load_state_dict(torch.load(pth_path, map_location=device, weights_only=True))
            model.to(device).eval()
        else:  # pt
            pt_path = onnx_dir / f"{model_name}.pt"
            if not pt_path.exists():
                print(f"  [SKIP] {model_name} .pt not found ({pt_path}). Run train.py first.")
                continue
            model = torch.jit.load(str(pt_path), map_location=device)
            model.eval()

        _flatten_recurrent_weights(model)

        use_gpu = device.type == "cuda"
        disable_cudnn = use_gpu and fmt == "pt" and model_name in {"NaiveLSTM", "ImprovedLSTM"}

        if use_gpu:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

        # Warmup
        for i in range(min(N_WARMUP, n_windows)):
            window = torch.from_numpy(inputs[i : i + 1]).to(device)
            with torch.no_grad(), _cudnn_context(disable_cudnn):
                model(window)
            if use_gpu:
                torch.cuda.synchronize()

        # Benchmark
        latencies: list[float] = []
        predictions: list[float] = []

        for i in range(n_windows):
            window = torch.from_numpy(inputs[i : i + 1]).to(device)
            if use_gpu:
                # Record GPU kernel time only; capture output from the same call
                start_event.record()
                with torch.no_grad(), _cudnn_context(disable_cudnn):
                    out = model(window)
                end_event.record()
                torch.cuda.synchronize()
                lat = start_event.elapsed_time(end_event)
                pred = out.item()
            else:
                t0 = time.perf_counter()
                with torch.no_grad(), _cudnn_context(disable_cudnn):
                    pred = model(window).item()
                lat = (time.perf_counter() - t0) * 1000.0
            latencies.append(lat)
            predictions.append(pred)

        # RMSE
        preds_np = np.array(predictions, dtype=np.float32)
        rmse = float(np.sqrt(np.mean((preds_np - targets) ** 2)))

        stats = compute_stats(latencies)

        results.append(
            {
                "model": model_name,
                "format": fmt,
                "provider": f"Python.{fmt}.{provider_label}",
                "device": provider_label,
                "latency_ms": stats,
                "throughput_windows_per_sec": 1000.0 / stats["mean"] if stats.get("mean", 0) > 0 else 0.0,
                "rmse": rmse,
                "n_windows": n_windows,
            }
        )

        print(f"  {model_name:<14} {fmt:>3} {provider_label:>3}  mean={stats['mean']:.3f} ms  p99={stats['p99']:.3f} ms  rmse={rmse:.4f}")

    return results


def run_python_benchmarks(
    onnx_dir: Path,
    inputs: np.ndarray,
    targets: np.ndarray,
) -> list[dict[str, Any]]:
    """Run all Python (.pth and .pt) benchmarks and return a flat list of metric dicts."""
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    else:
        print("  CUDA not available; skipping GPU benchmarks for Python runtimes.")

    all_metrics: list[dict[str, Any]] = []

    for fmt in ("pth", "pt"):
        print(f"\n--- Python .{fmt} ---")
        for model_name in MODEL_NAMES:
            metrics = benchmark_python_format(model_name, fmt, onnx_dir, inputs, targets, devices)
            all_metrics.extend(metrics)

    return all_metrics


# =============================================================================
# C++ binary orchestration
# =============================================================================


def _python_cuda_lib_dirs() -> list[str]:
    """
    Return CUDA-related shared-library directories bundled in this Python env.

    PyTorch wheels ship CUDA/cuDNN libs under site-packages/nvidia/*/lib.
    Passing these paths to C++ subprocesses lets ONNX Runtime CUDA EP resolve
    libcudnn/libcublas when they are not installed system-wide.
    """
    site_packages = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.exists():
        return []

    lib_dirs = [str(p.resolve()) for p in sorted(nvidia_root.glob("*/lib")) if p.is_dir()]
    return list(dict.fromkeys(lib_dirs))


def _cpp_env_with_cuda_paths() -> dict[str, str]:
    """Build subprocess env with CUDA lib paths prepended to LD_LIBRARY_PATH."""
    env = os.environ.copy()
    extra_dirs = _python_cuda_lib_dirs()
    if not extra_dirs:
        return env

    existing = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = ":".join(extra_dirs + ([existing] if existing else []))
    return env


def run_cpp_binary(binary: Path, onnx_dir: Path, label: str) -> bool:
    """
    Run a C++ benchmark binary as a subprocess.

    Returns True if the binary ran successfully, False if it was skipped
    (binary missing or execution failed).
    """
    if not binary.exists():
        print(f"\n  [SKIP] {label} binary not found at {binary}")
        print("         Build it first (see README) then re-run this script.")
        return False

    print(f"\n--- {label} ---")
    try:
        subprocess.run([str(binary), str(onnx_dir)], check=True, env=_cpp_env_with_cuda_paths())
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [ERROR] {label} binary exited with code {e.returncode}")
        return False


def load_json_metrics(path: Path) -> list[dict[str, Any]]:
    """Load a metrics JSON file; return empty list if it doesn't exist."""
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


# =============================================================================
# System info
# =============================================================================


def _cpu_name() -> str:
    """Return a human-readable CPU name."""
    # /proc/cpuinfo is the most reliable on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def system_info() -> dict[str, str]:
    """Collect CPU, GPU, and CUDA version strings."""
    cpu = _cpu_name()
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        cuda_ver = torch.version.cuda or "unknown"
    else:
        gpu = "N/A"
        cuda_ver = "N/A"
    return {"cpu": cpu, "gpu": gpu, "cuda": cuda_ver}


# =============================================================================
# Summary table
# =============================================================================

# Canonical row order for the 8-row table
_ROW_ORDER: list[tuple[str, str]] = [
    ("Python (.pth)", "CPU"),
    ("Python (.pth)", "GPU"),
    ("Python (.pt)", "CPU"),
    ("Python (.pt)", "GPU"),
    ("ONNX / ORT (C++)", "CPU"),
    ("ONNX / ORT (C++)", "GPU"),
    ("TensorRT (C++)", "CPU"),
    ("TensorRT (C++)", "GPU"),
]


def _runtime_device_key(entry: dict[str, Any]) -> tuple[str, str]:
    """Extract (runtime_label, device_label) from a metrics entry."""
    provider: str = entry.get("provider", "")
    device: str = entry.get("device", "")

    if provider.startswith("Python.pth"):
        runtime = "Python (.pth)"
        dev = "GPU" if "GPU" in provider else "CPU"
    elif provider.startswith("Python.pt"):
        runtime = "Python (.pt)"
        dev = "GPU" if "GPU" in provider else "CPU"
    elif "TensorRT" in provider:
        runtime = "TensorRT (C++)"
        dev = "GPU"
    elif provider in ("CPUExecutionProvider",):
        runtime = "ONNX / ORT (C++)"
        dev = "CPU"
    elif provider in ("CUDAExecutionProvider",):
        runtime = "ONNX / ORT (C++)"
        dev = "GPU"
    else:
        runtime = provider
        dev = device or "?"

    return runtime, dev


def build_table(all_metrics: list[dict[str, Any]], stat_key: str) -> dict[tuple[str, str], dict[str, float]]:
    """
    Build a (runtime, device) -> {model_name: latency_ms} lookup.

    stat_key: "mean" or "p99"
    """
    table: dict[tuple[str, str], dict[str, float]] = {}
    for entry in all_metrics:
        model = entry.get("model", "?")
        key = _runtime_device_key(entry)
        lat_block = entry.get("latency_ms", {})
        val = lat_block.get(stat_key)
        if key not in table:
            table[key] = {}
        if val is not None:
            table[key][model] = val
    return table


def print_table(
    table: dict[tuple[str, str], dict[str, float]],
    title: str,
    unit: str = "ms",
) -> None:
    """Print a formatted comparison table."""
    col_w = 13  # width per model column

    header_models = "  ".join(f"{m:>{col_w}}" for m in MODEL_NAMES)
    sep = "-" * (26 + len(MODEL_NAMES) * (col_w + 2))

    print(f"\n{title}")
    print(f"{'Runtime':<22} {'Device':<7}  {header_models}")
    print(sep)

    for runtime, device in _ROW_ORDER:
        key = (runtime, device)
        row_data = table.get(key, {})
        cells = []
        for m in MODEL_NAMES:
            val = row_data.get(m)
            if val is not None:
                cells.append(f"{val:>{col_w}.3f}")
            else:
                cells.append(f"{'N/A':>{col_w}}")
        print(f"{runtime:<22} {device:<7}  {'  '.join(cells)}")


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference timing benchmark across all runtimes")
    p.add_argument("--onnx-dir", type=Path, default=DEFAULT_ONNX_DIR, help="Directory with .onnx, .pth, .pt files and holdout binaries")
    p.add_argument("--inference-bin", type=Path, default=DEFAULT_INFERENCE_BIN, help="Path to the ONNX Runtime C++ binary")
    p.add_argument("--trt-bin", type=Path, default=DEFAULT_TRT_BIN, help="Path to the TensorRT C++ binary")
    p.add_argument("--skip-python", action="store_true", help="Skip Python .pth/.pt benchmarks (just run C++ binaries and print results)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    onnx_dir: Path = args.onnx_dir

    print("=" * 60)
    print("Inference Benchmark")
    print("=" * 60)

    sysinfo = system_info()
    print(f"CPU  : {sysinfo['cpu']}")
    print(f"GPU  : {sysinfo['gpu']}")
    print(f"CUDA : {sysinfo['cuda']}")

    # Load holdout data (written by train.py)
    inputs, targets, n_windows = load_holdout(onnx_dir)
    print(f"\nHoldout: {n_windows} windows x {LOOKBACK} timesteps x {N_FEATURES} features")

    all_metrics: list[dict[str, Any]] = []

    # --- Python benchmarks ---
    if not args.skip_python:
        print("\n" + "=" * 60)
        print("Python inference benchmarks")
        print("=" * 60)
        py_metrics = run_python_benchmarks(onnx_dir, inputs, targets)
        all_metrics.extend(py_metrics)

        # Save Python results so they survive between runs
        py_out = onnx_dir / "python_metrics.json"
        with open(py_out, "w") as f:
            json.dump(py_metrics, f, indent=2)
        print(f"\nSaved {py_out}")
    else:
        # Load previously saved Python metrics if available
        py_cached = onnx_dir / "python_metrics.json"
        if py_cached.exists():
            py_metrics = load_json_metrics(py_cached)
            all_metrics.extend(py_metrics)
            print(f"\nLoaded cached Python metrics from {py_cached}")

    # --- ONNX Runtime C++ ---
    print("\n" + "=" * 60)
    print("ONNX Runtime C++ benchmark")
    print("=" * 60)
    ort_ok = run_cpp_binary(args.inference_bin, onnx_dir, "ONNX / ORT C++")
    if ort_ok:
        cpp_metrics = load_json_metrics(onnx_dir / "cpp_metrics.json")
        all_metrics.extend(cpp_metrics)

    # --- TensorRT C++ ---
    print("\n" + "=" * 60)
    print("TensorRT C++ benchmark")
    print("=" * 60)
    trt_ok = run_cpp_binary(args.trt_bin, onnx_dir, "TensorRT C++")
    if trt_ok:
        trt_metrics = load_json_metrics(onnx_dir / "trt_metrics.json")
        all_metrics.extend(trt_metrics)

    # --- Summary tables ---
    if not all_metrics:
        print("\nNo metrics collected. Run train.py first, then build the C++ binaries.")
        return

    mean_table = build_table(all_metrics, "mean")
    p99_table = build_table(all_metrics, "p99")

    print("\n" + "=" * 60)
    print("=== Inference Benchmark Summary ===")
    print(f"CPU  : {sysinfo['cpu']}")
    print(f"GPU  : {sysinfo['gpu']}")
    print(f"CUDA : {sysinfo['cuda']}")
    print("=" * 60)

    print_table(mean_table, "Mean latency (ms)  —  single window, batch_size=1")
    print_table(p99_table, "p99 latency (ms)")


if __name__ == "__main__":
    main()
