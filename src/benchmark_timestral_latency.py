"""Compare warmed CPU single-request latency with optional compilation and single-pass inference."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster

from compare_timestral import CHANNELS, MODEL_REVISIONS, ROOT, make_windows
from models.timestral import forecast_timestral


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile", action="store_true", help="Compile the transformer; first-call cost is recorded separately")
    parser.add_argument("--single-pass", action="store_true", help="Disable mirrored averaging; may change forecast accuracy")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/timestral-latency")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("threads and repeats must be positive")
    torch.set_num_threads(args.threads)
    path = ROOT / "data/input/data_signals.csv"
    df = pd.read_csv(path)
    x, y, indices = make_windows(df[CHANNELS].to_numpy(dtype=np.float32), 1600, 2000, 256)
    model = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", revision=MODEL_REVISIONS["5M"], device="cpu", local_files_only=True)
    if args.compile:
        # Compile the tensor-only stack, leaving input validation and normalization intact.
        model.model.transformer = torch.compile(model.model.transformer, fullgraph=True, dynamic=False)
    start = time.perf_counter()
    forecast_timestral(model, x[:1], symmetric_averaging=not args.single_pass)
    first_call_seconds = time.perf_counter() - start
    for i in range(10):
        forecast_timestral(model, x[i : i + 1], symmetric_averaging=not args.single_pass)
    samples, predictions = [], []
    for repeat in range(args.repeats):
        values = []
        for i, index in enumerate(indices):
            start = time.perf_counter_ns()
            pred = forecast_timestral(model, x[i : i + 1], symmetric_averaging=not args.single_pass)
            elapsed_ms = (time.perf_counter_ns() - start) / 1e6
            samples.append(dict(repeat=repeat, index=int(index), latency_ms=elapsed_ms))
            values.append(pred)
        predictions.append(np.concatenate(values))
    q = predictions[0]
    if not np.isfinite(q).all() or not np.all(np.diff(q, axis=1) >= 0):
        raise ValueError("Invalid output quantiles")
    if not all(np.allclose(q, other, rtol=1e-6, atol=1e-6) for other in predictions[1:]):
        raise ValueError("Predictions changed between repeated passes")
    samples_df = pd.DataFrame(samples)
    gpu = pd.read_csv(ROOT / "docs/timestral/predictions.csv")["Timestral-5M"].to_numpy()
    receipt = dict(
        model="Timestral-5M",
        revision=MODEL_REVISIONS["5M"],
        device="cpu",
        cpu=platform.processor(),
        torch_version=torch.__version__,
        threads=args.threads,
        compiled=args.compile,
        symmetric_averaging=not args.single_pass,
        lookback=256,
        horizon=1,
        batch_size=1,
        windows=len(x),
        repeats=args.repeats,
        timed_calls=len(samples_df),
        data_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        first_call_seconds=first_call_seconds,
        median_ms=float(samples_df.latency_ms.median()),
        p95_ms=float(samples_df.latency_ms.quantile(0.95)),
        p99_ms=float(samples_df.latency_ms.quantile(0.99)),
        max_ms=float(samples_df.latency_ms.max()),
        fraction_under_10ms=float((samples_df.latency_ms < 10).mean()),
        rmse=float(np.sqrt(np.mean((q[:, 1] - y) ** 2))),
        mae=float(np.mean(abs(q[:, 1] - y))),
        coverage_80=float(np.mean((y >= q[:, 0]) & (y <= q[:, 2]))),
        max_abs_difference_from_default_gpu=float(np.max(abs(q[:, 1] - gpu))),
        per_repeat=[dict(repeat=int(r), median_ms=float(g.latency_ms.median()), p95_ms=float(g.latency_ms.quantile(0.95))) for r, g in samples_df.groupby("repeat")],
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "run.json").write_text(json.dumps(receipt, indent=2) + "\n")
    samples_df.to_csv(args.output / "latencies.csv", index=False)
    pd.DataFrame(dict(index=indices, actual=y, p10=q[:, 0], median=q[:, 1], p90=q[:, 2])).to_csv(args.output / "predictions.csv", index=False)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
