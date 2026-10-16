"""Measure warmed Timestral-5M CPU accuracy, throughput and single-window latency."""

import json
import platform
import time

import numpy as np
import pandas as pd
import torch
from timestral import TimestralForecaster

from compare_timestral import CHANNELS, MODEL_REVISIONS, ROOT, make_windows
from models.timestral import forecast_timestral


def main() -> None:
    torch.set_num_threads(4)
    df = pd.read_csv(ROOT / "data/input/data_signals.csv")
    x, y, indices = make_windows(df[CHANNELS].to_numpy(dtype=np.float32), 1600, 2000, 256)
    start = time.perf_counter()
    f = TimestralForecaster.from_pretrained("mistralai/Timestral-5M", revision=MODEL_REVISIONS["5M"], device="cpu", local_files_only=True)
    load_seconds = time.perf_counter() - start
    forecast_timestral(f, x[:16])
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        q = np.concatenate([forecast_timestral(f, b) for b in np.array_split(x, 25)])
        timings.append(time.perf_counter() - start)
    latencies = []
    for _ in range(20):
        start = time.perf_counter()
        forecast_timestral(f, x[:1])
        latencies.append((time.perf_counter() - start) * 1000)
    assert next(f.model.parameters()).device.type == "cpu"
    gpu = pd.read_csv(ROOT / "docs/timestral/predictions.csv")["Timestral-5M"].to_numpy()
    result = dict(
        device="cpu",
        cpu=platform.processor(),
        threads=4,
        model="Timestral-5M",
        revision=MODEL_REVISIONS["5M"],
        windows=400,
        lookback=256,
        horizon=1,
        rmse=float(np.sqrt(np.mean((q[:, 1] - y) ** 2))),
        mae=float(np.mean(abs(q[:, 1] - y))),
        load_seconds=load_seconds,
        batch_size=16,
        total_inference_seconds=timings,
        median_total_seconds=float(np.median(timings)),
        single_forecast_median_ms=float(np.median(latencies)),
        single_forecast_p95_ms=float(np.percentile(latencies, 95)),
        max_abs_difference_from_gpu=float(np.max(abs(q[:, 1] - gpu))),
    )
    out = ROOT / "artifacts/timestral-cpu"
    out.mkdir(exist_ok=True)
    (out / "run.json").write_text(json.dumps(result, indent=2) + "\n")
    pd.DataFrame(dict(index=indices, actual=y, p10=q[:, 0], median=q[:, 1], p90=q[:, 2])).to_csv(out / "predictions.csv", index=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
