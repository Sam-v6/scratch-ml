from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from path import SCRATCH_HOME


def rescale_signal(x: np.ndarray, new_min: float = 1.0, new_max: float = 5.0) -> np.ndarray:
    """Rescale array x to range [new_min, new_max]."""
    old_min, old_max = np.min(x), np.max(x)
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


def generate_synthetic_dataframe(
    duration_s: float = 20.0,
    fs: int = 100,
    seed: int = 42,
    lag_steps: int = 5,
    noise_std: float = 0.1,
    rescale_min: float = 1.0,
    rescale_max: float = 5.0,
) -> pd.DataFrame:
    """
    Generate synthetic time-series feature/target data.

    Returns a DataFrame with columns:
        time, sine, square, triangle, target
    """
    rng = np.random.default_rng(seed)

    # Define time axis
    t = np.linspace(0.0, duration_s, int(duration_s * fs), endpoint=False)

    # Features
    s = np.sin(2 * np.pi * 1.0 * t) + noise_std * rng.standard_normal(len(t))
    q = signal.square(2 * np.pi * 2.0 * t, duty=0.5) + noise_std * rng.standard_normal(len(t))
    tri = signal.sawtooth(2 * np.pi * 1.0 * t, width=0.5) + noise_std * rng.standard_normal(len(t))

    # Memory term via lagged triangle
    tri_lag = np.roll(tri, lag_steps)
    tri_lag[:lag_steps] = tri_lag[lag_steps]

    # Nonlinear target + slow envelope
    y_base = 1.2 * s + 0.5 * (s * tri_lag) + 0.6 * (q * tri) + 0.3 * (tri**2)
    env = 1.0 + 0.4 * np.sin(2 * np.pi * 0.2 * t)
    y = env * y_base + noise_std * rng.standard_normal(len(t))

    # Rescale all channels
    s = rescale_signal(s, rescale_min, rescale_max)
    q = rescale_signal(q, rescale_min, rescale_max)
    tri = rescale_signal(tri, rescale_min, rescale_max)
    y = rescale_signal(y, rescale_min, rescale_max)

    return pd.DataFrame(
        {
            "time": t,
            "sine": s,
            "square": q,
            "triangle": tri,
            "target": y,
        }
    )


def plot_synthetic_signals(df: pd.DataFrame, out_path: str | Path) -> None:
    """Plot feature/target channels and save to PNG."""
    out_path = Path(out_path)
    fig, axs = plt.subplots(4, 1, figsize=(12, 8))

    axs[0].plot(df["time"], df["sine"])
    axs[0].set_title("Feature: Sine Wave")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Voltage [V]")
    axs[0].set_xlim(1, 20)
    axs[0].set_ylim(1, 5)

    axs[1].plot(df["time"], df["square"])
    axs[1].set_title("Feature: Square Wave")
    axs[1].set_xlabel("Time (s)")
    axs[1].set_ylabel("Voltage [V]")
    axs[1].set_xlim(1, 20)
    axs[1].set_ylim(1, 5)

    axs[2].plot(df["time"], df["triangle"])
    axs[2].set_title("Feature: Triangle Wave")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Voltage [V]")
    axs[2].set_xlim(1, 20)
    axs[2].set_ylim(1, 5)

    axs[3].plot(df["time"], df["target"])
    axs[3].set_title("Target: Combo Wave")
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("Voltage [V]")
    axs[3].set_xlim(1, 20)
    axs[3].set_ylim(1, 5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=800)
    plt.close(fig)


def main() -> None:
    df = generate_synthetic_dataframe()
    out_dir = SCRATCH_HOME / "data" / "input"
    out_dir.mkdir(parents=True, exist_ok=True)

    png_path = out_dir / "data_signals.png"
    csv_path = out_dir / "data_signals.csv"

    plot_synthetic_signals(df, png_path)
    print(df.head())
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    main()
