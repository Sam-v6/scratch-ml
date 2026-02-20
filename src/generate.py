import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

from path import SCRATCH_HOME

#########################################################################
# Create feature data
#########################################################################
inputs = {}  # We will store our data in this dict initially
np.random.seed(42)  # Fix the seed to some constant number like the answer to everything

# Define signal parameters
T = 20  # period (seconds)
fs = 100  # sampling frequency (Hz)
inputs["time"] = np.linspace(0, T, T * fs, endpoint=False)

# Generate sine wave
frequency_sine = 1  # frequency (Hz)
inputs["sine"] = np.sin(2 * np.pi * frequency_sine * inputs["time"]) + (0.1 * np.random.randn(len(inputs["time"])))

# Generate square wave
frequency_square = 2  # frequency (Hz)
duty_cycle = 0.5  # 50% high, 50% low
inputs["square"] = signal.square(2 * np.pi * frequency_square * inputs["time"], duty=duty_cycle) + (0.1 * np.random.randn(len(inputs["time"])))

# Generate triangle wave
frequency_triangle = 1  # frequency (Hz)
inputs["triangle"] = signal.sawtooth(2 * np.pi * frequency_triangle * inputs["time"], width=0.5) + 0.1 * np.random.randn(len(inputs["time"]))

#########################################################################
# Generate target wave as a non-linear combonation of the inputs
#########################################################################
# Short names
t = inputs["time"]
s = inputs["sine"]
q = inputs["square"]
tri = inputs["triangle"]

# Nonlinear mix of the three inputs (no lags, no filters)
#   - s * tri         : interaction term (nonlinear)
#   - q * tri         : regime-sensitive interaction (square flips sign)
#   - tri**2          : even nonlinear term (always ≥ 0)
#   - s               : linear piece for a baseline correlation
y_base = 1.2 * s + 0.5 * (s * tri) + 0.6 * (q * tri) + 0.3 * (tri**2)

# Add in noise to act like this measurement was sampled
noise = 0.1 * np.random.randn(len(t))

# Add memory
# NOTE: Create a delayed version of signal, so the target depends not only on the current values of sine/square/triangle, but also on some past values...
# this forces LSTM to actually use its memory instead of just mapping current inputs directly
tri_lag = np.roll(tri, 5)  # shift triangle wave 5 samples to the right
tri_lag[:5] = tri_lag[5]  # fix the first 5 entries (so they don’t wrap around)

# Add in the lag
y_base = 1.2 * s + 0.5 * (s * tri_lag) + 0.6 * (q * tri) + 0.3 * (tri**2)

# NOTE: If we just left y base as is, target is stationary (same non-linear formula applies), but now the strenght of the relationship changes throughout time...abs
# making it harder for the LSTM to match
env = 1.0 + 0.4 * np.sin(2 * np.pi * 0.2 * t)

# Final target
inputs["target"] = env * y_base + noise


#########################################################################
# Rescale data to between 1-10 (let's act like this is a 1-5 V measurement
#########################################################################
def rescale_signal(x: np.ndarray, new_min: float = 1, new_max: float = 5) -> np.ndarray:
    """Rescale array x to range [new_min, new_max]."""
    old_min, old_max = np.min(x), np.max(x)
    return (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min


# Rescale all signals to [1, 5]
for key in ["sine", "square", "triangle", "target"]:
    inputs[key] = rescale_signal(inputs[key], 1, 5)

#########################################################################
# Plot
#########################################################################
# Let's plot the signals/measurements
fig, axs = plt.subplots(4, 1, figsize=(12, 8))

# Sine
axs[0].plot(inputs["time"], inputs["sine"])
axs[0].set_title("Feature: Sine Wave")
axs[0].set_xlabel("Time (s)")
axs[0].set_ylabel("Voltage [V]")
axs[0].set_xlim(1, 20)
axs[0].set_ylim(1, 5)
# Square
axs[1].plot(inputs["time"], inputs["square"])
axs[1].set_title("Feature: Square Wave")
axs[1].set_xlabel("Time (s)")
axs[1].set_ylabel("Voltage [V]")
axs[1].set_xlim(1, 20)
axs[1].set_ylim(1, 5)
# Triangle
axs[2].plot(inputs["time"], inputs["triangle"])
axs[2].set_title("Feature: Triangle Wave")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Voltage [V]")
axs[2].set_xlim(1, 20)
axs[2].set_ylim(1, 5)
# Target
axs[3].plot(inputs["time"], inputs["target"])
axs[3].set_title("Target: Combo Wave")
axs[3].set_xlabel("Time (s)")
axs[3].set_ylabel("Voltage [V]")
axs[3].set_xlim(1, 20)
axs[3].set_ylim(1, 5)
# Save
plt.tight_layout()
plt.savefig(os.path.join(SCRATCH_HOME, "data", "input", "data_signals.png"), dpi=800)

# Save to csv
df = pd.DataFrame(inputs)
print(df.head())
df.to_csv(os.path.join(SCRATCH_HOME, "data", "input", "data_signals.csv"), index=False)
