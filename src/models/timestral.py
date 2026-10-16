"""Local zero-shot next-step forecasting; import the optional package only when loading."""

from typing import Protocol

import numpy as np
import torch


class QuantileForecast(Protocol):
    def quantile(self, level: float) -> torch.Tensor: ...


class Forecaster(Protocol):
    def forecast(self, context: np.ndarray, *, prediction_length: int, past_covariates: np.ndarray) -> QuantileForecast: ...


def forecast_timestral(forecaster: Forecaster, windows: np.ndarray) -> np.ndarray:
    """Return p10/p50/p90 for windows (B,T,4): sine, square, triangle, target."""
    forecast = forecaster.forecast(windows[:, None, :, 3], prediction_length=1, past_covariates=windows[:, :, :3].transpose(0, 2, 1))
    return np.stack([forecast.quantile(q).detach().cpu().numpy()[:, 0, 0] for q in (0.1, 0.5, 0.9)], axis=1)
