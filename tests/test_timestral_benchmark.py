import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from compare_timestral import make_windows, plot_results
from models.timestral import forecast_timestral


class ForecastTests(unittest.TestCase):
    def test_windows_end_before_target_and_keep_split_boundary(self) -> None:
        data = np.arange(80, dtype=np.float32).reshape(20, 4)
        x, y, indices = make_windows(data, 8, 12, 4)
        np.testing.assert_array_equal(indices, [8, 9, 10, 11])
        np.testing.assert_array_equal(x[0], data[4:8])
        np.testing.assert_array_equal(y, data[8:12, 3])
        changed = data.copy()
        changed[8:] = -999
        x_changed, _, _ = make_windows(changed, 8, 9, 4)
        np.testing.assert_array_equal(x[:1], x_changed)

    def test_plot_includes_every_requested_model(self) -> None:
        names = [f"Model-{i}" for i in range(10)]
        predictions = pd.DataFrame({"time": [0, 1], "actual": [1, 2], **{name: [1, 2] for name in names}})
        metrics = pd.DataFrame({"model": names, "rmse": [0.1] * 10, "mae": [0.1] * 10})
        titles = []

        def capture(figure: object, path: Path, **kwargs: object) -> None:
            if path.name == "forecasts.png":
                titles.extend(ax.get_title() for ax in figure.axes)

        with patch("matplotlib.figure.Figure.savefig", capture):
            plot_results(predictions, metrics, {"Model-0": [0.2]}, Path("/tmp"))
        self.assertTrue(any("Model-9" in title for title in titles))

    def test_adapter_uses_target_history_and_only_past_sensors(self) -> None:
        x = np.arange(48, dtype=np.float32).reshape(2, 6, 4)
        calls = []

        class Result:
            def quantile(self, level: float) -> torch.Tensor:
                return torch.full((2, 1, 1), level)

        class Recorder:
            def forecast(self, context: np.ndarray, **kwargs: object) -> Result:
                calls.append((context, kwargs))
                return Result()

        result = forecast_timestral(Recorder(), x)
        np.testing.assert_array_equal(calls[0][0], x[:, :, 3][:, None, :])
        np.testing.assert_array_equal(calls[0][1]["past_covariates"], x[:, :, :3].transpose(0, 2, 1))
        self.assertEqual(calls[0][1]["prediction_length"], 1)
        self.assertNotIn("known_covariates", calls[0][1])
        self.assertTrue(calls[0][1].get("symmetric_averaging", True))
        forecast_timestral(Recorder(), x, symmetric_averaging=False)
        self.assertFalse(calls[1][1]["symmetric_averaging"])
        np.testing.assert_allclose(result, [[0.1, 0.5, 0.9]] * 2)


if __name__ == "__main__":
    unittest.main()
