# Base imports
import json
import logging
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from ray import tune
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from common.paths import ARTIFACT_PATH, PROJECT_ROOT
from models.lstm_regressor import LSTMRegressor


class LstmModel:
	"""
	Trainer class for LSTM model.
	"""

	def __init__(self, lookback: int) -> None:
		self.lookback = lookback

	###################################
	# Data
	###################################
	def _get_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
		"Gets data from CSV and returns numpy arrays."

		# Load in training data
		df = pd.read_csv(PROJECT_ROOT / "data" / "data_signals.csv")

		# Assign time series data
		x = df[["sine", "square", "triangle"]].to_numpy(dtype=np.float32)
		y = df[["target"]].to_numpy(dtype=np.float32)

		# Splittys (but since it's time series we don't need anything extra here)
		train_end = int(0.8 * len(df))
		x_train, x_val = x[:train_end, :], x[train_end:, :]
		y_train, y_val = y[:train_end, :], y[train_end:, :]

		return x_train, y_train, x_val, y_val

	def _make_windows(self, x: np.ndarray, y: np.ndarray, lookback: int, horizon: int = 1, stride: int = 1, as_torch: bool = True) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
		"""
		Build sliding windows for sequence models.

		Args:
		x: np.ndarray of shape (T, x_dim)
		y: np.ndarray of shape (T,) or (T, y_dim)
		lookback: number of past steps per input window
		horizon: number of future steps to predict (default: 1 = next-step)
		stride: shift between consecutive windows (default: 1)
		as_torch: if True, return torch.float32 tensors, else numpy arrays

		Returns:
		X: (N of windows (batches), lookback, x_features)
		Y: (N of windows, y_features) if horizon==1 else (N of windows, horizon, y_features)

		Notes:
		No shuffling; preserves time order (good for forecasting).
		No leakage: each target window starts exactly after each input window.
		Number of windows (or batches) is N = ((total number of samples - lookback - horizon)/stride) + 1
		"""
		x = np.asarray(x)
		y = np.asarray(y)

		if y.ndim == 1:
			y = y.reshape(-1, 1)

		T = x.shape[0]

		max_start = T - lookback - horizon + 1
		if max_start <= 0:
			raise ValueError(f"Not enough timesteps: T={T}, lookback={lookback}, horizon={horizon}")

		X_list, Y_list = [], []
		for start in range(0, max_start, stride):
			end = start + lookback
			tgt_end = end + horizon
			X_list.append(x[start:end])  # (lookback, x_dim)
			Y_list.append(y[end:tgt_end])  # (horizon, y_dim)

		X = np.stack(X_list, axis=0)  # (N, lookback, x_dim)
		Y = np.stack(Y_list, axis=0)  # (N, horizon, y_dim)

		if horizon == 1:
			Y = Y[:, 0, :]  # (N, y_dim)

		if as_torch:
			X = torch.from_numpy(X).to(torch.float32)
			Y = torch.from_numpy(Y).to(torch.float32)

		return X, Y

	def transform_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
		x_train, y_train, x_val, y_val = self._get_data()

		# Standarize features
		# NOTE: Calculates the mean and standard deviation for each feature in the training set and applies scaling transformation
		# X' = (X - mean of feature) / std of feature --> after scaling each features has mean of 0 is and std of 1 ish, all features get normalized to similar range
		scaler = StandardScaler().fit(x_train)  # We want to scale only on training data to avoid leakage
		x_train_scaled_np = scaler.transform(x_train)
		x_val_scaled_np = scaler.transform(x_val)

		# Makes overlapping windows of data (with targets as n+1) then converts to tensors that live on CPU
		x_seq_train_torch, y_seq_train_torch = self._make_windows(x_train_scaled_np, y_train, lookback=self.lookback, horizon=1, stride=1, as_torch=True)
		x_seq_val_torch, y_seq_val_torch = self._make_windows(x_val_scaled_np, y_val, lookback=self.lookback, horizon=1, stride=1, as_torch=True)

		return x_seq_train_torch, y_seq_train_torch, x_seq_val_torch, y_seq_val_torch, scaler

	##################################
	# Training
	##################################
	def _create_dataloaders(self, x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, batch_size: int, g: torch.Generator) -> tuple[DataLoader, DataLoader]:
		######################################################################
		# Create dataloaders
		######################################################################
		# Create data loaders for batching
		train_loader = DataLoader(
			TensorDataset(x_train, y_train),
			batch_size=batch_size,
			shuffle=True,  # We want random mini batches so GD doesn't overfit to specific ordering patterns, lets shuffle
			generator=g,  # Fixes the shuffle
			num_workers=0,  # Eliminate worker non-determinism
			pin_memory=True,  # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
		)

		val_loader = DataLoader(
			TensorDataset(x_val, y_val),
			batch_size=batch_size,
			shuffle=False,  # In eval we aren't updating the weights, so it doesn't really matter if we imply ordering or not
			num_workers=0,  # Eliminate worker non-determinism
			pin_memory=True,  # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
		)

		return train_loader, val_loader

	def _create_lstm_model(self, device: torch.device, hidden_size: int, num_layers: int, dropout: float) -> LSTMRegressor:
		model = LSTMRegressor(
			input_size=3,
			hidden_size=hidden_size,
			num_layers=num_layers,
			dropout=dropout,
		).to(device)

		return model

	def _train_epoch(self, model: torch.nn.Module, device: torch.device, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader) -> float:
		model.train()  # Setting some dropout layers to go to 0 to prevent overfitting & normalize activations of previous layer
		train_loss_sum = 0.0  # For each epoch we want to zero out the training loss since we are starting fresh on the dataset
		n = 0
		for xb, yb in train_loader:
			# allows CPU to to continue exec'ing code while data transfer to GPU goes concurrently (requires CPU pinned memory)
			xb = xb.to(device, non_blocking=True)
			yb = yb.to(device, non_blocking=True)
			optimizer.zero_grad()  # For each batch we compute loss and take the gradient of that to update model weights, then make a model update, so zero out gradients from prior batch
			preds = model(xb)  # Forward pass of predictions, this does the call forward fcn
			loss = criterion(preds, yb)  # Computes loss of predictions vs actuals, this is the average loss over the batch
			loss.backward()  # Backpropagation, computes gradients of loss w.r.t. each model parameter and stores in param.grad
			optimizer.step()  # Reads gradients and updates model params based on optimizer config (learning rate, etc)
			train_loss_sum += loss.item() * xb.size(
				0
			)  # Want the sum of loss over samples, because at end of training we divide by total samples to get exact dataset loss, even when last batch may not be exactly batch size
			n += xb.size(0)  # We want to record number of batches we went through, since we may decide to break out early or something
		train_loss = train_loss_sum / n  # We divide the accumualted loss for the batch by the number of batches we actually got through

		return train_loss

	def _validate_epoch(self, model: torch.nn.Module, device: torch.device, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, val_loader: DataLoader) -> float:
		model.eval()  # Disable droput layers and activation normalization
		val_loss_sum = 0.0  # Zero out validation loss for each epoch
		n = 0
		with torch.no_grad():  # We aren't updating weights, so we don't need to compute gradients, this saves memory and computations
			for xb, yb in val_loader:
				xb = xb.to(device, non_blocking=True)  # non_blocking allows CPU to to continue exec'ing code while data transfer to GPU goes concurrently (requires CPU pinned memory)
				yb = yb.to(device, non_blocking=True)
				preds = model(xb)
				loss = criterion(preds, yb)
				val_loss_sum += loss.item() * xb.size(0)
				n += xb.size(0)  # We want to record number of batches we went through, since we may decide to break out early or somethin
		val_loss = val_loss_sum / n  # We divide the accumualted loss for the batch by the number of batches we actually got through

		return val_loss

	def _plot_losses(self, epoch: int, train_rmse_hist: list[float], val_rmse_hist: list[float], save_dir: Path) -> Path:
		matplotlib.use("Agg")  # Matplotlib runs headless
		fig = plt.figure()
		plt.plot(range(1, epoch + 1), train_rmse_hist, label="train_rmse")
		plt.plot(range(1, epoch + 1), val_rmse_hist, label="val_rmse")
		plt.xlabel("Epoch")
		plt.ylabel("RMSE")
		plt.title("Training/Validation RMSE")
		plt.legend()
		plt.tight_layout()
		loss_plot_path = save_dir / "loss_curve.png"
		plt.savefig(loss_plot_path, dpi=150)
		plt.close(fig)

		return loss_plot_path

	def train_trial(self, config: dict, data_ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler], RAY_HPO: bool = False) -> None | float:
		"""Train one model that can be used with our without Ray Tune."""

		#########################################################################
		# Set seeds
		#########################################################################
		base_seed = 42
		np.random.seed(base_seed)
		# torch.manual_seed(base_seed)
		g = torch.Generator()  # Creates a generator that fixes the shuffle in torch Dataloader
		g.manual_seed(base_seed)

		#########################################################################
		# Create dataloaders
		#########################################################################
		x_train, y_train, x_val, y_val, scaler = data_ref  # Pull out data from Ray store
		train_loader, val_loader = self._create_dataloaders(
			x_train,
			y_train,
			x_val,
			y_val,
			int(config["batch_size"]),
			g,
		)

		######################################################################
		# Create model, loss, optimizer
		######################################################################
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # this is the Ray-assigned GPU (Ray sets CUDA_VISIBLE_DEVICES)
		criterion = torch.nn.MSELoss()

		# LSTM model
		model = self._create_lstm_model(
			device=device,
			hidden_size=int(config["hidden_size"]),
			num_layers=int(config["num_layers"]),
			dropout=float(config["dropout"]),
		)

		# Adapative moment estimation, makes sure we step opposite smoothed gradient and shrink/grow step based on how noisy each model parameter's gradient has been
		optimizer = torch.optim.Adam(model.parameters())

		######################################################################
		# Training Loop
		######################################################################
		epochs = int(config["epochs"])
		best_val_loss = float("inf")
		train_rmse_hist, val_rmse_hist = [], []

		# Initiate the MLflow run context
		for epoch in range(1, epochs + 1):
			# Training & Validation
			train_loss = self._train_epoch(model, device, criterion, optimizer, train_loader)
			val_loss = self._validate_epoch(model, device, criterion, optimizer, val_loader)

			# Create performance metrics
			train_rmse = float(np.sqrt(train_loss))
			val_rmse = float(np.sqrt(val_loss))
			train_rmse_hist.append(train_rmse)
			val_rmse_hist.append(val_rmse)

			# We want to report metrics every epoch regardless if we are checkpointing
			metrics = {
				"val_rmse": float(val_rmse_hist[-1]),
				"train_rmse": float(train_rmse_hist[-1]),
				"epoch": int(epoch),
			}

			######################################################################
			# Tune logging (with MLflow callback this is all mirrored there as well)
			# NOTE: By calling tune.report here effectively once per epoch, that becomes our time scale!
			######################################################################
			if RAY_HPO:
				# Report to Ray (MLflow mirroring is handled by Ray's MLflowCallback)
				tune.report(metrics)
			else:
				# Update saved model
				best_model_path = ARTIFACT_PATH / "lstm.pt"
				if np.mean(val_rmse_hist) < best_val_loss:
					best_val_loss = np.mean(val_rmse_hist)
					torch.save(model.state_dict(), best_model_path)

			# Status for screen
			if epoch % 10 == 0:
				logging.info(f"Epoch {epoch:02d} | train RMSE: {train_rmse:.6f} | val RMSE: {val_rmse:.6f}")

		#########################################################################
		# If this is not Ray orchestrated, log mlflow params, register model, and log onnx artifact
		#########################################################################
		if not RAY_HPO:
			# Plot the training results (we don't want to do this with HPO because Ray records these as metrics)
			self._plot_losses(epoch, train_rmse_hist, val_rmse_hist, ARTIFACT_PATH)

			####################################################
			# ONNX export
			####################################################
			# Use a batch=1 example for LSTM safety
			x_example_tensor = x_train[:10]  # torch.Tensor
			example_input_t = x_example_tensor[:1].detach().cpu().float()
			onnx_path = ARTIFACT_PATH / "lstm.onnx"

			torch.onnx.export(
				model.to("cpu").eval(),
				example_input_t,
				onnx_path,
				input_names=["input"],
				output_names=["output"],
				opset_version=int(config.get("onnx_opset", 18)),  # guess
				dynamo=True,
			)

			return best_val_loss

	def train_model(self) -> None:
		"""Main training function."""
		# Get data
		x_train, y_train, x_val, y_val, scaler = self.transform_data()

		# Load model params from json
		config_path = PROJECT_ROOT / "data" / "model_params.json"
		with open(config_path) as f:
			model_params_config = json.load(f)

		# Train
		self.train_trial(
			config=model_params_config,
			data_ref=(x_train, y_train, x_val, y_val, scaler),
			RAY_HPO=False,
		)

	##################################
	# Predictions
	##################################
	@torch.no_grad()
	def _predict_ordered_windows(model: torch.nn.Module, X: torch.Tensor, device: torch.device) -> np.ndarray:
		model.eval()
		# If large, do mini-batches; for small, one shot is fine:
		X = X.to(device)

		# NOTE:
		# Squeeze removes the dimensions of size 1 so goes from (N, 1) --> (N,)
		# Detach makes a new tensor that does not have gradient info
		# CPU moves it from a CUDA tensor to cpu mem
		# Converts back to numpy
		yhat = model(X).squeeze(-1).detach().cpu().numpy()  # (N,)
		return yhat

	def _make_predictions(
		self, device: torch.device, model: torch.nn.Module, x_seq_train_torch: torch.Tensor, x_seq_val_torch: torch.Tensor, df: pd.DataFrame, train_end: int
	) -> tuple[np.ndarray, np.ndarray, pd.Series, np.ndarray, np.ndarray, float]:
		# Get predictions in chronological (non-shuffled) order
		yhat_train = self._predict_ordered_windows(model, x_seq_train_torch, device)  # (N_tr,)
		yhat_val = self._predict_ordered_windows(model, x_seq_val_torch, device)  # (N_va,)

		# Map window-indexed predictions to absolute time indices
		time = df["time"]
		N_tr = len(yhat_train)  # training prediction count
		N_va = len(yhat_val)  # validation prediction count

		# For horizon=1, each window predicts the sample right after the window:
		# train windows cover indices [0 .. train_end-1], so their targets are at:
		t_idx_train = np.arange(self.lookback, self.lookback + N_tr)  # absolute:  lookback .. train_end-1
		# val windows are built on the val segment, so offset by train_end:
		t_idx_val = train_end + np.arange(self.lookback, self.lookback + N_va)  # absolute:  train_end+lookback .. T-1

		# Determine normalized RMSE (by mean)
		nRMSE = (best_val / df["target"].mean()) * 100

		return yhat_train, yhat_val, time, t_idx_train, t_idx_val, nRMSE

	def _plot_predictions(self, df: pd.DataFrame, yhat_train: np.ndarray, yhat_val: np.ndarray, time: pd.Series, t_idx_train: np.ndarray, t_idx_val: np.ndarray, nRMSE: float) -> None:
		# Plot against the real time axis
		y_plot = df["target"]
		fig, ax = plt.subplots(figsize=(12, 5))
		ax.plot(time, y_plot, label="Actual", linewidth=1.5, alpha=0.4)  # blue by default
		ax.plot(time[t_idx_train], yhat_train, label="Train Predictions", linewidth=1.0, color="red")
		ax.plot(time[t_idx_val], yhat_val, label="Validation Predictions", linewidth=1.0, color="green")
		ax.set_xlabel("Time (s)")
		ax.set_ylabel("Voltage (V)")
		ax.set_xlim(1, 20)
		ax.set_ylim(1, 5)
		ax.set_title(f"Predictions Aligned to Real Time\nRMSE: {round(best_val, 2)} V, nRMSE (mean): {round(nRMSE, 2)}%")
		ax.legend()
		plt.tight_layout()
		plt.savefig(ARTIFACT_PATH / "predictions.png", dpi=300)

	def predict(self) -> None:
		# Load back model params
		config_path = PROJECT_ROOT / "data" / "model_params.json"
		with open(config_path) as f:
			config = json.load(f)

		# Instatiate model
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		model = self._create_lstm_model(
			device=device,
			hidden_size=int(config["hidden_size"]),
			num_layers=int(config["num_layers"]),
			dropout=float(config["dropout"]),
		)

		# Reload state dict
		model.load_state_dict(torch.load(ARTIFACT_PATH / "lstm.pt", map_location="cpu"))

		# Get data
		x_seq_train_torch, y_seq_train_torch, x_seq_val_torch, y_seq_val_torch, scaler = self.transform_data()

		# Make predictions
		df = pd.read_csv(PROJECT_ROOT / "data" / "data_signals.csv")
		train_end = int(0.8 * len(df))
		yhat_train, yhat_val, time, t_idx_train, t_idx_val, nRMSE = self._make_predictions(
			device,
			model,
			x_seq_train_torch,
			x_seq_val_torch,
			df,
			train_end,
		)

		# PLot predictions
		self._plot_predictions(
			df,
			yhat_train,
			yhat_val,
			time,
			t_idx_train,
			t_idx_val,
			nRMSE,
		)


if __name__ == "__main__":
	# Setup logging
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	# Train model
	lstmModel = LstmModel()
	lstmModel.train_model()

	# Run predictions
	lstmModel.predict()
