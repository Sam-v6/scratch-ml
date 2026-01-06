# Base imports
import json
import logging
import os
from contextlib import nullcontext
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from ray import tune
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from models.lstm import LSTMRegressor
from paths import PROJECT_ROOT


def _mlflow_run_context() -> mlflow.ActiveRun | nullcontext:
	"""Start an MLflow run only if none is active (plays nice with Ray's MLflowCallback)."""
	if mlflow.active_run() is None:
		return mlflow.start_run()
	else:
		return nullcontext()


def _make_windows(x: np.ndarray, y: np.ndarray, lookback: int, horizon: int = 1, stride: int = 1, as_torch: bool = True) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
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


def transform_data(base_seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
	######################################################################
	# Load data and do splits
	######################################################################
	input_path = PROJECT_ROOT / "data" / "input"

	# Load in training data
	df = pd.read_csv(os.path.join(input_path, "data_signals.csv"))

	# Assign time series data
	x = df[["sine", "square", "triangle"]].to_numpy(dtype=np.float32)
	y = df[["target"]].to_numpy(dtype=np.float32)

	# Splittys (but since it's time series we don't need anything extra here)
	train_end = int(0.8 * len(df))
	x_train, x_val = x[:train_end, :], x[train_end:, :]
	y_train, y_val = y[:train_end, :], y[train_end:, :]

	# Standarize features
	# NOTE: Calculates the mean and standard deviation for each feature in the training set and applies scaling transformation
	# X' = (X - mean of feature) / std of feature --> after scaling each features has mean of 0 is and std of 1 ish, all features get normalized to similar range
	scaler = StandardScaler().fit(x_train)  # We want to scale only on training data to avoid leakage
	x_train_scaled_np = scaler.transform(x_train)
	x_val_scaled_np = scaler.transform(x_val)

	# Makes overlapping windows of data (with targets as n+1) then converts to tensors that live on CPU
	lookback = 256
	x_seq_train_torch, y_seq_train_torch = _make_windows(x_train_scaled_np, y_train, lookback=lookback, horizon=1, stride=1, as_torch=True)
	x_seq_val_torch, y_seq_val_torch = _make_windows(x_val_scaled_np, y_val, lookback=lookback, horizon=1, stride=1, as_torch=True)

	return x_seq_train_torch, y_seq_train_torch, x_seq_val_torch, y_seq_val_torch, scaler


def _create_dataloaders(x_train: torch.Tensor, y_train: torch.Tensor, x_val: torch.Tensor, y_val: torch.Tensor, batch_size: int, g: torch.Generator) -> tuple[DataLoader, DataLoader]:
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


def _create_lstm_model(device: torch.device, hidden_size: int, num_layers: int, dropout: float) -> LSTMRegressor:
	model = LSTMRegressor(
		input_size=3,
		hidden_size=hidden_size,
		num_layers=num_layers,
		dropout=dropout,
	).to(device)

	return model


def _train_epoch(model: torch.nn.Module, device: torch.device, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, train_loader: DataLoader) -> float:
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


def _validate_epoch(model: torch.nn.Module, device: torch.device, criterion: torch.nn.Module, optimizer: torch.optim.Optimizer, val_loader: DataLoader) -> float:
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


def _plot_losses(epoch: int, train_rmse_hist: list[float], val_rmse_hist: list[float], save_dir: Path) -> Path:
	import matplotlib.pyplot as plt

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


def train_trial(config: dict, data_ref: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler], base_seed: int, RAY_HPO: bool = False) -> None:
	"""Train one model that can be used with our without Ray Tune."""

	#########################################################################
	# Setup MLflow if this is not a Ray orchestrated trial
	#########################################################################
	if not RAY_HPO:
		mlflow_db_path = PROJECT_ROOT / "log" / "mlflow.db"
		mlflow_tracking_uri = f"sqlite:///{mlflow_db_path}"
		mlflow.set_tracking_uri(mlflow_tracking_uri)

	#########################################################################
	# Set seeds
	#########################################################################
	np.random.seed(base_seed)
	# torch.manual_seed(base_seed)
	g = torch.Generator()  # Creates a generator that fixes the shuffle in torch Dataloader
	g.manual_seed(base_seed)

	#########################################################################
	# Create dataloaders
	#########################################################################
	x_train, y_train, x_val, y_val, scaler = data_ref  # Pull out data from Ray store
	train_loader, val_loader = _create_dataloaders(
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
	model = _create_lstm_model(
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
	with _mlflow_run_context():
		for epoch in range(1, epochs + 1):
			# Training & Validation
			train_loss = _train_epoch(model, device, criterion, optimizer, train_loader)
			val_loss = _validate_epoch(model, device, criterion, optimizer, val_loader)

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
				best_model_path = PROJECT_ROOT / "data" / "training" / "lstm.pt"
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
			ARTIFACT_PATH = PROJECT_ROOT / "artifacts"
			ARTIFACT_PATH.mkdir(parents=True, exist_ok=True)
			_plot_losses(epoch, train_rmse_hist, val_rmse_hist, ARTIFACT_PATH)

			####################################################
			# Build an MLflow signature using a small CPU batch
			####################################################
			model.eval()

			# Choose a small slice for signature + input example
			x_signature_tensor = x_train[: int(config.get("batch_size", 64))]  # torch.Tensor
			x_example_tensor = x_train[:10]  # torch.Tensor

			# Run the model forward with tensors
			with torch.no_grad():
				y_pred_np = model(x_signature_tensor.to(device)).detach().cpu().numpy()

			# Convert inputs to NumPy for MLflow
			x_signature_np = x_signature_tensor.detach().cpu().numpy()  # np.ndarray
			x_example_np = x_example_tensor.detach().cpu().numpy()  # np.ndarray

			# Signature expects numpy (or other) hence why we converted
			signature = mlflow.models.infer_signature(x_signature_np, y_pred_np)

			# Log the trained model (CPU for portability)
			mlflow.pytorch.log_model(
				pytorch_model=model.to("cpu"),
				signature=signature,
				input_example=x_example_np,
			)

			####################################################
			# ONNX export
			####################################################
			# Use a batch=1 example for LSTM safety
			example_input_t = x_example_tensor[:1].detach().cpu().float()
			onnx_path = ARTIFACT_PATH / "model.onnx"

			torch.onnx.export(
				model.to("cpu").eval(),
				example_input_t,
				onnx_path,
				input_names=["input"],
				output_names=["output"],
				opset_version=int(config.get("onnx_opset", 18)),  # guess
				dynamo=True,
			)
			mlflow.log_artifact(onnx_path)


if __name__ == "__main__":
	# Setup logging
	logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

	# Get data
	base_seed = 42
	x_train, y_train, x_val, y_val, scaler = transform_data(base_seed)

	# Load model params from json
	config_path = PROJECT_ROOT / "data" / "model_params.json"
	with open(config_path) as f:
		model_params_config = json.load(f)

	# Train
	train_trial(
		config=model_params_config,
		data_ref=(x_train, y_train, x_val, y_val, scaler),
		base_seed=base_seed,
		RAY_HPO=False,
	)
