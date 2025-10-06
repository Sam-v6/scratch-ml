# Base imports
import os
import time
import tempfile
import random
import logging

# Common imports
import numpy as np
import pandas as pd
from pathlib import Path

# ML imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib # Save pkl

# Local imports
from models.lstm import LSTMRegressor

def make_windows(x, y, lookback: int, horizon: int = 1, stride: int = 1, as_torch: bool = True):
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
        - No shuffling; preserves time order (good for forecasting).
        - No leakage: each target window starts exactly after each input window.
        - Number of windows (or batches) is N = ((total number of samples - lookback - horizon)/stride) + 1
    """
    x = np.asarray(x)
    y = np.asarray(y)

    if y.ndim == 1:
        y = y.reshape(-1, 1)

    T = x.shape[0]
    x_dim = x.shape[1]
    y_dim = y.shape[1]

    max_start = T - lookback - horizon + 1
    if max_start <= 0:
        raise ValueError(f"Not enough timesteps: T={T}, lookback={lookback}, horizon={horizon}")

    X_list, Y_list = [], []
    for start in range(0, max_start, stride):
        end = start + lookback
        tgt_end = end + horizon
        X_list.append(x[start:end])           # (lookback, x_dim)
        Y_list.append(y[end:tgt_end])         # (horizon, y_dim)

    X = np.stack(X_list, axis=0)              # (N, lookback, x_dim)
    Y = np.stack(Y_list, axis=0)              # (N, horizon, y_dim)

    if horizon == 1:
        Y = Y[:, 0, :]                        # (N, y_dim)

    if as_torch:
        X = torch.from_numpy(X).to(torch.float32)
        Y = torch.from_numpy(Y).to(torch.float32)

    return X, Y

def transform_data(base_seed):
    ######################################################################
    # Load data and do splits
    ######################################################################
    input_path = os.path.join(os.getenv('SCRATCH_HOME'), 'data', 'input')

    # Load in training data
    df = pd.read_csv(os.path.join(input_path, 'data_signals.csv'))
    
    # Assign time series data
    x = df[['sine', 'square', 'triangle']].to_numpy(dtype=np.float32)
    y = df[['target']].to_numpy(dtype=np.float32)

    # Splittys (but since it's time series we don't need anything extra here)
    train_end = int(0.8 * len(df))
    x_train, x_val = x[:train_end, :], x[train_end:, :]
    y_train, y_val = y[:train_end, :], y[train_end:, :]

    # Standarize features
    # NOTE: Calculates the mean and standard deviation for each feature in the training set and applies scaling transformation
    # X' = (X - mean of feature) / std of feature --> after scaling each features has mean of 0 is and std of 1 ish, all features get normalized to similar range
    scaler = StandardScaler().fit(x_train) # We want to scale only on training data to avoid leakage
    x_train_scaled_np = scaler.transform(x_train)
    x_val_scaled_np   = scaler.transform(x_val)

    # Makes overlapping windows of data (with targets as n+1) then converts to tensors that live on CPU
    lookback = 256
    x_seq_train_torch, y_seq_train_torch = make_windows(x_train_scaled_np, y_train, lookback=lookback, horizon=1, stride=1, as_torch=True)
    x_seq_val_torch, y_seq_val_torch = make_windows(x_val_scaled_np, y_val, lookback=lookback, horizon=1, stride=1, as_torch=True)

    return x_seq_train_torch, y_seq_train_torch, x_seq_val_torch, y_seq_val_torch, scaler

def create_dataloaders(x_train, y_train, x_val, y_val, batch_size, g):

    ######################################################################
    # Create dataloaders
    ######################################################################
    # Create data loaders for batching
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,                       # We want random mini batches so GD doesn't overfit to specific ordering patterns, lets shuffle
        generator=g,                        # Fixes the shuffle
        num_workers=0,                      # Eliminate worker non-determinism
        pin_memory=True,                    # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
        )  
    
    val_loader = DataLoader(
        TensorDataset(x_val, y_val),
        batch_size=batch_size,
        shuffle=False,                      # In eval we aren't updating the weights, so it doesn't really matter if we imply ordering or not
        num_workers=0,                      # Eliminate worker non-determinism
        pin_memory=True,                    # Batches are allocated on page-locked ("pinned") memory on the host, allows GPU driver to perform faster async DMA
        )       

    return train_loader, val_loader

def create_lstm_model(device, hidden_size, num_layers, dropout):

    model = LSTMRegressor(
        input_size=3,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        ).to(device)

    return model

def train_epoch(model, device, criterion, optimizer, train_loader) -> float:

    model.train()                   # Setting some dropout layers to go to 0 to prevent overfitting & normalize activations of previous layer
    train_loss_sum = 0.0            # For each epoch we want to zero out the training loss since we are starting fresh on the dataset
    n = 0
    for xb, yb in train_loader:
        # allows CPU to to continue exec'ing code while data transfer to GPU goes concurrently (requires CPU pinned memory)
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad()       # For each batch we compute loss and take the gradient of that to update model weights, then make a model update, so zero out gradients from prior batch
        preds = model(xb)           # Forward pass of predictions, this does the call forward fcn
        loss = criterion(preds, yb) # Computes loss of predictions vs actuals, this is the average loss over the batch
        loss.backward()             # Backpropagation, computes gradients of loss w.r.t. each model parameter and stores in param.grad
        optimizer.step()            # Reads gradients and updates model params based on optimizer config (learning rate, etc)
        train_loss_sum += loss.item() * xb.size(0)  # Want the sum of loss over samples, because at end of training we divide by total samples to get exact dataset loss, even when last batch may not be exactly batch size
        n += xb.size(0)             # We want to record number of batches we went through, since we may decide to break out early or something
    train_loss = train_loss_sum / n  # We divide the accumualted loss for the batch by the number of batches we actually got through

    return train_loss

def validate_epoch(model, device, criterion, optimizer, val_loader) -> float:
    model.eval()                   # Disable droput layers and activation normalization
    val_loss_sum = 0.0             # Zero out validation loss for each epoch
    n = 0
    with torch.no_grad():          # We aren't updating weights, so we don't need to compute gradients, this saves memory and computations
        for xb, yb in val_loader:
            xb = xb.to(device, non_blocking=True)  # non_blocking allows CPU to to continue exec'ing code while data transfer to GPU goes concurrently (requires CPU pinned memory)
            yb = yb.to(device, non_blocking=True)
            preds = model(xb)
            loss = criterion(preds, yb)
            val_loss_sum += loss.item() * xb.size(0)
            n += xb.size(0) # We want to record number of batches we went through, since we may decide to break out early or somethin
    val_loss = val_loss_sum / n  # We divide the accumualted loss for the batch by the number of batches we actually got through

    return val_loss

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    print('wut')
