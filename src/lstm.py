import torch.nn as nn

class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        # LSTM reads a sequence of vectors (len = T, dim = input_size) and produces a sequence of hidden states (dim = hidden_size)
        # B = batch size or number of windows
        # T = samples in a batch/window
        # F = feature dimension
        self.lstm = nn.LSTM(
            input_size=input_size,        # features per timestep, F
            hidden_size=hidden_size,      # size of LSTM hidden state, H
            num_layers=num_layers,        # stack depth: 1..N LSTM layers
            dropout=dropout if num_layers > 1 else 0.0,  # dropout *between* stacked layers
            batch_first=True,             # expect x shaped (B, T, F) instead of (T, B, F)
        )
        # Final linear head maps the LAST timestep's hidden state (H) → scalar target (1)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (B, T, F)
        out, (h_n, c_n) = self.lstm(x)
        # out: (B, T, H)  hidden states for every timestep
        # h_n: (num_layers, B, H) final hidden state per layer (top layer is h_n[-1])
        # c_n: (num_layers, B, H) final cell state per layer (not used here)

        last = out[:, -1, :]             # take the hidden state at the final timestep: (B, H)
        yhat = self.head(last)           # project to scalar: (B, 1)
        return yhat