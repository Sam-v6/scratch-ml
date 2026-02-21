import torch
import torch.nn as nn


class NaiveLSTM(nn.Module):
    """
    Baseline LSTM: two stacked layers, final hidden state -> linear head.

    The LSTM unrolls over T timesteps, producing a hidden state h_t at each
    step.  This naive version discards h_1 ... h_{T-1} and only uses h_T.
    That's wasteful -- the improved version (improved_lstm.py) fixes it with
    attention.

    Training recipe (applied in train.py, not here):
        - Adam, lr=1e-3
        - No gradient clipping, no LR scheduler
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,
        num_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # nn.LSTM stacks `num_layers` LSTM cells.  dropout is applied *between*
        # stacked layers (not within a single layer), so it has no effect when
        # num_layers == 1.
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,  # expect (B, T, F), not (T, B, F)
        )

        # Map the final hidden state H -> scalar prediction
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:   (B, T, F)
        # Explicitly initialise h₀/c₀ with x.new_zeros so the hidden state is
        # always on the same device as the input.  torch.zeros would bake in a
        # fixed CPU device when traced with torch.jit.trace.
        B = x.size(0)
        h0 = x.new_zeros(self.lstm.num_layers, B, self.lstm.hidden_size)
        c0 = x.new_zeros(self.lstm.num_layers, B, self.lstm.hidden_size)
        self.lstm.flatten_parameters()  # ensure contiguous weight layout for cuDNN
        out, _ = self.lstm(x, (h0, c0))
        # out: (B, T, H)  -- one hidden vector per input timestep
        # We only use the LAST one, discarding all earlier information
        return self.head(out[:, -1, :])  # (B, 1)
