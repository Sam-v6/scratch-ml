import torch
import torch.nn as nn


class ImprovedLSTM(nn.Module):
    """
    LSTM with soft attention over all hidden states.

    Key upgrades over NaiveLSTM:
      - Attention  -- instead of using only h_T, compute a weighted average of
                      h_1 ... h_T.  The weights (alpha) are learned scalars that
                      sum to 1 over the time axis.
                        scores  = attn(out)           # (B, T, 1)
                        alpha   = softmax(scores, T)  # (B, T, 1), sums to 1
                        context = sum_t alpha_t * h_t # (B, H)
      - LayerNorm  -- stabilise the hidden-state distribution before attention
      - Capacity   -- hidden_size=64, num_layers=3 vs. 32/2 in the naive model

    Training recipe (applied in train.py):
        - Gradient clipping: clip_grad_norm_(max_norm=1.0)
        - LR scheduler:      ReduceLROnPlateau(factor=0.5, patience=20)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        # LayerNorm over the feature dimension H (not over the batch/time dims)
        self.norm = nn.LayerNorm(hidden_size)

        # Single linear layer that scores each hidden state with one scalar.
        # No bias -- the softmax normalises the scores anyway.
        self.attn = nn.Linear(hidden_size, 1, bias=False)

        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x:   (B, T, F)
        out, _ = self.lstm(x)
        # out: (B, T, H)  -- hidden state at every timestep

        out = self.norm(out)
        # out: (B, T, H)  -- normalised

        scores = self.attn(out)
        # scores: (B, T, 1)  -- one importance score per timestep

        alpha = torch.softmax(scores, dim=1)
        # alpha: (B, T, 1)  -- attention weights; sum to 1 over the T dimension

        context = (alpha * out).sum(dim=1)
        # context: (B, H)  -- weighted average of all hidden states

        return self.head(context)  # (B, 1)
