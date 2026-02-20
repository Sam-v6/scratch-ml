import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding from "Attention is All You Need"
    (Vaswani et al., 2017).

    Self-attention is position-agnostic — swapping two input timesteps would
    produce the same output.  Positional encoding fixes this by adding a
    deterministic signal that uniquely encodes each position.

    For position `pos` and embedding dimension index `i`:
        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    The table is pre-computed once at construction and stored as a buffer
    (not a learned parameter), so it requires no gradient and moves to GPU
    automatically with .to(device).
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()  # (max_len, 1)

        # Frequency denominators in log-space for numerical stability.
        # div_term[i] = 1 / 10000^(2i/d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # (d_model/2,)

        pe[:, 0::2] = torch.sin(position * div_term)  # even indices → sine
        pe[:, 1::2] = torch.cos(position * div_term)  # odd  indices → cosine

        # Add a batch dimension so broadcasting works: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        # Slice the PE table to the actual sequence length T
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class TimeSeriesTransformer(nn.Module):
    """
    Encoder-only Transformer for time-series regression.

    Unlike the LSTM, the Transformer processes all T timesteps *in parallel*
    using multi-head self-attention.  Every timestep attends to every other
    timestep in a single forward pass — no sequential computation.

    Architecture:
        input_proj   — Linear(F, d_model)  embeds each timestep into d_model dims
        pos_enc      — sinusoidal positional encoding (adds position info)
        encoder      — N × TransformerEncoderLayer
                         each layer = MultiHeadAttention + FFN + two LayerNorms
        mean pool    — average all T output tokens into one vector
        head         — Linear(d_model, 1)

    TransformerEncoderLayer details:
        - d_model=64, nhead=8 → each head has dimension 64/8 = 8
        - dim_feedforward=256 → the inner FFN projects 64→256→64
        - Post-norm (LayerNorm after residual connection, matching the original paper)
    """

    def __init__(self, input_size: int, d_model: int = 64, nhead: int = 8, num_layers: int = 3, dim_feedforward: int = 256, dropout: float = 0.1) -> None:
        super().__init__()

        # Project each F-dimensional input timestep into d_model dimensions.
        # This is analogous to the token embedding in NLP Transformers.
        self.input_proj = nn.Linear(input_size, d_model)

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,  # CRITICAL: without this PyTorch expects (T, B, d_model)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = self.input_proj(x)  # (B, T, d_model)
        x = self.pos_enc(x)  # (B, T, d_model)  — position info added
        x = self.encoder(x)  # (B, T, d_model)  — self-attention applied
        x = x.mean(dim=1)  # (B, d_model)      — mean pool over time
        return self.head(x)  # (B, 1)
