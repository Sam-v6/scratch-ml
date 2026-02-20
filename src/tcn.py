import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):
    """
    One block of a TCN: two dilated causal Conv1d layers with a residual connection.

    Key ideas:

    Causal convolution — the output at time t must only depend on inputs at
    times ≤ t (no future leakage).  Standard Conv1d with padding=p pads
    symmetrically (left AND right), so the kernel can see p/2 steps into the
    future.  We fix this by setting padding=0 and manually left-padding only:

        F.pad(x, (pad, 0))   ← pad `pad` steps on the left, 0 on the right

    where pad = dilation * (kernel_size - 1).  After the convolution, the
    sequence length is restored:
        output length = T + pad − dilation*(kernel_size−1) = T  ✓

    Dilated convolution — dilation d means the kernel skips d−1 samples
    between each element it reads.  A kernel of size k with dilation d covers
    a span of d*(k−1)+1 timesteps.  Stacking blocks with d=1,2,4,8,...
    grows the receptive field exponentially without adding parameters.

    Weight normalisation — decouples the weight's magnitude (g) from its
    direction (v/‖v‖), which can stabilise and speed up training.
    Uses nn.utils.parametrizations.weight_norm (the modern API).

    Residual connection — adds the block input to the output; if the block
    can't improve on the identity, it simply learns weights near zero.
    A 1×1 conv is used when the channel dimensions differ.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float = 0.1) -> None:
        super().__init__()

        # Amount to left-pad before each convolution to maintain sequence length
        self.pad = dilation * (kernel_size - 1)

        self.conv1 = nn.utils.parametrizations.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=0)  # padding handled manually
        )
        self.conv2 = nn.utils.parametrizations.weight_norm(nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation, padding=0))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # 1×1 conv maps in_channels → out_channels for the residual path
        # when the two don't match; otherwise the residual is the identity
        self.downsample = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in, T)  — Conv1d uses channels-first ordering

        # First dilated causal conv
        out = F.pad(x, (self.pad, 0))  # (B, C_in, T + pad)
        out = self.dropout(self.relu(self.conv1(out)))  # (B, C_out, T)

        # Second dilated causal conv
        out = F.pad(out, (self.pad, 0))  # (B, C_out, T + pad)
        out = self.dropout(self.relu(self.conv2(out)))  # (B, C_out, T)

        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)  # (B, C_out, T)


class TCN(nn.Module):
    """
    Temporal Convolutional Network — stack of TemporalBlocks with exponentially
    increasing dilation.

    From "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
    for Sequence Modeling" (Bai et al., 2018).

    Dilation schedule for num_levels=6, kernel_size=3:
        Level 0: dilation=1,  span=3,   receptive field contribution = 2
        Level 1: dilation=2,  span=5,   contribution = 4
        Level 2: dilation=4,  span=9,   contribution = 8
        Level 3: dilation=8,  span=17,  contribution = 16
        Level 4: dilation=16, span=33,  contribution = 32
        Level 5: dilation=32, span=65,  contribution = 64
        Total receptive field: 2+4+8+16+32+64 = 126 timesteps
        (comfortably covers the lookback window of 256)

    The public interface takes (B, T, F) like the other models;
    internally it permutes to (B, F, T) for Conv1d.
    """

    def __init__(self, input_size: int, n_filters: int = 64, kernel_size: int = 3, num_levels: int = 6, dropout: float = 0.1) -> None:
        super().__init__()

        layers = []
        for i in range(num_levels):
            in_ch = input_size if i == 0 else n_filters  # first block: F → n_filters
            out_ch = n_filters
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
        self.network = nn.Sequential(*layers)

        self.head = nn.Linear(n_filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F)
        x = x.permute(0, 2, 1)  # (B, F, T)  — Conv1d needs channels-first
        x = self.network(x)  # (B, n_filters, T)
        x = x.mean(dim=-1)  # (B, n_filters)  — mean pool over time
        return self.head(x)  # (B, 1)
