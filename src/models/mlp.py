import torch
import torch.nn as nn


class MLP(nn.Module):
	def __init__(self, in_dim: int = 3, hidden: int = 32, out_dim: int = 1) -> None:
		super().__init__()
		self.net = torch.nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.net(x)
