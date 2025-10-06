import torch.nn

class MLP(nn.Module):
    def __init__(self, in_dim=3, hidden=32, out_dim=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x):
        return self.net(x)