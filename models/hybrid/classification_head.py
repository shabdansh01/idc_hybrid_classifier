import torch.nn as nn

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, 1)
        )

    def forward(self, x):
        return self.net(x)
