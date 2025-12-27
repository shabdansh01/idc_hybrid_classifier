import torch.nn as nn

class ClassificationHead(nn.Module):
    """MLP classification head."""
    
    def __init__(self,
                 input_dim: int,
                 hidden_dims: list = [512, 256],
                 num_classes: int = 1,
                 dropout: float = 0.4,
                 use_layernorm: bool = True):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.head = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.head(x)
