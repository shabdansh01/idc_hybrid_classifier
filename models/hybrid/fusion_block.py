import torch.nn as nn

class FusionBlock(nn.Module):
    def __init__(self, d_cnn, d_vit, d_fuse):
        super().__init__()
        self.cnn_proj = nn.Linear(d_cnn, d_fuse)
        self.vit_proj = nn.Linear(d_vit, d_fuse)
        self.norm = nn.LayerNorm(2 * d_fuse)

    def forward(self, f_cnn, f_vit):
        f1 = self.cnn_proj(f_cnn)
        f2 = self.vit_proj(f_vit)
        fused = self.norm(torch.cat([f1, f2], dim=-1))
        return fused
