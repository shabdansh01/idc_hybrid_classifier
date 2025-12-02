class FusionBlock(nn.Module):
    """Feature fusion with alignment."""
    
    def __init__(self,
                 cnn_dim: int,
                 vit_dim: int,
                 project_dim: int = 512,
                 use_alignment: bool = True):
        super().__init__()
        
        self.use_alignment = use_alignment
        
        if use_alignment:
            # Project both streams to common dimension
            self.cnn_norm = nn.LayerNorm(cnn_dim)
            self.vit_norm = nn.LayerNorm(vit_dim)
            
            self.cnn_proj = nn.Linear(cnn_dim, project_dim)
            self.vit_proj = nn.Linear(vit_dim, project_dim)
            
            self.fused_dim = project_dim * 2
        else:
            # Simple concatenation
            self.fused_dim = cnn_dim + vit_dim
    
    def forward(self, cnn_features, vit_features):
        """Fuse CNN and ViT features."""
        if self.use_alignment:
            # Normalize
            cnn_features = self.cnn_norm(cnn_features)
            vit_features = self.vit_norm(vit_features)
            
            # Project
            cnn_proj = self.cnn_proj(cnn_features)
            vit_proj = self.vit_proj(vit_features)
            
            # Concatenate
            fused = torch.cat([cnn_proj, vit_proj], dim=1)
        else:
            fused = torch.cat([cnn_features, vit_features], dim=1)
        
        return fused

