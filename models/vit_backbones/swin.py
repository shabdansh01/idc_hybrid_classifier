import torch
import torch.nn as nn
import timm



class SwinTransformerBackbone(nn.Module):
    """Swin Transformer for global feature extraction."""
    
    def __init__(self,
                 model_name='swin_tiny_patch4_window7_224',
                 pretrained=True,
                 img_size=56,
                 freeze_layers=8,
                 feature_dim=768):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size
        )
        
        self.feature_dim = feature_dim
        
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, num_layers):
        """Freeze first N transformer blocks."""
        # Freeze patch embed
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        
        # Freeze layers
        for i, layer in enumerate(self.backbone.layers):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)

def create_swin(vit_config):
    """Factory function to create Swin backbone from config."""
    return SwinTransformerBackbone(
        model_name=vit_config['backbone'],
        pretrained=vit_config['pretrained'],
        img_size=vit_config['img_size'],
        freeze_layers=vit_config['freeze_layers'],
        feature_dim=vit_config['feature_dim'],
    )


if __name__ == "__main__":
    # ------------------------------
    # Dummy config
    # ------------------------------
    vit_config = {
        "backbone": "swin_tiny_patch4_window7_224",
        "pretrained": True,
        "img_size": 224,
        "freeze_layers": 2,   # freeze first N Swin stages
        "feature_dim": 768,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------
    # Create model
    # ------------------------------
    model = create_swin(vit_config)
    model.to(device)
    model.eval()

    print("\nSwin Transformer model created successfully")
    print(f"Expected feature dim: {vit_config['feature_dim']}")

    # ------------------------------
    # Count frozen vs trainable params
    # ------------------------------
    frozen = sum(not p.requires_grad for p in model.parameters())
    trainable = sum(p.requires_grad for p in model.parameters())

    print(f"Frozen params: {frozen}")
    print(f"Trainable params: {trainable}")

    # ------------------------------
    # Dummy input
    # ------------------------------
    x = torch.randn(1, 3, vit_config["img_size"], vit_config["img_size"]).to(device)

    # ------------------------------
    # Forward pass
    # ------------------------------
    with torch.no_grad():
        features = model(x)

    print("\nForward pass successful")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {features.shape}")

    # ------------------------------
    # Shape check
    # ------------------------------
    # timm Swin with num_classes=0 returns [B, C]
    assert features.ndim == 2, "Expected global feature vector output"
    assert features.shape[1] == vit_config["feature_dim"], \
        "Feature dimension mismatch!"

    print("\n✅ Swin Transformer backbone loaded and ran correctly")