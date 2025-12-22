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
    return SwinBackbone(
        model_name=vit_config['backbone'],
        pretrained=vit_config['pretrained'],
        img_size = config['img_size'],
        freeze_layers=vit_config['freeze_layers'],
        feature_dim=vit_config['feature_dim'],
    )