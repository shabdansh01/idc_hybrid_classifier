class DeiTBackbone(nn.Module):
    """DeiT (Data-efficient Image Transformer) backbone."""
    
    def __init__(self,
                 model_name='deit_small_patch16_224',
                 pretrained=True,
                 img_size=56,
                 freeze_layers=8,
                 feature_dim=384):
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
        """Freeze first N blocks."""
        for param in self.backbone.patch_embed.parameters():
            param.requires_grad = False
        
        for i in range(min(num_layers, len(self.backbone.blocks))):
            for param in self.backbone.blocks[i].parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)
