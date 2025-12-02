class EfficientNetV2Backbone(nn.Module):
    """EfficientNetV2 backbone for local feature extraction."""
    
    def __init__(self,
                 model_name='efficientnetv2_s',
                 pretrained=True,
                 freeze_stages=2,
                 feature_dim=512):
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        self.feature_dim = feature_dim
        
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)
    
    def _freeze_stages(self, num_stages):
        """Freeze first N blocks."""
        blocks = list(self.backbone.blocks)
        num_freeze = min(num_stages, len(blocks))
        
        for i in range(num_freeze):
            for param in blocks[i].parameters():
                param.requires_grad = False
    
    def forward(self, x):
        return self.backbone(x)
