import torch
import torch.nn as nn
import timm

class ConvNeXtBackbone(nn.Module):
    """ConvNeXt backbone for local feature extraction."""
    
    def __init__(self, 
                 model_name='convnext_tiny',
                 pretrained=True,
                 freeze_stages=2,
                 feature_dim=768):
        super().__init__()
        
        # Load pretrained model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        self.feature_dim = feature_dim
        
        # Freeze early stages
        if freeze_stages > 0:
            self._freeze_stages(freeze_stages)
    
    def _freeze_stages(self, num_stages):
        """Freeze first N stages."""
        # ConvNeXt has 4 stages
        for i in range(num_stages):
            stage = getattr(self.backbone.stages, str(i), None)
            if stage is not None:
                for param in stage.parameters():
                    param.requires_grad = False
    
    def forward(self, x):
        """Forward pass."""
        features = self.backbone(x)
        return features

