import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights, ConvNeXt_Base_Weights

class ConvNeXtBackbone(nn.Module):
    def __init__(self, version='tiny', pretrained=True, fine_tune=True):
        """
        Args:
            version (str): 'tiny', 'small', or 'base'.
            pretrained (bool): Whether to use ImageNet weights.
            fine_tune (bool): If False, freezes the backbone weights.
        """
        super(ConvNeXtBackbone, self).__init__()
        
        self.version = version.lower()
        
        # Select weights and model based on version
        if self.version == 'tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_tiny(weights=weights)
        elif self.version == 'small':
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_small(weights=weights)
        elif self.version == 'base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_base(weights=weights)
        else:
            raise ValueError(f"Unsupported ConvNeXt version: {version}")

        # Extract the input features of the final linear layer to know embedding size
        # Structure of convnext classifier is: Sequential(LayerNorm2d, Flatten, Linear)
        self.out_features = self.model.classifier[2].in_features
        
        # Replace the classifier with Identity so we get the pooled features
        # Note: ConvNeXt performs global pooling *before* the classifier block in the `features` part,
        # but the classifier block contains the final LayerNorm. 
        # We want the output after the LayerNorm but before the Linear projection.
        
        # We keep the LayerNorm (index 0) and Flatten (index 1), remove Linear (index 2)
        self.model.classifier[2] = nn.Identity()

        # Freeze weights if not fine-tuning
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Returns shape: (Batch_Size, out_features)
        return self.model(x)

    def get_output_dim(self):
        return self.out_features