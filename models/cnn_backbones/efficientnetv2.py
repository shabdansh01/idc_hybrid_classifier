import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights, EfficientNet_V2_L_Weights

class EfficientNetV2Backbone(nn.Module):
    def __init__(self, version='s', pretrained=True, fine_tune=True):
        """
        Args:
            version (str): 's' (small), 'm' (medium), or 'l' (large).
            pretrained (bool): Whether to use ImageNet weights.
            fine_tune (bool): If False, freezes the backbone weights.
        """
        super(EfficientNetV2Backbone, self).__init__()
        
        self.version = version.lower()
        
        # Select weights and model based on version
        if self.version == 's':
            weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_v2_s(weights=weights)
        elif self.version == 'm':
            weights = EfficientNet_V2_M_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_v2_m(weights=weights)
        elif self.version == 'l':
            weights = EfficientNet_V2_L_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.efficientnet_v2_l(weights=weights)
        else:
            raise ValueError(f"Unsupported EfficientNetV2 version: {version}")

        # Structure of EfficientNetV2 classifier is: Sequential(Dropout, Linear)
        # We need the in_features of the Linear layer to know our embedding size
        self.out_features = self.model.classifier[1].in_features
        
        # Remove the classification head (Dropout + Linear)
        # We replace the whole classifier block with Identity.
        # EfficientNet V2 puts the AvgPool before the classifier, so this leaves us with the feature vector.
        self.model.classifier = nn.Identity()

        # Freeze weights if not fine-tuning
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        # Returns shape: (Batch_Size, out_features)
        return self.model(x)

    def get_output_dim(self):
        return self.out_features