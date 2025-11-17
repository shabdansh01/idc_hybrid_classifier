import timm
import torch.nn as nn

def create_convnext():
    model = timm.create_model("convnext_tiny", pretrained=True)
    model.reset_classifier(num_classes=0)
    return model
