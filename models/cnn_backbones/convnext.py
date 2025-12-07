import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
)

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone that returns ONLY local feature maps.
    Supports loading fine-tuned weights.
    """

    def __init__(self, version='tiny', pretrained=True, fine_tune=True, checkpoint_path=None):
        super().__init__()

        version = version.lower()

        # ------------------------------
        # Load model with ImageNet weights
        # ------------------------------
        if version == 'tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_tiny(weights=weights)
            self.out_channels = 768
        elif version == 'small':
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_small(weights=weights)
            self.out_channels = 768
        elif version == 'base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_base(weights=weights)
            self.out_channels = 1024
        else:
            raise ValueError(f"Unsupported ConvNeXt version: {version}")

        # Remove classifier (we want local features)
        self.model.classifier = nn.Identity()

        # ------------------------------
        # Load finetuned checkpoint
        # ------------------------------
        if checkpoint_path is not None:
            print(f"Loading finetuned weights from: {checkpoint_path}")

            state = torch.load(checkpoint_path, map_location="cpu")

            # Some checkpoints store weights under a key "model_state_dict"
            if "state_dict" in state:
                state = state["state_dict"]
            if "model" in state:
                state = state["model"]

            # Remove classifier keys if they exist
            filtered_state = {
                k: v for k, v in state.items()
                if "classifier" not in k
            }

            missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        # ------------------------------
        # Freeze if needed
        # ------------------------------
        if not fine_tune:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model.features(x)

    def get_output_dim(self):
        return self.out_channels



if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image

    # ---------------------------
    # 1. Device
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # ---------------------------
    # 2. Load an image
    # ---------------------------
    img_path = "breast_images/datasets/paultimothymooney/breast-histopathology-images/versions/1/8867/1/8867_idx5_x451_y901_class1.png"   # <-- change to your image path
    img = Image.open(img_path).convert("RGB")

    print("Image size: ",img.size)
    # ---------------------------
    # 3. ConvNeXt preprocessing
    # ---------------------------
    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
    ])

    x = preprocess(img).unsqueeze(0).to(device)   # shape: (1,3,224,224)

    # ---------------------------
    # 4. Initialize backbone
    # ---------------------------
    backbone = ConvNeXtBackbone(
        version='tiny',     # tiny/small/base
        pretrained=True,
        fine_tune=False
    ).to(device)

    # ---------------------------
    # 5. Forward pass
    # ---------------------------
    features = backbone(x)

    print("Local feature map shape:", features.shape)
    print("Output channels:", backbone.get_output_dim())
