import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    ConvNeXt_Small_Weights,
    ConvNeXt_Base_Weights,
)
from huggingface_hub import snapshot_download
import safetensors.torch
import os

class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt backbone that returns ONLY local feature maps.
    Supports loading fine-tuned weights from local checkpoint or Hugging Face repo.
    """

    def __init__(self, model_name='convnext_tiny', pretrained=True, freeze_stages=0, feature_dim=768, hf_repo=None):
        super().__init__()

        model_name = model_name.lower()

        # ------------------------------
        # Load model with ImageNet weights
        # ------------------------------
        if model_name == 'convnext_tiny':
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_tiny(weights=weights)
            self.out_channels = 768
        elif model_name == 'convnext_small':
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_small(weights=weights)
            self.out_channels = 768
        elif model_name == 'convnext_base':
            weights = ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.model = models.convnext_base(weights=weights)
            self.out_channels = 1024
        else:
            raise ValueError(f"Unsupported ConvNeXt model_name: {model_name}")

        self.feature_dim = feature_dim  # For compatibility
        assert self.out_channels == feature_dim, f"feature_dim {feature_dim} mismatches out_channels {self.out_channels}"

        # Remove classifier (we want local features)
        self.model.classifier = nn.Identity()

        # ------------------------------
        # Load finetuned checkpoint from HF
        # ------------------------------
        state = None
        if hf_repo is not None:
            print(f"Downloading fine-tuned weights from Hugging Face repo: {hf_repo}")
            local_dir = snapshot_download(repo_id=hf_repo, local_dir_use_symlinks=False)
            safetensors_path = os.path.join(local_dir, "model.safetensors")
            if os.path.exists(safetensors_path):
                state = safetensors.torch.load_file(safetensors_path)
                print(f"Loaded state dict from {safetensors_path}")
            else:
                # Fallback to pytorch_model.bin
                bin_path = os.path.join(local_dir, "pytorch_model.bin")
                if os.path.exists(bin_path):
                    state = torch.load(bin_path, map_location="cpu")
                    print(f"Loaded state dict from {bin_path}")
                else:
                    raise FileNotFoundError(f"No model file found in {local_dir}")

        if state is not None:
            # Handle nested keys
            if "state_dict" in state:
                state = state["state_dict"]
            if "model" in state:
                state = state["model"]

            # Filter out classifier/head keys
            filtered_state = {
                k: v for k, v in state.items()
                if not any(exclude in k for exclude in ["classifier", "head", "fc", "logits"])
            }

            missing, unexpected = self.model.load_state_dict(filtered_state, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
            if missing:
                print("Note: Missing keys likely from classifier/head; expected for backbone.")

        # ------------------------------
        # Freeze stages if specified
        # ------------------------------
        if freeze_stages > 0:
            # ConvNeXt features has stages: 0-7 (downsample layers)
            for i in range(freeze_stages):
                for param in self.model.features[i].parameters():
                    param.requires_grad = False
            print(f"Froze first {freeze_stages} stages.")

    def forward(self, x):
        return self.model.features(x)

    def get_output_dim(self):
        return self.out_channels

def create_convnext(cnn_config):
    """Factory function to create ConvNeXt backbone from config."""
    hf_repo = "FatimaK6/breast-cancer-convnext-tiny"  # Hardcoded for now; make configurable
    return ConvNeXtBackbone(
        model_name=cnn_config['backbone'],
        pretrained=cnn_config['pretrained'],
        freeze_stages=cnn_config['freeze_stages'],
        feature_dim=cnn_config['feature_dim'],
        hf_repo=hf_repo
    )
if __name__ == "__main__":
    import torch

    # ------------------------------
    # Dummy config (matches your factory)
    # ------------------------------
    cnn_config = {
        "backbone": "convnext_tiny",
        "pretrained": True,
        "freeze_stages": 2,
        "feature_dim": 768,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # ------------------------------
    # Create model
    # ------------------------------
    model = create_convnext(cnn_config)
    model.to(device)
    model.eval()

    print("\nModel created successfully")
    print(f"Output feature dim: {model.get_output_dim()}")

    # ------------------------------
    # Check frozen parameters
    # ------------------------------
    frozen = sum(not p.requires_grad for p in model.parameters())
    trainable = sum(p.requires_grad for p in model.parameters())

    print(f"Frozen params: {frozen}")
    print(f"Trainable params: {trainable}")

    # ------------------------------
    # Dummy input
    # ------------------------------
    x = torch.randn(1, 3, 224, 224).to(device)

    # ------------------------------
    # Forward pass
    # ------------------------------
    with torch.no_grad():
        features = model(x)

    print("\nForward pass successful")
    print(f"Input shape: {x.shape}")
    print(f"Output feature map shape: {features.shape}")

    # ------------------------------
    # Expected output for ConvNeXt-Tiny
    # ------------------------------
    # Typically: [B, 768, 7, 7] for 224x224 input
    assert features.shape[1] == cnn_config["feature_dim"], \
        "Feature dimension mismatch!"

    print("\n✅ ConvNeXt backbone loaded weights and ran correctly")