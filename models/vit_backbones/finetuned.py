import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel


class ViTBackbone(nn.Module):
    """
    ViT backbone that returns a GLOBAL feature vector.

    Default model: Falah/vit-base-breast-cancer
    (ViT-B/16 fine-tuned for breast cancer histopathology classification)

    Assumes input x is a float tensor of shape (B, 3, H, W) already:
      - resized to the expected size (typically 224x224)
      - normalized with ImageNet mean/std (or whatever the HF model expects)
    """

    def __init__(
        self,
        model_name: str = "Falah/vit-base-breast-cancer",
        pretrained: bool = True,
        fine_tune: bool = True,
        checkpoint_path: str = None,
        use_cls_token: bool = True,
    ):
        super().__init__()

        self.use_cls_token = use_cls_token

        # ------------------------------
        # Load model
        # ------------------------------
        if pretrained:
            # This will pull the ViT encoder weights (no classifier head)
            self.model = AutoModel.from_pretrained(model_name)
        else:
            config = AutoConfig.from_pretrained(model_name)
            self.model = AutoModel.from_config(config)

        self.out_dim = self.model.config.hidden_size

        # ------------------------------
        # Optionally load custom checkpoint
        # ------------------------------
        if checkpoint_path is not None:
            print(f"Loading finetuned weights from: {checkpoint_path}")
            state = torch.load(checkpoint_path, map_location="cpu")

            # Handle common nesting conventions
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict) and "model" in state:
                state = state["model"]

            missing, unexpected = self.model.load_state_dict(state, strict=False)
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        # ------------------------------
        # Freeze if needed
        # ------------------------------
        if not fine_tune:
            for p in self.model.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, H, W) tensor, already resized & normalized.
        Returns: (B, hidden_size) global feature vector.
        """
        outputs = self.model(pixel_values=x)
        last_hidden = outputs.last_hidden_state  # (B, seq_len, hidden)

        if self.use_cls_token:
            # CLS token is the first token
            global_feat = last_hidden[:, 0]  # (B, hidden)
        else:
            # Mean pool over patch tokens (excluding CLS)
            global_feat = last_hidden[:, 1:].mean(dim=1)  # (B, hidden)

        return global_feat

    def get_output_dim(self) -> int:
        return self.out_dim


if __name__ == "__main__":
    import torchvision.transforms as T
    from PIL import Image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    img_path = "breast_images/datasets/paultimothymooney/breast-histopathology-images/versions/1/8867/1/8867_idx5_x451_y901_class1.png"  # <-- change this
    img = Image.open(img_path).convert("RGB")


    preprocess = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    x = preprocess(img).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    backbone = ViTBackbone(
        model_name="Falah/vit-base-breast-cancer",
        pretrained=True,
        fine_tune=False,
        use_cls_token=True,  # or False for mean pooling
    ).to(device)


    global_feat = backbone(x)

    print("Global feature shape:", global_feat.shape)  # (1, hidden_size)
    print("Output dim:", backbone.get_output_dim())
