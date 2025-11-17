import timm

def create_swin():
    model = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True, num_classes=0)
    return model
