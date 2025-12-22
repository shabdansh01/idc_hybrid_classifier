from models.cnn_backbones.convnext import create_convnext
from models.vit_backbones.swin import create_swin
from models.hybrid.hybrid_model import HybridNet

def create_backbones(config,cnn_name, vit_name):
    # cnn = create_convnext()
    cnn = create_convnext(config['model']['cnn'])
    vit = create_swin(config['model']['vit'])
    return cnn, vit

def build_hybrid(cnn_name, vit_name, fusion_dim=512, hidden_dim=512):
    cnn, vit = create_backbones(cnn_name, vit_name)
    return HybridNet(cnn, vit, fusion_dim, hidden_dim)


