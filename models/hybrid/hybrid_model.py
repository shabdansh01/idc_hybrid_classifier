import torch.nn as nn
from models.hybrid.fusion_block import FusionBlock
from models.hybrid.classification_head import ClassificationHead

class HybridNet(nn.Module):
    def __init__(self, cnn, vit, fusion_dim=512, hidden_dim=512):
        super().__init__()
        self.cnn = cnn
        self.vit = vit
        
        d_cnn = cnn.num_features
        d_vit = vit.num_features

        self.fusion = FusionBlock(d_cnn, d_vit, fusion_dim)
        self.classifier = ClassificationHead(2 * fusion_dim, hidden_dim)

    def forward(self, x):
        f_cnn = self.cnn(x)
        f_vit = self.vit(x)
        fused = self.fusion(f_cnn, f_vit)
        return self.classifier(fused)
