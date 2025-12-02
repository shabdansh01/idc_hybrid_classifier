class HybridIDCClassifier(nn.Module):
    """Hybrid CNN-ViT model for IDC classification."""
    
    def __init__(self, config):
        super().__init__()
        
        # CNN Stream (Local Features)
        cnn_config = config['model']['cnn']
        if 'convnext' in cnn_config['backbone']:
            self.cnn_backbone = ConvNeXtBackbone(
                model_name=cnn_config['backbone'],
                pretrained=cnn_config['pretrained'],
                freeze_stages=cnn_config['freeze_stages'],
                feature_dim=cnn_config['feature_dim']
            )
        else:
            self.cnn_backbone = EfficientNetV2Backbone(
                model_name=cnn_config['backbone'],
                pretrained=cnn_config['pretrained'],
                freeze_stages=cnn_config['freeze_stages'],
                feature_dim=cnn_config['feature_dim']
            )
        
        # ViT Stream (Global Features)
        vit_config = config['model']['vit']
        if 'swin' in vit_config['backbone']:
            self.vit_backbone = SwinTransformerBackbone(
                model_name=vit_config['backbone'],
                pretrained=vit_config['pretrained'],
                img_size=vit_config['img_size'],
                freeze_layers=vit_config['freeze_layers'],
                feature_dim=vit_config['feature_dim']
            )
        else:
            self.vit_backbone = DeiTBackbone(
                model_name=vit_config['backbone'],
                pretrained=vit_config['pretrained'],
                img_size=vit_config['img_size'],
                freeze_layers=vit_config['freeze_layers'],
                feature_dim=vit_config['feature_dim']
            )
        
        # Fusion Block
        fusion_config = config['model']['fusion']
        self.fusion = FusionBlock(
            cnn_dim=self.cnn_backbone.feature_dim,
            vit_dim=self.vit_backbone.feature_dim,
            project_dim=fusion_config['project_dim'],
            use_alignment=fusion_config['use_alignment']
        )
        
        # Classification Head
        head_config = config['model']['head']
        self.classifier = ClassificationHead(
            input_dim=self.fusion.fused_dim,
            hidden_dims=head_config['hidden_dims'],
            num_classes=config['data']['num_classes'],
            dropout=head_config['dropout'],
            use_layernorm=head_config['use_layernorm']
        )
    
    def forward(self, x):
        """Forward pass through hybrid model."""
        # Extract features from both streams
        cnn_features = self.cnn_backbone(x)
        vit_features = self.vit_backbone(x)
        
        # Fuse features
        fused_features = self.fusion(cnn_features, vit_features)
        
        # Classify
        logits = self.classifier(fused_features)
        
        return logits
    
    def unfreeze_all(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

