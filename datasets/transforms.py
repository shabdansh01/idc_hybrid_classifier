import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class IDCTransforms:
    """Augmentation transforms for IDC patches."""
    
    @staticmethod
    def get_train_transforms(config):
        """Get training augmentations."""
        transforms = []
        
        # Geometric
        transforms.append(A.Rotate(limit=config['rotation'], p=0.7))
        if config.get('horizontal_flip', True):
            transforms.append(A.HorizontalFlip(p=0.5))
        if config.get('vertical_flip', True):
            transforms.append(A.VerticalFlip(p=0.5))
        
        # Color
        if config.get('color_jitter', 0) > 0:
            transforms.append(A.ColorJitter(
                brightness=config['color_jitter'],
                contrast=config['color_jitter'],
                saturation=config['color_jitter'],
                hue=config['color_jitter'] * 0.5,
                p=0.5
            ))
        
        # Noise
        if config.get('gaussian_noise', 0) > 0:
            transforms.append(A.GaussNoise(
                var_limit=(0, config['gaussian_noise'] * 255),
                p=0.3
            ))
        
        # Elastic
        if config.get('elastic_transform', False):
            transforms.append(A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=5,
                p=0.3
            ))
        
        # Cutout
        if config.get('cutout', False):
            transforms.append(A.CoarseDropout(
                max_holes=8,
                max_height=8,
                max_width=8,
                p=0.3
            ))
        
        # Normalize and convert
        transforms.extend([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        return A.Compose(transforms)
    
    @staticmethod
    def get_val_transforms():
        """Get validation transforms."""
        return A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_tta_transforms(num_augments=8):
        """Get test-time augmentation transforms."""
        tta_transforms = []
        
        # Original
        tta_transforms.append(A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]))
        
        # Flips
        for hflip in [False, True]:
            for vflip in [False, True]:
                if not hflip and not vflip:
                    continue  # Skip original
                if len(tta_transforms) >= num_augments:
                    break
                    
                ops = []
                if hflip:
                    ops.append(A.HorizontalFlip(p=1.0))
                if vflip:
                    ops.append(A.VerticalFlip(p=1.0))
                ops.extend([
                    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ToTensorV2()
                ])
                tta_transforms.append(A.Compose(ops))
        
        # Rotations
        for angle in [90, 180, 270]:
            if len(tta_transforms) >= num_augments:
                break
            tta_transforms.append(A.Compose([
                A.Rotate(limit=(angle, angle), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]))
        
        return tta_transforms[:num_augments]

