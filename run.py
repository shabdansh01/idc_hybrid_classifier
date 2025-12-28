# Standard library
import argparse
import yaml
from pathlib import Path

# Core ML
import torch
import torch.nn as nn
import numpy as np

# Project-specific imports (you must have these modules)
from datasets.idc_dataset import IDCDataset
from datasets.transforms import IDCTransforms
from models.hybrid.hybrid_model import HybridIDCClassifier
from training.engine import TrainingEngine
from training.losses import FocalLoss
from utils.stain_normalization import MacenkoStainNormalizer


DEFAULT_CONFIG = "configs/default.yaml"

def parse_args():
    parser = argparse.ArgumentParser(description='IDC Hybrid Classifier')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                       help='Path to config file')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'eval', 'predict'],
                       help='Mode: train, eval, or predict')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint for eval/predict')
    parser.add_argument('--data_dir', type=str, default=None,
                       help='Override data directory')
    return parser.parse_args()


def load_config(config_path):
    """Load YAML config."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        # Use default config
        return yaml.safe_load(DEFAULT_CONFIG)


def setup_directories(config):
    """Create necessary directories."""
    for path_key in ['output_dir', 'checkpoint_dir', 'log_dir']:
        Path(config['paths'][path_key]).mkdir(parents=True, exist_ok=True)


def train(config):
    """Main training function."""
    print("=" * 80)
    print("IDC HYBRID CNN-VIT CLASSIFIER - TRAINING")
    print("=" * 80)
    
    # Set seed
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create stain normalizer
    print("\n[1/6] Preparing stain normalization...")
    # stain_normalizer = MacenkoStainNormalizer()
    stain_normalizer = None
    # Fit on a reference image (you should provide this)
    # stain_normalizer.fit(reference_image)
    
    # Create datasets
    print("[2/6] Loading datasets...")
    train_transforms = IDCTransforms.get_train_transforms(config['data']['augmentation']['train'])
    val_transforms = IDCTransforms.get_val_transforms()
    print(config['paths']['data_dir'])
    train_dataset = IDCDataset(
        data_dir=Path(config['paths']['data_dir']) / 'train',
        split='train',
        transform=train_transforms,
        stain_normalizer=stain_normalizer,
        target_size=config['data']['target_size']
    )
    
    val_dataset = IDCDataset(
        data_dir=Path(config['paths']['data_dir']) / 'val',
        split='val',
        transform=val_transforms,
        stain_normalizer=stain_normalizer,
        target_size=config['data']['target_size']
    )
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=config['training']['pin_memory']
    )
    
    # Create model
    print("[3/6] Building hybrid model...")
    model = HybridIDCClassifier(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    print("[4/6] Setting up optimizer...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
        betas=config['training']['optimizer']['betas']
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['training']['num_epochs'],
        eta_min=config['training']['scheduler']['min_lr']
    )
    
    # Loss function
    if config['training']['loss']['type'] == 'focal':
        criterion = FocalLoss()
    else:
        pos_weight = torch.tensor([config['training']['loss']['pos_weight']]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Training engine
    engine = TrainingEngine(model, optimizer, criterion, device, config)
    
    # Training loop
    print(f"[5/6] Starting training for {config['training']['num_epochs']} epochs...")
    best_auc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config['training']['num_epochs'] + 1):
        # Train
        train_metrics = engine.train_epoch(train_loader, epoch)
        
        # Validate
        val_metrics = engine.validate(val_loader)
        
        # Scheduler step
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch}/{config['training']['num_epochs']}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1']:.4f}, "
              f"AUC: {train_metrics['auc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1']:.4f}, "
              f"AUC: {val_metrics['auc']:.4f}")
        
        # Save checkpoint
        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            
            checkpoint_path = Path(config['paths']['checkpoint_dir']) / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_auc': best_auc,
                'config': config
            }, checkpoint_path)
            print(f"✓ Saved best model (AUC: {best_auc:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= config['training']['early_stopping']['patience']:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    print("\n[6/6] Training complete!")
    print(f"Best validation AUC: {best_auc:.4f}")
    print("=" * 80)


def main():
    """Main entry point."""
    args = parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override data_dir if provided
    if args.data_dir:
        config['paths']['data_dir'] = args.data_dir
    
    # Setup directories
    setup_directories(config)
    
    # Run mode
    if args.mode == 'train':
        torch.cuda.empty_cache()
        train(config)
    elif args.mode == 'eval':
        print("Evaluation mode - implement evaluate() function")
    elif args.mode == 'predict':
        print("Prediction mode - implement predict() function")


if __name__ == '__main__':
    main()