import torch
import numpy as np
from tqdm import tqdm

# Metrics
from training.metrics import MetricsCalculator


class TrainingEngine:
    """Training and validation engine."""
    
    def __init__(self, model, optimizer, criterion, device, config):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        # self.scaler = torch.cuda.amp.GradScaler() if config['training']['mixed_precision'] else None
        self.scaler = torch.amp.GradScaler('cuda') if config['training']['mixed_precision'] else None
    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)
            
            # Forward pass
            if self.scaler is not None:
                # with torch.cuda.amp.autocast():
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    if torch.isnan(outputs).any() or torch.isinf(outputs).any():  # Check logits pre-loss
                        print(f"[DEBUG] NaN/Inf in outputs! Max/Min: {outputs.max().item():.2f}/{outputs.min().item():.2f}")
                        print(f"[DEBUG] Images mean/std: {images.mean().item():.4f}/{images.std().item():.4f}")
                        torch.save(images, 'debug_batch0.pt')  # Load later: import torch; img = torch.load('debug_batch0.pt')[0].cpu().permute(1,2,0).numpy()
                        return {'loss': float('nan'), 'debug_halt': True}
                    loss = self.criterion(outputs, targets)
                    if torch.isnan(loss).any():
                        print(f"[DEBUG] NaN detected! Logits max/min: {outputs.max().item():.2f}/{outputs.min().item():.2f}")  # Fixed: outputs, not logits
                        print(f"[DEBUG] Targets sample: {targets[:5]}")  # [B,1] tensor, values 0./1.
                        print(f"[DEBUG] Inputs mean/std: {images.mean().item():.4f}/{images.std().item():.4f}")  # Fixed: images, not inputs
                        # Optional: Save for viz (add import matplotlib.pyplot as plt if needed)
                        torch.save(images, 'debug_inputs.pt')
                        # Early exit for debug (remove later)
                        return {'loss': float('nan'), 'debug_halt': True}  # Halt epoch on first NaN
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training'].get('gradient_clip'):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                
                self.optimizer.zero_grad()
                loss.backward()
                
                if self.config['training'].get('gradient_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['training']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Compute metrics
        avg_loss = running_loss / len(dataloader)
        metrics = MetricsCalculator.compute_metrics(
            np.array(all_preds).squeeze(),
            np.array(all_targets).squeeze()
        )
        metrics['loss'] = avg_loss
        
        return metrics
    
    @torch.no_grad()
    def validate(self, dataloader):
        """Validate model."""
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        for images, targets in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)
            
            running_loss += loss.item()
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
        
        avg_loss = running_loss / len(dataloader)
        metrics = MetricsCalculator.compute_metrics(
            np.array(all_preds).squeeze(),
            np.array(all_targets).squeeze()
        )
        metrics['loss'] = avg_loss
        
        return metrics