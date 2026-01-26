import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

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

        self.scaler = (
            torch.amp.GradScaler('cuda')
            if config['training']['mixed_precision']
            else None
        )

        # ✅ TensorBoard: create run-specific directory
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_log_dir = Path(config['paths']['log_dir'])
        self.log_dir = base_log_dir / run_name
        self.log_dir.mkdir(parents=True, exist_ok=True)

        print(f"[TensorBoard] Logging to: {self.log_dir}")
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []

        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
        for images, targets in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

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

            running_loss += loss.item()
            preds = torch.sigmoid(outputs).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())

            pbar.set_postfix({'loss': loss.item()})

        avg_loss = running_loss / len(dataloader)
        metrics = MetricsCalculator.compute_metrics(
            np.array(all_preds).squeeze(),
            np.array(all_targets).squeeze()
        )
        metrics['loss'] = avg_loss

        # 🔹 TensorBoard (TRAIN)
        self.writer.add_scalar("Train/Loss", metrics['loss'], epoch)
        self.writer.add_scalar("Train/Accuracy", metrics['accuracy'], epoch)
        self.writer.add_scalar("Train/F1", metrics['f1'], epoch)
        self.writer.add_scalar("Train/AUC", metrics['auc'], epoch)

        if self.optimizer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("Train/LR", lr, epoch)

        self.writer.flush()
        return metrics

    @torch.no_grad()
    def validate(self, dataloader, epoch=None):
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

        # 🔹 TensorBoard (VAL)
        if epoch is not None:
            self.writer.add_scalar("Val/Loss", metrics['loss'], epoch)
            self.writer.add_scalar("Val/Accuracy", metrics['accuracy'], epoch)
            self.writer.add_scalar("Val/F1", metrics['f1'], epoch)
            self.writer.add_scalar("Val/AUC", metrics['auc'], epoch)
            self.writer.flush()

        return metrics