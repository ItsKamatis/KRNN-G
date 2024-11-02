import torch
import mlflow
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm

from src.model.krnn_v5 import KRNN, ModelConfig
from src.utils.experiment_v5 import ExperimentManager, MemoryOptimizer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    """Immutable trainer configuration."""
    epochs: int = 50
    early_stop_patience: int = 10
    checkpoint_dir: Path = Path("checkpoints")
    gradient_clip: float = 1.0
    save_best: bool = True


class Trainer:
    """Handles model training and evaluation."""

    def __init__(self, model: KRNN, config: TrainerConfig):
        self.model = model
        self.config = config
        self.device = model.config.device

        # Create checkpoint directory
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tracking
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Setup optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=model.config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, train_loader, val_loader, experiment: ExperimentManager) -> None:
        """Run training loop with experiment tracking."""
        memory_opt = MemoryOptimizer()

        for epoch in range(self.config.epochs):
            with memory_opt:
                # Training phase
                train_metrics = self._train_epoch(train_loader, epoch)
                mlflow.log_metrics(
                    {f"train_{k}": v for k, v in train_metrics.items()},
                    step=epoch
                )

                # Validation phase
                val_metrics = self._validate(val_loader)
                mlflow.log_metrics(
                    {f"val_{k}": v for k, v in val_metrics.items()},
                    step=epoch
                )

                # Learning rate scheduling
                self.scheduler.step(val_metrics['loss'])

                # Save best model
                if self.config.save_best and val_metrics['loss'] < self.best_val_loss:
                    self.save_checkpoint('best.pt')
                    self.best_val_loss = val_metrics['loss']
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Early stopping check
                if self.patience_counter >= self.config.early_stop_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

                # Log epoch summary
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss = {train_metrics['loss']:.4f}, "
                    f"Val Loss = {val_metrics['loss']:.4f}, "
                    f"Val Accuracy = {val_metrics['accuracy']:.4f}"
                )

    def _train_epoch(self, train_loader, epoch: int) -> Dict[str, float]:
        """Train single epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        with tqdm(train_loader, desc=f"Epoch {epoch}") as pbar:
            for features, targets in pbar:
                # Move data to device
                features = features.to(self.device)
                targets = targets.to(self.device).squeeze()

                # Forward pass
                self.optimizer.zero_grad()
                logits, _ = self.model(features)
                loss = self.criterion(logits, targets)

                # Backward pass
                loss.backward()

                # Gradient clipping
                if self.config.gradient_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.gradient_clip
                    )

                self.optimizer.step()

                # Track metrics
                total_loss += loss.item()
                _, predicted = torch.max(logits, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'acc': f"{100. * correct / total:.2f}%"
                })

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100. * correct / total
        }

    @torch.no_grad()
    def _validate(self, val_loader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for features, targets in val_loader:
            # Move data to device
            features = features.to(self.device)
            targets = targets.to(self.device).squeeze()

            # Forward pass
            logits, _ = self.model(features)
            loss = self.criterion(logits, targets)

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100. * correct / total
        }

    def save_checkpoint(self, filename: str) -> None:
        """Save model checkpoint."""
        path = self.config.checkpoint_dir / filename
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.model.config,
            'best_val_loss': self.best_val_loss
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, filename: str) -> None:
        """Load model checkpoint."""
        path = self.config.checkpoint_dir / filename
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.best_val_loss = checkpoint['best_val_loss']
        logger.info(f"Loaded checkpoint from {path}")