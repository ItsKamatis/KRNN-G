# src/utils/trainer_v5.py
import torch
import torch.nn as nn
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class Trainer:
    """
    Handles the training loop for the KRNN Regressor.
    """

    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        """
        Args:
            model: The PyTorch model to train.
            config: Configuration dictionary (containing 'training' params).
        """
        self.model = model
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        # Move model to device
        self.model.to(self.device)

        # Optimization
        lr = config.get('training', {}).get('learning_rate', 0.001)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        # --- CRITICAL: MSELoss for Regression ---
        self.criterion = nn.MSELoss()

    def train_epoch(self, loader: torch.utils.data.DataLoader) -> float:
        """Runs one epoch of training."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, (features, targets) in enumerate(loader):
            features = features.to(self.device)

            # --- THE CRITICAL FIX ---
            # Squeeze [Batch, 1] -> [Batch] to match model output
            targets = targets.to(self.device).squeeze(-1)
            # ------------------------

            # Reset gradients
            self.optimizer.zero_grad()

            # Forward pass
            preds = self.model(features)

            # Calculate loss (Regression: MSE)
            loss = self.criterion(preds, targets)

            # Backward pass
            loss.backward()

            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader: torch.utils.data.DataLoader) -> float:
        """Runs validation."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for features, targets in loader:
                features = features.to(self.device)

                # --- APPLY HERE TOO ---
                targets = targets.to(self.device).squeeze(-1)
                # ----------------------

                preds = self.model(features)
                loss = self.criterion(preds, targets)
                total_loss += loss.item()

        return total_loss / len(loader)

    def train(self,
              train_loader: torch.utils.data.DataLoader,
              val_loader: torch.utils.data.DataLoader,
              experiment: Any = None) -> Dict[str, float]:
        """
        Main training loop that 'main_pipeline.py' calls.
        """
        epochs = self.config.get('training', {}).get('epochs', 10)
        logger.info(f"Starting training for {epochs} epochs on {self.device}...")

        best_val_loss = float('inf')
        train_loss = 0.0

        for epoch in range(epochs):
            # 1. Train
            train_loss = self.train_epoch(train_loader)

            # 2. Validate
            val_loss = self.validate(val_loader)

            # 3. Log
            logger.info(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

            if experiment:
                experiment.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss
                }, step=epoch)

            # 4. Save Best Model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Optionally save checkpoint here if needed

        return {
            'loss': best_val_loss,
            'final_train_loss': train_loss
        }