# src/utils/trainer_v5.py
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from typing import Dict

logger = logging.getLogger(__name__)


class Trainer:
    """
     specialized Trainer for Heteroscedastic Regression.
     Minimizes Gaussian Negative Log Likelihood (GNLL).
    """

    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.device = model.config.device

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )

        # Loss Function: Gaussian NLL
        # L = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)
        self.criterion = nn.GaussianNLLLoss()

    def train(self, train_loader, val_loader) -> Dict[str, float]:
        """Run full training loop."""
        epochs = self.config['training']['epochs']
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # --- Training Step ---
            self.model.train()
            total_train_loss = 0
            for X, y in train_loader:
                X, y = X.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()

                # Model returns Tuple (mu, sigma)
                mu, sigma = self.model(X)

                # Reshape y to match mu/sigma if needed (Batch,)
                if y.dim() > 1: y = y.squeeze()
                if mu.dim() > 1: mu = mu.squeeze()
                if sigma.dim() > 1: sigma = sigma.squeeze()

                # Calculate GNLL Loss (requires variance = sigma^2)
                loss = self.criterion(mu, y, sigma.pow(2))

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # --- Validation Step ---
            val_metrics = self.validate(val_loader)
            current_val_loss = val_metrics['loss']

            # --- Logging ---
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | Train GNLL: {avg_train_loss:.4f} | Val GNLL: {current_val_loss:.4f}")

            # --- Early Stopping & Checkpointing ---
            self.scheduler.step(current_val_loss)

            if current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                # Save best model
                import os
                os.makedirs(self.config['paths']['checkpoints'], exist_ok=True)
                torch.save(self.model.state_dict(), f"{self.config['paths']['checkpoints']}/best_model.pt")
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stop_patience']:
                    logger.info("Early stopping triggered.")
                    break

        return {'loss': best_val_loss}

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        for X, y in val_loader:
            X, y = X.to(self.device), y.to(self.device)
            mu, sigma = self.model(X)

            if y.dim() > 1: y = y.squeeze()
            if mu.dim() > 1: mu = mu.squeeze()
            if sigma.dim() > 1: sigma = sigma.squeeze()

            loss = self.criterion(mu, y, sigma.pow(2))
            total_loss += loss.item()

        return {'loss': total_loss / len(val_loader)}