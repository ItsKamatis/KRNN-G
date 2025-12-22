# src/model/krnn_v5.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple, Dict
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for the KRNN Model."""
    feature_dim: int  # Renamed from input_dim to match train_local_v5.py
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Added defaults because train_local_v5.py doesn't pass these
    bidirectional: bool = True
    num_classes: int = 3  # 1: Up, 2: Stable, 3: Down
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class KRNN(nn.Module):
    """K-rare class nearest neighbor enhanced RNN."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Calculate dimensions
        self.rnn_hidden = config.hidden_dim
        self.rnn_directions = 2 if config.bidirectional else 1

        # Feature projection
        self.feature_proj = nn.Linear(config.feature_dim, config.hidden_dim)

        # GRU layer
        self.rnn = nn.GRU(
            input_size=config.hidden_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )

        # Output layers
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(
            config.hidden_dim * self.rnn_directions,
            config.num_classes
        )

        self._init_weights()
        self.to(config.device)

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with attention mechanism."""
        # Project features
        x = self.feature_proj(x)
        x = F.relu(x)

        # RNN processing
        rnn_out, _ = self.rnn(x)

        # Self-attention
        # Query: rnn_out, Key: rnn_out
        # Score = Q @ K.T / sqrt(d)
        attention = F.softmax(
            torch.bmm(rnn_out, rnn_out.transpose(1, 2)) /
            torch.sqrt(torch.tensor(self.rnn_hidden, dtype=torch.float32, device=x.device)),
            dim=2
        )

        # Apply attention
        context = torch.bmm(attention, rnn_out)

        # Classification using the last timestep of the context
        pooled = context[:, -1]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits, attention

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Generate predictions with no gradient tracking."""
        self.eval()
        with torch.no_grad():
            logits, _ = self(x)
            return F.softmax(logits, dim=1)


class KRNNPredictor:
    """Handles training and prediction."""

    def __init__(self, model: KRNN):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=1e-3,  # Default LR, usually driven by config but hardcoded in prev version
            weight_decay=0.01
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(self.model.config.device)
            # targets from DataLoader might be [Batch, 1], need [Batch] for CrossEntropy
            targets = targets.to(self.model.config.device).view(-1)

            # Since targets are 1, 2, 3, we might need to shift to 0, 1, 2 for CrossEntropy
            # Check if your data is 0-indexed or 1-indexed.
            # Usually CrossEntropy expects 0..N-1.
            # Assuming data is 1..3, we substract 1.
            if targets.min() >= 1:
                targets = targets - 1

            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(features)
            loss = self.criterion(logits, targets.long())

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        return {
            'loss': total_loss / len(train_loader),
            'accuracy': correct / total if total > 0 else 0
        }

    @torch.no_grad()
    def validate(self, val_loader) -> Dict[str, float]:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, targets) in enumerate(val_loader):
            features = features.to(self.model.config.device)
            targets = targets.to(self.model.config.device).view(-1)

            if targets.min() >= 1:
                targets = targets - 1

            # Forward pass
            logits, _ = self.model(features)
            loss = self.criterion(logits, targets.long())

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total if total > 0 else 0
        }

        self.scheduler.step(metrics['loss'])
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'config': self.model.config
        }, path)

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.model.config.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])