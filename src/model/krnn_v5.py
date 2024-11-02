import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Immutable model configuration."""
    feature_dim: int
    hidden_dim: int = 128
    num_classes: int = 3
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    learning_rate: float = 0.001
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
        """
        Forward pass with attention mechanism.

        Args:
            x: Input tensor of shape (batch_size, seq_len, feature_dim)

        Returns:
            tuple: (predictions, attention_weights)
        """
        # Project features
        x = self.feature_proj(x)
        x = F.relu(x)

        # RNN processing
        rnn_out, _ = self.rnn(x)

        # Self-attention
        attention = F.softmax(
            torch.bmm(rnn_out, rnn_out.transpose(1, 2)) /
            torch.sqrt(torch.tensor(self.rnn_hidden, dtype=torch.float32)),
            dim=2
        )

        # Apply attention
        context = torch.bmm(attention, rnn_out)

        # Classification
        pooled = context[:, -1]  # Use last timestep
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
            lr=model.config.learning_rate,
            weight_decay=0.01
        )
        self.scheduler = torch.optim.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_step(self,
                   features: torch.Tensor,
                   targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        logits, _ = self.model(features)
        loss = self.criterion(logits, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()
        return loss.item(), logits

    @torch.no_grad()
    def validate(self, features: torch.Tensor, targets: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """Validation step."""
        self.model.eval()
        logits, _ = self.model(features)
        loss = self.criterion(logits, targets)
        return loss.item(), logits

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