# src/model/krnn_v5.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from typing import Tuple
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the K-Parallel RNN (KRNN) Regressor."""
    input_dim: int              # Number of input features
    hidden_dim: int = 64        # Hidden dimension for each RNN
    num_layers: int = 2         # Number of RNN layers
    dropout: float = 0.2        # Dropout rate
    k_dups: int = 3             # 'K' in KRNN: Number of parallel RNNs
    output_dim: int = 1         # Regression output (Predicting Log Return)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class KParallelEncoder(nn.Module):
    """
    The Core 'KRNN' Logic.
    Based on pytorch_krnn.py: Creates K parallel RNNs and averages their outputs.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.k_dups = config.k_dups
        self.hidden_dim = config.hidden_dim
        self.device = config.device

        # Create K independent RNN instances
        # We use ModuleList to ensure PyTorch registers them as trainable parameters
        self.rnn_modules = nn.ModuleList()
        for _ in range(self.k_dups):
            self.rnn_modules.append(
                nn.GRU(
                    input_size=config.input_dim,
                    hidden_size=config.hidden_dim,
                    num_layers=config.num_layers,
                    dropout=config.dropout if config.num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=False  # Standard KRNN is usually unidirectional, can be toggled
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [Batch, Seq_Len, Features]
        Returns:
            Averaged Hidden Representation [Batch, Hidden_Dim]
        """
        # x shape: [batch, seq, feature]

        parallel_outputs = []

        # 1. Run input through each of the K RNNs independently
        for rnn in self.rnn_modules:
            # rnn_out: [batch, seq, hidden]
            # h_n: [layers, batch, hidden]
            out, _ = rnn(x)

            # We take the output of the last time step
            last_step_out = out[:, -1, :]  # [batch, hidden]
            parallel_outputs.append(last_step_out)

        # 2. Stack them: [batch, hidden, K]
        stacked_outputs = torch.stack(parallel_outputs, dim=-1)

        # 3. Aggregation: Calculate the MEAN across the K duplicates
        # This is the "Ensemble" effect that justifies the KRNN name.
        # Shape becomes [batch, hidden]
        krnn_representation = torch.mean(stacked_outputs, dim=-1)

        return krnn_representation


class KRNNRegressor(nn.Module):
    """
    The Full Regression Model.
    Wraps the K-Parallel Encoder with a Linear Regression Head.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # The K-Parallel Encoder
        self.encoder = KParallelEncoder(config)

        # Regression Head (Hidden -> 1)
        self.regressor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.output_dim)
        )

        self.to(config.device)

    def forward(self, x: torch.Tensor):
        # 1. Encode (K-Parallel RNNs)
        encoded_features = self.encoder(x)

        # 2. Regress (Predict Log Return)
        prediction = self.regressor(encoded_features)

        return prediction.squeeze(-1)  # [Batch]

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
        attention = F.softmax(
            torch.bmm(rnn_out, rnn_out.transpose(1, 2)) /
            torch.sqrt(torch.tensor(self.rnn_hidden, dtype=torch.float32, device=x.device)),
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
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
            # Removed verbose=True to address deprecation warning
        )
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self, train_loader) -> dict:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(self.model.config.device)
            targets = targets.to(self.model.config.device).squeeze(1)  # Squeeze target tensor

            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(features)
            loss = self.criterion(logits, targets)

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
            'accuracy': correct / total
        }

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (features, targets) in enumerate(val_loader):
            features = features.to(self.model.config.device)
            targets = targets.to(self.model.config.device).squeeze(1)  # Squeeze target tensor

            # Forward pass
            logits, _ = self.model(features)
            loss = self.criterion(logits, targets)

            # Track metrics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

        metrics = {
            'loss': total_loss / len(val_loader),
            'accuracy': correct / total
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
