# src/model/krnn_v5.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    """Configuration for the Probabilistic K-Parallel RNN."""
    input_dim: int
    hidden_dim: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    k_dups: int = 3
    output_dim: int = 2  # Mu and Sigma
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

class KParallelEncoder(nn.Module):
    """
    The Core 'KRNN' Logic.
    Creates K parallel RNNs and averages their outputs to reduce variance.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.k_dups = config.k_dups
        self.hidden_dim = config.hidden_dim
        self.device = config.device

        # Create K independent RNN instances
        self.rnn_modules = nn.ModuleList()
        for _ in range(self.k_dups):
            self.rnn_modules.append(
                nn.GRU(
                    input_size=config.input_dim,
                    hidden_size=config.hidden_dim,
                    num_layers=config.num_layers,
                    dropout=config.dropout if config.num_layers > 1 else 0,
                    batch_first=True,
                    bidirectional=False
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len, Features]
        Returns:
            Averaged Hidden Representation [Batch, Hidden_Dim]
        """
        parallel_outputs = []
        for rnn in self.rnn_modules:
            out, _ = rnn(x)
            # Take the output of the last time step
            last_step_out = out[:, -1, :]  # [batch, hidden]
            parallel_outputs.append(last_step_out)

        # Stack and Mean Pool
        stacked_outputs = torch.stack(parallel_outputs, dim=-1)
        krnn_representation = torch.mean(stacked_outputs, dim=-1)
        return krnn_representation

class KRNNRegressor(nn.Module):
    """
    Probabilistic KRNN Regressor.
    Predicts both Expected Return (Mu) and Volatility (Sigma).
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.encoder = KParallelEncoder(config)

        # We need 2 outputs: Mu (Mean) and Sigma (Std Dev)
        self.regressor = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 2)
        )
        self.to(config.device)

    def forward(self, x: torch.Tensor):
        # 1. Encode
        encoded = self.encoder(x)

        # 2. Predict parameters
        out = self.regressor(encoded)  # [Batch, 2]

        # Split output
        mu = out[:, 0]  # Predicted Mean Return
        log_var = out[:, 1]  # Raw output for volatility

        # Enforce positivity for Sigma using Softplus
        # Add epsilon to prevent division by zero in loss function
        sigma = F.softplus(log_var) + 1e-6

        return mu, sigma