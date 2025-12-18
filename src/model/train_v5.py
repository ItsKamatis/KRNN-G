# train_v5.py
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from torch import nn
from src.utils.experiment_v5 import ExperimentManager, MemoryOptimizer
from src.model.krnn_v5 import KRNN, ModelConfig
from src.data.dataset_v5 import DataModule, DataConfig
from src.utils.trainer_v5 import Trainer, TrainerConfig
from src.utils.metrics_v5 import MetricsCalculator

logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config

        # OPTIMIZER
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=0.001
        )

        # CRITICAL CHANGE:
        # Use MSELoss for Regression (Predicting Prices/Returns)
        self.criterion = nn.MSELoss()

    def train_step(self, batch_x, batch_y):
        self.model.train()
        self.optimizer.zero_grad()

        # Forward
        preds = self.model(batch_x)

        # Loss (Mean Squared Error)
        loss = self.criterion(preds, batch_y)

        # Backward
        loss.backward()
        self.optimizer.step()

        return loss.item()

def load_config(path: str) -> Dict[str, Any]:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def train(config_path: str) -> None:
    """Main training function."""
    # Load configuration
    config = load_config(config_path)

    # Setup experiment
    experiment = ExperimentManager(config)

    with experiment, MemoryOptimizer():
        # Initialize data
        data = DataModule(DataConfig(
            batch_size=config['training']['batch_size'],
            sequence_length=config['data']['window_size']
        ))
        data.setup(Path(config['paths']['data']))
        dataloaders = data.get_dataloaders()

        # Initialize model
        model = KRNN(ModelConfig(
            feature_dim=len(data.train_dataset.feature_cols),
            hidden_dim=config['model']['hidden_dim'],
            num_layers=config['model']['num_layers'],
            dropout=config['model']['dropout'],
            bidirectional=config['model']['bidirectional']
        ))

        # Initialize trainer
        trainer = Trainer(
            model=model,
            config=TrainerConfig(
                epochs=config['training']['epochs'],
                checkpoint_dir=Path(config['paths']['checkpoints'])
            )
        )

        # Train model
        trainer.train(
            train_loader=dataloaders['train'],
            val_loader=dataloaders['val'],
            experiment=experiment
        )

        # Final evaluation
        metrics = MetricsCalculator()
        test_metrics = trainer._validate(dataloaders['test'])
        logger.info(f"Test metrics: {test_metrics}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_v5.yaml", help="Path to config file")
    args = parser.parse_args()

    train(args.config)