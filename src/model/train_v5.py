# train_v5.py
import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

import torch

from src.utils.experiment_v5 import ExperimentManager, MemoryOptimizer
from src.model.krnn_v5 import KRNN, ModelConfig
from src.data.dataset_v5 import DataModule, DataConfig
from src.utils.trainer_v5 import Trainer, TrainerConfig
from src.utils.metrics_v5 import MetricsCalculator

logger = logging.getLogger(__name__)


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