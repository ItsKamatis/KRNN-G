# train_local_v5.py
import torch
import logging
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.model.krnn_v5 import KRNN, ModelConfig, KRNNPredictor
from src.data.dataset_v5 import DataModule, DataConfig
from src.utils.experiment_v5 import ExperimentManager, MemoryOptimizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_data_exists(config: dict) -> None:
    """Ensure training data exists, create if not."""
    data_dir = Path(config['paths']['data'])
    if all((data_dir / f'{split}.parquet').exists() for split in ['train', 'validation', 'test']):
        return

    logger.info("Preparing training data...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Generate synthetic data for testing
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    n_samples = len(dates)

    # Define a list of tickers
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA']
    assigned_tickers = np.random.choice(tickers, size=n_samples)

    data = pd.DataFrame({
        'Date': dates,
        'Ticker': assigned_tickers,
        'Open': np.random.normal(100, 10, n_samples).cumsum(),
        'High': np.random.normal(100, 10, n_samples).cumsum() + 2,
        'Low': np.random.normal(100, 10, n_samples).cumsum() - 2,
        'Close': np.random.normal(100, 10, n_samples).cumsum(),
        'Volume': np.random.lognormal(10, 1, n_samples),
        'RSI': np.random.uniform(0, 100, n_samples),
        'MACD': np.random.normal(0, 1, n_samples),
        'BB_UPPER': np.random.normal(105, 5, n_samples),
        'BB_LOWER': np.random.normal(95, 5, n_samples),
        'ATR': np.abs(np.random.normal(0, 1, n_samples))
    })

    # Introduce missing values randomly for testing
    missing_rate = 0.01  # 1% missing data
    num_missing = int(missing_rate * n_samples * len(data.columns))
    for _ in range(num_missing):
        idx = np.random.randint(0, n_samples)
        col = np.random.choice(data.columns.drop(['Date', 'Ticker', 'Label'], errors='ignore'))
        data.at[idx, col] = np.nan

    # Add labels (1: Up, 2: Stable, 3: Down)
    returns = data['Close'].pct_change()
    data['Label'] = pd.qcut(returns, q=3, labels=[1, 2, 3]).fillna(2).astype(int)

    # Drop rows with missing values and reset index
    data_clean = data.dropna().reset_index(drop=True)
    logger.info(f"Dropped {len(data) - len(data_clean)} rows due to missing values.")

    # Sort data by Ticker and Date
    data_clean = data_clean.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

    # Split data
    train_cutoff = int(len(data_clean) * 0.7)
    val_cutoff = int(len(data_clean) * 0.85)

    train_data = data_clean[:train_cutoff]
    val_data = data_clean[train_cutoff:val_cutoff]
    test_data = data_clean[val_cutoff:]

    # Ensure there is enough data after cleaning
    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise ValueError("Not enough data to create train/validation/test splits after cleaning.")

    # Scale features
    scaler = StandardScaler()
    feature_cols = [col for col in data.columns if col not in ['Date', 'Ticker', 'Label']]
    scaler.fit(train_data[feature_cols])

    # Save splits
    for split, df in [('train', train_data), ('validation', val_data), ('test', test_data)]:
        df_scaled = df.copy()
        df_scaled[feature_cols] = scaler.transform(df[feature_cols])
        df_scaled.to_parquet(data_dir / f'{split}.parquet', index=False)
        logger.info(f"Saved {split} data with {len(df_scaled)} samples")


def train_local(config_path: str) -> dict:
    """Run complete training pipeline."""
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Ensure data exists
    ensure_data_exists(config)

    # Setup experiment tracking
    experiment = ExperimentManager(config)
    logger.info(f"MLflow experiment: {config['mlflow']['experiment_name']}")

    try:
        with experiment, MemoryOptimizer():
            # Initialize data module
            data = DataModule(DataConfig(
                batch_size=config['data']['batch_size'],
                sequence_length=config['data']['window_size'],
                num_workers=config['hardware']['num_workers'],
                pin_memory=config['hardware']['pin_memory']
            ))
            data.setup(Path(config['paths']['data']))

            # Retrieve dataloaders
            loaders = data.get_dataloaders()
            train_loader = loaders['train']
            val_loader = loaders['val']
            test_loader = loaders['test']

            # Create model
            model = KRNN(ModelConfig(
                feature_dim=len(data.train_dataset.feature_cols),
                hidden_dim=config['model']['hidden_dim'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout'],
                device=str(device)
            )).to(device)

            # Initialize predictor
            predictor = KRNNPredictor(model)

            # Training loop
            best_val_loss = float('inf')
            patience_counter = 0

            for epoch in range(config['training']['epochs']):
                # Train
                train_metrics = predictor.train_epoch(train_loader)
                experiment.log_metrics(train_metrics, step=epoch)

                # Validate
                val_metrics = predictor.validate(val_loader)
                experiment.log_metrics(val_metrics, step=epoch)

                # Early stopping
                if val_metrics['loss'] < best_val_loss:
                    best_val_loss = val_metrics['loss']
                    patience_counter = 0
                    predictor.save_checkpoint(
                        Path(config['paths']['checkpoints']) / 'best_model.pt'
                    )
                else:
                    patience_counter += 1
                    if patience_counter >= config['training']['early_stop_patience']:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

                logger.info(
                    f"Epoch {epoch}: train_loss={train_metrics['loss']:.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"train_accuracy={train_metrics['accuracy']:.4f}, "
                    f"val_accuracy={val_metrics['accuracy']:.4f}"
                )

            # Test final model
            test_metrics = predictor.validate(test_loader)
            experiment.log_metrics({'test_' + k: v for k, v in test_metrics.items()})
            logger.info(f"Final test metrics: {test_metrics}")
            return test_metrics

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_v5.yaml', help='Path to config file')
    args = parser.parse_args()

    metrics = train_local(args.config)
    logger.info(f"Final test metrics: {metrics}")
