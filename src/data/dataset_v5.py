import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader


@dataclass(frozen=True)
class DataConfig:
    """Immutable dataset configuration."""
    batch_size: int = 64
    sequence_length: int = 10
    train_shuffle: bool = True
    num_workers: int = 4
    pin_memory: bool = True


class StockDataset(Dataset):
    """Memory efficient dataset for stock data."""

    def __init__(self,
                 features: pd.DataFrame,
                 sequence_length: int,
                 target_column: str = 'Label'):
        self.features = features
        self.sequence_length = sequence_length
        self.target_column = target_column

        # Get feature columns (exclude Date, Ticker, and target)
        self.feature_cols = [
            col for col in features.columns
            if col not in ['Date', 'Ticker', target_column]
        ]

        # Group by ticker and get valid indices
        self.valid_indices = self._get_valid_indices()

    def _get_valid_indices(self) -> list:
        """Get indices where we have enough history."""
        groups = self.features.groupby('Ticker')
        valid_indices = []

        for _, group in groups:
            # Only use indices where we have enough history
            for i in range(self.sequence_length - 1, len(group)):
                valid_indices.append(i)

        return valid_indices

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sequence of features and target."""
        index = self.valid_indices[idx]
        ticker = self.features.iloc[index]['Ticker']

        # Get ticker's data
        ticker_data = self.features[self.features['Ticker'] == ticker]

        # Get sequence
        end_idx = ticker_data.index.get_loc(self.features.index[index])
        start_idx = end_idx - self.sequence_length + 1

        # Extract features and target
        feature_sequence = ticker_data.iloc[start_idx:end_idx + 1][self.feature_cols].values
        target = ticker_data.iloc[end_idx][self.target_column]

        return (
            torch.FloatTensor(feature_sequence),
            torch.LongTensor([target - 1])  # Adjust for 0-based indexing
        )


class DataModule:
    """Handles data loading and preparation."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, data_dir: Path) -> None:
        """Load and prepare datasets."""
        # Load data splits
        train_data = pd.read_parquet(data_dir / 'train.parquet')
        val_data = pd.read_parquet(data_dir / 'validation.parquet')
        test_data = pd.read_parquet(data_dir / 'test.parquet')

        # Create datasets
        self.train_dataset = StockDataset(
            train_data,
            self.config.sequence_length
        )
        self.val_dataset = StockDataset(
            val_data,
            self.config.sequence_length
        )
        self.test_dataset = StockDataset(
            test_data,
            self.config.sequence_length
        )

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        """Create data loaders for each split."""
        if not all([self.train_dataset, self.val_dataset, self.test_dataset]):
            raise RuntimeError("Datasets not initialized. Call setup() first.")

        return {
            'train': DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.train_shuffle,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            ),
            'val': DataLoader(
                self.val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            ),
            'test': DataLoader(
                self.test_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        }


class TimeSeriesSplitter:
    """Handles time series cross-validation splits."""

    def __init__(self,
                 n_splits: int = 5,
                 gap_days: int = 5,
                 test_size: int = 30):
        self.n_splits = n_splits
        self.gap_days = gap_days
        self.test_size = test_size

    def split(self, df: pd.DataFrame) -> list:
        """Generate time series splits."""
        splits = []
        unique_dates = sorted(df['Date'].unique())

        # Calculate split points
        total_periods = len(unique_dates)
        test_start_idx = total_periods - self.test_size

        for i in range(self.n_splits):
            # Calculate validation period
            val_end_idx = test_start_idx - (i * self.gap_days)
            val_start_idx = val_end_idx - self.test_size

            if val_start_idx <= 0:
                break

            # Get date ranges
            train_dates = unique_dates[:val_start_idx - self.gap_days]
            val_dates = unique_dates[val_start_idx:val_end_idx]
            test_dates = unique_dates[test_start_idx:]

            # Create masks
            train_mask = df['Date'].isin(train_dates)
            val_mask = df['Date'].isin(val_dates)
            test_mask = df['Date'].isin(test_dates)

            splits.append((
                df[train_mask],
                df[val_mask],
                df[test_mask]
            ))

        return splits