# src/data/dataset_v5.py
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DataConfig:
    batch_size: int = 64
    sequence_length: int = 10
    train_shuffle: bool = True
    num_workers: int = 0
    pin_memory: bool = False
    device: str = "cpu"  # Added field to control storage location


class StockDataset(Dataset):
    """
    Optimized GPU-Resident Dataset.
    """

    def __init__(
            self,
            features: pd.DataFrame,
            sequence_length: int,
            target_column: str = 'Target',
            device: str = 'cpu'  # Accept device
    ):
        self.sequence_length = sequence_length
        self.device = device

        # 1. Identify Feature Columns
        self.feature_cols = [
            col for col in features.columns
            if col not in ['Date', 'Ticker', target_column]
               and pd.api.types.is_numeric_dtype(features[col])
        ]

        # 2. Sort
        df_sorted = features.sort_values(by=['Ticker', 'Date']).reset_index(drop=True)

        # 3. Pre-allocate lists
        all_feats_list = []
        all_targets_list = []

        grouped = df_sorted.groupby('Ticker')

        for ticker, group in grouped:
            group_feats = group[self.feature_cols].values.astype(np.float32)
            group_targets = group[target_column].values.astype(np.float32)

            num_samples = len(group)
            if num_samples <= sequence_length:
                continue

            # Creating windows
            for i in range(sequence_length, num_samples + 1):
                window_feats = group_feats[i - sequence_length: i]
                target_val = group_targets[i - 1]

                all_feats_list.append(window_feats)
                all_targets_list.append(target_val)

        if len(all_feats_list) == 0:
            raise ValueError("Not enough data to create sequences.")

        # 4. Stack and Move to Device IMMEDIATELY
        # This is the "Pro Move". We send ~350MB to VRAM once.
        self.X = torch.from_numpy(np.stack(all_feats_list)).to(self.device)
        self.y = torch.from_numpy(np.array(all_targets_list)).to(self.device)

        # Calculate memory usage in MB
        mem_usage = (self.X.element_size() * self.X.nelement() + self.y.element_size() * self.y.nelement()) / (
                    1024 * 1024)
        logger.info(f"Dataset loaded to {self.device}. Usage: {mem_usage:.2f} MB")

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Zero overhead access, data is already on GPU
        return self.X[idx], self.y[idx]


class DataModule:
    """Handles data loading and preparation."""

    def __init__(self, config: DataConfig):
        self.config = config
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, data_dir: Path) -> None:
        try:
            train_data = pd.read_parquet(data_dir / 'train.parquet')
            val_data = pd.read_parquet(data_dir / 'validation.parquet')
            test_data = pd.read_parquet(data_dir / 'test.parquet')
        except FileNotFoundError:
            raise RuntimeError("Data files not found.")

        # Pass the device config down to the dataset
        self.train_dataset = StockDataset(train_data, self.config.sequence_length, device=self.config.device)
        self.val_dataset = StockDataset(val_data, self.config.sequence_length, device=self.config.device)
        self.test_dataset = StockDataset(test_data, self.config.sequence_length, device=self.config.device)

    def get_dataloaders(self) -> Dict[str, DataLoader]:
        return {
            'train': DataLoader(
                self.train_dataset,
                batch_size=self.config.batch_size,
                shuffle=self.config.train_shuffle,
                num_workers=0,  # MUST be 0 for GPU-resident data
                pin_memory=False  # MUST be False for GPU-resident data
            ),
            'val': DataLoader(self.val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0,
                              pin_memory=False),
            'test': DataLoader(self.test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0,
                               pin_memory=False)
        }