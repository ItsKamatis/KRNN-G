import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.metrics import matthews_corrcoef, confusion_matrix


@dataclass(frozen=True)
class TradingMetrics:
    """Trading performance metrics."""
    accuracy: float
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1: Dict[str, float]
    mcc: float
    confusion: np.ndarray
    timestamp: pd.Timestamp


class MetricsCalculator:
    """Calculates and tracks trading metrics."""

    def __init__(self, class_names: list = ['up', 'neutral', 'down']):
        self.class_names = class_names

    def calculate_metrics(self,
                          y_true: np.ndarray,
                          y_pred: np.ndarray,
                          timestamp: Optional[pd.Timestamp] = None) -> TradingMetrics:
        """Calculate comprehensive trading metrics."""
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Calculate confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Calculate per-class metrics
        precision, recall, f1 = self._calculate_class_metrics(conf_matrix)

        # Calculate MCC
        mcc = matthews_corrcoef(y_true, y_pred)

        # Calculate accuracy
        accuracy = np.mean(y_true == y_pred)

        return TradingMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            mcc=mcc,
            confusion=conf_matrix,
            timestamp=timestamp or pd.Timestamp.now()
        )

    def _calculate_class_metrics(self,
                                 conf_matrix: np.ndarray) -> Tuple[Dict[str, float],
    Dict[str, float],
    Dict[str, float]]:
        """Calculate precision, recall, and F1 for each class."""
        precision = {}
        recall = {}
        f1 = {}

        for i, class_name in enumerate(self.class_names):
            # True positives
            tp = conf_matrix[i, i]
            # False positives
            fp = conf_matrix[:, i].sum() - tp
            # False negatives
            fn = conf_matrix[i, :].sum() - tp

            # Calculate metrics
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

            precision[class_name] = prec
            recall[class_name] = rec
            f1[class_name] = f1_score

        return precision, recall, f1

    def calculate_rolling_metrics(self,
                                  y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  timestamps: np.ndarray,
                                  window: int = 20) -> pd.DataFrame:
        """Calculate metrics over rolling window."""
        metrics_list = []

        for i in range(window, len(y_true)):
            window_true = y_true[i - window:i]
            window_pred = y_pred[i - window:i]
            metrics = self.calculate_metrics(
                window_true,
                window_pred,
                timestamp=timestamps[i]
            )

            metrics_list.append({
                'timestamp': metrics.timestamp,
                'accuracy': metrics.accuracy,
                'mcc': metrics.mcc,
                **{f"precision_{k}": v for k, v in metrics.precision.items()},
                **{f"recall_{k}": v for k, v in metrics.recall.items()},
                **{f"f1_{k}": v for k, v in metrics.f1.items()}
            })

        return pd.DataFrame(metrics_list)

    @staticmethod
    def calculate_returns(prices: np.ndarray,
                          positions: np.ndarray) -> Tuple[float, float]:
        """Calculate trading returns and Sharpe ratio."""
        returns = np.diff(prices) / prices[:-1]
        strategy_returns = returns * positions[:-1]

        total_return = np.prod(1 + strategy_returns) - 1
        sharpe_ratio = np.sqrt(252) * np.mean(strategy_returns) / np.std(strategy_returns)

        return total_return, sharpe_ratio