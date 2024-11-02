import os
import mlflow
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExperimentContext:
    """Immutable experiment context."""
    run_id: str
    start_time: datetime
    config: Dict[str, Any]
    artifacts_path: Path


class ExperimentManager:
    """Manages experiment lifecycle and MLflow tracking."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.context = None
        self._setup_tracking()

    def _setup_tracking(self) -> None:
        """Configure MLflow tracking."""
        mlflow.set_tracking_uri(self.config['mlflow']['tracking_uri'])
        mlflow.set_experiment(self.config['mlflow']['experiment_name'])

    def __enter__(self) -> ExperimentContext:
        """Start experiment tracking."""
        # Start MLflow run
        run = mlflow.start_run()

        # Create experiment context
        self.context = ExperimentContext(
            run_id=run.info.run_id,
            start_time=datetime.now(),
            config=self.config,
            artifacts_path=Path(run.info.artifact_uri).resolve()
        )

        # Log configuration and initial info
        self._log_initial_info()
        return self.context

    def __exit__(self, exc_type, exc_val, exc_tb):
        """End experiment tracking."""
        if exc_type is not None:
            mlflow.set_tag('status', 'FAILED')
            mlflow.log_param('error_type', exc_type.__name__)
            mlflow.log_param('error_message', str(exc_val))
            logger.error(f"Experiment failed: {str(exc_val)}")
        else:
            mlflow.set_tag('status', 'COMPLETED')

        mlflow.end_run()
        self.context = None

    def _log_initial_info(self) -> None:
        """Log initial experiment information."""
        # Log configuration
        flat_config = self._flatten_dict(self.config)
        mlflow.log_params(flat_config)

        # Log git info if available
        try:
            git_commit = os.popen('git rev-parse HEAD').read().strip()
            git_branch = os.popen('git rev-parse --abbrev-ref HEAD').read().strip()
            mlflow.set_tags({
                'git_commit': git_commit,
                'git_branch': git_branch
            })
        except:
            logger.warning("Unable to log git information")

        # Log start time
        mlflow.log_param('start_time', self.context.start_time.isoformat())

    def _flatten_dict(self, d: Dict, parent_key: str = '') -> Dict[str, str]:
        """Flatten nested dictionary for MLflow logging."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            else:
                items.append((new_key, str(v)))
        return dict(items)


# Memory optimization context manager
class MemoryOptimizer:
    """Optimizes memory usage during experiment."""

    def __enter__(self):
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()