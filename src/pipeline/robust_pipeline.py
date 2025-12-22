import numpy as np
import pandas as pd
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple

# Import your existing modules
from src.data.data_collector_v5 import DataCollector
from src.utils.predict_v5 import Predictor
from src.risk.scenario_engine import CVaRScenarioReducer
from src.portfolio.optimizer import MeanCVaROptimizer

logger = logging.getLogger(__name__)


class RobustPortfolioPipeline:
    def __init__(self, config_path: str):
        """
        Args:
            config_path: Path to the .yaml configuration file.
        """
        self.config_path = config_path

        # Load Config Dictionary
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.collector = DataCollector(self.config)

        # --- FIX: Locate the Model File ---
        # The Predictor requires a string path to the .pt file, not a dict.
        checkpoint_dir = Path(self.config['paths']['checkpoints']).resolve()  # .resolve() gives the full Absolute Path
        logger.info(f"Looking for model checkpoints in: {checkpoint_dir}")

        # Create directory if it doesn't exist (just to be safe)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Heuristic: Look for 'best_model.pt' or the most recently modified .pt file
        model_path = checkpoint_dir / "best_model.pt"

        if not model_path.exists():
            # If explicit 'best_model.pt' doesn't exist, find the newest .pt file
            pt_files = list(checkpoint_dir.glob("*.pt"))
            if pt_files:
                model_path = sorted(pt_files, key=lambda x: x.stat().st_mtime)[-1]
                logger.info(f"Auto-detected latest model checkpoint: {model_path}")
            else:
                # This is where your error came from. Now it tells you EXACTLY where it looked.
                raise FileNotFoundError(
                    f"\n[CRITICAL ERROR] No .pt model files found in:\n"
                    f"--> {checkpoint_dir}\n"
                    f"Solution: You must run 'train_local_v5.py' first to generate a model!"
                )
        else:
            logger.info(f"Using model checkpoint: {model_path}")


        # Initialize Predictor with the PATH string, not the config dict
        self.predictor = Predictor(str(model_path), self.config_path)

        # Risk Settings
        self.target_scenarios = 50
        self.confidence_level = 0.95
        self.target_return = 0.001

    def get_robust_scenarios(self, current_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates forward-looking scenarios by combining:
        1. KRNN Point Prediction (The Signal)
        2. Reduced Historical Residuals (The Noise/Risk)
        """
        # 1. Get KRNN Prediction
        # predict_next_day expects features, usually DataFrame or array.
        # Check predict_v5.py input requirements if this fails.
        # Assuming current_features is appropriate format.
        predicted_mean = self.predictor.predict_next_day(current_features)

        # 2. Get Historical Residuals (Using recent returns as proxy for now)
        historical_returns = self.collector.get_recent_returns(window=1000)

        # Center history to treat it as "Error Distribution"
        centered_history = historical_returns - np.mean(historical_returns, axis=0)

        # 3. Reduce Scenarios
        # We use the mean of assets (Market Proxy) to guide the reduction
        market_proxy = np.mean(centered_history, axis=1)

        reducer = CVaRScenarioReducer(
            target_data=market_proxy,
            num_scenarios=self.target_scenarios,
            alpha=self.confidence_level
        )
        reduced_values, probs = reducer.reduce()

        # Map reduced 1D values back to full N-dimensional asset residuals
        selected_indices = []
        for val in reduced_values:
            # Find historical day closest to this reduced scenario value
            idx = (np.abs(market_proxy - val)).argmin()
            selected_indices.append(idx)

        reduced_residuals = centered_history[selected_indices]

        # 4. Final Scenarios = Prediction + Residuals
        final_scenarios = predicted_mean + reduced_residuals
        return final_scenarios, probs

    def run(self):
        logger.info("Starting Robust Portfolio Optimization...")

        # 1. Fetch latest market data
        features, _ = self.collector.get_latest_batch()

        # 2. Generate Weighted Scenarios
        scenarios, probabilities = self.get_robust_scenarios(features)

        logger.info(f"Generated {len(scenarios)} scenarios.")

        # 3. Optimize
        optimizer = MeanCVaROptimizer(confidence_level=self.confidence_level)
        result = optimizer.optimize(
            scenarios=scenarios,
            probabilities=probabilities,
            target_return=self.target_return
        )

        if result:
            logger.info("Optimization Success!")
            logger.info(f"Weights: {result['weights']}")
            logger.info(f"Explicit CVaR: {result['CVaR_Optimal']:.6f}")
            logger.info(f"VaR: {result['VaR']:.6f}")
        else:
            logger.error("Optimization failed to converge.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pipeline = RobustPortfolioPipeline("config_v5.yaml")
    pipeline.run()