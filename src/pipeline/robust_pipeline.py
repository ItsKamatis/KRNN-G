import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple

# Import your existing modules
from src.data.data_collector_v5 import DataCollector
from src.model.krnn_v5 import KRNN
from src.utils.predict_v5 import Predictor

# Import the NEW Robust Risk modules
from src.risk.scenario_engine import CVaRScenarioReducer
from src.portfolio.optimizer import MeanCVaROptimizer

logger = logging.getLogger(__name__)


class RobustPortfolioPipeline:
    def __init__(self, config: dict):
        self.config = config
        self.collector = DataCollector(config)

        # Load the trained KRNN model
        self.predictor = Predictor(config)
        self.predictor.load_model(config['paths']['checkpoints'])

        # Risk Settings
        self.target_scenarios = 50  # M << N (Compression)
        self.confidence_level = 0.95
        self.target_return = 0.001  # Daily target (e.g., 0.1%)

    def get_robust_scenarios(self, current_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates forward-looking scenarios by combining:
        1. KRNN Point Prediction (The Signal)
        2. Reduced Historical Residuals (The Noise/Risk)
        """
        # 1. Get the 'Best Guess' for tomorrow using KRNN
        # shape: (1, N_assets)
        predicted_mean = self.predictor.predict_next_day(current_features)

        # 2. Get Historical Residuals (Errors)
        # We need to see how the model failed in the past to know how it might fail tomorrow.
        # Ideally, you store these during validation. For now, let's assume we calculate
        # them from the validation set or recent history.

        # [Placeholder: Calculate residuals = True_History - Predicted_History]
        # For this tutor example, let's assume we pull raw historical returns as a proxy for error distribution
        # In a perfect world, you use: residuals = y_true - y_pred
        historical_returns = self.collector.get_recent_returns(window=1000)  # shape (1000, N_assets)

        # Center the history around 0 to represent "Uncertainty" around the prediction
        centered_history = historical_returns - np.mean(historical_returns, axis=0)

        # 3. Apply CVaR Scenario Reduction to the NOISE
        # We want to find the 50 "Critical Error Scenarios" that define our risk.

        # We reduce based on the Portfolio-level error or per-asset.
        # The paper suggests reducing the joint distribution.
        # For simplicity in this step, let's treat the 'target' for reduction as the
        # average market return or a representative index to simplify the 1D reduction logic.
        market_proxy = np.mean(centered_history, axis=1)  # Reduce based on "Systemic Risk"

        reducer = CVaRScenarioReducer(
            target_data=market_proxy,
            num_scenarios=self.target_scenarios,
            alpha=self.confidence_level
        )

        # These are the "Indices" or "Values" of the critical scenarios in 1D
        reduced_values, probs = reducer.reduce()

        # Now we map these back to the full N-asset universe.
        # We need to select the full vector of asset returns that corresponds to these reduced 1D points.
        # (This is a simplified "Selection" logic. Full multidimensional reduction is slower).

        # Find the indices in history that are closest to our reduced values
        # This effectively picks the 50 historical days that best represent the tail risk.
        selected_indices = []
        for val in reduced_values:
            idx = (np.abs(market_proxy - val)).argmin()
            selected_indices.append(idx)

        # 4. Construct Final Scenarios
        # Scenario_j = Prediction + Residual_j
        reduced_residuals = centered_history[selected_indices]  # (M, N_assets)

        # Broadcast: (1, N) + (M, N) -> (M, N)
        final_scenarios = predicted_mean + reduced_residuals

        return final_scenarios, probs

    def run(self):
        logger.info("Starting Robust Portfolio Optimization...")

        # 1. Fetch latest market data
        features, _ = self.collector.get_latest_batch()

        # 2. Generate Weighted Scenarios
        scenarios, probabilities = self.get_robust_scenarios(features)

        logger.info(f"Generated {len(scenarios)} scenarios with probability sum {np.sum(probabilities):.4f}")

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

            # [Optional] 4. Validate with Explicit Bounds (Naumova)
            # You can call src.risk.moment_bounds here to check if this CVaR
            # is within the theoretical [Min, Max] derived from moments.
        else:
            logger.error("Optimization failed to converge.")


if __name__ == "__main__":
    # Load your config.yaml here
    import yaml

    with open("config_v5.yaml", "r") as f:
        config = yaml.safe_load(f)

    pipeline = RobustPortfolioPipeline(config)
    pipeline.run()