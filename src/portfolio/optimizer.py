import numpy as np
import scipy.optimize as opt
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class MeanCVaROptimizer:
    """
    Portfolio Optimizer that minimizes Conditional Value at Risk (CVaR)
    using the Weighted Rockafellar & Uryasev (2000) formulation.

    This version accepts arbitrary scenario probabilities, enabling
    Distributionally Robust Optimization (DRO) when coupled with
    Scenario Reduction or Discrete Moment Problem solvers.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.alpha = confidence_level

    def optimize(self,
                 scenarios: np.ndarray,
                 probabilities: np.ndarray,
                 target_return: float) -> Dict:
        """
        Solves the Linear Program to find optimal weights.

        Args:
            scenarios: (M, N) array of asset returns in M scenarios.
            probabilities: (M,) array of probabilities for each scenario.
                           MUST sum to approximately 1.0.
            target_return: Minimum weighted expected return required.

        Returns:
            Dictionary containing weights and risk metrics.
        """
        M, N_assets = scenarios.shape

        # Validation
        if len(probabilities) != M:
            raise ValueError(f"Probabilities length ({len(probabilities)}) must match scenarios ({M})")
        if not np.isclose(np.sum(probabilities), 1.0, atol=1e-4):
            logger.warning(f"Probabilities sum to {np.sum(probabilities):.4f}, not 1.0. Renormalizing.")
            probabilities = probabilities / np.sum(probabilities)

        # --- 1. Define Decision Variables ---
        # We use scipy.optimize.linprog which solves: min c @ x s.t. A_ub @ x <= b_ub
        #
        # Variable Vector x structure:
        # [gamma,  z_1...z_M,  w_1...w_N]
        #   1        M vars      N vars
        #
        # gamma: Value at Risk (VaR)
        # z_j:   Tail loss auxiliary variable for scenario j
        # w_i:   Weight of asset i

        num_vars = 1 + M + N_assets
        idx_gamma = 0
        idx_z_start = 1
        idx_z_end = 1 + M
        idx_w_start = 1 + M

        # --- 2. Objective Function ---
        # Min: gamma + (1 / (1 - alpha)) * Sum(p_j * z_j)
        # Note: The standard 1/T is replaced by p_j here [cite: 496]
        c = np.zeros(num_vars)
        c[idx_gamma] = 1.0

        # The coefficient for each z_j is p_j / (1 - alpha)
        scaling_factor = 1.0 / (1.0 - self.alpha)
        c[idx_z_start:idx_z_end] = probabilities * scaling_factor

        # --- 3. Inequality Constraints (A_ub @ x <= b_ub) ---
        A_ub = []
        b_ub = []

        # Constraint A: Loss Definition
        # z_j >= Loss_j - gamma
        # Rearranged for linprog (<=): -z_j - gamma + Loss_j <= 0
        # Loss_j = - (Sum(w_i * r_ij))  (Negative Portfolio Return)
        # So: -z_j - gamma - Sum(w_i * r_ij) <= 0

        # We build this row by row for each scenario j
        for j in range(M):
            row = np.zeros(num_vars)
            row[idx_gamma] = -1.0  # Coeff for gamma
            row[idx_z_start + j] = -1.0  # Coeff for z_j

            # Coeff for weights: The negative return of assets in this scenario
            # Because Loss = -(w @ r), and we moved Loss to LHS, it becomes -r
            row[idx_w_start:] = -scenarios[j, :]

            A_ub.append(row)
            b_ub.append(0.0)

        # Constraint B: Minimum Expected Return
        # Sum(p_j * (w @ r_j)) >= target
        # Rearranged (<=): - Sum(w_i * ExpectedReturn_i) <= -target
        # Where ExpectedReturn_i = Sum_j(p_j * r_ij)

        # Calculate weighted expected return for each asset
        # (M,) @ (M, N) -> (N,)
        asset_expected_returns = probabilities @ scenarios

        row_ret = np.zeros(num_vars)
        row_ret[idx_w_start:] = -asset_expected_returns
        A_ub.append(row_ret)
        b_ub.append(-target_return)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # --- 4. Equality Constraints (A_eq @ x == b_eq) ---
        # Sum(w) = 1.0 (Full Investment)
        A_eq = np.zeros((1, num_vars))
        A_eq[0, idx_w_start:] = 1.0
        b_eq = np.array([1.0])

        # --- 5. Bounds ---
        bounds = []
        bounds.append((None, None))  # Gamma (VaR) is free, can be negative (profit)

        # z_j >= 0 (Loss exceeds VaR or 0)
        for _ in range(M):
            bounds.append((0, None))

        # Weights: Allow Long/Short (-1.0 to 1.0)
        # This implies Gross Exposure can be up to 200%?
        # (e.g. 1.0 Long, -1.0 Short is invalid sum=0, but 1.0 Long, 0 Short is valid)
        # Let's keep your previous setting of -1 to 1 per asset.
        for _ in range(N_assets):
            bounds.append((-1.0, 1.0))

        # --- 6. Solve ---
        # Using 'highs' method which is robust for large LPs
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not result.success:
            logger.error(f"Optimization failed: {result.message}")
            return None

        optimal_weights = result.x[idx_w_start:]

        # Explicitly calculate the metrics using the result
        # Note: result.x[idx_gamma] is VaR
        # result.fun is the optimized CVaR value

        return {
            "weights": optimal_weights,
            "VaR": result.x[idx_gamma],
            "CVaR_Optimal": result.fun,
            "status": result.message
        }