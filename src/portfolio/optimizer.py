# src/portfolio/optimizer.py
import numpy as np
import scipy.optimize as opt
import logging
from typing import Dict, Tuple

logger = logging.getLogger(__name__)


class MeanCVaROptimizer:
    """
    Portfolio Optimizer that minimizes Conditional Value at Risk (CVaR).
    Methodology: Rockafellar & Uryasev (2000) Linear Programming formulation.
    Updates: Allows Short Selling and Leverage.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.alpha = confidence_level

    def optimize(self,
                 expected_returns: np.ndarray,
                 scenarios: np.ndarray,
                 target_return: float) -> Dict:
        """
        Solves the Linear Program to find optimal weights (Long/Short).
        """
        T, N = scenarios.shape

        # --- 1. Define Decision Variables ---
        # x = [gamma, z_1...z_T, w_1...w_N]
        num_vars = 1 + T + N
        idx_gamma = 0
        idx_z_start = 1
        idx_z_end = 1 + T
        idx_w_start = 1 + T

        # --- 2. Objective Function ---
        # Min: gamma + (1/((1-alpha)T)) * sum(z)
        c = np.zeros(num_vars)
        c[idx_gamma] = 1.0
        c[idx_z_start:idx_z_end] = 1.0 / ((1 - self.alpha) * T)

        # --- 3. Inequality Constraints ---
        A_ub = []
        b_ub = []

        # Constraint A: Loss Constraints (z_t >= Loss_t - gamma)
        # Note: With shorting, Loss can be negative (Profit)
        for t in range(T):
            row = np.zeros(num_vars)
            row[idx_gamma] = -1.0
            row[idx_z_start + t] = -1.0
            row[idx_w_start:] = -scenarios[t, :]  # Works for negative weights too
            A_ub.append(row)
            b_ub.append(0.0)

        # Constraint B: Minimum Return
        # With Shorting, we can target positive returns even if assets are negative
        effective_target = target_return

        row_ret = np.zeros(num_vars)
        row_ret[idx_w_start:] = -expected_returns
        A_ub.append(row_ret)
        b_ub.append(-effective_target)

        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # --- 4. Equality Constraints ---
        # Sum(w) = 1.0 (Net Long Exposure = 100%)
        # This allows, e.g., 150% Long, -50% Short -> Net 100%
        A_eq = np.zeros((1, num_vars))
        A_eq[0, idx_w_start:] = 1.0
        b_eq = np.array([1.0])

        # --- 5. Bounds ---
        bounds = []
        bounds.append((None, None))  # Gamma
        for _ in range(T):
            bounds.append((0, None))  # z_t must be positive (it's the tail loss)

        # CHANGED: Allow weights between -1.0 (Short) and 1.0 (Long)
        # This enables 100% leverage per asset (Gross can be > 1)
        for _ in range(N):
            bounds.append((-1.0, 1.0))

            # --- 6. Solve ---
        logger.info(f"Solving Long/Short Mean-CVaR for {N} assets...")
        result = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if not result.success:
            logger.error(f"Optimization failed: {result.message}")
            return None

        optimal_weights = result.x[idx_w_start:]
        # No zero-clipping/normalization here to preserve short positions logic
        return {
            "weights": optimal_weights,
            "VaR": result.x[idx_gamma],
            "CVaR_Optimal": result.fun,
            "status": result.message
        }