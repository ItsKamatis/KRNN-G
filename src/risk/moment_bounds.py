import numpy as np
import scipy.optimize as optimize
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MomentBounds:
    """Stores the estimated bounds for VaR and CVaR (Standardized Units)."""
    confidence_level: float
    wc_var_z: float  # Worst-Case VaR (Z-score)
    wc_cvar_z: float  # Worst-Case CVaR (Z-score)
    moments: List[float]


class DiscreteMomentSolver:
    """
    Solves the Discrete Moment Problem (DMP) to find Worst-Case Risk Measures.

    Methodology:
    Constructs a Linear Program (LP) to maximize the probability mass in the tail (for VaR)
    or the expected tail loss (for CVaR) subject to the exact moment constraints
    derived from the data.

    This provides a 'Robust' upper bound on risk, assuming the distribution could
    be the worst possible shape consistent with the observed mean, variance, skew, and kurtosis.
    """

    def __init__(self, support_points: int = 1000, support_range: Tuple[float, float] = (-10.0, 10.0)):
        self.n_points = support_points
        self.range = support_range

    def fit_and_estimate(self, data: np.ndarray, confidence_level: float = 0.99, n_moments: int = 4) -> MomentBounds:
        # 1. Calculate Standardized Moments from Data (Z-scores)
        # Data is assumed to be residuals, but we enforce standardization for stability
        mu = np.mean(data)
        sigma = np.std(data) + 1e-6
        std_data = (data - mu) / sigma

        # Moments: [1 (prob), 0 (mean), 1 (var), skew, kurt, ...]
        moments = [1.0]
        for k in range(1, n_moments + 1):
            moments.append(np.mean(np.power(std_data, k)))

        # 2. Define Discrete Support Grid (Z-score space)
        z_grid = np.linspace(self.range[0], self.range[1], self.n_points)

        # 3. Solve for Worst-Case VaR (Z)
        wc_var_z = self._find_worst_case_var_z(z_grid, moments, confidence_level)

        # 4. Solve for Worst-Case CVaR (Z)
        wc_cvar_z = self._solve_worst_case_cvar_lp(z_grid, moments, wc_var_z, confidence_level)

        return MomentBounds(
            confidence_level=confidence_level,
            wc_var_z=wc_var_z,
            wc_cvar_z=wc_cvar_z,
            moments=moments
        )

    def _find_worst_case_var_z(self, z_grid: np.ndarray, moments: List[float], alpha: float) -> float:
        """Finds smallest threshold z such that Max P(Z > z) <= 1 - alpha."""
        target_prob = 1.0 - alpha

        # Scan downwards from max z
        for i in range(len(z_grid) - 1, -1, -1):
            threshold = z_grid[i]
            # Maximize P(Z > threshold)
            c = -1.0 * (z_grid > threshold).astype(float)
            max_prob = self._solve_lp(c, z_grid, moments)

            # If we CAN construct a distribution with P(tail) >= target,
            # then the VaR (boundary) must be at least this high.
            if max_prob >= target_prob:
                return threshold
        return z_grid[0]

    def _solve_worst_case_cvar_lp(self, z_grid: np.ndarray, moments: List[float], var_threshold: float,
                                  alpha: float) -> float:
        """Maximizes Expected Shortfall given the VaR threshold."""
        # Objective: Maximize E[Z | Z > VaR] * (1-alpha) approx
        mask = z_grid >= var_threshold
        # We want to maximize sum(p_i * z_i) over the tail
        c = -1.0 * (z_grid * mask)

        expected_tail_loss = self._solve_lp(c, z_grid, moments)
        return expected_tail_loss / (1.0 - alpha)

    def _solve_lp(self, c: np.ndarray, z_grid: np.ndarray, moments: List[float]) -> float:
        # Vandermonde Matrix for moment constraints
        A_eq = np.vstack([np.power(z_grid, k) for k in range(len(moments))])
        b_eq = np.array(moments)
        bounds = [(0, 1) for _ in range(len(z_grid))]

        try:
            # Using 'highs' method for reliable simplex/interior-point solution
            res = optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            return -res.fun if res.success else 0.0
        except:
            return 0.0