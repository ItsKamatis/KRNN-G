import numpy as np
import cvxpy as cp
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


class CVaRScenarioReducer:
    """
    Implements the Pavlikov-Uryasev (2014) algorithm for scenario reduction
    by minimizing the CVaR distance between the historical distribution and
    the reduced discrete distribution.

    Ref: 'Integrating Moment Problems for Risk Optimization', Section 5
    """

    def __init__(self, target_data: np.ndarray, num_scenarios: int, alpha: float = 0.95):
        """
        Args:
            target_data: 1D array of historical returns (The "Truth").
            num_scenarios: The number of scenarios to reduce down to (M << N).
            alpha: Confidence level for CVaR (e.g., 0.95 focuses on the worst 5%).
        """
        self.target = np.sort(target_data)  # Sorting is crucial for quantile logic
        self.M = num_scenarios
        self.N = len(target_data)
        self.alpha = alpha

        # Validation
        if self.M >= self.N:
            logger.warning("Target scenarios >= History length. Reduction may not be useful.")

    def _step_optimize_probabilities(self, current_support: np.ndarray) -> np.ndarray:
        """
        Step 2 of Algorithm: Fix locations (support), find optimal probabilities[cite: 334].
        Solves a Linear Program to minimize CVaR distance.
        """
        # Distance matrix D[i, j] = |x_i - y_j|
        # Broadcasting: (M, 1) - (1, N) -> (M, N)
        D = np.abs(current_support[:, None] - self.target[None, :])

        # Variables
        p = cp.Variable(self.M)  # Probabilities of our scenarios
        gamma = cp.Variable()  # Auxiliary variable for CVaR
        v = cp.Variable(self.N)  # Auxiliary variable for tail losses

        # The Objective: Minimize CVaR of the error distribution
        # Formula: gamma + (1 / (1-alpha)N) * sum(v) [cite: 350]
        scaling_factor = 1.0 / ((1 - self.alpha) * self.N)
        objective = gamma + scaling_factor * cp.sum(v)

        # Expected error for each historical point j if mapped to our scenarios
        # This represents sum(p_i * |x_i - y_j|)
        expected_error_per_j = p @ D

        constraints = [
            v >= expected_error_per_j - gamma,
            v >= 0,
            cp.sum(p) == 1.0,
            p >= 0.0
        ]

        prob = cp.Problem(cp.Minimize(objective), constraints)
        # Solve with a robust solver (ECOS is default, CLARABEL is newer/better if avail)
        prob.solve()

        if prob.status != 'optimal':
            logger.warning(f"Probability optimization failed: {prob.status}")
            return np.ones(self.M) / self.M  # Fallback to uniform

        return p.value

    def _step_optimize_locations(self, current_probs: np.ndarray) -> np.ndarray:
        """
        Step 1 of Algorithm: Fix probabilities, find optimal locations[cite: 328].
        Heuristic: The optimal location for a probability chunk is the mean
        of the target data falling into that quantile bucket.
        """
        new_support = np.zeros(self.M)

        # We divide the sorted history into M buckets based on the cumulative probability
        # of our reduced scenarios.
        cum_p = np.cumsum(current_probs)

        # Map cumulative probs to indices in the sorted history array
        # e.g., if p=[0.5, 0.5], split history at index N/2
        split_indices = np.searchsorted(np.linspace(0, 1, self.N), cum_p)

        start_idx = 0
        for i in range(self.M):
            end_idx = split_indices[i] if i < self.M - 1 else self.N

            # Safety check for empty buckets
            if end_idx > start_idx:
                # The "atom" location is the average of the history in this bucket
                new_support[i] = np.mean(self.target[start_idx:end_idx])
            else:
                # If a bucket is empty, keep previous support or interpolate
                new_support[i] = self.target[min(start_idx, self.N - 1)]

            start_idx = end_idx

        return np.sort(new_support)

    def reduce(self, max_iter: int = 10, tol: float = 1e-4) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main Iterative Loop[cite: 355].
        Alternates between optimizing probabilities and locations.

        Returns:
            (scenarios, probabilities)
        """
        # Initialization: Spread support points evenly across range
        current_support = np.linspace(np.min(self.target), np.max(self.target), self.M)
        current_probs = np.ones(self.M) / self.M

        for k in range(max_iter):
            prev_support = current_support.copy()

            # Alternating minimization
            current_probs = self._step_optimize_probabilities(current_support)
            current_support = self._step_optimize_locations(current_probs)

            # Check convergence (change in support locations)
            diff = np.linalg.norm(current_support - prev_support)
            if diff < tol:
                logger.info(f"Scenario reduction converged in {k} iterations.")
                break

        return current_support, current_probs