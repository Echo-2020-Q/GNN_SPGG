from __future__ import annotations

import numpy as np

from Project1.env import Observation


class UniformAllocationPolicy:
    """Allocate each pool's resources uniformly over itself and its neighbors."""

    def allocate(self, observation: Observation) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=np.float64)
        row_sums = local_mask.sum(axis=1, keepdims=True)
        return local_mask / row_sums

    def __call__(self, observation: Observation) -> np.ndarray:
        return self.allocate(observation)


class ProportionalContributionPolicy:
    """Allocate resources in proportion to local unit investment, with uniform fallback."""

    def allocate(self, observation: Observation) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        unit_investment = np.asarray(observation["unit_investment"], dtype=np.float64)

        allocation = np.zeros_like(local_mask, dtype=np.float64)
        weighted_contributions = local_mask.astype(np.float64) * unit_investment[None, :]
        row_sums = weighted_contributions.sum(axis=1, keepdims=True)

        positive_rows = row_sums.squeeze(-1) > 0.0
        if np.any(positive_rows):
            allocation[positive_rows] = weighted_contributions[positive_rows] / row_sums[positive_rows]

        if np.any(~positive_rows):
            fallback = local_mask[~positive_rows].astype(np.float64)
            allocation[~positive_rows] = fallback / fallback.sum(axis=1, keepdims=True)

        return allocation

    def __call__(self, observation: Observation) -> np.ndarray:
        return self.allocate(observation)
