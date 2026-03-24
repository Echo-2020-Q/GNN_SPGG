from __future__ import annotations

import numpy as np

from Project1.env import Observation


def _uniform_allocation(local_mask: np.ndarray) -> np.ndarray:
    weights = np.asarray(local_mask, dtype=np.float64)
    row_sums = weights.sum(axis=1, keepdims=True)
    return weights / row_sums


def _proportional_allocation(local_mask: np.ndarray, unit_investment: np.ndarray) -> np.ndarray:
    mask = np.asarray(local_mask, dtype=bool)
    contributions = np.asarray(unit_investment, dtype=np.float64)

    allocation = np.zeros_like(mask, dtype=np.float64)
    weighted_contributions = mask.astype(np.float64) * contributions[None, :]
    row_sums = weighted_contributions.sum(axis=1, keepdims=True)

    positive_rows = row_sums.squeeze(-1) > 0.0
    if np.any(positive_rows):
        allocation[positive_rows] = weighted_contributions[positive_rows] / row_sums[positive_rows]

    if np.any(~positive_rows):
        allocation[~positive_rows] = _uniform_allocation(mask[~positive_rows])

    return allocation


class UniformAllocationPolicy:
    """Allocate each pool's resources uniformly over itself and its neighbors."""

    def allocate(self, observation: Observation) -> np.ndarray:
        return _uniform_allocation(np.asarray(observation["local_mask"], dtype=bool))

    def __call__(self, observation: Observation) -> np.ndarray:
        return self.allocate(observation)


class ProportionalContributionPolicy:
    """Allocate resources in proportion to local unit investment, with uniform fallback."""

    def allocate(self, observation: Observation) -> np.ndarray:
        return _proportional_allocation(
            local_mask=np.asarray(observation["local_mask"], dtype=bool),
            unit_investment=np.asarray(observation["unit_investment"], dtype=np.float64),
        )

    def __call__(self, observation: Observation) -> np.ndarray:
        return self.allocate(observation)


class ConstantMixAllocationPolicy:
    """Blend uniform and proportional allocation with a constant omega."""

    def __init__(self, omega: float = 0.5):
        if not 0.0 <= omega <= 1.0:
            raise ValueError("omega must be in [0, 1].")
        self.omega = float(omega)

    def allocate(self, observation: Observation) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        unit_investment = np.asarray(observation["unit_investment"], dtype=np.float64)
        uniform = _uniform_allocation(local_mask)
        proportional = _proportional_allocation(local_mask, unit_investment)
        return (self.omega * uniform) + ((1.0 - self.omega) * proportional)

    def __call__(self, observation: Observation) -> np.ndarray:
        return self.allocate(observation)


class PoolPowerMixAllocationPolicy:
    """Blend uniform and proportional allocation with omega_i = (clip(pool_raw_i, 0, p_max) / p_max)^k."""

    def __init__(self, power_k: float = 19.0):
        if power_k < 0.0:
            raise ValueError("power_k must be non-negative.")
        self.power_k = float(power_k)

    def allocate(self, observation: Observation) -> np.ndarray:
        local_mask = np.asarray(observation["local_mask"], dtype=bool)
        unit_investment = np.asarray(observation["unit_investment"], dtype=np.float64)
        pool_raw = np.asarray(observation["pool_raw"], dtype=np.float64)
        p_max = float(np.asarray(observation["p_max"], dtype=np.float64))
        if p_max <= 0.0:
            raise ValueError("observation['p_max'] must be positive.")

        uniform = _uniform_allocation(local_mask)
        proportional = _proportional_allocation(local_mask, unit_investment)
        omega = np.power(np.clip(pool_raw, 0.0, p_max) / p_max, self.power_k)
        return (omega[:, None] * uniform) + ((1.0 - omega)[:, None] * proportional)

    def __call__(self, observation: Observation) -> np.ndarray:
        return self.allocate(observation)
