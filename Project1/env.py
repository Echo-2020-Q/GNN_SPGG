from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union

import numpy as np


GraphInput = Union[
    Mapping[int, Iterable[int]],
    Sequence[Iterable[int]],
    Sequence[Tuple[int, int]],
]
Observation = Dict[str, np.ndarray]


@dataclass
class RewardConfig:
    lambda_payoff: float = 1.0
    lambda_cooperation: float = 0.0
    lambda_gini: float = 0.0
    epsilon: float = 1e-8


@dataclass
class SPGGConfig:
    alpha: float = 0.0
    r: float = 1.0
    p_max: float = 10.0
    resource_consumption_mode: str = "fixed"
    resource_consumption_fixed_mode: str = "constant"
    resource_consumption_fixed: float = 0.0
    resource_consumption_degree_multiplier: float = 0.0
    resource_consumption_rate: float = 0.0
    resource_consumption_threshold: float = 0.0
    strategy_update_rule: str = "fermi"
    beta: float = 1.0
    q_learning_rate: float = 0.1
    q_learning_discount: float = 0.95
    q_learning_epsilon: float = 0.05
    q_learning_initial_value: float = 0.0
    episode_length: int = 100
    initial_resource: float = 10.0
    initial_cooperation_prob: float = 0.5
    num_nodes: int | None = None
    target_mean_degree: float | None = None
    reward: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if self.p_max <= 0.0:
            raise ValueError("p_max must be positive.")
        if self.resource_consumption_mode not in {"fixed", "proportional", "piecewise_linear"}:
            raise ValueError(
                "resource_consumption_mode must be one of {'fixed', 'proportional', 'piecewise_linear'}."
            )
        if self.resource_consumption_fixed_mode not in {"constant", "degree_scaled"}:
            raise ValueError(
                "resource_consumption_fixed_mode must be one of {'constant', 'degree_scaled'}."
            )
        if self.resource_consumption_fixed < 0.0:
            raise ValueError("resource_consumption_fixed must be non-negative.")
        if self.resource_consumption_degree_multiplier < 0.0:
            raise ValueError("resource_consumption_degree_multiplier must be non-negative.")
        if self.resource_consumption_rate < 0.0:
            raise ValueError("resource_consumption_rate must be non-negative.")
        if self.resource_consumption_threshold < 0.0:
            raise ValueError("resource_consumption_threshold must be non-negative.")
        if self.strategy_update_rule not in {"fermi", "q_learning", "q_learning_2x2", "imitate_best"}:
            raise ValueError(
                "strategy_update_rule must be one of {'fermi', 'q_learning', 'q_learning_2x2', 'imitate_best'}."
            )
        if not 0.0 <= self.q_learning_rate <= 1.0:
            raise ValueError("q_learning_rate must be in [0, 1].")
        if not 0.0 <= self.q_learning_discount <= 1.0:
            raise ValueError("q_learning_discount must be in [0, 1].")
        if not 0.0 <= self.q_learning_epsilon <= 1.0:
            raise ValueError("q_learning_epsilon must be in [0, 1].")
        if self.episode_length <= 0:
            raise ValueError("episode_length must be positive.")
        if not 0.0 <= self.initial_cooperation_prob <= 1.0:
            raise ValueError("initial_cooperation_prob must be in [0, 1].")
        if self.num_nodes is not None and self.num_nodes <= 0:
            raise ValueError("num_nodes must be positive when provided.")
        if self.target_mean_degree is not None and self.target_mean_degree < 0.0:
            raise ValueError("target_mean_degree must be non-negative when provided.")


@dataclass(frozen=True)
class GraphData:
    num_nodes: int
    neighbors: tuple[tuple[int, ...], ...]
    degrees: np.ndarray
    local_mask: np.ndarray
    adjacency_matrix: np.ndarray


def _empty_adjacency(num_nodes: int) -> Dict[int, set[int]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    return {node: set() for node in range(num_nodes)}


def _finalize_adjacency(adjacency: Mapping[int, Iterable[int]]) -> Dict[int, List[int]]:
    return {node: sorted(int(neighbor) for neighbor in neighbors) for node, neighbors in adjacency.items()}


def make_grid_graph(rows: int, cols: int, periodic: bool = False) -> Dict[int, List[int]]:
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive.")

    adjacency: Dict[int, set[int]] = {index: set() for index in range(rows * cols)}

    def node_id(row: int, col: int) -> int:
        return row * cols + col

    offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
    for row in range(rows):
        for col in range(cols):
            source = node_id(row, col)
            for d_row, d_col in offsets:
                nbr_row = row + d_row
                nbr_col = col + d_col
                if periodic:
                    nbr_row %= rows
                    nbr_col %= cols
                elif not (0 <= nbr_row < rows and 0 <= nbr_col < cols):
                    continue

                target = node_id(nbr_row, nbr_col)
                if target != source:
                    adjacency[source].add(target)
                    adjacency[target].add(source)

    return {node: sorted(neighbors) for node, neighbors in adjacency.items()}


def make_random_regular_graph(
    num_nodes: int,
    degree: int,
    seed: int | None = None,
    max_attempts: int = 1_000,
) -> Dict[int, List[int]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if degree < 0 or degree >= num_nodes:
        raise ValueError("degree must satisfy 0 <= degree < num_nodes.")
    if (num_nodes * degree) % 2 != 0:
        raise ValueError("num_nodes * degree must be even for a regular graph.")
    if degree == 0:
        return _finalize_adjacency(_empty_adjacency(num_nodes))
    if degree == num_nodes - 1:
        complete = _empty_adjacency(num_nodes)
        for source in range(num_nodes):
            for target in range(source + 1, num_nodes):
                complete[source].add(target)
                complete[target].add(source)
        return _finalize_adjacency(complete)

    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        adjacency = _empty_adjacency(num_nodes)
        stubs = np.repeat(np.arange(num_nodes), degree)
        rng.shuffle(stubs)
        valid = True

        for index in range(0, stubs.size, 2):
            source = int(stubs[index])
            target = int(stubs[index + 1])
            if source == target or target in adjacency[source]:
                valid = False
                break
            adjacency[source].add(target)
            adjacency[target].add(source)

        if valid and all(len(neighbors) == degree for neighbors in adjacency.values()):
            return _finalize_adjacency(adjacency)

    raise RuntimeError("Failed to sample a simple random regular graph. Increase max_attempts.")


def make_erdos_renyi_graph(
    num_nodes: int,
    edge_prob: float,
    seed: int | None = None,
) -> Dict[int, List[int]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError("edge_prob must be in [0, 1].")

    rng = np.random.default_rng(seed)
    adjacency = _empty_adjacency(num_nodes)
    for source in range(num_nodes):
        for target in range(source + 1, num_nodes):
            if rng.random() < edge_prob:
                adjacency[source].add(target)
                adjacency[target].add(source)
    return _finalize_adjacency(adjacency)


def make_watts_strogatz_graph(
    num_nodes: int,
    degree: int,
    rewiring_prob: float,
    seed: int | None = None,
) -> Dict[int, List[int]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if degree < 0 or degree >= num_nodes:
        raise ValueError("degree must satisfy 0 <= degree < num_nodes.")
    if degree % 2 != 0:
        raise ValueError("Watts-Strogatz degree must be even.")
    if not 0.0 <= rewiring_prob <= 1.0:
        raise ValueError("rewiring_prob must be in [0, 1].")

    rng = np.random.default_rng(seed)
    adjacency = _empty_adjacency(num_nodes)
    half_degree = degree // 2

    for source in range(num_nodes):
        for offset in range(1, half_degree + 1):
            target = (source + offset) % num_nodes
            adjacency[source].add(target)
            adjacency[target].add(source)

    for source in range(num_nodes):
        for offset in range(1, half_degree + 1):
            target = (source + offset) % num_nodes
            if rng.random() >= rewiring_prob:
                continue
            if target not in adjacency[source]:
                continue

            adjacency[source].remove(target)
            adjacency[target].remove(source)

            forbidden = adjacency[source] | {source}
            candidates = [node for node in range(num_nodes) if node not in forbidden]
            if not candidates:
                adjacency[source].add(target)
                adjacency[target].add(source)
                continue

            new_target = int(rng.choice(candidates))
            adjacency[source].add(new_target)
            adjacency[new_target].add(source)

    return _finalize_adjacency(adjacency)


def make_barabasi_albert_graph(
    num_nodes: int,
    attachments_per_new_node: int,
    seed: int | None = None,
) -> Dict[int, List[int]]:
    if num_nodes <= 0:
        raise ValueError("num_nodes must be positive.")
    if attachments_per_new_node <= 0:
        raise ValueError("attachments_per_new_node must be positive.")
    if attachments_per_new_node >= num_nodes:
        raise ValueError("attachments_per_new_node must be smaller than num_nodes.")

    rng = np.random.default_rng(seed)
    initial_size = attachments_per_new_node + 1
    adjacency = _empty_adjacency(num_nodes)

    for source in range(initial_size):
        for target in range(source + 1, initial_size):
            adjacency[source].add(target)
            adjacency[target].add(source)

    degree_targets: list[int] = []
    for node in range(initial_size):
        degree_targets.extend([node] * len(adjacency[node]))

    for new_node in range(initial_size, num_nodes):
        chosen_targets: set[int] = set()
        while len(chosen_targets) < attachments_per_new_node:
            chosen_targets.add(int(rng.choice(degree_targets)))

        for target in chosen_targets:
            adjacency[new_node].add(target)
            adjacency[target].add(new_node)

        degree_targets.extend([new_node] * attachments_per_new_node)
        for target in chosen_targets:
            degree_targets.append(target)

    return _finalize_adjacency(adjacency)


def gini_coefficient(values: np.ndarray, epsilon: float = 1e-8) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        values = values.reshape(-1)
    if values.size == 0:
        return 0.0

    total = float(values.sum())
    if total <= 0.0:
        return 0.0

    sorted_values = np.sort(values)
    rank = np.arange(1, sorted_values.size + 1, dtype=np.float64)
    numerator = float(np.dot((2.0 * rank) - sorted_values.size - 1.0, sorted_values))
    denominator = (sorted_values.size * total) + (0.5 * float(epsilon))
    return float(numerator / denominator)


class SPGGEnv:
    """Gym-like environment for the SPGG model with centralized allocation."""

    def __init__(self, config: SPGGConfig, graph: GraphInput):
        self.config = config
        self.graph = self._normalize_graph(graph, config.num_nodes)
        self.num_nodes = self.graph.num_nodes
        self.graph_mean_degree = float(self.graph.degrees.mean()) if self.num_nodes > 0 else 0.0
        self.target_mean_degree = (
            float(config.target_mean_degree)
            if config.target_mean_degree is not None
            else self.graph_mean_degree
        )
        self.resource_norm_reference = self._compute_resource_norm_reference()
        self._refresh_static_caches()
        self.rng = np.random.default_rng()

        self._step_count = 0
        self._nominal_strategies = np.zeros(self.num_nodes, dtype=np.int8)
        self._resources = np.zeros(self.num_nodes, dtype=np.float64)
        self._q_values = self._initialize_q_values()
        self._q_learning_previous_actions = np.zeros(self.num_nodes, dtype=np.int8)
        self._current_observation: Observation | None = None

    def reset(
        self,
        initial_resources: float | Sequence[float] | np.ndarray | None = None,
        initial_strategies: int | Sequence[int] | np.ndarray | None = None,
        seed: int | None = None,
    ) -> Observation:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._step_count = 0
        self._resources = self._coerce_resource_vector(initial_resources)
        self._nominal_strategies = self._coerce_strategy_vector(initial_strategies)
        self._q_values = self._initialize_q_values()
        self._current_observation = self._precompute_observation(self._nominal_strategies, self._resources)
        self._q_learning_previous_actions = self._current_observation["x_actual"].astype(np.int8, copy=True)
        return self._copy_observation(self._current_observation)

    def step(self, allocation_matrix: np.ndarray | Sequence[Sequence[float]]) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self._current_observation is None:
            raise RuntimeError("Call reset() before step().")

        observation = self._current_observation
        allocation = self._validate_allocation(allocation_matrix)

        transfer_matrix = allocation * observation["pool_grown"][:, None]
        income = transfer_matrix.sum(axis=0)
        resources_after_exchange = observation["resources"] - observation["investment"] + income
        consumption = self._compute_effective_consumption(observation["resources"], resources_after_exchange)
        payoff = income - observation["investment"]
        next_resources = resources_after_exchange - consumption

        next_nominal = self._update_nominal_strategies(
            observation["x_nominal"],
            observation["x_actual"].astype(np.int8, copy=False),
            payoff,
        )

        self._step_count += 1
        done = self._step_count >= self.config.episode_length

        next_observation = self._precompute_observation(next_nominal, next_resources)
        next_gini = float(np.asarray(next_observation["gini"]).item())
        next_actual_cooperation = float(next_observation["x_actual"].mean())
        current_actual_cooperation = float(observation["x_actual"].mean())
        reward = self._planner_reward(
            payoff,
            next_observation["x_actual"],
            next_resources,
            next_resource_gini=next_gini,
        )

        info = {
            "allocation_matrix": allocation.copy(),
            "transfer_matrix": transfer_matrix.copy(),
            "income": income.copy(),
            "consumption": consumption.copy(),
            "payoff": payoff.copy(),
            "gini": next_gini,
            "actual_cooperation_rate": next_actual_cooperation,
            "actual_cooperation_rate_current": current_actual_cooperation,
            "nominal_strategies_next": next_nominal.copy(),
            "resources_next": next_resources.copy(),
            "reward_components": {
                "mean_consumption": float(consumption.mean()),
                "mean_payoff": float(payoff.mean()),
                "actual_cooperation_rate_next": next_actual_cooperation,
                "gini_next_resources": next_gini,
            },
        }

        self._nominal_strategies = next_nominal
        self._resources = next_resources
        self._current_observation = next_observation
        return self._copy_observation(next_observation), reward, done, info

    def get_observation(self) -> Observation:
        if self._current_observation is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")
        return self._copy_observation(self._current_observation)

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "graph": {node: list(neighbors) for node, neighbors in enumerate(self.graph.neighbors)},
            "rng_state": self.rng.bit_generator.state,
            "step_count": int(self._step_count),
            "nominal_strategies": self._nominal_strategies.copy(),
            "resources": self._resources.copy(),
            "q_values": self._q_values.copy(),
            "q_learning_previous_actions": self._q_learning_previous_actions.copy(),
            "current_observation": (
                self._copy_observation(self._current_observation)
                if self._current_observation is not None
                else None
            ),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        config_payload = dict(state_dict["config"])
        reward_payload = dict(config_payload.pop("reward"))
        self.config = SPGGConfig(
            reward=RewardConfig(**reward_payload),
            **config_payload,
        )
        self.graph = self._normalize_graph(state_dict["graph"], self.config.num_nodes)
        self.num_nodes = self.graph.num_nodes
        self.graph_mean_degree = float(self.graph.degrees.mean()) if self.num_nodes > 0 else 0.0
        self.target_mean_degree = (
            float(self.config.target_mean_degree)
            if self.config.target_mean_degree is not None
            else self.graph_mean_degree
        )
        self.resource_norm_reference = self._compute_resource_norm_reference()
        self._refresh_static_caches()
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state_dict["rng_state"]
        self._step_count = int(state_dict["step_count"])
        self._nominal_strategies = np.asarray(state_dict["nominal_strategies"], dtype=np.int8).copy()
        self._resources = np.asarray(state_dict["resources"], dtype=np.float64).copy()
        self._q_values = np.asarray(state_dict["q_values"], dtype=np.float64).copy()
        self._q_learning_previous_actions = np.asarray(
            state_dict["q_learning_previous_actions"],
            dtype=np.int8,
        ).copy()
        current_observation = state_dict.get("current_observation")
        self._current_observation = (
            self._copy_observation(current_observation)
            if current_observation is not None
            else None
        )

    def _coerce_resource_vector(
        self,
        initial_resources: float | Sequence[float] | np.ndarray | None,
    ) -> np.ndarray:
        if initial_resources is None:
            return np.full(self.num_nodes, self.config.initial_resource, dtype=np.float64)

        resources = np.asarray(initial_resources, dtype=np.float64)
        if resources.ndim == 0:
            if float(resources) < 0.0:
                raise ValueError("Initial resource scalar must be non-negative.")
            return np.full(self.num_nodes, float(resources), dtype=np.float64)
        if resources.shape != (self.num_nodes,):
            raise ValueError(f"Initial resources must have shape ({self.num_nodes},).")
        if np.any(resources < 0.0):
            raise ValueError("Initial resources must be non-negative.")
        return resources.astype(np.float64, copy=True)

    def _coerce_strategy_vector(
        self,
        initial_strategies: int | Sequence[int] | np.ndarray | None,
    ) -> np.ndarray:
        if initial_strategies is None:
            samples = self.rng.random(self.num_nodes) < self.config.initial_cooperation_prob
            return samples.astype(np.int8)

        strategies = np.asarray(initial_strategies, dtype=np.int8)
        if strategies.ndim == 0:
            if int(strategies) not in (0, 1):
                raise ValueError("Initial strategy scalar must be 0 or 1.")
            return np.full(self.num_nodes, int(strategies), dtype=np.int8)
        if strategies.shape != (self.num_nodes,):
            raise ValueError(f"Initial strategies must have shape ({self.num_nodes},).")
        if np.any((strategies != 0) & (strategies != 1)):
            raise ValueError("Strategies must be binary.")
        return strategies.astype(np.int8, copy=True)

    def _precompute_observation(self, nominal_strategies: np.ndarray, resources: np.ndarray) -> Observation:
        actual_strategies = nominal_strategies.astype(np.float64, copy=False) * (
            resources >= self._thresholds_float64
        ).astype(np.float64, copy=False)

        investment_base = self._thresholds_float64 + self.config.alpha * (resources - self._thresholds_float64)
        investment = actual_strategies * np.minimum(resources, np.maximum(0.0, investment_base))
        unit_investment = investment / self._thresholds_float64

        pool_raw = self._local_mask_float64 @ unit_investment
        pool_grown = np.minimum((1.0 + self.config.r) * pool_raw, self.config.p_max)
        pool_raw_norm = np.minimum(pool_raw, self.config.p_max) / self.config.p_max
        resource_norm = resources / self.resource_norm_reference
        strategy_norm = np.divide(
            investment,
            resources,
            out=np.zeros_like(resources, dtype=np.float64),
            where=resources > 0.0,
        )
        resource_gini = gini_coefficient(resources, self.config.reward.epsilon)

        return {
            "x_nominal": nominal_strategies.astype(np.int8, copy=True),
            "x_actual": actual_strategies.astype(np.float64, copy=True),
            "resources": resources.astype(np.float64, copy=True),
            "investment": investment.astype(np.float64, copy=True),
            "unit_investment": unit_investment.astype(np.float64, copy=True),
            "pool_raw": pool_raw.astype(np.float64, copy=True),
            "pool_grown": pool_grown.astype(np.float64, copy=True),
            "degrees": self._degrees_int64.copy(),
            "pool_raw_norm": pool_raw_norm.astype(np.float64, copy=True),
            "resource_norm": resource_norm.astype(np.float64, copy=True),
            "degree_norm": self._degree_norm_float64.copy(),
            "strategy_norm": strategy_norm.astype(np.float64, copy=True),
            "gini": np.asarray(resource_gini, dtype=np.float64),
            "p_max": self._p_max_scalar.copy(),
            "local_mask": self._local_mask_bool.copy(),
        }

    def _validate_allocation(self, allocation_matrix: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        allocation = np.asarray(allocation_matrix, dtype=np.float64)
        if allocation.shape != (self.num_nodes, self.num_nodes):
            raise ValueError(
                f"Allocation matrix must have shape ({self.num_nodes}, {self.num_nodes})."
            )
        if np.any(allocation < -1e-8):
            raise ValueError("Allocation matrix cannot contain negative entries.")
        invalid_entries = np.abs(allocation[~self.graph.local_mask]) > 1e-8
        if np.any(invalid_entries):
            raise ValueError("Allocation matrix may only allocate to self or graph neighbors.")

        allocation = allocation.copy()
        allocation[~self.graph.local_mask] = 0.0
        row_sums = allocation.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("Each pool row must sum to 1 across its local neighborhood.")
        return allocation

    def _compute_effective_consumption(
        self,
        resources: np.ndarray,
        resources_after_exchange: np.ndarray,
    ) -> np.ndarray:
        desired_consumption = self._compute_nominal_consumption(resources)
        return np.minimum(desired_consumption, resources_after_exchange)

    def _compute_nominal_consumption(self, resources: np.ndarray) -> np.ndarray:
        resources = np.asarray(resources, dtype=np.float64)
        fixed_component = self._compute_fixed_consumption_component()
        if self.config.resource_consumption_mode == "fixed":
            return fixed_component
        if self.config.resource_consumption_mode == "proportional":
            return self.config.resource_consumption_rate * resources
        if self.config.resource_consumption_mode == "piecewise_linear":
            excess_resources = np.maximum(resources - self.config.resource_consumption_threshold, 0.0)
            return fixed_component + (self.config.resource_consumption_rate * excess_resources)
        raise RuntimeError(
            "Unsupported resource consumption mode: {0}".format(self.config.resource_consumption_mode)
        )

    def _compute_fixed_consumption_component(self) -> np.ndarray:
        return self._fixed_consumption_component_float64

    def _compute_fixed_consumption_component_array(self) -> np.ndarray:
        if self.config.resource_consumption_fixed_mode == "constant":
            return np.full(self.num_nodes, self.config.resource_consumption_fixed, dtype=np.float64)
        if self.config.resource_consumption_fixed_mode == "degree_scaled":
            return self.config.resource_consumption_degree_multiplier * self._degrees_float64
        raise RuntimeError(
            "Unsupported resource_consumption_fixed_mode: {0}".format(self.config.resource_consumption_fixed_mode)
        )

    def _compute_resource_norm_reference(self) -> float:
        denominator = self.config.alpha + self.config.resource_consumption_rate
        if denominator <= 1e-8:
            return max(self.config.p_max, 1e-8)

        theoretical_max = (
            self.config.p_max - ((1.0 - self.config.alpha) * (self.graph_mean_degree + 1.0))
        ) / denominator
        if theoretical_max <= 1e-8:
            return max(self.config.p_max, 1e-8)
        return float(theoretical_max)

    def _synchronous_fermi_update(self, nominal_strategies: np.ndarray, payoff: np.ndarray) -> np.ndarray:
        next_nominal = nominal_strategies.astype(np.int8, copy=True)
        for node, neighbors in enumerate(self.graph.neighbors):
            if not neighbors:
                continue

            sampled_neighbor = int(self.rng.choice(neighbors))
            payoff_delta = float(payoff[sampled_neighbor] - payoff[node])
            scaled_delta = np.clip(self.config.beta * payoff_delta, -60.0, 60.0)
            adoption_probability = 1.0 / (1.0 + np.exp(-scaled_delta))
            if self.rng.random() < adoption_probability:
                next_nominal[node] = nominal_strategies[sampled_neighbor]
        return next_nominal

    def _update_nominal_strategies(
        self,
        nominal_strategies: np.ndarray,
        actual_strategies: np.ndarray,
        payoff: np.ndarray,
    ) -> np.ndarray:
        if self.config.strategy_update_rule == "fermi":
            return self._synchronous_fermi_update(nominal_strategies, payoff)
        if self.config.strategy_update_rule == "q_learning":
            return self._q_learning_update(actual_strategies, payoff)
        if self.config.strategy_update_rule == "q_learning_2x2":
            return self._q_learning_2x2_update(actual_strategies, payoff)
        if self.config.strategy_update_rule == "imitate_best":
            return self._imitate_best_update(nominal_strategies, payoff)
        raise RuntimeError("Unsupported strategy update rule: {0}".format(self.config.strategy_update_rule))

    def _q_learning_update(self, actual_strategies: np.ndarray, payoff: np.ndarray) -> np.ndarray:
        previous_q_values = self._q_values.copy()
        next_q_values = previous_q_values.copy()
        next_nominal = actual_strategies.astype(np.int8, copy=True)

        for node in range(self.num_nodes):
            action = int(actual_strategies[node])
            best_future_value = float(previous_q_values[node].max())
            td_target = float(payoff[node]) + (self.config.q_learning_discount * best_future_value)
            next_q_values[node, action] = (
                (1.0 - self.config.q_learning_rate) * previous_q_values[node, action]
                + (self.config.q_learning_rate * td_target)
            )
            next_nominal[node] = self._select_q_learning_action(next_q_values[node])

        self._q_values = next_q_values
        return next_nominal

    def _q_learning_2x2_update(self, actual_strategies: np.ndarray, payoff: np.ndarray) -> np.ndarray:
        previous_q_values = self._q_values.copy()
        next_q_values = previous_q_values.copy()
        state_actions = self._q_learning_previous_actions.astype(np.int8, copy=True)
        next_nominal = actual_strategies.astype(np.int8, copy=True)

        for node in range(self.num_nodes):
            state = int(state_actions[node])
            action = int(actual_strategies[node])
            next_state = action
            best_future_value = float(previous_q_values[node, next_state].max())
            td_target = float(payoff[node]) + (self.config.q_learning_discount * best_future_value)
            next_q_values[node, state, action] = (
                (1.0 - self.config.q_learning_rate) * previous_q_values[node, state, action]
                + (self.config.q_learning_rate * td_target)
            )
            next_nominal[node] = self._select_q_learning_action(next_q_values[node, next_state])

        self._q_values = next_q_values
        self._q_learning_previous_actions = actual_strategies.astype(np.int8, copy=True)
        return next_nominal

    def _imitate_best_update(self, nominal_strategies: np.ndarray, payoff: np.ndarray) -> np.ndarray:
        next_nominal = nominal_strategies.astype(np.int8, copy=True)
        for node, neighbors in enumerate(self.graph.neighbors):
            candidate_nodes = np.asarray((node, *neighbors), dtype=np.int64)
            candidate_payoff = payoff[candidate_nodes]
            best_payoff = float(candidate_payoff.max())
            best_candidates = candidate_nodes[np.isclose(candidate_payoff, best_payoff)]
            selected = int(self.rng.choice(best_candidates))
            next_nominal[node] = nominal_strategies[selected]
        return next_nominal

    def _select_q_learning_action(self, q_values: np.ndarray) -> np.int8:
        if self.rng.random() < self.config.q_learning_epsilon:
            return np.int8(self.rng.integers(0, 2))

        best_value = float(q_values.max())
        best_actions = np.flatnonzero(np.isclose(q_values, best_value))
        return np.int8(self.rng.choice(best_actions))

    def _initialize_q_values(self) -> np.ndarray:
        q_shape = (self.num_nodes, 2, 2) if self.config.strategy_update_rule == "q_learning_2x2" else (self.num_nodes, 2)
        return np.full(
            q_shape,
            self.config.q_learning_initial_value,
            dtype=np.float64,
        )

    def _planner_reward(
        self,
        payoff: np.ndarray,
        next_actual_strategies: np.ndarray,
        next_resources: np.ndarray,
        next_resource_gini: float | None = None,
    ) -> float:
        reward_config = self.config.reward
        gini_penalty = 0.0
        if reward_config.lambda_gini != 0.0:
            resource_gini = (
                float(next_resource_gini)
                if next_resource_gini is not None
                else gini_coefficient(next_resources, reward_config.epsilon)
            )
            gini_penalty = reward_config.lambda_gini * resource_gini
        return float(
            reward_config.lambda_payoff * payoff.mean()
            + reward_config.lambda_cooperation * next_actual_strategies.mean()
            - gini_penalty
        )

    def _refresh_static_caches(self) -> None:
        self._degrees_int64 = self.graph.degrees.astype(np.int64, copy=True)
        self._degrees_float64 = self.graph.degrees.astype(np.float64, copy=True)
        self._thresholds_float64 = self._degrees_float64 + 1.0
        self._local_mask_bool = self.graph.local_mask.copy()
        self._local_mask_float64 = self._local_mask_bool.astype(np.float64, copy=True)
        degree_reference = max(self.target_mean_degree, 1e-8)
        self._degree_norm_float64 = (self._degrees_float64 - degree_reference) / degree_reference
        self._fixed_consumption_component_float64 = self._compute_fixed_consumption_component_array()
        self._p_max_scalar = np.asarray(self.config.p_max, dtype=np.float64)

    @staticmethod
    def _copy_observation(observation: Observation) -> Observation:
        return {key: value.copy() for key, value in observation.items()}

    @staticmethod
    def _normalize_graph(graph: GraphInput, num_nodes_hint: int | None) -> GraphData:
        neighbor_sets: list[set[int]]

        if isinstance(graph, Mapping):
            all_nodes = set(graph.keys())
            for neighbors in graph.values():
                all_nodes.update(int(node) for node in neighbors)

            if not all_nodes and num_nodes_hint is None:
                raise ValueError("Cannot infer graph size from an empty mapping.")

            inferred_num_nodes = (max(all_nodes) + 1) if all_nodes else 0
            num_nodes = max(inferred_num_nodes, num_nodes_hint or 0)
            neighbor_sets = [set() for _ in range(num_nodes)]
            for source, neighbors in graph.items():
                source_id = int(source)
                if source_id < 0 or source_id >= num_nodes:
                    raise ValueError("Node ids must be within [0, num_nodes).")
                for target in neighbors:
                    target_id = int(target)
                    SPGGEnv._add_undirected_edge(neighbor_sets, source_id, target_id)
        else:
            graph_items = list(graph)
            is_edge_list = SPGGEnv._looks_like_edge_list(graph_items)
            if is_edge_list:
                if graph_items:
                    max_node = max(max(int(u), int(v)) for u, v in graph_items)
                    inferred_num_nodes = max_node + 1
                else:
                    if num_nodes_hint is None:
                        raise ValueError("Empty edge list requires num_nodes in config.")
                    inferred_num_nodes = 0
                num_nodes = max(inferred_num_nodes, num_nodes_hint or 0)
                neighbor_sets = [set() for _ in range(num_nodes)]
                for source, target in graph_items:
                    SPGGEnv._add_undirected_edge(neighbor_sets, int(source), int(target))
            else:
                num_nodes = max(len(graph_items), num_nodes_hint or 0)
                neighbor_sets = [set() for _ in range(num_nodes)]
                for source, neighbors in enumerate(graph_items):
                    for target in neighbors:
                        SPGGEnv._add_undirected_edge(neighbor_sets, source, int(target))

        neighbors = tuple(tuple(sorted(neighbor_set)) for neighbor_set in neighbor_sets)
        degrees = np.asarray([len(neighbor_set) for neighbor_set in neighbors], dtype=np.int64)
        local_mask = np.eye(len(neighbors), dtype=bool)
        adjacency_matrix = np.zeros((len(neighbors), len(neighbors)), dtype=np.float64)
        for source, neighbor_list in enumerate(neighbors):
            for target in neighbor_list:
                local_mask[source, target] = True
                adjacency_matrix[source, target] = 1.0

        return GraphData(
            num_nodes=len(neighbors),
            neighbors=neighbors,
            degrees=degrees,
            local_mask=local_mask,
            adjacency_matrix=adjacency_matrix,
        )

    @staticmethod
    def _looks_like_edge_list(graph_items: list[Any]) -> bool:
        if not graph_items:
            return True

        for item in graph_items:
            if not isinstance(item, tuple):
                return False
            if len(item) != 2:
                return False
            if not all(isinstance(entry, (int, np.integer)) for entry in item):
                return False
        return True

    @staticmethod
    def _add_undirected_edge(neighbor_sets: list[set[int]], source: int, target: int) -> None:
        num_nodes = len(neighbor_sets)
        if source < 0 or target < 0 or source >= num_nodes or target >= num_nodes:
            raise ValueError("Node ids must be within [0, num_nodes).")
        if source == target:
            return
        neighbor_sets[source].add(target)
        neighbor_sets[target].add(source)
