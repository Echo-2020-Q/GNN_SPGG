from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
from multiprocessing.connection import wait
from multiprocessing import shared_memory
from time import perf_counter
import traceback
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from Project1.env import (
    SPGGConfig,
    SPGGEnv,
    make_barabasi_albert_graph,
    make_erdos_renyi_graph,
    make_random_regular_graph,
    make_watts_strogatz_graph,
)
from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig
from Project1.policies.rule_based import (
    ConstantMixAllocationPolicy,
    PoolPowerMixAllocationPolicy,
    ProportionalContributionPolicy,
    UniformAllocationPolicy,
)

from .config import DomainRandomizationConfig, GraphTD3Config, WorkerConfig
from .data import (
    REPLAY_OBSERVATION_DTYPES,
    TensorActionRecord,
    TensorReplayActionRecord,
    TensorReplayBatch,
    TensorTransition,
    stack_tensor_transitions,
)
from .exploration import LogitSpaceExplorer


def _clone_graph_from_env(env: SPGGEnv) -> dict[int, list[int]]:
    return {node: list(neighbors) for node, neighbors in enumerate(env.graph.neighbors)}


def _clone_graph_dict(graph: Mapping[int, list[int] | tuple[int, ...]]) -> dict[int, list[int]]:
    return {int(node): [int(neighbor) for neighbor in neighbors] for node, neighbors in graph.items()}


def _copy_module_state_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _serialize_module_state(state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy() for key, value in state_dict.items()}


def _deserialize_module_state(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device="cpu") for key, value in state_dict.items()}


def _configure_rollout_worker_runtime(device: str, num_threads: int | None) -> None:
    resolved_device = torch.device(device)
    if resolved_device.type == "cuda":
        torch.cuda.set_device(resolved_device)
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))


def _numpy_dtype_for_observation_field(key: str) -> np.dtype:
    dtype = REPLAY_OBSERVATION_DTYPES[key]
    if dtype == torch.bool:
        return np.dtype(np.bool_)
    if dtype == torch.float32:
        return np.dtype(np.float32)
    raise TypeError("Unsupported observation dtype for key '{0}': {1}".format(key, dtype))


def _serialize_inference_observation_batch(observations: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not observations:
        raise ValueError("observations must contain at least one item.")

    payload: dict[str, np.ndarray] = {}
    first_num_nodes: int | None = None
    for key in REPLAY_OBSERVATION_DTYPES:
        arrays: list[np.ndarray] = []
        for observation in observations:
            if key not in observation:
                raise KeyError("Observation is missing inference field '{0}'.".format(key))
            array = np.asarray(observation[key], dtype=_numpy_dtype_for_observation_field(key))
            if key == "local_mask":
                if array.ndim != 2 or array.shape[0] != array.shape[1]:
                    raise ValueError("Observation field 'local_mask' must be a square matrix.")
                if first_num_nodes is None:
                    first_num_nodes = int(array.shape[0])
                elif int(array.shape[0]) != first_num_nodes:
                    raise ValueError("Inference batch requires all observations to share the same node count.")
            arrays.append(array)
        payload[key] = np.ascontiguousarray(np.stack(arrays, axis=0))
    return payload


def _normalize_serialized_inference_batch(observations_batch: Mapping[str, Any]) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    batch_size: int | None = None
    num_nodes: int | None = None
    for key in REPLAY_OBSERVATION_DTYPES:
        if key not in observations_batch:
            raise KeyError("Inference batch is missing field '{0}'.".format(key))
        array = np.asarray(observations_batch[key], dtype=_numpy_dtype_for_observation_field(key))
        if array.ndim < 1:
            raise ValueError("Inference batch field '{0}' must include a batch dimension.".format(key))
        if batch_size is None:
            batch_size = int(array.shape[0])
        elif int(array.shape[0]) != batch_size:
            raise ValueError("Inference batch fields must share the same batch dimension.")
        if key == "local_mask":
            if array.ndim != 3 or array.shape[1] != array.shape[2]:
                raise ValueError("Inference batch field 'local_mask' must have shape [batch, num_nodes, num_nodes].")
            num_nodes = int(array.shape[1])
        elif num_nodes is not None and array.ndim >= 2 and int(array.shape[1]) != num_nodes:
            raise ValueError(
                "Inference batch field '{0}' must share the same num_nodes dimension as local_mask.".format(key)
            )
        payload[key] = np.ascontiguousarray(array)
    return payload


def _serialized_inference_batch_metadata(observations_batch: Mapping[str, Any]) -> tuple[int, int]:
    batch_size: int | None = None
    num_nodes: int | None = None
    for key in REPLAY_OBSERVATION_DTYPES:
        if key not in observations_batch:
            raise KeyError("Inference batch is missing field '{0}'.".format(key))
        array = np.asarray(observations_batch[key])
        if array.ndim < 1:
            raise ValueError("Inference batch field '{0}' must include a batch dimension.".format(key))
        if batch_size is None:
            batch_size = int(array.shape[0])
        elif int(array.shape[0]) != batch_size:
            raise ValueError("Inference batch fields must share the same batch dimension.")
        if key == "local_mask":
            if array.ndim != 3 or array.shape[1] != array.shape[2]:
                raise ValueError("Inference batch field 'local_mask' must have shape [batch, num_nodes, num_nodes].")
            num_nodes = int(array.shape[1])
        elif num_nodes is not None and array.ndim >= 2 and int(array.shape[1]) != num_nodes:
            raise ValueError(
                "Inference batch field '{0}' must share the same num_nodes dimension as local_mask.".format(key)
            )
    assert batch_size is not None
    assert num_nodes is not None
    return batch_size, num_nodes


def _concat_serialized_inference_batches(batches: list[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    if not batches:
        raise ValueError("batches must contain at least one item.")
    reference_batch_size, reference_num_nodes = _serialized_inference_batch_metadata(batches[0])
    normalized_batches: list[Mapping[str, Any]] = [batches[0]]
    for batch in batches[1:]:
        batch_size, num_nodes = _serialized_inference_batch_metadata(batch)
        if num_nodes != reference_num_nodes:
            raise ValueError("Inference batches must share the same num_nodes before concatenation.")
        if batch_size <= 0:
            raise ValueError("Inference batches must contain at least one item.")
        normalized_batches.append(batch)
    if reference_batch_size <= 0:
        raise ValueError("Inference batches must contain at least one item.")
    return {
        key: np.ascontiguousarray(np.concatenate([np.asarray(batch[key]) for batch in normalized_batches], axis=0))
        for key in REPLAY_OBSERVATION_DTYPES
    }


def _slice_action_record(action: TensorActionRecord, index: int) -> TensorActionRecord:
    return TensorActionRecord(
        logits=action.logits[index],
        allocation=action.allocation[index],
        transfers=action.transfers[index],
        incoming=action.incoming[index],
        ego_mask=action.ego_mask[index],
        pool_values=action.pool_values[index],
    )


class RolloutInferenceClient:
    def __init__(
        self,
        connection,
        timeout_seconds: float,
        device: torch.device | str = "cpu",
        worker_id: int | None = None,
    ):
        self._connection = connection
        self._timeout_seconds = float(timeout_seconds)
        self._device = torch.device(device)
        self._worker_id = worker_id

    def infer_logits(self, observation: Mapping[str, Any]) -> tuple[torch.Tensor, int]:
        logits_batch, batch_size = self.infer_logits_batch([observation])
        return logits_batch[0], batch_size

    def infer_logits_batch(self, observations: list[Mapping[str, Any]]) -> tuple[torch.Tensor, int]:
        return self.infer_logits_tensor_batch(_serialize_inference_observation_batch(observations))

    def infer_logits_tensor_batch(self, observations_batch: Mapping[str, Any]) -> tuple[torch.Tensor, int]:
        payload_batch = dict(observations_batch)
        batch_size, _ = _serialized_inference_batch_metadata(payload_batch)
        if batch_size <= 0:
            raise ValueError("observations_batch must contain at least one item.")
        self._connection.send(
            {
                "command": "infer_logits_batch",
                "observations_batch": payload_batch,
            }
        )
        if not self._connection.poll(self._timeout_seconds):
            raise TimeoutError(
                "Timed out waiting {0:.1f}s for rollout inference response{1}.".format(
                    self._timeout_seconds,
                    "" if self._worker_id is None else " for worker {0}".format(self._worker_id),
                )
            )
        response = self._connection.recv()
        status = str(response.get("status", "ok"))
        if status != "ok":
            error_payload = dict(response.get("error", {}))
            raise RuntimeError(
                "Rollout inference server failed{0} with {1}: {2}\n{3}".format(
                    "" if self._worker_id is None else " for worker {0}".format(self._worker_id),
                    str(error_payload.get("type", "Exception")),
                    str(error_payload.get("message", "")),
                    str(error_payload.get("traceback", "")),
                )
            )
        payload = dict(response.get("payload", {}))
        return (
            torch.as_tensor(payload["logits"], dtype=torch.float32, device=self._device),
            int(payload.get("batch_size", 1)),
        )

    def close(self) -> None:
        try:
            self._connection.close()
        except OSError:
            pass


def _serialize_remote_exception(exc: Exception) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def _close_shared_memory_handles(handles: list[shared_memory.SharedMemory], unlink: bool) -> None:
    for handle in handles:
        try:
            handle.close()
        except FileNotFoundError:
            pass
        except BufferError:
            pass
        if unlink:
            try:
                handle.unlink()
            except FileNotFoundError:
                pass


def _tensor_to_shared_memory_payload(tensor: torch.Tensor) -> tuple[dict[str, Any], shared_memory.SharedMemory]:
    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    array = cpu_tensor.numpy()
    shm = shared_memory.SharedMemory(create=True, size=max(int(array.nbytes), 1))
    shared_array = np.ndarray(array.shape, dtype=array.dtype, buffer=shm.buf)
    shared_array[...] = array
    return (
        {
            "name": shm.name,
            "shape": list(array.shape),
            "dtype": array.dtype.str,
        },
        shm,
    )


def _shared_memory_payload_to_tensor(payload: Mapping[str, Any]) -> tuple[torch.Tensor, shared_memory.SharedMemory]:
    shm = shared_memory.SharedMemory(name=str(payload["name"]))
    array = np.ndarray(
        tuple(int(item) for item in payload["shape"]),
        dtype=np.dtype(str(payload["dtype"])),
        buffer=shm.buf,
    )
    return torch.from_numpy(array), shm


def _serialize_replay_batch_to_shared_memory(batch: TensorReplayBatch) -> tuple[dict[str, Any], list[shared_memory.SharedMemory]]:
    handles: list[shared_memory.SharedMemory] = []

    def _store_tensor(tensor: torch.Tensor) -> dict[str, Any]:
        payload, handle = _tensor_to_shared_memory_payload(tensor)
        handles.append(handle)
        return payload

    return (
        {
            "obs": {key: _store_tensor(value) for key, value in batch.obs.items()},
            "action": {
                "allocation": _store_tensor(batch.action.allocation),
            },
            "reward": _store_tensor(batch.reward),
            "next_obs": {key: _store_tensor(value) for key, value in batch.next_obs.items()},
            "done": _store_tensor(batch.done),
            "is_demo": _store_tensor(batch.is_demo),
            "collapse_flag": _store_tensor(batch.collapse_flag),
            "topology_id": _store_tensor(batch.topology_id),
            "pool_power_demo_flag": _store_tensor(batch.pool_power_demo_flag),
            "demo_return_target": _store_tensor(batch.demo_return_target),
            "demo_return_valid": _store_tensor(batch.demo_return_valid),
        },
        handles,
    )


def _deserialize_replay_batch_from_shared_memory(
    payload: dict[str, Any],
) -> tuple[TensorReplayBatch, tuple[shared_memory.SharedMemory, ...]]:
    handles: list[shared_memory.SharedMemory] = []

    def _load_tensor(item: Mapping[str, Any]) -> torch.Tensor:
        tensor, handle = _shared_memory_payload_to_tensor(item)
        handles.append(handle)
        return tensor

    try:
        batch = TensorReplayBatch(
            obs={key: _load_tensor(value) for key, value in dict(payload["obs"]).items()},
            action=TensorReplayActionRecord(
                allocation=_load_tensor(payload["action"]["allocation"]),
            ),
            reward=_load_tensor(payload["reward"]),
            next_obs={key: _load_tensor(value) for key, value in dict(payload["next_obs"]).items()},
            done=_load_tensor(payload["done"]),
            is_demo=_load_tensor(payload["is_demo"]),
            collapse_flag=_load_tensor(payload["collapse_flag"]),
            topology_id=_load_tensor(payload["topology_id"]),
            pool_power_demo_flag=_load_tensor(payload["pool_power_demo_flag"]),
            demo_return_target=_load_tensor(payload["demo_return_target"]),
            demo_return_valid=_load_tensor(payload["demo_return_valid"]),
        )
    except Exception:
        _close_shared_memory_handles(handles, unlink=False)
        raise
    return batch, tuple(handles)


def _serialize_rollout_result(result: RolloutResult) -> tuple[dict[str, Any], list[shared_memory.SharedMemory]]:
    replay_payload, handles = _serialize_replay_batch_to_shared_memory(result.replay_batch)
    return {
        "replay_batch": replay_payload,
        "metrics": dict(result.metrics),
    }, handles


def _deserialize_rollout_result(payload: dict[str, Any]) -> RolloutResult:
    replay_batch, handles = _deserialize_replay_batch_from_shared_memory(dict(payload["replay_batch"]))
    return RolloutResult(
        replay_batch=replay_batch,
        metrics=dict(payload["metrics"]),
        shared_memory_handles=handles,
    )

@dataclass(frozen=True)
class RolloutResult:
    replay_batch: TensorReplayBatch
    metrics: dict[str, Any]
    shared_memory_handles: tuple[shared_memory.SharedMemory, ...] = ()

    def release_shared_memory(self) -> None:
        _close_shared_memory_handles(list(self.shared_memory_handles), unlink=True)


@dataclass
class _CollectedTransitionRecord:
    slot_index: int
    transition: TensorTransition


class RandomizedEnvFactory:
    def __init__(
        self,
        base_config: SPGGConfig,
        base_graph: dict[int, list[int]],
        randomization: DomainRandomizationConfig | None = None,
    ):
        self.base_config = base_config
        self.base_graph = _clone_graph_dict(base_graph)
        self.randomization = randomization or DomainRandomizationConfig(enabled=False)
        self._fixed_graph_bank = self._build_fixed_graph_bank()
        self._fixed_graph_bank_round_robin_positions = {
            network_type: 0 for network_type in self._fixed_graph_bank
        }

    @classmethod
    def from_env(
        cls,
        env: SPGGEnv,
        randomization: DomainRandomizationConfig | None = None,
    ) -> "RandomizedEnvFactory":
        return cls(
            base_config=env.config,
            base_graph=_clone_graph_from_env(env),
            randomization=randomization,
        )

    def sample_environment(self, rng: np.random.Generator) -> tuple[SPGGEnv, dict[str, Any]]:
        if not self.randomization.enabled:
            env = SPGGEnv(self.base_config, self.base_graph)
            return env, {"network_type": "fixed", "num_nodes": env.num_nodes}

        network_probabilities = None
        if self.randomization.network_type_weights is not None:
            weight_array = np.asarray(self.randomization.network_type_weights, dtype=np.float64)
            network_probabilities = weight_array / weight_array.sum()
        network_type = str(rng.choice(self.randomization.network_types, p=network_probabilities))
        bank_entry = self._sample_graph_bank_entry(network_type, rng)
        graph_bank_index = None
        if bank_entry is not None:
            graph_bank_index, num_nodes, graph = bank_entry
        else:
            num_nodes = int(rng.choice(self.randomization.num_nodes_choices))
            graph = self._sample_graph(network_type, num_nodes, rng)
        config = self._sample_config(rng, num_nodes=num_nodes)
        env = SPGGEnv(config, graph)
        metadata = {
            "network_type": network_type,
            "num_nodes": num_nodes,
        }
        if graph_bank_index is not None:
            metadata["graph_bank_index"] = int(graph_bank_index)
        return env, metadata

    def _fixed_graph_bank_enabled(self) -> bool:
        return bool(self.randomization.enabled) and bool(self.randomization.fixed_graph_bank_enabled)

    def _build_fixed_graph_bank(self) -> dict[str, tuple[tuple[int, dict[int, list[int]]], ...]]:
        if not self._fixed_graph_bank_enabled():
            return {}

        bank_size = int(self.randomization.fixed_graph_bank_size_per_type)
        seed_base = int(self.randomization.fixed_graph_bank_seed)
        graph_bank: dict[str, tuple[tuple[int, dict[int, list[int]]], ...]] = {}
        for network_type in self.randomization.network_types:
            type_seed = seed_base + sum((index + 1) * ord(char) for index, char in enumerate(str(network_type)))
            type_rng = np.random.default_rng(type_seed)
            entries: list[tuple[int, dict[int, list[int]]]] = []
            for _ in range(bank_size):
                num_nodes = int(type_rng.choice(self.randomization.num_nodes_choices))
                entries.append((num_nodes, self._sample_graph(str(network_type), num_nodes, type_rng)))
            graph_bank[str(network_type)] = tuple(entries)
        return graph_bank

    def _sample_graph_bank_entry(
        self,
        network_type: str,
        rng: np.random.Generator,
    ) -> tuple[int, int, dict[int, list[int]]] | None:
        entries = self._fixed_graph_bank.get(str(network_type))
        if not entries:
            return None

        if self.randomization.fixed_graph_bank_sampling == "round_robin":
            current_position = int(self._fixed_graph_bank_round_robin_positions.get(str(network_type), 0))
            bank_index = current_position % len(entries)
            self._fixed_graph_bank_round_robin_positions[str(network_type)] = (bank_index + 1) % len(entries)
        else:
            bank_index = int(rng.integers(0, len(entries)))

        num_nodes, graph = entries[bank_index]
        return bank_index, int(num_nodes), _clone_graph_dict(graph)

    def _sample_graph(self, network_type: str, num_nodes: int, rng: np.random.Generator) -> dict[int, list[int]]:
        seed = int(rng.integers(0, 2**31 - 1))
        if network_type == "regular":
            degree = int(rng.choice(self.randomization.regular_degree_choices))
            return make_random_regular_graph(num_nodes, degree=degree, seed=seed)
        if network_type == "erdos_renyi":
            mean_degree = float(rng.choice(self.randomization.er_mean_degree_choices))
            edge_prob = 0.0 if num_nodes <= 1 else min(max(mean_degree / max(num_nodes - 1, 1), 0.0), 1.0)
            return make_erdos_renyi_graph(num_nodes, edge_prob=edge_prob, seed=seed)
        if network_type == "small_world":
            degree = int(rng.choice(self.randomization.ws_degree_choices))
            rewiring_prob = float(rng.choice(self.randomization.ws_rewiring_choices))
            return make_watts_strogatz_graph(num_nodes, degree=degree, rewiring_prob=rewiring_prob, seed=seed)
        if network_type == "scale_free":
            attachments = int(rng.choice(self.randomization.ba_attachment_choices))
            return make_barabasi_albert_graph(num_nodes, attachments_per_new_node=attachments, seed=seed)
        raise ValueError("Unsupported randomized network_type: {0}".format(network_type))

    def _sample_config(self, rng: np.random.Generator, num_nodes: int) -> SPGGConfig:
        config = replace(self.base_config, num_nodes=num_nodes)
        if self.randomization.initial_resource_range is not None:
            config = replace(
                config,
                initial_resource=float(rng.uniform(*self.randomization.initial_resource_range)),
            )
        if self.randomization.initial_cooperation_prob_range is not None:
            config = replace(
                config,
                initial_cooperation_prob=float(rng.uniform(*self.randomization.initial_cooperation_prob_range)),
            )
        if self.randomization.alpha_range is not None:
            config = replace(config, alpha=float(rng.uniform(*self.randomization.alpha_range)))
        if self.randomization.r_range is not None:
            config = replace(config, r=float(rng.uniform(*self.randomization.r_range)))
        if self.randomization.p_max_range is not None:
            config = replace(config, p_max=float(rng.uniform(*self.randomization.p_max_range)))
        return config


class RolloutWorker:
    def __init__(
        self,
        actor: GNNAllocationPolicy,
        explorer: LogitSpaceExplorer,
        env_factory: RandomizedEnvFactory,
        config: WorkerConfig,
        train_config: GraphTD3Config,
        device: torch.device | str = "cpu",
        inference_client: RolloutInferenceClient | None = None,
    ):
        self.actor = actor.to(device)
        self.actor.eval()
        self.explorer = explorer
        self.env_factory = env_factory
        self.config = config
        self.train_config = train_config
        self.device = torch.device(device)
        self.inference_client = inference_client
        self.rng = np.random.default_rng(config.seed)
        self.num_envs_per_worker = int(config.num_envs_per_worker)
        self.total_env_steps = 0
        self.actor_version = 0
        self.envs: list[SPGGEnv | None] = [None for _ in range(self.num_envs_per_worker)]
        self.env_metadatas: list[dict[str, Any]] = [{} for _ in range(self.num_envs_per_worker)]
        self.observations: list[dict[str, np.ndarray] | None] = [None for _ in range(self.num_envs_per_worker)]
        self.uniform_policy = UniformAllocationPolicy()
        self.proportional_policy = ProportionalContributionPolicy()
        self.constant_mix_policy = ConstantMixAllocationPolicy(train_config.warmup_constant_mix_omega)
        self.pool_power_mix_policy = PoolPowerMixAllocationPolicy(train_config.warmup_pool_power_k)
        self.current_warmup_behavior_sources: list[str | None] = [None for _ in range(self.num_envs_per_worker)]
        self.teacher_takeover_release_env_step: int | None = None

    def sync_actor(
        self,
        actor_state_dict: dict[str, torch.Tensor],
        version: int,
        teacher_takeover_release_env_step: int | None = None,
    ) -> None:
        self.actor.load_state_dict(_deserialize_module_state(actor_state_dict))
        self.actor.eval()
        self.actor_version = version
        self.teacher_takeover_release_env_step = (
            None if teacher_takeover_release_env_step is None else int(teacher_takeover_release_env_step)
        )

    def set_env_factory(self, env_factory: RandomizedEnvFactory, reset_environment: bool = True) -> None:
        self.env_factory = env_factory
        if reset_environment:
            self.envs = [None for _ in range(self.num_envs_per_worker)]
            self.env_metadatas = [{} for _ in range(self.num_envs_per_worker)]
            self.observations = [None for _ in range(self.num_envs_per_worker)]
            self.current_warmup_behavior_sources = [None for _ in range(self.num_envs_per_worker)]

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "worker_id": int(self.config.worker_id),
                "seed": int(self.config.seed),
                "rollout_steps_per_sync": int(self.config.rollout_steps_per_sync),
                "num_envs_per_worker": int(self.config.num_envs_per_worker),
                "noise_scale_multiplier": float(self.config.noise_scale_multiplier),
            },
            "rng_state": self.rng.bit_generator.state,
            "total_env_steps": int(self.total_env_steps),
            "actor_version": int(self.actor_version),
            "actor_state_dict": _serialize_module_state(_copy_module_state_to_cpu(self.actor)),
            "teacher_takeover_release_env_step": (
                None
                if self.teacher_takeover_release_env_step is None
                else int(self.teacher_takeover_release_env_step)
            ),
            "env_metadata_list": [dict(item) for item in self.env_metadatas],
            "observation_list": [
                ({key: np.asarray(value).copy() for key, value in observation.items()} if observation is not None else None)
                for observation in self.observations
            ],
            "env_state_list": [env.state_dict() if env is not None else None for env in self.envs],
            "current_warmup_behavior_source_list": list(self.current_warmup_behavior_sources),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state_dict["rng_state"]
        self.total_env_steps = int(state_dict["total_env_steps"])
        self.actor_version = int(state_dict["actor_version"])
        self.actor.load_state_dict(_deserialize_module_state(dict(state_dict["actor_state_dict"])))
        self.actor.eval()
        release_env_step = state_dict.get("teacher_takeover_release_env_step")
        self.teacher_takeover_release_env_step = None if release_env_step is None else int(release_env_step)
        env_metadata_list = state_dict.get("env_metadata_list")
        if env_metadata_list is None:
            env_metadata_list = [dict(state_dict.get("env_metadata", {}))]
        observation_list = state_dict.get("observation_list")
        if observation_list is None:
            observation_list = [state_dict.get("observation")]
        env_state_list = state_dict.get("env_state_list")
        if env_state_list is None:
            env_state_list = [state_dict.get("env_state")]
        current_warmup_behavior_source_list = state_dict.get("current_warmup_behavior_source_list")
        if current_warmup_behavior_source_list is None:
            current_warmup_behavior_source_list = [state_dict.get("current_warmup_behavior_source")]

        if len(env_state_list) != self.num_envs_per_worker:
            raise ValueError(
                "Checkpoint env count {0} does not match worker num_envs_per_worker {1}.".format(
                    len(env_state_list),
                    self.num_envs_per_worker,
                )
            )

        self.env_metadatas = [dict(item) for item in env_metadata_list]
        self.observations = [
            ({key: np.asarray(value).copy() for key, value in observation.items()} if observation is not None else None)
            for observation in observation_list
        ]
        self.envs = []
        for env_state in env_state_list:
            if env_state is None:
                self.envs.append(None)
            else:
                env = SPGGEnv(self.env_factory.base_config, self.env_factory.base_graph)
                env.load_state_dict(env_state)
                self.envs.append(env)
        self.current_warmup_behavior_sources = list(current_warmup_behavior_source_list)

    def _total_rollout_env_steps(self) -> int:
        return int(self.train_config.total_updates) * int(self.train_config.steps_per_update) * int(self.train_config.num_workers)

    def _current_teacher_takeover_prob(self, global_env_step: int) -> float:
        if not bool(self.train_config.teacher_takeover_enabled):
            return 0.0

        warmup_end_step = int(self.train_config.warmup_steps)
        total_rollout_env_steps = self._total_rollout_env_steps()
        decay_end_step = int(round(float(total_rollout_env_steps) * float(self.train_config.teacher_takeover_decay_end_fraction)))
        decay_end_step = max(warmup_end_step, decay_end_step)
        decay_duration = max(0, int(decay_end_step - warmup_end_step))
        current_step = max(0, int(global_env_step))

        if current_step < warmup_end_step:
            return float(self.train_config.teacher_takeover_start_prob)
        if bool(self.train_config.adaptive_teacher_release_enabled) and self.teacher_takeover_release_env_step is None:
            return float(self.train_config.teacher_takeover_start_prob)

        release_env_step = (
            int(self.teacher_takeover_release_env_step)
            if self.teacher_takeover_release_env_step is not None
            else warmup_end_step
        )
        release_env_step = max(warmup_end_step, release_env_step)
        if decay_duration <= 0:
            return float(self.train_config.teacher_takeover_end_prob)
        decay_finish_step = release_env_step + decay_duration
        if current_step <= release_env_step:
            return float(self.train_config.teacher_takeover_start_prob)
        if current_step >= decay_finish_step:
            return float(self.train_config.teacher_takeover_end_prob)

        progress = float(current_step - release_env_step) / float(decay_duration)
        progress = min(max(progress, 0.0), 1.0)
        start_prob = float(self.train_config.teacher_takeover_start_prob)
        end_prob = float(self.train_config.teacher_takeover_end_prob)
        return start_prob + (end_prob - start_prob) * progress

    def _annotate_demo_return_targets(
        self,
        records: list[_CollectedTransitionRecord],
        *,
        mode: str,
        n_step: int,
    ) -> None:
        by_slot: dict[int, list[int]] = {}
        for index, record in enumerate(records):
            by_slot.setdefault(int(record.slot_index), []).append(index)

        gamma = float(self.train_config.gamma)
        horizon = max(1, int(n_step))

        def _reward_at(record_index: int) -> float:
            return float(records[record_index].transition.reward.item())

        def _done_at(record_index: int) -> bool:
            return bool(records[record_index].transition.done.item() > 0.5)

        for slot_indices in by_slot.values():
            if mode == "n_step":
                for local_index, record_index in enumerate(slot_indices):
                    discounted_return = 0.0
                    discount = 1.0
                    for step_offset in range(horizon):
                        future_index_position = local_index + step_offset
                        if future_index_position >= len(slot_indices):
                            break
                        future_record_index = slot_indices[future_index_position]
                        discounted_return += discount * _reward_at(future_record_index)
                        if _done_at(future_record_index):
                            break
                        discount *= gamma
                    records[record_index].transition = replace(
                        records[record_index].transition,
                        demo_return_target=torch.tensor(discounted_return, dtype=torch.float32, device="cpu"),
                        demo_return_valid=torch.tensor(True, dtype=torch.bool, device="cpu"),
                    )
                continue

            episode_start = 0
            while episode_start < len(slot_indices):
                episode_end = None
                for cursor in range(episode_start, len(slot_indices)):
                    if _done_at(slot_indices[cursor]):
                        episode_end = cursor
                        break
                if episode_end is None:
                    for cursor in range(episode_start, len(slot_indices)):
                        record_index = slot_indices[cursor]
                        records[record_index].transition = replace(
                            records[record_index].transition,
                            demo_return_target=torch.tensor(0.0, dtype=torch.float32, device="cpu"),
                            demo_return_valid=torch.tensor(False, dtype=torch.bool, device="cpu"),
                        )
                    break

                episode_rewards = [_reward_at(slot_indices[cursor]) for cursor in range(episode_start, episode_end + 1)]
                for cursor in range(episode_start, episode_end + 1):
                    discounted_return = 0.0
                    discount = 1.0
                    for reward_value in episode_rewards[cursor - episode_start :]:
                        discounted_return += discount * reward_value
                        discount *= gamma
                    record_index = slot_indices[cursor]
                    records[record_index].transition = replace(
                        records[record_index].transition,
                        demo_return_target=torch.tensor(discounted_return, dtype=torch.float32, device="cpu"),
                        demo_return_valid=torch.tensor(True, dtype=torch.bool, device="cpu"),
                    )
                episode_start = episode_end + 1

    def collect(
        self,
        num_steps: int,
        global_warmup_steps: int = 0,
        forced_behavior_source: str | None = None,
        mark_as_demo: bool | None = None,
        count_env_steps: bool = True,
        global_env_start_step: int = 0,
        demo_return_target_mode: str | None = None,
        demo_return_n_step: int | None = None,
    ) -> RolloutResult:
        collect_start = perf_counter()
        rewards: list[float] = []
        completed_episodes = 0
        behavior_source_counts: dict[str, int] = {}
        cooperation_rates: list[float] = []
        mean_resources: list[float] = []
        gini_values: list[float] = []
        mean_payoffs: list[float] = []
        mean_pool_growns: list[float] = []
        mean_pool_raws: list[float] = []
        transition_records: list[_CollectedTransitionRecord] = []
        warmup_budget = max(0, int(global_warmup_steps))
        env_step_seconds = 0.0
        inference_wait_seconds = 0.0
        inference_request_build_seconds = 0.0
        local_policy_forward_seconds = 0.0
        action_to_numpy_seconds = 0.0
        transition_encode_seconds = 0.0
        inference_batch_sizes: list[int] = []
        collected_steps = 0
        teacher_takeover_probs: list[float] = []
        open_episode_indices_by_slot: dict[int, list[int]] = {
            slot_index: [] for slot_index in range(self.num_envs_per_worker)
        }
        compute_demo_returns = (
            forced_behavior_source is not None
            and bool(mark_as_demo if mark_as_demo is not None else True)
            and str(forced_behavior_source) == str(self.train_config.demo_collection_behavior_source)
            and demo_return_target_mode in {"n_step", "mc"}
        )
        resolved_demo_return_mode = str(demo_return_target_mode) if compute_demo_returns else None
        resolved_demo_return_n_step = max(1, int(demo_return_n_step or self.train_config.demo_critic_pretrain_n_step))

        while True:
            if collected_steps < num_steps:
                batch_slots = list(range(min(self.num_envs_per_worker, num_steps - collected_steps)))
            elif resolved_demo_return_mode == "mc":
                batch_slots = [
                    slot_index
                    for slot_index, episode_indices in open_episode_indices_by_slot.items()
                    if episode_indices
                ]
            else:
                break

            if not batch_slots:
                break
            for slot_index in batch_slots:
                self._ensure_environment(slot_index)

            actions_by_slot: dict[int, TensorActionRecord] = {}
            is_demo_by_slot: dict[int, bool] = {}
            behavior_source_by_slot: dict[int, str] = {}
            actor_slots: list[int] = []
            actor_observations: list[dict[str, np.ndarray]] = []
            actor_behavior_sources: list[str] = []
            actor_global_step_offsets: list[int] = []

            for batch_offset, slot_index in enumerate(batch_slots):
                observation = self.observations[slot_index]
                assert observation is not None
                if forced_behavior_source is not None:
                    actions_by_slot[slot_index] = self._sample_warmup_action(observation, str(forced_behavior_source))
                    is_demo_by_slot[slot_index] = True if mark_as_demo is None else bool(mark_as_demo)
                    behavior_source_by_slot[slot_index] = str(forced_behavior_source)
                    behavior_source_counts[str(forced_behavior_source)] = (
                        behavior_source_counts.get(str(forced_behavior_source), 0) + 1
                    )
                    continue
                is_warmup = (collected_steps + batch_offset) < warmup_budget
                if is_warmup:
                    behavior_source = self._resolve_warmup_behavior_source(slot_index)
                    actions_by_slot[slot_index] = self._sample_warmup_action(observation, behavior_source)
                    is_demo_by_slot[slot_index] = True
                    behavior_source_by_slot[slot_index] = behavior_source
                    behavior_source_counts[behavior_source] = behavior_source_counts.get(behavior_source, 0) + 1
                    continue
                current_global_step = int(global_env_start_step) + int(collected_steps + batch_offset)
                teacher_takeover_prob = self._current_teacher_takeover_prob(current_global_step)
                teacher_takeover_probs.append(float(teacher_takeover_prob))
                if teacher_takeover_prob > 0.0 and (
                    self.rng.random() < teacher_takeover_prob
                ):
                    behavior_source = str(self.train_config.teacher_takeover_behavior_source)
                    actions_by_slot[slot_index] = self._sample_warmup_action(observation, behavior_source)
                    is_demo_by_slot[slot_index] = True
                    behavior_source_by_slot[slot_index] = behavior_source
                    behavior_source_counts[behavior_source] = behavior_source_counts.get(behavior_source, 0) + 1
                    continue
                actor_slots.append(slot_index)
                actor_observations.append(observation)
                actor_behavior_sources.append("actor_logits")
                actor_global_step_offsets.append(batch_offset)

            if actor_slots:
                if self.inference_client is not None:
                    inference_request_build_start = perf_counter()
                    actor_observation_batch = _serialize_inference_observation_batch(actor_observations)
                    inference_request_build_seconds += perf_counter() - inference_request_build_start
                    inference_wait_start = perf_counter()
                    logits_batch, inference_batch_size = self.inference_client.infer_logits_tensor_batch(
                        actor_observation_batch
                    )
                    inference_wait_seconds += perf_counter() - inference_wait_start
                    inference_batch_sizes.append(int(inference_batch_size))
                    ego_mask_batch = torch.as_tensor(
                        actor_observation_batch["local_mask"],
                        dtype=torch.bool,
                        device=self.device,
                    )
                    pool_values_batch = torch.as_tensor(
                        actor_observation_batch["pool_grown"],
                        dtype=torch.float32,
                        device=self.device,
                    )
                else:
                    policy_forward_start = perf_counter()
                    with torch.inference_mode():
                        logits_batch = self.actor.rollout_logits_batch(actor_observations)
                    local_policy_forward_seconds += perf_counter() - policy_forward_start
                    inference_batch_sizes.append(len(actor_slots))
                    ego_mask_batch = torch.stack(
                        [
                            torch.as_tensor(
                                self.observations[slot_index]["local_mask"],
                                dtype=torch.bool,
                                device=self.device,
                            )
                            for slot_index in actor_slots
                        ],
                        dim=0,
                    )
                    pool_values_batch = torch.stack(
                        [
                            torch.as_tensor(
                                self.observations[slot_index]["pool_grown"],
                                dtype=torch.float32,
                                device=self.device,
                            )
                            for slot_index in actor_slots
                        ],
                        dim=0,
                    )
                noise_std_batch = [
                    self.explorer.current_noise_std(
                        base_std=self.train_config.rollout_logit_noise_std,
                        step=self.total_env_steps + actor_global_step_offsets[actor_batch_index],
                        decay=self.train_config.rollout_noise_decay,
                        multiplier=self.config.noise_scale_multiplier,
                    )
                    for actor_batch_index in range(len(actor_slots))
                ]
                batched_action = self.explorer.apply_to_logits(
                    logits=logits_batch,
                    ego_mask=ego_mask_batch,
                    pool_values=pool_values_batch,
                    noise_std=noise_std_batch,
                    noise_clip=self.train_config.rollout_logit_noise_clip,
                )
                for actor_batch_index, slot_index in enumerate(actor_slots):
                    actions_by_slot[slot_index] = _slice_action_record(batched_action, actor_batch_index)
                    is_demo_by_slot[slot_index] = False
                    behavior_source = actor_behavior_sources[actor_batch_index]
                    behavior_source_by_slot[slot_index] = behavior_source
                    behavior_source_counts[behavior_source] = behavior_source_counts.get(behavior_source, 0) + 1

            for slot_index in batch_slots:
                observation = self.observations[slot_index]
                env = self.envs[slot_index]
                action = actions_by_slot[slot_index]
                assert observation is not None
                assert env is not None

                env_step_start = perf_counter()
                action_to_numpy_start = perf_counter()
                allocation_numpy = action.allocation.detach().cpu().numpy()
                action_to_numpy_seconds += perf_counter() - action_to_numpy_start
                next_observation, reward, done, info = env.step(allocation_numpy)
                env_step_seconds += perf_counter() - env_step_start

                transition_encode_start = perf_counter()
                actual_cooperation_rate = float(
                    info.get("actual_cooperation_rate", np.asarray(next_observation["x_actual"]).mean())
                )
                collapse_flag = actual_cooperation_rate < float(self.train_config.replay_collapse_fc_threshold)
                transition = TensorTransition.from_step(
                    obs=observation,
                    action=action,
                    reward=float(reward),
                    next_obs=next_observation,
                    done=bool(done),
                    is_demo=bool(is_demo_by_slot.get(slot_index, False)),
                    collapse_flag=collapse_flag,
                    topology_name=str(self.env_metadatas[slot_index].get("network_type", "unknown")),
                    pool_power_demo_flag=bool(
                        is_demo_by_slot.get(slot_index, False)
                        and behavior_source_by_slot.get(slot_index) == "pool_power_mix"
                    ),
                )
                transition_encode_seconds += perf_counter() - transition_encode_start
                transition_records.append(
                    _CollectedTransitionRecord(slot_index=int(slot_index), transition=transition)
                )
                open_episode_indices_by_slot[int(slot_index)].append(len(transition_records) - 1)

                rewards.append(float(reward))
                cooperation_rates.append(actual_cooperation_rate)
                mean_resources.append(float(np.asarray(next_observation["resources"]).mean()))
                gini_values.append(float(info.get("gini", np.asarray(next_observation["gini"]).item())))
                mean_payoffs.append(float(np.asarray(info.get("payoff", np.zeros_like(next_observation["resources"]))).mean()))
                mean_pool_growns.append(float(np.asarray(next_observation["pool_grown"]).mean()))
                mean_pool_raws.append(float(np.asarray(next_observation["pool_raw"]).mean()))
                if count_env_steps:
                    self.total_env_steps += 1
                collected_steps += 1
                self.observations[slot_index] = next_observation
                if done:
                    completed_episodes += 1
                    open_episode_indices_by_slot[int(slot_index)] = []
                    if collected_steps < num_steps or resolved_demo_return_mode != "mc":
                        self._reset_environment(slot_index)

        if compute_demo_returns and transition_records:
            self._annotate_demo_return_targets(
                transition_records,
                mode=str(resolved_demo_return_mode),
                n_step=resolved_demo_return_n_step,
            )

        metrics = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "episodes_completed": float(completed_episodes),
            "env_steps": float(self.total_env_steps),
            "steps_collected": float(len(transition_records)),
            "mean_actual_cooperation_rate": float(np.mean(cooperation_rates)) if cooperation_rates else 0.0,
            "mean_resource": float(np.mean(mean_resources)) if mean_resources else 0.0,
            "mean_gini": float(np.mean(gini_values)) if gini_values else 0.0,
            "mean_payoff": float(np.mean(mean_payoffs)) if mean_payoffs else 0.0,
            "mean_pool_grown": float(np.mean(mean_pool_growns)) if mean_pool_growns else 0.0,
            "mean_pool_raw": float(np.mean(mean_pool_raws)) if mean_pool_raws else 0.0,
            "collect_wall_seconds": float(perf_counter() - collect_start),
            "env_step_seconds": float(env_step_seconds),
            "inference_wait_seconds": float(inference_wait_seconds),
            "inference_request_build_seconds": float(inference_request_build_seconds),
            "local_policy_forward_seconds": float(local_policy_forward_seconds),
            "action_to_numpy_seconds": float(action_to_numpy_seconds),
            "transition_encode_seconds": float(transition_encode_seconds),
            "inference_batch_size_mean": float(np.mean(inference_batch_sizes)) if inference_batch_sizes else 0.0,
            "inference_batch_size_max": float(max(inference_batch_sizes)) if inference_batch_sizes else 0.0,
            "behavior_source_counts": behavior_source_counts,
            "teacher_takeover_prob_mean": float(np.mean(teacher_takeover_probs)) if teacher_takeover_probs else 0.0,
        }
        replay_stack_start = perf_counter()
        replay_batch = stack_tensor_transitions([record.transition for record in transition_records])
        metrics["stack_transitions_seconds"] = float(perf_counter() - replay_stack_start)
        return RolloutResult(replay_batch=replay_batch, metrics=metrics)

    def _ensure_environment(self, slot_index: int) -> None:
        if self.envs[slot_index] is None or self.observations[slot_index] is None:
            self._reset_environment(slot_index)

    def _reset_environment(self, slot_index: int) -> None:
        env, env_metadata = self.env_factory.sample_environment(self.rng)
        reset_seed = int(self.rng.integers(0, 2**31 - 1))
        self.envs[slot_index] = env
        self.env_metadatas[slot_index] = env_metadata
        self.observations[slot_index] = env.reset(seed=reset_seed)
        self.current_warmup_behavior_sources[slot_index] = None
        if self.train_config.warmup_selection_granularity == "per_episode":
            self.current_warmup_behavior_sources[slot_index] = self._select_warmup_behavior_source()

    def _resolve_warmup_behavior_source(self, slot_index: int) -> str:
        if self.train_config.warmup_selection_granularity == "per_step":
            return self._select_warmup_behavior_source()
        if self.current_warmup_behavior_sources[slot_index] is None:
            self.current_warmup_behavior_sources[slot_index] = self._select_warmup_behavior_source()
        return str(self.current_warmup_behavior_sources[slot_index])

    def _select_warmup_behavior_source(self) -> str:
        if self.train_config.warmup_behavior_mode == "random_only":
            return "random_logits"

        candidates: list[str] = []
        weights: list[float] = []
        behavior_weights = (
            ("uniform", self.train_config.warmup_uniform_prob),
            ("proportional", self.train_config.warmup_proportional_prob),
            ("constant_mix", self.train_config.warmup_constant_mix_prob),
            ("pool_power_mix", self.train_config.warmup_pool_power_mix_prob),
            ("random_logits", self.train_config.warmup_random_logits_prob),
        )
        for name, weight in behavior_weights:
            if weight > 0.0:
                candidates.append(name)
                weights.append(float(weight))

        weight_array = np.asarray(weights, dtype=np.float64)
        weight_array = weight_array / weight_array.sum()
        return str(self.rng.choice(candidates, p=weight_array))

    def _sample_warmup_action(self, observation: Mapping[str, Any], behavior_source: str):

        if behavior_source == "random_logits":
            return self.explorer.sample_random_logits_action(
                ego_mask=observation["local_mask"],
                pool_values=observation["pool_grown"],
                rng=self.rng,
                device=self.device,
            )

        if behavior_source == "uniform":
            heuristic_allocation = self.uniform_policy.allocate(observation)
        elif behavior_source == "proportional":
            heuristic_allocation = self.proportional_policy.allocate(observation)
        elif behavior_source == "constant_mix":
            heuristic_allocation = self.constant_mix_policy.allocate(observation)
        elif behavior_source == "pool_power_mix":
            heuristic_allocation = self.pool_power_mix_policy.allocate(observation)
        else:
            raise ValueError("Unsupported warm-up behavior source: {0}".format(behavior_source))

        return self.explorer.action_from_allocation(
            allocation=heuristic_allocation,
            ego_mask=observation["local_mask"],
            pool_values=observation["pool_grown"],
            noise_std=self.train_config.warmup_logit_noise_std,
            noise_clip=self.train_config.warmup_logit_noise_clip,
            device=self.device,
        )


def _parallel_rollout_inference_server_main(
    control_connection,
    worker_connections: tuple[Any, ...],
    actor_config: GNNPolicyConfig,
    actor_state_dict: dict[str, Any],
    device: str,
    batch_timeout_ms: float,
    num_threads: int | None,
) -> None:
    _configure_rollout_worker_runtime(device=device, num_threads=num_threads)
    actor = GNNAllocationPolicy(actor_config)
    actor.load_state_dict(_deserialize_module_state(actor_state_dict))
    actor = actor.to(device)
    actor.eval()

    all_connections = [control_connection, *worker_connections]
    batch_timeout_seconds = max(float(batch_timeout_ms), 0.0) / 1000.0

    def _send_error(connection, exc: Exception) -> None:
        try:
            connection.send({"status": "error", "error": _serialize_remote_exception(exc)})
        except (BrokenPipeError, EOFError, OSError):
            pass

    def _process_pending_requests(pending_requests: list[tuple[Any, dict[str, np.ndarray]]]) -> None:
        if not pending_requests:
            return
        grouped_requests: dict[int, dict[str, Any]] = {}
        for connection, observations_batch in pending_requests:
            try:
                batch_size, num_nodes = _serialized_inference_batch_metadata(observations_batch)
            except Exception as exc:
                _send_error(connection, exc)
                continue
            group = grouped_requests.setdefault(
                num_nodes,
                {
                    "observation_batches": [],
                    "requests": [],
                },
            )
            start_index = sum(int(batch["local_mask"].shape[0]) for batch in group["observation_batches"])
            group["observation_batches"].append(observations_batch)
            end_index = start_index + batch_size
            group["requests"].append((connection, start_index, end_index))

        try:
            with torch.inference_mode():
                for group in grouped_requests.values():
                    observations_batch = _concat_serialized_inference_batches(list(group["observation_batches"]))
                    logits_batch = actor.rollout_logits_tensor_batch(observations_batch)
                    batch_size = int(observations_batch["local_mask"].shape[0])
                    for connection, start_index, end_index in group["requests"]:
                        try:
                            connection.send(
                                {
                                    "status": "ok",
                                    "payload": {
                                        "logits": logits_batch[start_index:end_index].detach().cpu().numpy().copy(),
                                        "batch_size": int(batch_size),
                                    },
                                }
                            )
                        except (BrokenPipeError, EOFError, OSError):
                            pass
        except Exception as exc:
            for connection, _ in pending_requests:
                _send_error(connection, exc)

    try:
        while True:
            ready_connections = list(wait(all_connections))
            pending_requests: list[tuple[Any, list[dict[str, np.ndarray]]]] = []
            should_close = False

            while True:
                for connection in ready_connections:
                    try:
                        message = connection.recv()
                    except EOFError:
                        continue
                    command = str(message["command"])
                    if command == "infer_logits":
                        pending_requests.append(
                            (connection, _serialize_inference_observation_batch([dict(message["observation"])]))
                        )
                        continue
                    if command == "infer_logits_batch":
                        if "observations_batch" in message:
                            pending_requests.append((connection, dict(message["observations_batch"])))
                        else:
                            pending_requests.append(
                                (connection, _serialize_inference_observation_batch([dict(item) for item in message["observations"]]))
                            )
                        continue
                    if command == "sync_actor":
                        try:
                            actor.load_state_dict(_deserialize_module_state(dict(message["actor_state_dict"])))
                            actor.eval()
                            connection.send({"status": "ok", "payload": None})
                        except Exception as exc:
                            _send_error(connection, exc)
                        continue
                    if command == "close":
                        try:
                            connection.send({"status": "ok", "payload": None})
                        except (BrokenPipeError, EOFError, OSError):
                            pass
                        if connection is control_connection:
                            should_close = True
                        continue
                    _send_error(connection, ValueError("Unsupported inference command: {0}".format(command)))

                if should_close or batch_timeout_seconds <= 0.0:
                    break
                ready_connections = list(wait(all_connections, timeout=batch_timeout_seconds))
                if not ready_connections:
                    break

            _process_pending_requests(pending_requests)
            if should_close:
                break
    finally:
        for connection in all_connections:
            try:
                connection.close()
            except OSError:
                pass


class ParallelRolloutInferenceServer:
    def __init__(
        self,
        actor: GNNAllocationPolicy,
        train_config: GraphTD3Config,
        device: str,
        num_clients: int,
    ):
        self._ctx = mp.get_context("spawn")
        self._device = str(device)
        self._rpc_timeout_seconds = float(train_config.worker_rpc_timeout_seconds)
        control_parent, control_child = self._ctx.Pipe()
        self._control_connection = control_parent
        server_connections: list[Any] = []
        self._worker_connections: list[Any | None] = []
        for _ in range(num_clients):
            server_connection, worker_connection = self._ctx.Pipe()
            server_connections.append(server_connection)
            self._worker_connections.append(worker_connection)

        actor_state = _serialize_module_state(_copy_module_state_to_cpu(actor))
        self._process = self._ctx.Process(
            target=_parallel_rollout_inference_server_main,
            args=(
                control_child,
                tuple(server_connections),
                actor.config,
                actor_state,
                self._device,
                float(train_config.rollout_inference_batch_timeout_ms),
                train_config.rollout_num_threads,
            ),
        )
        self._process.start()
        control_child.close()
        for connection in server_connections:
            connection.close()

    def take_worker_connection(self, worker_index: int):
        connection = self._worker_connections[worker_index]
        if connection is None:
            raise RuntimeError("Worker connection {0} has already been claimed.".format(worker_index))
        self._worker_connections[worker_index] = None
        return connection

    def _recv_response(self) -> Any:
        if not self._control_connection.poll(self._rpc_timeout_seconds):
            raise TimeoutError(
                "Timed out waiting {0:.1f}s for rollout inference server RPC response on device {1}.".format(
                    self._rpc_timeout_seconds,
                    self._device,
                )
            )
        response = self._control_connection.recv()
        status = str(response.get("status", "ok"))
        if status != "ok":
            error_payload = dict(response.get("error", {}))
            raise RuntimeError(
                "Rollout inference server on {0} failed with {1}: {2}\n{3}".format(
                    self._device,
                    str(error_payload.get("type", "Exception")),
                    str(error_payload.get("message", "")),
                    str(error_payload.get("traceback", "")),
                )
            )
        return response.get("payload")

    def sync_actor(self, actor_state_dict: dict[str, torch.Tensor]) -> None:
        self._control_connection.send(
            {
                "command": "sync_actor",
                "actor_state_dict": _serialize_module_state(actor_state_dict),
            }
        )
        self._recv_response()

    def close(self) -> None:
        if getattr(self, "_control_connection", None) is None:
            return
        try:
            self._control_connection.send({"command": "close"})
            try:
                self._recv_response()
            except Exception:
                pass
        except (BrokenPipeError, EOFError, OSError):
            pass
        except Exception:
            pass
        try:
            self._control_connection.close()
        except OSError:
            pass
        for connection in self._worker_connections:
            if connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _parallel_rollout_worker_main(
    connection,
    actor_config: GNNPolicyConfig,
    actor_state_dict: dict[str, Any],
    env_factory: RandomizedEnvFactory,
    config: WorkerConfig,
    train_config: GraphTD3Config,
    device: str,
    inference_connection=None,
) -> None:
    _configure_rollout_worker_runtime(device=device, num_threads=train_config.rollout_num_threads)
    actor = GNNAllocationPolicy(actor_config)
    actor.load_state_dict(_deserialize_module_state(actor_state_dict))
    inference_client = None
    if inference_connection is not None:
        inference_client = RolloutInferenceClient(
            inference_connection,
            timeout_seconds=float(train_config.worker_rpc_timeout_seconds),
            device="cpu",
            worker_id=config.worker_id,
        )
    worker = RolloutWorker(
        actor=actor,
        explorer=LogitSpaceExplorer(),
        env_factory=env_factory,
        config=config,
        train_config=train_config,
        device="cpu" if inference_client is not None else device,
        inference_client=inference_client,
    )

    try:
        while True:
            try:
                message = connection.recv()
            except EOFError:
                break
            command = str(message["command"])
            if command == "close":
                break
            try:
                if command == "sync_actor":
                    worker.sync_actor(
                        actor_state_dict=dict(message["actor_state_dict"]),
                        version=int(message["version"]),
                        teacher_takeover_release_env_step=message.get("teacher_takeover_release_env_step"),
                    )
                    connection.send({"status": "ok", "payload": None})
                    continue
                if command == "set_env_factory":
                    worker.set_env_factory(
                        env_factory=message["env_factory"],
                        reset_environment=bool(message.get("reset_environment", True)),
                    )
                    connection.send({"status": "ok", "payload": None})
                    continue
                if command == "collect":
                    collect_result = worker.collect(
                        num_steps=int(message["num_steps"]),
                        global_warmup_steps=int(message.get("global_warmup_steps", 0)),
                        forced_behavior_source=message.get("forced_behavior_source"),
                        mark_as_demo=message.get("mark_as_demo"),
                        count_env_steps=bool(message.get("count_env_steps", True)),
                        global_env_start_step=int(message.get("global_env_start_step", 0)),
                        demo_return_target_mode=message.get("demo_return_target_mode"),
                        demo_return_n_step=message.get("demo_return_n_step"),
                    )
                    shared_memory_serialize_start = perf_counter()
                    serialized_result, shared_memory_handles = _serialize_rollout_result(collect_result)
                    serialized_result["metrics"]["shared_memory_serialize_seconds"] = float(
                        perf_counter() - shared_memory_serialize_start
                    )
                    try:
                        connection.send({"status": "ok", "payload": serialized_result})
                    except Exception:
                        _close_shared_memory_handles(shared_memory_handles, unlink=True)
                        raise
                    finally:
                        _close_shared_memory_handles(shared_memory_handles, unlink=False)
                    continue
                if command == "state_dict":
                    connection.send({"status": "ok", "payload": worker.state_dict()})
                    continue
                if command == "load_state_dict":
                    worker.load_state_dict(dict(message["state_dict"]))
                    connection.send({"status": "ok", "payload": None})
                    continue
                raise ValueError("Unsupported worker command: {0}".format(command))
            except Exception as exc:
                try:
                    connection.send(
                        {
                            "status": "error",
                            "error": _serialize_remote_exception(exc),
                        }
                    )
                except (BrokenPipeError, EOFError, OSError):
                    break
    finally:
        if inference_client is not None:
            inference_client.close()
        connection.close()


class ParallelRolloutWorker:
    def __init__(
        self,
        actor: GNNAllocationPolicy,
        env_factory: RandomizedEnvFactory,
        config: WorkerConfig,
        train_config: GraphTD3Config,
        device: str = "cpu",
        inference_connection=None,
    ):
        self.config = config
        self._device = str(device)
        self._rpc_timeout_seconds = float(train_config.worker_rpc_timeout_seconds)
        self._ctx = mp.get_context("spawn")
        parent_connection, child_connection = self._ctx.Pipe()
        self._connection = parent_connection
        actor_state = _serialize_module_state(_copy_module_state_to_cpu(actor))
        self._process = self._ctx.Process(
            target=_parallel_rollout_worker_main,
            args=(
                child_connection,
                actor.config,
                actor_state,
                env_factory,
                config,
                train_config,
                self._device,
                inference_connection,
            ),
        )
        self._process.start()
        child_connection.close()
        if inference_connection is not None:
            try:
                inference_connection.close()
            except OSError:
                pass
        self._collect_inflight = False

    @property
    def connection(self):
        return self._connection

    def _raise_remote_error(self, error_payload: Mapping[str, Any]) -> None:
        raise RuntimeError(
            "Remote worker {0} failed with {1}: {2}\n{3}".format(
                self.config.worker_id,
                str(error_payload.get("type", "Exception")),
                str(error_payload.get("message", "")),
                str(error_payload.get("traceback", "")),
            )
        )

    def _recv_response(self, timeout_seconds: float | None = None, ready: bool = False) -> Any:
        if not ready:
            timeout = self._rpc_timeout_seconds if timeout_seconds is None else float(timeout_seconds)
            if not self._connection.poll(timeout):
                raise TimeoutError(
                    "Timed out waiting {0:.1f}s for worker {1} RPC response.".format(
                        timeout,
                        self.config.worker_id,
                    )
                )
        response = self._connection.recv()
        status = str(response.get("status", "ok"))
        if status != "ok":
            self._raise_remote_error(dict(response.get("error", {})))
        return response.get("payload")

    def _request_response(self, payload: dict[str, Any]) -> Any:
        self._connection.send(payload)
        return self._recv_response()

    def sync_actor(
        self,
        actor_state_dict: dict[str, torch.Tensor],
        version: int,
        teacher_takeover_release_env_step: int | None = None,
    ) -> None:
        self._request_response(
            {
                "command": "sync_actor",
                "actor_state_dict": _serialize_module_state(actor_state_dict),
                "version": int(version),
                "teacher_takeover_release_env_step": (
                    None
                    if teacher_takeover_release_env_step is None
                    else int(teacher_takeover_release_env_step)
                ),
            }
        )

    def set_env_factory(self, env_factory: RandomizedEnvFactory, reset_environment: bool = True) -> None:
        self._request_response(
            {
                "command": "set_env_factory",
                "env_factory": env_factory,
                "reset_environment": bool(reset_environment),
            }
        )

    def start_collect(
        self,
        num_steps: int,
        global_warmup_steps: int = 0,
        forced_behavior_source: str | None = None,
        mark_as_demo: bool | None = None,
        count_env_steps: bool = True,
        global_env_start_step: int = 0,
        demo_return_target_mode: str | None = None,
        demo_return_n_step: int | None = None,
    ) -> None:
        if self._collect_inflight:
            raise RuntimeError("Collect request already in flight for worker {0}.".format(self.config.worker_id))
        self._connection.send(
            {
                "command": "collect",
                "num_steps": int(num_steps),
                "global_warmup_steps": int(global_warmup_steps),
                "forced_behavior_source": forced_behavior_source,
                "mark_as_demo": mark_as_demo,
                "count_env_steps": bool(count_env_steps),
                "global_env_start_step": int(global_env_start_step),
                "demo_return_target_mode": demo_return_target_mode,
                "demo_return_n_step": demo_return_n_step,
            }
        )
        self._collect_inflight = True

    def finish_collect(self) -> RolloutResult:
        if not self._collect_inflight:
            raise RuntimeError("No in-flight collect request for worker {0}.".format(self.config.worker_id))
        try:
            payload = self._recv_response()
            deserialize_start = perf_counter()
            result = _deserialize_rollout_result(payload)
            result.metrics["shared_memory_deserialize_seconds"] = float(perf_counter() - deserialize_start)
            return result
        finally:
            self._collect_inflight = False

    def finish_collect_ready(self) -> RolloutResult:
        if not self._collect_inflight:
            raise RuntimeError("No in-flight collect request for worker {0}.".format(self.config.worker_id))
        try:
            payload = self._recv_response(ready=True)
            deserialize_start = perf_counter()
            result = _deserialize_rollout_result(payload)
            result.metrics["shared_memory_deserialize_seconds"] = float(perf_counter() - deserialize_start)
            return result
        finally:
            self._collect_inflight = False

    def collect(
        self,
        num_steps: int,
        global_warmup_steps: int = 0,
        forced_behavior_source: str | None = None,
        mark_as_demo: bool | None = None,
        count_env_steps: bool = True,
        global_env_start_step: int = 0,
        demo_return_target_mode: str | None = None,
        demo_return_n_step: int | None = None,
    ) -> RolloutResult:
        self.start_collect(
            num_steps=num_steps,
            global_warmup_steps=global_warmup_steps,
            forced_behavior_source=forced_behavior_source,
            mark_as_demo=mark_as_demo,
            count_env_steps=count_env_steps,
            global_env_start_step=global_env_start_step,
            demo_return_target_mode=demo_return_target_mode,
            demo_return_n_step=demo_return_n_step,
        )
        return self.finish_collect()

    def state_dict(self) -> dict[str, Any]:
        if self._collect_inflight:
            raise RuntimeError("Cannot fetch worker state while collect is in flight.")
        return self._request_response({"command": "state_dict"})

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self._collect_inflight:
            raise RuntimeError("Cannot load worker state while collect is in flight.")
        self._request_response(
            {
                "command": "load_state_dict",
                "state_dict": state_dict,
            }
        )

    def close(self) -> None:
        if getattr(self, "_connection", None) is None:
            return
        try:
            if self._collect_inflight:
                try:
                    payload = self._recv_response(timeout_seconds=1.0)
                    if isinstance(payload, dict) and "replay_batch" in payload:
                        result = _deserialize_rollout_result(payload)
                        result.release_shared_memory()
                except Exception:
                    pass
                finally:
                    self._collect_inflight = False
            self._connection.send({"command": "close"})
        except (BrokenPipeError, EOFError, OSError):
            pass
        except Exception:
            pass
        try:
            self._connection.close()
        except OSError:
            pass
        self._process.join(timeout=5.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=5.0)

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
