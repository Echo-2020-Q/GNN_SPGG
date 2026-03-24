from __future__ import annotations

from dataclasses import dataclass, replace
import multiprocessing as mp
from multiprocessing import shared_memory
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
from .data import TensorActionRecord, TensorReplayBatch, TensorTransition, stack_tensor_transitions
from .exploration import LogitSpaceExplorer


def _clone_graph_from_env(env: SPGGEnv) -> dict[int, list[int]]:
    return {node: list(neighbors) for node, neighbors in enumerate(env.graph.neighbors)}


def _copy_module_state_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _serialize_module_state(state_dict: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy().copy() for key, value in state_dict.items()}


def _deserialize_module_state(state_dict: dict[str, Any]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, device="cpu") for key, value in state_dict.items()}


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
                "logits": _store_tensor(batch.action.logits),
                "allocation": _store_tensor(batch.action.allocation),
                "transfers": _store_tensor(batch.action.transfers),
                "incoming": _store_tensor(batch.action.incoming),
                "ego_mask": _store_tensor(batch.action.ego_mask),
                "pool_values": _store_tensor(batch.action.pool_values),
            },
            "reward": _store_tensor(batch.reward),
            "next_obs": {key: _store_tensor(value) for key, value in batch.next_obs.items()},
            "done": _store_tensor(batch.done),
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
            action=TensorActionRecord(
                logits=_load_tensor(payload["action"]["logits"]),
                allocation=_load_tensor(payload["action"]["allocation"]),
                transfers=_load_tensor(payload["action"]["transfers"]),
                incoming=_load_tensor(payload["action"]["incoming"]),
                ego_mask=_load_tensor(payload["action"]["ego_mask"]),
                pool_values=_load_tensor(payload["action"]["pool_values"]),
            ),
            reward=_load_tensor(payload["reward"]),
            next_obs={key: _load_tensor(value) for key, value in dict(payload["next_obs"]).items()},
            done=_load_tensor(payload["done"]),
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


class RandomizedEnvFactory:
    def __init__(
        self,
        base_config: SPGGConfig,
        base_graph: dict[int, list[int]],
        randomization: DomainRandomizationConfig | None = None,
    ):
        self.base_config = base_config
        self.base_graph = {node: list(neighbors) for node, neighbors in base_graph.items()}
        self.randomization = randomization or DomainRandomizationConfig(enabled=False)

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
        num_nodes = int(rng.choice(self.randomization.num_nodes_choices))
        graph = self._sample_graph(network_type, num_nodes, rng)
        config = self._sample_config(rng, num_nodes=num_nodes)
        env = SPGGEnv(config, graph)
        metadata = {
            "network_type": network_type,
            "num_nodes": num_nodes,
        }
        return env, metadata

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
    ):
        self.actor = actor.to(device)
        self.actor.eval()
        self.explorer = explorer
        self.env_factory = env_factory
        self.config = config
        self.train_config = train_config
        self.device = torch.device(device)
        self.rng = np.random.default_rng(config.seed)
        self.total_env_steps = 0
        self.actor_version = 0
        self.env: SPGGEnv | None = None
        self.env_metadata: dict[str, Any] = {}
        self.observation: dict[str, np.ndarray] | None = None
        self.uniform_policy = UniformAllocationPolicy()
        self.proportional_policy = ProportionalContributionPolicy()
        self.constant_mix_policy = ConstantMixAllocationPolicy(train_config.warmup_constant_mix_omega)
        self.pool_power_mix_policy = PoolPowerMixAllocationPolicy(train_config.warmup_pool_power_k)
        self.current_warmup_behavior_source: str | None = None

    def sync_actor(self, actor_state_dict: dict[str, torch.Tensor], version: int) -> None:
        self.actor.load_state_dict(_deserialize_module_state(actor_state_dict))
        self.actor.eval()
        self.actor_version = version

    def set_env_factory(self, env_factory: RandomizedEnvFactory, reset_environment: bool = True) -> None:
        self.env_factory = env_factory
        if reset_environment:
            self.env = None
            self.env_metadata = {}
            self.observation = None
            self.current_warmup_behavior_source = None

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": {
                "worker_id": int(self.config.worker_id),
                "seed": int(self.config.seed),
                "rollout_steps_per_sync": int(self.config.rollout_steps_per_sync),
                "noise_scale_multiplier": float(self.config.noise_scale_multiplier),
            },
            "rng_state": self.rng.bit_generator.state,
            "total_env_steps": int(self.total_env_steps),
            "actor_version": int(self.actor_version),
            "actor_state_dict": _serialize_module_state(_copy_module_state_to_cpu(self.actor)),
            "env_metadata": dict(self.env_metadata),
            "observation": (
                {key: np.asarray(value).copy() for key, value in self.observation.items()}
                if self.observation is not None
                else None
            ),
            "env_state": self.env.state_dict() if self.env is not None else None,
            "current_warmup_behavior_source": self.current_warmup_behavior_source,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.rng = np.random.default_rng()
        self.rng.bit_generator.state = state_dict["rng_state"]
        self.total_env_steps = int(state_dict["total_env_steps"])
        self.actor_version = int(state_dict["actor_version"])
        self.actor.load_state_dict(_deserialize_module_state(dict(state_dict["actor_state_dict"])))
        self.actor.eval()
        self.env_metadata = dict(state_dict.get("env_metadata", {}))
        observation = state_dict.get("observation")
        self.observation = (
            {key: np.asarray(value).copy() for key, value in observation.items()}
            if observation is not None
            else None
        )
        env_state = state_dict.get("env_state")
        if env_state is None:
            self.env = None
        else:
            self.env = SPGGEnv(self.env_factory.base_config, self.env_factory.base_graph)
            self.env.load_state_dict(env_state)
        self.current_warmup_behavior_source = state_dict.get("current_warmup_behavior_source")

    def collect(self, num_steps: int, global_warmup_steps: int = 0) -> RolloutResult:
        rewards: list[float] = []
        completed_episodes = 0
        behavior_source_counts: dict[str, int] = {}
        cooperation_rates: list[float] = []
        mean_resources: list[float] = []
        gini_values: list[float] = []
        mean_payoffs: list[float] = []
        mean_pool_growns: list[float] = []
        mean_pool_raws: list[float] = []
        transitions: list[TensorTransition] = []
        warmup_budget = max(0, int(global_warmup_steps))

        for step_index in range(num_steps):
            self._ensure_environment()
            assert self.observation is not None

            is_warmup = step_index < warmup_budget
            behavior_source = "actor_logits"

            if is_warmup:
                behavior_source = self._resolve_warmup_behavior_source()
                action = self._sample_warmup_action(behavior_source)
            else:
                with torch.no_grad():
                    policy_output = self.actor.deterministic_action(self.observation)
                noise_std = self.explorer.current_noise_std(
                    base_std=self.train_config.rollout_logit_noise_std,
                    step=self.total_env_steps,
                    decay=self.train_config.rollout_noise_decay,
                    multiplier=self.config.noise_scale_multiplier,
                )
                action = self.explorer.apply_to_policy_output(
                    policy_output=policy_output,
                    ego_mask=torch.as_tensor(self.observation["local_mask"], dtype=torch.bool, device=self.device),
                    pool_values=torch.as_tensor(self.observation["pool_grown"], dtype=torch.float32, device=self.device),
                    noise_std=noise_std,
                    noise_clip=self.train_config.rollout_logit_noise_clip,
                )

            behavior_source_counts[behavior_source] = behavior_source_counts.get(behavior_source, 0) + 1
            next_observation, reward, done, info = self.env.step(action.allocation.detach().cpu().numpy())
            transition = TensorTransition.from_step(
                obs=self.observation,
                action=action,
                reward=float(reward),
                next_obs=next_observation,
                done=bool(done),
            )
            transitions.append(transition)

            rewards.append(float(reward))
            cooperation_rates.append(float(info.get("actual_cooperation_rate", np.asarray(next_observation["x_actual"]).mean())))
            mean_resources.append(float(np.asarray(next_observation["resources"]).mean()))
            gini_values.append(float(info.get("gini", np.asarray(next_observation["gini"]).item())))
            mean_payoffs.append(float(np.asarray(info.get("payoff", np.zeros_like(next_observation["resources"]))).mean()))
            mean_pool_growns.append(float(np.asarray(next_observation["pool_grown"]).mean()))
            mean_pool_raws.append(float(np.asarray(next_observation["pool_raw"]).mean()))
            self.total_env_steps += 1
            self.observation = next_observation
            if done:
                completed_episodes += 1
                self._reset_environment()

        metrics = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "episodes_completed": float(completed_episodes),
            "env_steps": float(self.total_env_steps),
            "steps_collected": float(len(transitions)),
            "mean_actual_cooperation_rate": float(np.mean(cooperation_rates)) if cooperation_rates else 0.0,
            "mean_resource": float(np.mean(mean_resources)) if mean_resources else 0.0,
            "mean_gini": float(np.mean(gini_values)) if gini_values else 0.0,
            "mean_payoff": float(np.mean(mean_payoffs)) if mean_payoffs else 0.0,
            "mean_pool_grown": float(np.mean(mean_pool_growns)) if mean_pool_growns else 0.0,
            "mean_pool_raw": float(np.mean(mean_pool_raws)) if mean_pool_raws else 0.0,
            "behavior_source_counts": behavior_source_counts,
        }
        replay_batch = stack_tensor_transitions(transitions)
        return RolloutResult(replay_batch=replay_batch, metrics=metrics)

    def _ensure_environment(self) -> None:
        if self.env is None or self.observation is None:
            self._reset_environment()

    def _reset_environment(self) -> None:
        self.env, self.env_metadata = self.env_factory.sample_environment(self.rng)
        reset_seed = int(self.rng.integers(0, 2**31 - 1))
        self.observation = self.env.reset(seed=reset_seed)
        self.current_warmup_behavior_source = None
        if self.train_config.warmup_selection_granularity == "per_episode":
            self.current_warmup_behavior_source = self._select_warmup_behavior_source()

    def _resolve_warmup_behavior_source(self) -> str:
        if self.train_config.warmup_selection_granularity == "per_step":
            return self._select_warmup_behavior_source()
        if self.current_warmup_behavior_source is None:
            self.current_warmup_behavior_source = self._select_warmup_behavior_source()
        return self.current_warmup_behavior_source

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

    def _sample_warmup_action(self, behavior_source: str):
        assert self.observation is not None

        if behavior_source == "random_logits":
            return self.explorer.sample_random_logits_action(
                ego_mask=self.observation["local_mask"],
                pool_values=self.observation["pool_grown"],
                rng=self.rng,
                device=self.device,
            )

        if behavior_source == "uniform":
            heuristic_allocation = self.uniform_policy.allocate(self.observation)
        elif behavior_source == "proportional":
            heuristic_allocation = self.proportional_policy.allocate(self.observation)
        elif behavior_source == "constant_mix":
            heuristic_allocation = self.constant_mix_policy.allocate(self.observation)
        elif behavior_source == "pool_power_mix":
            heuristic_allocation = self.pool_power_mix_policy.allocate(self.observation)
        else:
            raise ValueError("Unsupported warm-up behavior source: {0}".format(behavior_source))

        return self.explorer.action_from_allocation(
            allocation=heuristic_allocation,
            ego_mask=self.observation["local_mask"],
            pool_values=self.observation["pool_grown"],
            noise_std=self.train_config.warmup_logit_noise_std,
            noise_clip=self.train_config.warmup_logit_noise_clip,
            device=self.device,
        )


def _parallel_rollout_worker_main(
    connection,
    actor_config: GNNPolicyConfig,
    actor_state_dict: dict[str, Any],
    env_factory: RandomizedEnvFactory,
    config: WorkerConfig,
    train_config: GraphTD3Config,
    device: str,
) -> None:
    actor = GNNAllocationPolicy(actor_config)
    actor.load_state_dict(_deserialize_module_state(actor_state_dict))
    worker = RolloutWorker(
        actor=actor,
        explorer=LogitSpaceExplorer(),
        env_factory=env_factory,
        config=config,
        train_config=train_config,
        device=device,
    )

    try:
        while True:
            message = connection.recv()
            command = str(message["command"])
            if command == "close":
                break
            try:
                if command == "sync_actor":
                    worker.sync_actor(
                        actor_state_dict=dict(message["actor_state_dict"]),
                        version=int(message["version"]),
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
                    serialized_result, shared_memory_handles = _serialize_rollout_result(
                        worker.collect(
                            num_steps=int(message["num_steps"]),
                            global_warmup_steps=int(message.get("global_warmup_steps", 0)),
                        )
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
                connection.send(
                    {
                        "status": "error",
                        "error": _serialize_remote_exception(exc),
                    }
                )
    finally:
        connection.close()


class ParallelRolloutWorker:
    def __init__(
        self,
        actor: GNNAllocationPolicy,
        env_factory: RandomizedEnvFactory,
        config: WorkerConfig,
        train_config: GraphTD3Config,
        device: str = "cpu",
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
            ),
        )
        self._process.start()
        child_connection.close()
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

    def sync_actor(self, actor_state_dict: dict[str, torch.Tensor], version: int) -> None:
        self._request_response(
            {
                "command": "sync_actor",
                "actor_state_dict": _serialize_module_state(actor_state_dict),
                "version": int(version),
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

    def start_collect(self, num_steps: int, global_warmup_steps: int = 0) -> None:
        if self._collect_inflight:
            raise RuntimeError("Collect request already in flight for worker {0}.".format(self.config.worker_id))
        self._connection.send(
            {
                "command": "collect",
                "num_steps": int(num_steps),
                "global_warmup_steps": int(global_warmup_steps),
            }
        )
        self._collect_inflight = True

    def finish_collect(self) -> RolloutResult:
        if not self._collect_inflight:
            raise RuntimeError("No in-flight collect request for worker {0}.".format(self.config.worker_id))
        try:
            payload = self._recv_response()
            return _deserialize_rollout_result(payload)
        finally:
            self._collect_inflight = False

    def finish_collect_ready(self) -> RolloutResult:
        if not self._collect_inflight:
            raise RuntimeError("No in-flight collect request for worker {0}.".format(self.config.worker_id))
        try:
            payload = self._recv_response(ready=True)
            return _deserialize_rollout_result(payload)
        finally:
            self._collect_inflight = False

    def collect(self, num_steps: int, global_warmup_steps: int = 0) -> RolloutResult:
        self.start_collect(num_steps=num_steps, global_warmup_steps=global_warmup_steps)
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
