from __future__ import annotations

from dataclasses import replace
from typing import Any

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
from Project1.policies.gnn_rl import GNNAllocationPolicy

from .config import DomainRandomizationConfig, GraphTD3Config, WorkerConfig
from .data import Transition
from .exploration import LogitSpaceExplorer
from .replay import ReplayBuffer


def _clone_graph_from_env(env: SPGGEnv) -> dict[int, list[int]]:
    return {node: list(neighbors) for node, neighbors in enumerate(env.graph.neighbors)}


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

        network_type = str(rng.choice(self.randomization.network_types))
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
        replay_buffer: ReplayBuffer,
        explorer: LogitSpaceExplorer,
        env_factory: RandomizedEnvFactory,
        config: WorkerConfig,
        train_config: GraphTD3Config,
        device: torch.device | str = "cpu",
    ):
        self.actor = actor.to(device)
        self.actor.eval()
        self.replay_buffer = replay_buffer
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

    def sync_actor(self, actor_state_dict: dict[str, torch.Tensor], version: int) -> None:
        self.actor.load_state_dict(actor_state_dict)
        self.actor.eval()
        self.actor_version = version

    def collect(self, num_steps: int) -> dict[str, float]:
        rewards: list[float] = []
        completed_episodes = 0

        for _ in range(num_steps):
            self._ensure_environment()
            assert self.observation is not None

            if self.total_env_steps < self.train_config.warmup_steps:
                action = self.explorer.sample_random_logits_action(
                    ego_mask=self.observation["local_mask"],
                    pool_values=self.observation["pool_grown"],
                    rng=self.rng,
                    device=self.device,
                )
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

            next_observation, reward, done, info = self.env.step(action.allocation.detach().cpu().numpy())
            transition = Transition(
                obs=self.observation,
                action=action.to_numpy(),
                reward=float(reward),
                next_obs=next_observation,
                done=bool(done),
                info=info,
                metadata={
                    "worker_id": self.config.worker_id,
                    "actor_version": self.actor_version,
                    **self.env_metadata,
                },
            )
            self.replay_buffer.add(transition)

            rewards.append(float(reward))
            self.total_env_steps += 1
            self.observation = next_observation
            if done:
                completed_episodes += 1
                self._reset_environment()

        return {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "episodes_completed": float(completed_episodes),
            "env_steps": float(self.total_env_steps),
        }

    def _ensure_environment(self) -> None:
        if self.env is None or self.observation is None:
            self._reset_environment()

    def _reset_environment(self) -> None:
        self.env, self.env_metadata = self.env_factory.sample_environment(self.rng)
        reset_seed = int(self.rng.integers(0, 2**31 - 1))
        self.observation = self.env.reset(seed=reset_seed)
