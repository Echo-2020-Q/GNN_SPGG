from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from Project1.env import Observation
from Project1.policies.gnn_rl import (
    MLP,
    ObservationGraphBuilder,
    TwoLayerGraphNetBackbone,
    ensure_batched_backbone_output,
    extract_batched_center_chunk_ego_subgraphs,
)


def _masked_edge_mean_by_receiver(edge_features: Tensor, edge_mask: Tensor) -> Tensor:
    mask = edge_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
    if edge_features.ndim == 4:
        counts = edge_mask.sum(dim=1).clamp_min(1).to(dtype=edge_features.dtype).unsqueeze(-1)
        return (edge_features * mask).sum(dim=1) / counts
    counts = edge_mask.sum(dim=0).clamp_min(1).to(dtype=edge_features.dtype).unsqueeze(-1)
    return (edge_features * mask).sum(dim=0) / counts


def _masked_global_edge_mean(edge_features: Tensor, edge_mask: Tensor) -> Tensor:
    mask = edge_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
    if edge_features.ndim == 4:
        count = edge_mask.sum(dim=(1, 2)).clamp_min(1).to(dtype=edge_features.dtype).unsqueeze(-1)
        return (edge_features * mask).sum(dim=(1, 2)) / count
    count = edge_mask.sum().clamp_min(1).to(dtype=edge_features.dtype)
    return (edge_features * mask).sum(dim=(0, 1)) / count


@dataclass
class GraphActionCriticConfig:
    state_hidden_dim: int = 64
    action_hidden_dim: int = 64
    pool_hidden_dim: int = 64
    q_hidden_dim: int = 64

    def __post_init__(self) -> None:
        if self.state_hidden_dim <= 0:
            raise ValueError("state_hidden_dim must be positive.")
        if self.action_hidden_dim <= 0:
            raise ValueError("action_hidden_dim must be positive.")
        if self.pool_hidden_dim <= 0:
            raise ValueError("pool_hidden_dim must be positive.")
        if self.q_hidden_dim <= 0:
            raise ValueError("q_hidden_dim must be positive.")


class GraphActionCritic(nn.Module):
    """Graph Q-network for variable-degree simplex-constrained joint actions."""

    def __init__(self, config: GraphActionCriticConfig):
        super().__init__()
        self.config = config
        self.graph_builder = ObservationGraphBuilder()
        hidden_dim = config.state_hidden_dim
        self.state_encoder = TwoLayerGraphNetBackbone(
            global_input_dim=self.graph_builder.global_input_dim,
            node_input_dim=self.graph_builder.node_input_dim,
            edge_input_dim=self.graph_builder.edge_input_dim,
            hidden_dim=hidden_dim,
        )
        self.action_encoder = MLP((2 * hidden_dim) + 3, config.action_hidden_dim, config.action_hidden_dim)
        self.pool_encoder = MLP(
            (2 * hidden_dim) + config.action_hidden_dim + hidden_dim + 2,
            config.pool_hidden_dim,
            config.pool_hidden_dim,
        )
        self.q_head = MLP(hidden_dim + config.pool_hidden_dim, config.q_hidden_dim, 1)

    def forward(self, obs: Observation, allocation_matrix: Tensor | np.ndarray) -> Tensor:
        device = next(self.parameters()).device
        graph_input = self.graph_builder.build(obs, device=device)
        backbone_output = self.state_encoder(graph_input)
        allocation = torch.as_tensor(allocation_matrix, dtype=torch.float32, device=device)
        q_values = self._forward_tensor_batch_from_backbone_output(
            ensure_batched_backbone_output(backbone_output),
            allocation.unsqueeze(0),
        )
        return q_values[0]

    def _forward_batch_same_size(
        self,
        observations: Sequence[Observation],
        actions: Sequence[Tensor | np.ndarray],
    ) -> Tensor:
        if not observations:
            raise ValueError("observations must contain at least one item.")
        if len(observations) != len(actions):
            raise ValueError("observations and actions must have the same length.")

        device = next(self.parameters()).device
        graph_input = self.graph_builder.build_batch(list(observations), device=device)
        allocation = torch.stack([torch.as_tensor(action, dtype=torch.float32, device=device) for action in actions], dim=0)
        return self._forward_tensor_batch_from_graph_input(graph_input, allocation)

    def _forward_tensor_batch_from_graph_input(
        self,
        graph_input,
        allocation: Tensor,
    ) -> Tensor:
        backbone_output = self.state_encoder(graph_input)
        return self._forward_tensor_batch_from_backbone_output(backbone_output, allocation)

    def _forward_tensor_batch_from_backbone_output(
        self,
        backbone_output,
        allocation: Tensor,
    ) -> Tensor:
        backbone_output = ensure_batched_backbone_output(backbone_output)
        if allocation.ndim == 2:
            allocation = allocation.unsqueeze(0)
        if allocation.ndim != 3:
            raise ValueError("allocation must have shape [batch_size, num_nodes, num_nodes].")

        batch_size, num_nodes = backbone_output.node_embeddings.shape[:2]
        device = backbone_output.node_embeddings.device
        ego_subgraph = extract_batched_center_chunk_ego_subgraphs(
            backbone_output,
            torch.arange(num_nodes, device=device, dtype=torch.int64),
        )
        local_node_mask = ego_subgraph.local_node_mask

        local_alloc = allocation[
            ego_subgraph.batch_indices[:, None],
            ego_subgraph.center_indices[:, None],
            ego_subgraph.member_indices,
        ].unsqueeze(-1)
        local_alloc = local_alloc * local_node_mask.unsqueeze(-1).to(dtype=allocation.dtype)
        local_transfers = local_alloc * ego_subgraph.pool_value.view(-1, 1, 1)
        local_nodes = ego_subgraph.local_node_features[:, :, :-1]
        center_indicator = ego_subgraph.local_node_features[:, :, -1:]
        local_edge_context = _masked_edge_mean_by_receiver(
            ego_subgraph.local_edge_features,
            ego_subgraph.local_edge_mask,
        )

        action_inputs = torch.cat(
            [
                local_nodes,
                local_edge_context,
                local_alloc,
                local_transfers,
                center_indicator,
            ],
            dim=-1,
        )
        encoded_actions = self.action_encoder(action_inputs)
        weighted_action_summary = (local_alloc * encoded_actions).sum(dim=1)
        ego_edge_summary = _masked_global_edge_mean(
            ego_subgraph.local_edge_features,
            ego_subgraph.local_edge_mask,
        )
        flat_indices = torch.arange(ego_subgraph.member_indices.size(0), device=device)
        center_node = local_nodes[flat_indices, ego_subgraph.center_local_indices]
        pool_size = local_node_mask.sum(dim=1).to(dtype=local_nodes.dtype).unsqueeze(-1)
        pool_value = ego_subgraph.pool_value.view(-1, 1)

        pool_input = torch.cat(
            [
                backbone_output.global_embedding[ego_subgraph.batch_indices],
                center_node,
                weighted_action_summary,
                ego_edge_summary,
                pool_value,
                pool_size,
            ],
            dim=-1,
        )
        pool_tokens = self.pool_encoder(pool_input).view(batch_size, num_nodes, -1)
        graph_pool = pool_tokens.mean(dim=1)
        q_input = torch.cat([backbone_output.global_embedding, graph_pool], dim=-1)
        return self.q_head(q_input).squeeze(-1)

    def forward_tensor_batch(
        self,
        observations: Mapping[str, Tensor],
        allocation_matrix: Tensor | np.ndarray,
    ) -> Tensor:
        device = next(self.parameters()).device
        graph_input = self.graph_builder.build_tensor_batch(observations, device=device)
        allocation = torch.as_tensor(allocation_matrix, dtype=torch.float32, device=device)
        return self._forward_tensor_batch_from_graph_input(graph_input, allocation)

    def forward_batch(
        self,
        observations: Sequence[Observation],
        actions: Sequence[Tensor | np.ndarray],
    ) -> Tensor:
        if len(observations) != len(actions):
            raise ValueError("observations and actions must have the same length.")
        if not observations:
            return torch.empty(0, dtype=torch.float32, device=next(self.parameters()).device)

        grouped_indices: dict[int, list[int]] = {}
        for index, observation in enumerate(observations):
            num_nodes = int(np.asarray(observation["local_mask"]).shape[0])
            grouped_indices.setdefault(num_nodes, []).append(index)

        outputs: list[Tensor | None] = [None] * len(observations)
        for indices in grouped_indices.values():
            group_observations = [observations[index] for index in indices]
            group_actions = [actions[index] for index in indices]
            group_values = self._forward_batch_same_size(group_observations, group_actions)
            for local_index, original_index in enumerate(indices):
                outputs[original_index] = group_values[local_index]

        return torch.stack([value for value in outputs if value is not None], dim=0)


class TwinCritic(nn.Module):
    def __init__(self, critic1: GraphActionCritic, critic2: GraphActionCritic):
        super().__init__()
        self.critic1 = critic1
        self.critic2 = critic2

    def forward_batch(
        self,
        observations: Sequence[Observation],
        actions: Sequence[Tensor | np.ndarray],
    ) -> tuple[Tensor, Tensor]:
        return (
            self.critic1.forward_batch(observations, actions),
            self.critic2.forward_batch(observations, actions),
        )

    def forward_tensor_batch(
        self,
        observations: Mapping[str, Tensor],
        actions: Tensor | np.ndarray,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.critic1.forward_tensor_batch(observations, actions),
            self.critic2.forward_tensor_batch(observations, actions),
        )
