from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from torch import Tensor, nn

from Project1.env import Observation
from Project1.policies.gnn_rl import (
    MLP,
    ObservationGraphBuilder,
    TwoLayerGraphNetBackbone,
    extract_ego_subgraph,
)


def _masked_edge_mean_by_receiver(edge_features: Tensor, edge_mask: Tensor) -> Tensor:
    mask = edge_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
    counts = edge_mask.sum(dim=0).clamp_min(1).to(dtype=edge_features.dtype).unsqueeze(-1)
    return (edge_features * mask).sum(dim=0) / counts


def _masked_global_edge_mean(edge_features: Tensor, edge_mask: Tensor) -> Tensor:
    mask = edge_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
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

        pool_tokens: list[Tensor] = []
        for center_index in range(backbone_output.node_embeddings.size(0)):
            ego_subgraph = extract_ego_subgraph(backbone_output, center_index=center_index)
            member_indices = ego_subgraph.member_indices

            local_alloc = allocation[center_index, member_indices].unsqueeze(-1)
            local_transfers = local_alloc * ego_subgraph.pool_value
            local_nodes = ego_subgraph.local_node_features[:, :-1]
            center_indicator = ego_subgraph.local_node_features[:, -1:]
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
            weighted_action_summary = (local_alloc * encoded_actions).sum(dim=0)
            ego_edge_summary = _masked_global_edge_mean(
                ego_subgraph.local_edge_features,
                ego_subgraph.local_edge_mask,
            )
            center_node = local_nodes[ego_subgraph.center_local_index]
            pool_size = local_nodes.new_tensor([float(member_indices.numel())])
            pool_value = ego_subgraph.pool_value.view(1)

            pool_input = torch.cat(
                [
                    backbone_output.global_embedding,
                    center_node,
                    weighted_action_summary,
                    ego_edge_summary,
                    pool_value,
                    pool_size,
                ],
                dim=0,
            )
            pool_tokens.append(self.pool_encoder(pool_input))

        graph_pool = torch.stack(pool_tokens, dim=0).mean(dim=0)
        q_input = torch.cat([backbone_output.global_embedding, graph_pool], dim=0)
        return self.q_head(q_input).squeeze(-1)

    def forward_batch(
        self,
        observations: Sequence[Observation],
        actions: Sequence[Tensor | np.ndarray],
    ) -> Tensor:
        values = [self.forward(obs, action) for obs, action in zip(observations, actions)]
        return torch.stack(values, dim=0)


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
