from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch.distributions import Dirichlet
from torch import Tensor, nn

from Project1.env import Observation


@dataclass
class GNNPolicyConfig:
    hidden_dim: int = 64
    num_message_passing_layers: int = 2
    local_hidden_dim: int | None = None
    score_hidden_dim: int | None = None
    critic_hidden_dim: int | None = None
    temperature: float = 1.0
    action_distribution: str = "softmax"
    dirichlet_alpha_floor: float = 1e-3

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_message_passing_layers != 2:
            raise ValueError("Current backbone is fixed to exactly two GraphNet blocks.")
        if self.local_hidden_dim is None:
            self.local_hidden_dim = self.hidden_dim
        if self.score_hidden_dim is None:
            self.score_hidden_dim = self.local_hidden_dim
        if self.critic_hidden_dim is None:
            self.critic_hidden_dim = self.hidden_dim
        if self.local_hidden_dim <= 0:
            raise ValueError("local_hidden_dim must be positive.")
        if self.score_hidden_dim <= 0:
            raise ValueError("score_hidden_dim must be positive.")
        if self.critic_hidden_dim <= 0:
            raise ValueError("critic_hidden_dim must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.action_distribution not in {"softmax", "dirichlet"}:
            raise ValueError("action_distribution must be one of {'softmax', 'dirichlet'}.")
        if self.dirichlet_alpha_floor <= 0.0:
            raise ValueError("dirichlet_alpha_floor must be positive.")


@dataclass(frozen=True)
class GraphTensorInput:
    """Dense GraphNet input for a single graph.

    Shapes:
    - global_features: [global_input_dim]
    - node_features: [num_nodes, node_input_dim]
    - edge_features: [num_nodes, num_nodes, edge_input_dim]
    - edge_mask: [num_nodes, num_nodes]
    - ego_mask: [num_nodes, num_nodes]
    - pool_values: [num_nodes]
    """

    global_features: Tensor
    node_features: Tensor
    edge_features: Tensor
    edge_mask: Tensor
    node_mask: Tensor
    ego_mask: Tensor
    pool_values: Tensor


@dataclass(frozen=True)
class GraphTensorState:
    """Intermediate GraphNet state with dense edge tensors."""

    global_features: Tensor
    node_features: Tensor
    edge_features: Tensor
    edge_mask: Tensor
    node_mask: Tensor


@dataclass(frozen=True)
class BackboneOutput:
    """Final backbone embeddings used by the local allocation head.

    Shapes:
    - global_embedding: [backbone_hidden_dim]
    - node_embeddings: [num_nodes, backbone_hidden_dim]
    - edge_embeddings: [num_nodes, num_nodes, backbone_hidden_dim]
    - edge_mask: [num_nodes, num_nodes]
    - ego_mask: [num_nodes, num_nodes]
    - pool_values: [num_nodes]
    """

    global_embedding: Tensor
    node_embeddings: Tensor
    edge_embeddings: Tensor
    edge_mask: Tensor
    node_mask: Tensor
    ego_mask: Tensor
    pool_values: Tensor


@dataclass(frozen=True)
class EgoSubgraph:
    """Induced ego-subgraph centered at one node.

    Shapes:
    - member_indices: [ego_num_nodes]
    - local_node_features: [ego_num_nodes, backbone_hidden_dim + 1]
    - local_edge_features: [ego_num_nodes, ego_num_nodes, backbone_hidden_dim]
    - local_edge_mask: [ego_num_nodes, ego_num_nodes]
    - local_global_features: [global_hidden_dim + backbone_hidden_dim + 2]
    """

    center_index: int
    center_local_index: int
    member_indices: Tensor
    local_node_features: Tensor
    local_node_mask: Tensor
    local_edge_features: Tensor
    local_edge_mask: Tensor
    local_global_features: Tensor
    pool_value: Tensor


@dataclass(frozen=True)
class BatchedEgoSubgraph:
    """Padded ego-subgraph batch centered at the same node index for multiple graphs."""

    center_index: int
    center_local_indices: Tensor
    member_indices: Tensor
    local_node_features: Tensor
    local_node_mask: Tensor
    local_edge_features: Tensor
    local_edge_mask: Tensor
    local_global_features: Tensor
    pool_value: Tensor


@dataclass(frozen=True)
class FlattenedBatchedEgoSubgraphs:
    """Flattened ego-subgraph batch over (graph, center) pairs."""

    batch_indices: Tensor
    center_indices: Tensor
    center_local_indices: Tensor
    member_indices: Tensor
    local_node_features: Tensor
    local_node_mask: Tensor
    local_edge_features: Tensor
    local_edge_mask: Tensor
    local_global_features: Tensor
    pool_value: Tensor


@dataclass(frozen=True)
class LocalGraphOutput:
    """Local tiny GraphNet output on one induced ego-subgraph."""

    global_embedding: Tensor
    node_embeddings: Tensor
    edge_embeddings: Tensor
    edge_mask: Tensor
    node_mask: Tensor


@dataclass
class PolicyOutput:
    allocation_matrix: Tensor
    transferred_resources: Tensor
    incoming_resources: Tensor
    value: Tensor
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
    logits: Tensor | None = None
    concentration: Tensor | None = None
    global_embedding: Tensor | None = None
    node_embeddings: Tensor | None = None
    edge_embeddings: Tensor | None = None


@dataclass(frozen=True)
class BatchedPolicyOutput:
    allocation_matrix: Tensor
    transferred_resources: Tensor
    incoming_resources: Tensor
    value: Tensor
    logits: Tensor | None = None
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
    concentration: Tensor | None = None



class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.net(inputs)


def _apply_partitioned_linear(
    linear: nn.Linear,
    inputs: Sequence[Tensor],
    input_dims: Sequence[int],
) -> Tensor:
    """Apply one linear layer to logical input parts without materializing their concat."""

    if len(inputs) != len(input_dims):
        raise ValueError("inputs and input_dims must have the same length.")
    if sum(int(item) for item in input_dims) != int(linear.in_features):
        raise ValueError("input_dims do not match the linear layer input width.")

    output: Tensor | None = None
    for index, (part, weight) in enumerate(zip(inputs, linear.weight.split(tuple(int(item) for item in input_dims), dim=1))):
        projected = F.linear(part, weight, linear.bias if index == 0 else None)
        output = projected if output is None else (output + projected)

    if output is None:
        raise ValueError("inputs must contain at least one tensor.")
    return output


def _apply_partitioned_mlp(
    mlp: MLP,
    inputs: Sequence[Tensor],
    input_dims: Sequence[int],
) -> Tensor:
    """Equivalent to mlp(torch.cat(inputs, dim=-1)) without allocating the concat tensor."""

    hidden = _apply_partitioned_linear(mlp.net[0], inputs, input_dims)
    hidden = mlp.net[1](hidden)
    return mlp.net[2](hidden)


class ObservationGraphBuilder:
    """Builds the dense GraphNet tensors (u, V, E) from the environment observation."""

    node_feature_names = (
        "pool_raw_norm",
        "resource_norm",
        "degree_norm",
        "strategy_norm",
    )
    global_feature_names = (
        "x_actual",
        "resource_norm",
        "pool_raw_norm",
        "gini",
    )

    @property
    def node_input_dim(self) -> int:
        return len(self.node_feature_names)

    @property
    def edge_input_dim(self) -> int:
        return 1

    @property
    def global_input_dim(self) -> int:
        return len(self.global_feature_names)

    def build(self, observation: Observation, device: torch.device) -> GraphTensorInput:
        ego_mask = torch.as_tensor(observation["local_mask"], dtype=torch.bool, device=device)
        if ego_mask.ndim != 2 or ego_mask.size(0) != ego_mask.size(1):
            raise ValueError("observation['local_mask'] must be a square matrix.")

        edge_mask = ego_mask
        node_mask = torch.ones(ego_mask.size(0), dtype=torch.bool, device=device)

        node_features = torch.stack(
            [
                torch.as_tensor(observation[key], dtype=torch.float32, device=device)
                for key in self.node_feature_names
            ],
            dim=-1,
        )
        global_features = torch.stack(
            [
                torch.as_tensor(observation[key], dtype=torch.float32, device=device).mean()
                for key in self.global_feature_names
            ],
            dim=0,
        )
        edge_features = edge_mask.to(dtype=torch.float32).unsqueeze(-1)
        pool_values = torch.as_tensor(observation["pool_grown"], dtype=torch.float32, device=device)

        return GraphTensorInput(
            global_features=global_features,
            node_features=node_features,
            edge_features=edge_features,
            edge_mask=edge_mask,
            node_mask=node_mask,
            ego_mask=ego_mask,
            pool_values=pool_values,
        )

    def build_batch(
        self,
        observations: list[Observation] | tuple[Observation, ...],
        device: torch.device,
    ) -> GraphTensorInput:
        if not observations:
            raise ValueError("observations must contain at least one item.")

        first_num_nodes = int(torch.as_tensor(observations[0]["local_mask"]).shape[0])
        for observation in observations:
            local_mask = torch.as_tensor(observation["local_mask"])
            if local_mask.ndim != 2 or local_mask.size(0) != local_mask.size(1):
                raise ValueError("Each observation['local_mask'] must be a square matrix.")
            if int(local_mask.size(0)) != first_num_nodes:
                raise ValueError("build_batch requires all observations to have the same number of nodes.")

        batch_size = len(observations)
        ego_mask = torch.stack(
            [
                torch.as_tensor(observation["local_mask"], dtype=torch.bool, device=device)
                for observation in observations
            ],
            dim=0,
        )
        edge_mask = ego_mask
        node_mask = torch.ones((batch_size, first_num_nodes), dtype=torch.bool, device=device)
        node_features = torch.stack(
            [
                torch.stack(
                    [
                        torch.as_tensor(observation[key], dtype=torch.float32, device=device)
                        for key in self.node_feature_names
                    ],
                    dim=-1,
                )
                for observation in observations
            ],
            dim=0,
        )
        global_features = torch.stack(
            [
                torch.stack(
                    [
                        torch.as_tensor(observation[key], dtype=torch.float32, device=device).mean()
                        for key in self.global_feature_names
                    ],
                    dim=0,
                )
                for observation in observations
            ],
            dim=0,
        )
        edge_features = edge_mask.to(dtype=torch.float32).unsqueeze(-1)
        pool_values = torch.stack(
            [
                torch.as_tensor(observation["pool_grown"], dtype=torch.float32, device=device)
                for observation in observations
            ],
            dim=0,
        )

        return GraphTensorInput(
            global_features=global_features,
            node_features=node_features,
            edge_features=edge_features,
            edge_mask=edge_mask,
            node_mask=node_mask,
            ego_mask=ego_mask,
            pool_values=pool_values,
        )

    def build_tensor_batch(
        self,
        observations: Mapping[str, Tensor],
        device: torch.device,
    ) -> GraphTensorInput:
        ego_mask = torch.as_tensor(observations["local_mask"], dtype=torch.bool, device=device)
        if ego_mask.ndim != 3 or ego_mask.size(1) != ego_mask.size(2):
            raise ValueError("Batched observation['local_mask'] must have shape [batch_size, num_nodes, num_nodes].")

        batch_size, num_nodes = ego_mask.shape[:2]
        node_mask = torch.ones((batch_size, num_nodes), dtype=torch.bool, device=device)
        node_features = torch.stack(
            [
                torch.as_tensor(observations[key], dtype=torch.float32, device=device)
                for key in self.node_feature_names
            ],
            dim=-1,
        )
        global_feature_values: list[Tensor] = []
        for key in self.global_feature_names:
            value = torch.as_tensor(observations[key], dtype=torch.float32, device=device)
            if value.ndim <= 1:
                global_feature_values.append(value.reshape(batch_size))
            else:
                global_feature_values.append(value.reshape(batch_size, -1).mean(dim=1))
        global_features = torch.stack(global_feature_values, dim=-1)
        pool_values = torch.as_tensor(observations["pool_grown"], dtype=torch.float32, device=device)
        edge_mask = ego_mask
        edge_features = edge_mask.to(dtype=torch.float32).unsqueeze(-1)

        return GraphTensorInput(
            global_features=global_features,
            node_features=node_features,
            edge_features=edge_features,
            edge_mask=edge_mask,
            node_mask=node_mask,
            ego_mask=ego_mask,
            pool_values=pool_values,
        )


def _masked_edge_sum_by_receiver(edge_features: Tensor, edge_mask: Tensor) -> Tensor:
    mask = edge_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
    if edge_features.ndim == 4:
        return (edge_features * mask).sum(dim=1)
    return (edge_features * mask).sum(dim=0)


def _masked_global_edge_normalized_sum(edge_features: Tensor, edge_mask: Tensor) -> Tensor:
    mask = edge_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
    if edge_features.ndim == 4:
        edge_count = edge_mask.sum(dim=(1, 2)).clamp_min(1).to(dtype=edge_features.dtype).unsqueeze(-1)
        return (edge_features * mask).sum(dim=(1, 2)) / edge_count
    edge_count = edge_mask.sum().clamp_min(1).to(dtype=edge_features.dtype)
    return (edge_features * mask).sum(dim=(0, 1)) / edge_count


class GraphNetBlock(nn.Module):
    """Single dense GraphNet block.

    Input shapes:
    - u: [global_input_dim]
    - V: [num_nodes, node_input_dim]
    - E: [num_nodes, num_nodes, edge_input_dim]
    - edge_mask: [num_nodes, num_nodes]

    Output shapes:
    - u': [global_output_dim]
    - V': [num_nodes, node_output_dim]
    - E': [num_nodes, num_nodes, edge_output_dim]
    """

    def __init__(
        self,
        global_input_dim: int,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int,
        global_output_dim: int,
        node_output_dim: int,
        edge_output_dim: int,
    ):
        super().__init__()
        self.edge_model = MLP(
            edge_input_dim + (2 * node_input_dim) + global_input_dim,
            hidden_dim,
            edge_output_dim,
        )
        self.node_model = MLP(
            node_input_dim + edge_output_dim + global_input_dim,
            hidden_dim,
            node_output_dim,
        )
        self.global_model = MLP(
            global_input_dim + node_output_dim + edge_output_dim,
            hidden_dim,
            global_output_dim,
        )

    def forward(self, state: GraphTensorState) -> GraphTensorState:
        squeeze_batch = state.node_features.ndim == 2
        if squeeze_batch:
            global_features = state.global_features.unsqueeze(0)
            node_features = state.node_features.unsqueeze(0)
            edge_features = state.edge_features.unsqueeze(0)
            edge_mask = state.edge_mask.unsqueeze(0)
            node_mask = state.node_mask.unsqueeze(0)
        else:
            global_features = state.global_features
            node_features = state.node_features
            edge_features = state.edge_features
            edge_mask = state.edge_mask
            node_mask = state.node_mask

        batch_size, num_nodes = node_features.shape[:2]
        updated_edges = _apply_partitioned_mlp(
            self.edge_model,
            (
                edge_features,
                node_features[:, :, None, :],
                node_features[:, None, :, :],
                global_features[:, None, None, :],
            ),
            (
                int(edge_features.size(-1)),
                int(node_features.size(-1)),
                int(node_features.size(-1)),
                int(global_features.size(-1)),
            ),
        )
        updated_edges = updated_edges * edge_mask.unsqueeze(-1).to(dtype=updated_edges.dtype)

        aggregated_edge_messages = _masked_edge_sum_by_receiver(updated_edges, edge_mask)
        updated_nodes = _apply_partitioned_mlp(
            self.node_model,
            (
                node_features,
                aggregated_edge_messages,
                global_features[:, None, :],
            ),
            (
                int(node_features.size(-1)),
                int(aggregated_edge_messages.size(-1)),
                int(global_features.size(-1)),
            ),
        )
        updated_nodes = updated_nodes * node_mask.unsqueeze(-1).to(dtype=updated_nodes.dtype)

        node_count = node_mask.sum(dim=1).clamp_min(1).to(dtype=updated_nodes.dtype).unsqueeze(-1)
        aggregated_nodes = updated_nodes.sum(dim=1) / node_count
        aggregated_edges = _masked_global_edge_normalized_sum(updated_edges, edge_mask)
        updated_global = _apply_partitioned_mlp(
            self.global_model,
            (
                global_features,
                aggregated_nodes,
                aggregated_edges,
            ),
            (
                int(global_features.size(-1)),
                int(aggregated_nodes.size(-1)),
                int(aggregated_edges.size(-1)),
            ),
        )
        if squeeze_batch:
            return GraphTensorState(
                global_features=updated_global.squeeze(0),
                node_features=updated_nodes.squeeze(0),
                edge_features=updated_edges.squeeze(0),
                edge_mask=edge_mask.squeeze(0),
                node_mask=node_mask.squeeze(0),
            )

        return GraphTensorState(
            global_features=updated_global,
            node_features=updated_nodes,
            edge_features=updated_edges,
            edge_mask=edge_mask,
            node_mask=node_mask,
        )


class TwoLayerGraphNetBackbone(nn.Module):
    """Two sequential GraphNet blocks producing u*, V*, E*."""

    def __init__(
        self,
        global_input_dim: int,
        node_input_dim: int,
        edge_input_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.block1 = GraphNetBlock(
            global_input_dim=global_input_dim,
            node_input_dim=node_input_dim,
            edge_input_dim=edge_input_dim,
            hidden_dim=hidden_dim,
            global_output_dim=hidden_dim,
            node_output_dim=hidden_dim,
            edge_output_dim=hidden_dim,
        )
        self.block2 = GraphNetBlock(
            global_input_dim=hidden_dim,
            node_input_dim=hidden_dim,
            edge_input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            global_output_dim=hidden_dim,
            node_output_dim=hidden_dim,
            edge_output_dim=hidden_dim,
        )

    def forward(self, graph_input: GraphTensorInput) -> BackboneOutput:
        state = GraphTensorState(
            global_features=graph_input.global_features,
            node_features=graph_input.node_features,
            edge_features=graph_input.edge_features,
            edge_mask=graph_input.edge_mask,
            node_mask=graph_input.node_mask,
        )
        state = self.block1(state)
        state = self.block2(state)
        return BackboneOutput(
            global_embedding=state.global_features,
            node_embeddings=state.node_features,
            edge_embeddings=state.edge_features,
            edge_mask=graph_input.edge_mask,
            node_mask=graph_input.node_mask,
            ego_mask=graph_input.ego_mask,
            pool_values=graph_input.pool_values,
        )


def extract_ego_subgraph(backbone_output: BackboneOutput, center_index: int) -> EgoSubgraph:
    """Extract the induced ego-subgraph S_i = {i} union N(i) from the final backbone embeddings."""

    if center_index < 0 or center_index >= backbone_output.node_embeddings.size(0):
        raise IndexError("center_index is out of range.")

    member_indices = backbone_output.ego_mask[center_index].nonzero(as_tuple=False).squeeze(-1)
    if member_indices.numel() == 0:
        raise RuntimeError("Each ego-subgraph must contain at least the center node.")

    center_local_matches = (member_indices == center_index).nonzero(as_tuple=False).squeeze(-1)
    if center_local_matches.numel() != 1:
        raise RuntimeError("Failed to identify the center node inside its ego-subgraph.")
    center_local_index = int(center_local_matches.item())

    local_node_embeddings = backbone_output.node_embeddings.index_select(0, member_indices)
    local_edge_features = backbone_output.edge_embeddings.index_select(0, member_indices).index_select(1, member_indices)
    local_edge_mask = backbone_output.edge_mask.index_select(0, member_indices).index_select(1, member_indices)

    center_indicator = torch.zeros(
        (member_indices.numel(), 1),
        dtype=local_node_embeddings.dtype,
        device=local_node_embeddings.device,
    )
    center_indicator[center_local_index, 0] = 1.0
    local_node_features = torch.cat([local_node_embeddings, center_indicator], dim=-1)
    local_node_mask = torch.ones(member_indices.numel(), dtype=torch.bool, device=local_node_embeddings.device)

    pool_value = backbone_output.pool_values[center_index].view(1)
    ego_size = local_node_embeddings.new_tensor([float(member_indices.numel())])
    local_global_features = torch.cat(
        [
            backbone_output.global_embedding,
            backbone_output.node_embeddings[center_index],
            pool_value,
            ego_size,
        ],
        dim=0,
    )

    return EgoSubgraph(
        center_index=center_index,
        center_local_index=center_local_index,
        member_indices=member_indices,
        local_node_features=local_node_features,
        local_node_mask=local_node_mask,
        local_edge_features=local_edge_features,
        local_edge_mask=local_edge_mask,
        local_global_features=local_global_features,
        pool_value=pool_value.squeeze(0),
    )


def extract_batched_ego_subgraph(backbone_output: BackboneOutput, center_index: int) -> BatchedEgoSubgraph:
    flattened = extract_batched_center_chunk_ego_subgraphs(backbone_output, [center_index])
    return BatchedEgoSubgraph(
        center_index=center_index,
        center_local_indices=flattened.center_local_indices,
        member_indices=flattened.member_indices,
        local_node_features=flattened.local_node_features,
        local_node_mask=flattened.local_node_mask,
        local_edge_features=flattened.local_edge_features,
        local_edge_mask=flattened.local_edge_mask,
        local_global_features=flattened.local_global_features,
        pool_value=flattened.pool_value,
    )


def ensure_batched_backbone_output(backbone_output: BackboneOutput) -> BackboneOutput:
    if backbone_output.node_embeddings.ndim == 3:
        return backbone_output
    if backbone_output.node_embeddings.ndim != 2:
        raise ValueError("BackboneOutput node_embeddings must have rank 2 or 3.")
    return BackboneOutput(
        global_embedding=backbone_output.global_embedding.unsqueeze(0),
        node_embeddings=backbone_output.node_embeddings.unsqueeze(0),
        edge_embeddings=backbone_output.edge_embeddings.unsqueeze(0),
        edge_mask=backbone_output.edge_mask.unsqueeze(0),
        node_mask=backbone_output.node_mask.unsqueeze(0),
        ego_mask=backbone_output.ego_mask.unsqueeze(0),
        pool_values=backbone_output.pool_values.unsqueeze(0),
    )


def extract_batched_center_chunk_ego_subgraphs(
    backbone_output: BackboneOutput,
    center_indices: Sequence[int] | Tensor,
) -> FlattenedBatchedEgoSubgraphs:
    backbone_output = ensure_batched_backbone_output(backbone_output)
    batch_size, num_nodes, hidden_dim = backbone_output.node_embeddings.shape
    device = backbone_output.node_embeddings.device
    center_indices_tensor = torch.as_tensor(center_indices, dtype=torch.int64, device=device)
    if center_indices_tensor.ndim != 1 or center_indices_tensor.numel() == 0:
        raise ValueError("center_indices must be a non-empty 1D tensor or sequence.")
    if int(center_indices_tensor.min().item()) < 0 or int(center_indices_tensor.max().item()) >= num_nodes:
        raise IndexError("center_indices contain an out-of-range value.")

    num_centers = int(center_indices_tensor.numel())
    member_mask = backbone_output.ego_mask[:, center_indices_tensor, :]
    member_mask_flat = member_mask.reshape(batch_size * num_centers, num_nodes)
    if not torch.all(member_mask_flat.any(dim=1)):
        raise RuntimeError("Each batched ego-subgraph must contain at least the center node.")

    base_indices = torch.arange(num_nodes, device=device).unsqueeze(0).expand(batch_size * num_centers, -1)
    masked_order = torch.where(member_mask_flat, base_indices, base_indices + num_nodes)
    sorted_member_indices = masked_order.argsort(dim=1)
    member_counts = member_mask_flat.sum(dim=1)
    max_members = int(member_counts.max().item())
    member_indices = sorted_member_indices[:, :max_members]
    local_node_mask = torch.gather(member_mask_flat, 1, member_indices)

    expanded_node_embeddings = backbone_output.node_embeddings.unsqueeze(1).expand(batch_size, num_centers, num_nodes, hidden_dim)
    expanded_node_embeddings = expanded_node_embeddings.reshape(batch_size * num_centers, num_nodes, hidden_dim)
    gather_nodes = member_indices.unsqueeze(-1).expand(batch_size * num_centers, max_members, hidden_dim)
    local_node_embeddings = torch.gather(expanded_node_embeddings, 1, gather_nodes)

    batch_indices = torch.arange(batch_size, device=device, dtype=torch.int64).unsqueeze(1).expand(batch_size, num_centers).reshape(-1)
    flat_center_indices = center_indices_tensor.unsqueeze(0).expand(batch_size, num_centers).reshape(-1)
    row_indices = member_indices.unsqueeze(2).expand(batch_size * num_centers, max_members, max_members)
    col_indices = member_indices.unsqueeze(1).expand(batch_size * num_centers, max_members, max_members)
    local_edge_features = backbone_output.edge_embeddings[batch_indices[:, None, None], row_indices, col_indices]
    local_edge_mask = backbone_output.edge_mask[batch_indices[:, None, None], row_indices, col_indices]
    local_edge_mask = local_edge_mask & local_node_mask.unsqueeze(1) & local_node_mask.unsqueeze(2)

    center_local_matches = (member_indices == flat_center_indices.unsqueeze(1)) & local_node_mask
    if not torch.all(center_local_matches.any(dim=1)):
        raise RuntimeError("Failed to identify the center node inside a batched ego-subgraph.")
    center_local_indices = center_local_matches.to(dtype=torch.int64).argmax(dim=1)

    center_indicator = torch.zeros(
        (batch_size * num_centers, max_members, 1),
        dtype=local_node_embeddings.dtype,
        device=device,
    )
    center_indicator[
        torch.arange(batch_size * num_centers, device=device),
        center_local_indices,
        0,
    ] = 1.0
    center_indicator = center_indicator * local_node_mask.unsqueeze(-1).to(dtype=local_node_embeddings.dtype)
    local_node_features = torch.cat([local_node_embeddings, center_indicator], dim=-1)

    pool_value = backbone_output.pool_values[batch_indices, flat_center_indices]
    ego_size = local_node_mask.sum(dim=1).to(dtype=local_node_embeddings.dtype).unsqueeze(-1)
    center_nodes = backbone_output.node_embeddings[batch_indices, flat_center_indices]
    local_global_features = torch.cat(
        [
            backbone_output.global_embedding[batch_indices],
            center_nodes,
            pool_value.unsqueeze(-1),
            ego_size,
        ],
        dim=-1,
    )

    return FlattenedBatchedEgoSubgraphs(
        batch_indices=batch_indices,
        center_indices=flat_center_indices,
        center_local_indices=center_local_indices,
        member_indices=member_indices,
        local_node_features=local_node_features,
        local_node_mask=local_node_mask,
        local_edge_features=local_edge_features,
        local_edge_mask=local_edge_mask,
        local_global_features=local_global_features,
        pool_value=pool_value,
    )


class EgoLocalGraphNet(nn.Module):
    """One-layer tiny GraphNet operating on the induced ego-subgraph."""

    def __init__(self, backbone_hidden_dim: int, local_hidden_dim: int):
        super().__init__()
        self.block = GraphNetBlock(
            global_input_dim=(2 * backbone_hidden_dim) + 2,
            node_input_dim=backbone_hidden_dim + 1,
            edge_input_dim=backbone_hidden_dim,
            hidden_dim=local_hidden_dim,
            global_output_dim=local_hidden_dim,
            node_output_dim=local_hidden_dim,
            edge_output_dim=local_hidden_dim,
        )

    def forward(
        self,
        ego_subgraph: EgoSubgraph | BatchedEgoSubgraph | FlattenedBatchedEgoSubgraphs,
    ) -> LocalGraphOutput:
        state = GraphTensorState(
            global_features=ego_subgraph.local_global_features,
            node_features=ego_subgraph.local_node_features,
            edge_features=ego_subgraph.local_edge_features,
            edge_mask=ego_subgraph.local_edge_mask,
            node_mask=ego_subgraph.local_node_mask,
        )
        state = self.block(state)
        return LocalGraphOutput(
            global_embedding=state.global_features,
            node_embeddings=state.node_features,
            edge_embeddings=state.edge_features,
            edge_mask=state.edge_mask,
            node_mask=state.node_mask,
        )


class ScoreReadout(nn.Module):
    """Maps concat(tilde_v_ij, tilde_v_ii) to a scalar score z_ij."""

    def __init__(self, local_hidden_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = MLP(2 * local_hidden_dim, hidden_dim, 1)

    def forward(self, local_node_embeddings: Tensor, center_embedding: Tensor) -> Tensor:
        if local_node_embeddings.ndim == 3:
            center_context = center_embedding.unsqueeze(1).expand(-1, local_node_embeddings.size(1), -1)
        else:
            center_context = center_embedding.unsqueeze(0).expand(local_node_embeddings.size(0), -1)
        score_inputs = torch.cat([local_node_embeddings, center_context], dim=-1)
        return self.mlp(score_inputs).squeeze(-1)


class AllocationHead(nn.Module):
    """Builds row-wise allocation parameters for either softmax or Dirichlet policies."""

    def __init__(self, temperature: float, dirichlet_alpha_floor: float):
        super().__init__()
        self.temperature = float(temperature)
        self.dirichlet_alpha_floor = float(dirichlet_alpha_floor)

    def logits(self, scores: Tensor) -> Tensor:
        return scores / self.temperature

    def concentration(self, scores: Tensor, valid_mask: Tensor | None = None) -> Tensor:
        concentration = F.softplus(self.logits(scores)) + self.dirichlet_alpha_floor
        if valid_mask is not None:
            concentration = torch.where(valid_mask, concentration, torch.zeros_like(concentration))
        return concentration

    def forward(self, scores: Tensor, pool_value: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits = self.logits(scores)
        allocation = torch.softmax(logits, dim=-1)
        if pool_value.ndim > 0:
            transferred = allocation * pool_value.unsqueeze(-1)
        else:
            transferred = allocation * pool_value
        return logits, allocation, transferred


class GlobalCritic(nn.Module):
    """Computes the scalar state value V_hat = f_V(u*)."""

    def __init__(self, global_hidden_dim: int, hidden_dim: int):
        super().__init__()
        self.value_head = MLP(global_hidden_dim, hidden_dim, 1)

    def forward(self, global_embedding: Tensor) -> Tensor:
        return self.value_head(global_embedding).squeeze(-1)


class GNNAllocationPolicy(nn.Module):
    """Deterministic GraphNet actor with induced ego-subgraph local allocation heads.

    Forward contract:
    - input observation keys follow Project1.env.Observation
    - output allocation_matrix has shape [num_nodes, num_nodes]
    - row i is supported only on S_i = {i} union N(i)
    - transferred_resources[i, j] = P_i * alpha_ij
    - value is the scalar critic output based on u*
    """

    def __init__(self, config: GNNPolicyConfig):
        super().__init__()
        self.config = config
        self.graph_builder = ObservationGraphBuilder()
        self.backbone = TwoLayerGraphNetBackbone(
            global_input_dim=self.graph_builder.global_input_dim,
            node_input_dim=self.graph_builder.node_input_dim,
            edge_input_dim=self.graph_builder.edge_input_dim,
            hidden_dim=config.hidden_dim,
        )
        self.local_graph_net = EgoLocalGraphNet(
            backbone_hidden_dim=config.hidden_dim,
            local_hidden_dim=config.local_hidden_dim,
        )
        self.score_readout = ScoreReadout(
            local_hidden_dim=config.local_hidden_dim,
            hidden_dim=config.score_hidden_dim,
        )
        self.allocation_head = AllocationHead(
            config.temperature,
            dirichlet_alpha_floor=config.dirichlet_alpha_floor,
        )
        self.critic = GlobalCritic(
            global_hidden_dim=config.hidden_dim,
            hidden_dim=config.critic_hidden_dim,
        )

    def build_graph_input(self, observation: Observation) -> GraphTensorInput:
        device = next(self.parameters()).device
        return self.graph_builder.build(observation, device=device)

    def build_graph_input_batch(self, observations: Sequence[Observation]) -> GraphTensorInput:
        device = next(self.parameters()).device
        return self.graph_builder.build_batch(list(observations), device=device)

    def encode_graph(self, graph_input: GraphTensorInput) -> BackboneOutput:
        return self.backbone(graph_input)

    def encode_graph_batch(self, graph_input: GraphTensorInput) -> BackboneOutput:
        return self.backbone(graph_input)

    def _rollout_logits_from_batched_backbone_output(self, backbone_output: BackboneOutput) -> Tensor:
        backbone_output = ensure_batched_backbone_output(backbone_output)
        batch_size, num_nodes = backbone_output.node_embeddings.shape[:2]
        dtype = backbone_output.node_embeddings.dtype
        device = backbone_output.node_embeddings.device
        masked_score_value = torch.finfo(dtype).min
        score_matrix = torch.full((batch_size, num_nodes, num_nodes), masked_score_value, dtype=dtype, device=device)
        ego_subgraph = extract_batched_center_chunk_ego_subgraphs(
            backbone_output,
            torch.arange(num_nodes, device=device, dtype=torch.int64),
        )
        local_output = self.local_graph_net(ego_subgraph)
        flat_indices = torch.arange(ego_subgraph.member_indices.size(0), device=device)
        center_embedding = local_output.node_embeddings[flat_indices, ego_subgraph.center_local_indices]
        local_scores = self.score_readout(local_output.node_embeddings, center_embedding)
        valid_local_nodes = ego_subgraph.local_node_mask & local_output.node_mask
        local_scores = torch.where(
            valid_local_nodes,
            local_scores,
            torch.full_like(local_scores, masked_score_value),
        )
        local_logits = self.allocation_head.logits(local_scores)

        scatter_batch_indices = ego_subgraph.batch_indices.unsqueeze(1).expand_as(ego_subgraph.member_indices)
        scatter_center_indices = ego_subgraph.center_indices.unsqueeze(1).expand_as(ego_subgraph.member_indices)
        score_matrix[
            scatter_batch_indices[valid_local_nodes],
            scatter_center_indices[valid_local_nodes],
            ego_subgraph.member_indices[valid_local_nodes],
        ] = local_logits[valid_local_nodes]

        return score_matrix

    def _compute_local_policy_inputs(
        self,
        backbone_output: BackboneOutput,
    ) -> tuple[BackboneOutput, FlattenedBatchedEgoSubgraphs, Tensor, Tensor, Tensor]:
        backbone_output = ensure_batched_backbone_output(backbone_output)
        batch_size, num_nodes = backbone_output.node_embeddings.shape[:2]
        device = backbone_output.node_embeddings.device
        masked_score_value = torch.finfo(backbone_output.node_embeddings.dtype).min
        ego_subgraph = extract_batched_center_chunk_ego_subgraphs(
            backbone_output,
            torch.arange(num_nodes, device=device, dtype=torch.int64),
        )
        local_output = self.local_graph_net(ego_subgraph)
        flat_indices = torch.arange(ego_subgraph.member_indices.size(0), device=device)
        center_embedding = local_output.node_embeddings[flat_indices, ego_subgraph.center_local_indices]
        local_scores = self.score_readout(local_output.node_embeddings, center_embedding)
        valid_local_nodes = ego_subgraph.local_node_mask & local_output.node_mask
        local_scores = torch.where(
            valid_local_nodes,
            local_scores,
            torch.full_like(local_scores, masked_score_value),
        )
        local_logits = self.allocation_head.logits(local_scores)
        local_concentration = self.allocation_head.concentration(local_scores, valid_local_nodes)
        return backbone_output, ego_subgraph, valid_local_nodes, local_logits, local_concentration

    @staticmethod
    def _normalize_local_simplex(local_allocation: Tensor, valid_local_nodes: Tensor, eps: float = 1e-8) -> Tensor:
        masked_allocation = torch.where(valid_local_nodes, local_allocation, torch.zeros_like(local_allocation))
        safe_allocation = torch.clamp(masked_allocation, min=eps)
        safe_allocation = torch.where(valid_local_nodes, safe_allocation, torch.zeros_like(safe_allocation))
        normalizer = safe_allocation.sum(dim=-1, keepdim=True).clamp_min(eps)
        return safe_allocation / normalizer

    def _sample_dirichlet_local_allocations(
        self,
        local_concentration: Tensor,
        valid_local_nodes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        local_allocation = torch.zeros_like(local_concentration)
        row_log_prob = local_concentration.new_zeros(local_concentration.size(0))
        row_entropy = local_concentration.new_zeros(local_concentration.size(0))
        for row_index in range(local_concentration.size(0)):
            valid_mask = valid_local_nodes[row_index]
            valid_count = int(valid_mask.sum().item())
            if valid_count <= 0:
                raise RuntimeError("Each local policy row must contain at least one valid action.")
            if valid_count == 1:
                local_allocation[row_index, valid_mask] = 1.0
                continue
            distribution = Dirichlet(local_concentration[row_index, valid_mask])
            sample = distribution.rsample()
            local_allocation[row_index, valid_mask] = sample
            row_log_prob[row_index] = distribution.log_prob(sample)
            row_entropy[row_index] = distribution.entropy()
        return local_allocation, row_log_prob, row_entropy

    def _evaluate_dirichlet_local_allocations(
        self,
        local_concentration: Tensor,
        valid_local_nodes: Tensor,
        local_allocation: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        normalized_allocation = self._normalize_local_simplex(local_allocation, valid_local_nodes)
        row_log_prob = local_concentration.new_zeros(local_concentration.size(0))
        row_entropy = local_concentration.new_zeros(local_concentration.size(0))
        for row_index in range(local_concentration.size(0)):
            valid_mask = valid_local_nodes[row_index]
            valid_count = int(valid_mask.sum().item())
            if valid_count <= 0:
                raise RuntimeError("Each local policy row must contain at least one valid action.")
            if valid_count == 1:
                normalized_allocation[row_index, valid_mask] = 1.0
                continue
            distribution = Dirichlet(local_concentration[row_index, valid_mask])
            action = normalized_allocation[row_index, valid_mask]
            row_log_prob[row_index] = distribution.log_prob(action)
            row_entropy[row_index] = distribution.entropy()
        return normalized_allocation, row_log_prob, row_entropy

    def _assemble_batched_output(
        self,
        backbone_output: BackboneOutput,
        ego_subgraph: FlattenedBatchedEgoSubgraphs,
        valid_local_nodes: Tensor,
        local_logits: Tensor,
        local_concentration: Tensor,
        local_allocation: Tensor,
        row_log_prob: Tensor | None = None,
        row_entropy: Tensor | None = None,
    ) -> BatchedPolicyOutput:
        backbone_output = ensure_batched_backbone_output(backbone_output)
        batch_size, num_nodes = backbone_output.node_embeddings.shape[:2]
        dtype = backbone_output.node_embeddings.dtype
        device = backbone_output.node_embeddings.device
        masked_score_value = torch.finfo(dtype).min
        allocation_matrix = torch.zeros((batch_size, num_nodes, num_nodes), dtype=dtype, device=device)
        transferred_resources = torch.zeros_like(allocation_matrix)
        score_matrix = torch.full((batch_size, num_nodes, num_nodes), masked_score_value, dtype=dtype, device=device)
        concentration_matrix = torch.zeros((batch_size, num_nodes, num_nodes), dtype=dtype, device=device)

        if ego_subgraph.pool_value.ndim > 0:
            local_transferred = local_allocation * ego_subgraph.pool_value.unsqueeze(-1)
        else:
            local_transferred = local_allocation * ego_subgraph.pool_value

        scatter_batch_indices = ego_subgraph.batch_indices.unsqueeze(1).expand_as(ego_subgraph.member_indices)
        scatter_center_indices = ego_subgraph.center_indices.unsqueeze(1).expand_as(ego_subgraph.member_indices)
        valid_positions = valid_local_nodes
        allocation_matrix[
            scatter_batch_indices[valid_positions],
            scatter_center_indices[valid_positions],
            ego_subgraph.member_indices[valid_positions],
        ] = local_allocation[valid_positions]
        transferred_resources[
            scatter_batch_indices[valid_positions],
            scatter_center_indices[valid_positions],
            ego_subgraph.member_indices[valid_positions],
        ] = local_transferred[valid_positions]
        score_matrix[
            scatter_batch_indices[valid_positions],
            scatter_center_indices[valid_positions],
            ego_subgraph.member_indices[valid_positions],
        ] = local_logits[valid_positions]
        concentration_matrix[
            scatter_batch_indices[valid_positions],
            scatter_center_indices[valid_positions],
            ego_subgraph.member_indices[valid_positions],
        ] = local_concentration[valid_positions]

        value = self.critic(backbone_output.global_embedding)
        incoming_resources = transferred_resources.sum(dim=1)
        log_prob: Tensor | None = None
        entropy: Tensor | None = None
        if row_log_prob is not None:
            log_prob = value.new_zeros(batch_size)
            log_prob.scatter_add_(0, ego_subgraph.batch_indices, row_log_prob.to(dtype=value.dtype))
        if row_entropy is not None:
            entropy = value.new_zeros(batch_size)
            entropy.scatter_add_(0, ego_subgraph.batch_indices, row_entropy.to(dtype=value.dtype))
        return BatchedPolicyOutput(
            allocation_matrix=allocation_matrix,
            transferred_resources=transferred_resources,
            incoming_resources=incoming_resources,
            value=value,
            logits=score_matrix,
            log_prob=log_prob,
            entropy=entropy,
            concentration=concentration_matrix,
        )

    def _forward_batched_backbone_output(self, backbone_output: BackboneOutput) -> BatchedPolicyOutput:
        backbone_output, ego_subgraph, valid_local_nodes, local_logits, local_concentration = self._compute_local_policy_inputs(
            backbone_output
        )
        if self.config.action_distribution == "dirichlet":
            local_allocation = local_concentration / local_concentration.sum(dim=-1, keepdim=True).clamp_min(
                self.config.dirichlet_alpha_floor
            )
        else:
            local_allocation = torch.softmax(local_logits, dim=-1)
        return self._assemble_batched_output(
            backbone_output,
            ego_subgraph,
            valid_local_nodes,
            local_logits,
            local_concentration,
            local_allocation,
        )

    def _sample_batched_backbone_output(self, backbone_output: BackboneOutput) -> BatchedPolicyOutput:
        backbone_output, ego_subgraph, valid_local_nodes, local_logits, local_concentration = self._compute_local_policy_inputs(
            backbone_output
        )
        if self.config.action_distribution == "dirichlet":
            local_allocation, row_log_prob, row_entropy = self._sample_dirichlet_local_allocations(
                local_concentration,
                valid_local_nodes,
            )
        else:
            local_allocation = torch.softmax(local_logits, dim=-1)
            row_log_prob = local_concentration.new_zeros(local_concentration.size(0))
            row_entropy = local_concentration.new_zeros(local_concentration.size(0))
        return self._assemble_batched_output(
            backbone_output,
            ego_subgraph,
            valid_local_nodes,
            local_logits,
            local_concentration,
            local_allocation,
            row_log_prob=row_log_prob,
            row_entropy=row_entropy,
        )

    def _forward_batched_graph_input(self, graph_input: GraphTensorInput) -> BatchedPolicyOutput:
        backbone_output = self.encode_graph_batch(graph_input)
        return self._forward_batched_backbone_output(backbone_output)

    def _forward_batch_same_size(self, observations: Sequence[Observation]) -> BatchedPolicyOutput:
        if not observations:
            raise ValueError("observations must contain at least one item.")

        graph_input = self.build_graph_input_batch(observations)
        return self._forward_batched_graph_input(graph_input)

    def rollout_logits_batch(self, observations: Sequence[Observation]) -> Tensor:
        if not observations:
            raise ValueError("observations must contain at least one item.")
        graph_input = self.build_graph_input_batch(observations)
        backbone_output = self.encode_graph_batch(graph_input)
        return self._rollout_logits_from_batched_backbone_output(backbone_output)

    def rollout_logits_tensor_batch(self, observations: Mapping[str, Tensor]) -> Tensor:
        device = next(self.parameters()).device
        graph_input = self.graph_builder.build_tensor_batch(observations, device=device)
        backbone_output = self.encode_graph_batch(graph_input)
        return self._rollout_logits_from_batched_backbone_output(backbone_output)

    def forward_tensor_batch(self, observations: Mapping[str, Tensor]) -> BatchedPolicyOutput:
        device = next(self.parameters()).device
        graph_input = self.graph_builder.build_tensor_batch(observations, device=device)
        return self._forward_batched_graph_input(graph_input)

    def rollout_logits(self, observation: Observation) -> Tensor:
        graph_input = self.build_graph_input(observation)
        backbone_output = self.encode_graph(graph_input)
        return self._rollout_logits_from_batched_backbone_output(backbone_output).squeeze(0)

    def forward(self, observation: Observation) -> PolicyOutput:
        graph_input = self.build_graph_input(observation)
        backbone_output = self.encode_graph(graph_input)
        batched_output = self._forward_batched_backbone_output(backbone_output)
        zero_scalar = batched_output.value[0].new_zeros(())

        return PolicyOutput(
            allocation_matrix=batched_output.allocation_matrix[0],
            transferred_resources=batched_output.transferred_resources[0],
            incoming_resources=batched_output.incoming_resources[0],
            value=batched_output.value[0],
            log_prob=zero_scalar if batched_output.log_prob is None else batched_output.log_prob[0],
            entropy=zero_scalar if batched_output.entropy is None else batched_output.entropy[0],
            logits=batched_output.logits[0] if batched_output.logits is not None else None,
            concentration=batched_output.concentration[0] if batched_output.concentration is not None else None,
            global_embedding=backbone_output.global_embedding,
            node_embeddings=backbone_output.node_embeddings,
            edge_embeddings=backbone_output.edge_embeddings,
        )

    def deterministic_action(self, observation: Observation) -> PolicyOutput:
        return self.forward(observation)

    def deterministic_action_tensor_batch(self, observations: Mapping[str, Tensor]) -> BatchedPolicyOutput:
        return self.forward_tensor_batch(observations)

    def deterministic_action_batch(self, observations: Sequence[Observation]) -> list[PolicyOutput]:
        batched_output = self._forward_batch_same_size(observations)
        outputs: list[PolicyOutput] = []
        batch_size = batched_output.allocation_matrix.size(0)
        for batch_index in range(batch_size):
            zero_scalar = batched_output.value[batch_index].new_zeros(())
            outputs.append(
                PolicyOutput(
                    allocation_matrix=batched_output.allocation_matrix[batch_index],
                    transferred_resources=batched_output.transferred_resources[batch_index],
                    incoming_resources=batched_output.incoming_resources[batch_index],
                    value=batched_output.value[batch_index],
                    log_prob=zero_scalar if batched_output.log_prob is None else batched_output.log_prob[batch_index],
                    entropy=zero_scalar if batched_output.entropy is None else batched_output.entropy[batch_index],
                    logits=batched_output.logits[batch_index] if batched_output.logits is not None else None,
                    concentration=(
                        batched_output.concentration[batch_index] if batched_output.concentration is not None else None
                    ),
                    global_embedding=None,
                    node_embeddings=None,
                    edge_embeddings=None,
                )
            )
        return outputs

    def evaluate_value(self, observation: Observation) -> Tensor:
        graph_input = self.build_graph_input(observation)
        backbone_output = self.encode_graph(graph_input)
        return self.critic(backbone_output.global_embedding)

    def sample_action_tensor_batch(self, observations: Mapping[str, Tensor]) -> BatchedPolicyOutput:
        device = next(self.parameters()).device
        graph_input = self.graph_builder.build_tensor_batch(observations, device=device)
        return self._sample_batched_backbone_output(self.encode_graph_batch(graph_input))

    def evaluate_action_tensor_batch(
        self,
        observations: Mapping[str, Tensor],
        allocation_matrix: Tensor,
    ) -> BatchedPolicyOutput:
        device = next(self.parameters()).device
        graph_input = self.graph_builder.build_tensor_batch(observations, device=device)
        action_tensor = torch.as_tensor(allocation_matrix, dtype=torch.float32, device=device)
        backbone_output, ego_subgraph, valid_local_nodes, local_logits, local_concentration = self._compute_local_policy_inputs(
            self.encode_graph_batch(graph_input)
        )
        gathered_allocation = action_tensor[
            ego_subgraph.batch_indices.unsqueeze(1).expand_as(ego_subgraph.member_indices),
            ego_subgraph.center_indices.unsqueeze(1).expand_as(ego_subgraph.member_indices),
            ego_subgraph.member_indices,
        ]
        if self.config.action_distribution != "dirichlet":
            zero_rows = local_concentration.new_zeros(local_concentration.size(0))
            return self._assemble_batched_output(
                backbone_output,
                ego_subgraph,
                valid_local_nodes,
                local_logits,
                local_concentration,
                self._normalize_local_simplex(gathered_allocation, valid_local_nodes),
                row_log_prob=zero_rows,
                row_entropy=zero_rows,
            )
        normalized_allocation, row_log_prob, row_entropy = self._evaluate_dirichlet_local_allocations(
            local_concentration,
            valid_local_nodes,
            gathered_allocation,
        )
        return self._assemble_batched_output(
            backbone_output,
            ego_subgraph,
            valid_local_nodes,
            local_logits,
            local_concentration,
            normalized_allocation,
            row_log_prob=row_log_prob,
            row_entropy=row_entropy,
        )

    def sample_action(self, observation: Observation) -> PolicyOutput:
        graph_input = self.build_graph_input(observation)
        backbone_output = self.encode_graph(graph_input)
        batched = self._sample_batched_backbone_output(backbone_output)
        zero_scalar = batched.value[0].new_zeros(())
        return PolicyOutput(
            allocation_matrix=batched.allocation_matrix[0],
            transferred_resources=batched.transferred_resources[0],
            incoming_resources=batched.incoming_resources[0],
            value=batched.value[0],
            log_prob=zero_scalar if batched.log_prob is None else batched.log_prob[0],
            entropy=zero_scalar if batched.entropy is None else batched.entropy[0],
            logits=batched.logits[0] if batched.logits is not None else None,
            concentration=batched.concentration[0] if batched.concentration is not None else None,
            global_embedding=None,
            node_embeddings=None,
            edge_embeddings=None,
        )
