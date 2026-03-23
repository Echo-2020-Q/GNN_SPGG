from __future__ import annotations

from dataclasses import dataclass

import torch
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
    dirichlet_concentration_scale: float = 1.0
    dirichlet_concentration_floor: float = 0.1

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
        if self.dirichlet_concentration_scale <= 0.0:
            raise ValueError("dirichlet_concentration_scale must be positive.")
        if self.dirichlet_concentration_floor <= 0.0:
            raise ValueError("dirichlet_concentration_floor must be positive.")


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
    ego_mask: Tensor
    pool_values: Tensor


@dataclass(frozen=True)
class GraphTensorState:
    """Intermediate GraphNet state with dense edge tensors."""

    global_features: Tensor
    node_features: Tensor
    edge_features: Tensor
    edge_mask: Tensor


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


class ObservationGraphBuilder:
    """Builds the dense GraphNet tensors (u, V, E) from the environment observation."""

    node_feature_names = (
        "x_nominal",
        "x_actual",
        "resources",
        "investment",
        "unit_investment",
        "pool_raw",
        "pool_grown",
        "degrees",
    )
    global_feature_names = (
        "x_nominal",
        "x_actual",
        "resources",
        "investment",
        "pool_raw",
        "pool_grown",
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

        num_nodes = ego_mask.size(0)
        diagonal = torch.eye(num_nodes, dtype=torch.bool, device=device)
        edge_mask = ego_mask & ~diagonal

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
            ego_mask=ego_mask,
            pool_values=pool_values,
        )


def _masked_edge_mean_by_receiver(edge_features: Tensor, edge_mask: Tensor) -> Tensor:
    mask = edge_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
    counts = edge_mask.sum(dim=0).clamp_min(1).to(dtype=edge_features.dtype).unsqueeze(-1)
    return (edge_features * mask).sum(dim=0) / counts


def _masked_global_edge_mean(edge_features: Tensor, edge_mask: Tensor) -> Tensor:
    mask = edge_mask.unsqueeze(-1).to(dtype=edge_features.dtype)
    count = edge_mask.sum().clamp_min(1).to(dtype=edge_features.dtype)
    return (edge_features * mask).sum(dim=(0, 1)) / count


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
        num_nodes = state.node_features.size(0)
        global_context_for_edges = state.global_features.view(1, 1, -1).expand(num_nodes, num_nodes, -1)
        sender_features = state.node_features[:, None, :].expand(num_nodes, num_nodes, -1)
        receiver_features = state.node_features[None, :, :].expand(num_nodes, num_nodes, -1)

        edge_inputs = torch.cat(
            [
                state.edge_features,
                sender_features,
                receiver_features,
                global_context_for_edges,
            ],
            dim=-1,
        )
        updated_edges = self.edge_model(edge_inputs)
        updated_edges = updated_edges * state.edge_mask.unsqueeze(-1).to(dtype=updated_edges.dtype)

        aggregated_edge_messages = _masked_edge_mean_by_receiver(updated_edges, state.edge_mask)
        global_context_for_nodes = state.global_features.view(1, -1).expand(num_nodes, -1)
        node_inputs = torch.cat(
            [
                state.node_features,
                aggregated_edge_messages,
                global_context_for_nodes,
            ],
            dim=-1,
        )
        updated_nodes = self.node_model(node_inputs)

        aggregated_nodes = updated_nodes.mean(dim=0)
        aggregated_edges = _masked_global_edge_mean(updated_edges, state.edge_mask)
        global_inputs = torch.cat(
            [
                state.global_features,
                aggregated_nodes,
                aggregated_edges,
            ],
            dim=-1,
        )
        updated_global = self.global_model(global_inputs)

        return GraphTensorState(
            global_features=updated_global,
            node_features=updated_nodes,
            edge_features=updated_edges,
            edge_mask=state.edge_mask,
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
        )
        state = self.block1(state)
        state = self.block2(state)
        return BackboneOutput(
            global_embedding=state.global_features,
            node_embeddings=state.node_features,
            edge_embeddings=state.edge_features,
            edge_mask=graph_input.edge_mask,
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
        local_edge_features=local_edge_features,
        local_edge_mask=local_edge_mask,
        local_global_features=local_global_features,
        pool_value=pool_value.squeeze(0),
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

    def forward(self, ego_subgraph: EgoSubgraph) -> LocalGraphOutput:
        state = GraphTensorState(
            global_features=ego_subgraph.local_global_features,
            node_features=ego_subgraph.local_node_features,
            edge_features=ego_subgraph.local_edge_features,
            edge_mask=ego_subgraph.local_edge_mask,
        )
        state = self.block(state)
        return LocalGraphOutput(
            global_embedding=state.global_features,
            node_embeddings=state.node_features,
            edge_embeddings=state.edge_features,
            edge_mask=state.edge_mask,
        )


class ScoreReadout(nn.Module):
    """Maps concat(tilde_v_ij, tilde_v_ii) to a scalar score z_ij."""

    def __init__(self, local_hidden_dim: int, hidden_dim: int):
        super().__init__()
        self.mlp = MLP(2 * local_hidden_dim, hidden_dim, 1)

    def forward(self, local_node_embeddings: Tensor, center_embedding: Tensor) -> Tensor:
        center_context = center_embedding.unsqueeze(0).expand(local_node_embeddings.size(0), -1)
        score_inputs = torch.cat([local_node_embeddings, center_context], dim=-1)
        return self.mlp(score_inputs).squeeze(-1)


class AllocationHead(nn.Module):
    """Applies ego-local softmax to obtain alpha_ij and x_ij = P_i * alpha_ij."""

    def forward(self, scores: Tensor, pool_value: Tensor) -> tuple[Tensor, Tensor]:
        allocation = torch.softmax(scores, dim=0)
        transferred = allocation * pool_value
        return allocation, transferred


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
        self.allocation_head = AllocationHead()
        self.critic = GlobalCritic(
            global_hidden_dim=config.hidden_dim,
            hidden_dim=config.critic_hidden_dim,
        )

    def build_graph_input(self, observation: Observation) -> GraphTensorInput:
        device = next(self.parameters()).device
        return self.graph_builder.build(observation, device=device)

    def encode_graph(self, graph_input: GraphTensorInput) -> BackboneOutput:
        return self.backbone(graph_input)

    def forward(self, observation: Observation) -> PolicyOutput:
        graph_input = self.build_graph_input(observation)
        backbone_output = self.encode_graph(graph_input)

        num_nodes = backbone_output.node_embeddings.size(0)
        dtype = backbone_output.node_embeddings.dtype
        device = backbone_output.node_embeddings.device
        masked_score_value = torch.finfo(dtype).min

        allocation_matrix = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
        transferred_resources = torch.zeros_like(allocation_matrix)
        score_matrix = torch.full((num_nodes, num_nodes), masked_score_value, dtype=dtype, device=device)

        for center_index in range(num_nodes):
            ego_subgraph = extract_ego_subgraph(backbone_output, center_index=center_index)
            local_output = self.local_graph_net(ego_subgraph)
            center_embedding = local_output.node_embeddings[ego_subgraph.center_local_index]
            local_scores = self.score_readout(local_output.node_embeddings, center_embedding)
            local_allocation, local_transferred = self.allocation_head(local_scores, ego_subgraph.pool_value)

            allocation_matrix[center_index, ego_subgraph.member_indices] = local_allocation
            transferred_resources[center_index, ego_subgraph.member_indices] = local_transferred
            score_matrix[center_index, ego_subgraph.member_indices] = local_scores

        value = self.critic(backbone_output.global_embedding)
        incoming_resources = transferred_resources.sum(dim=0)
        zero_scalar = value.new_zeros(())

        return PolicyOutput(
            allocation_matrix=allocation_matrix,
            transferred_resources=transferred_resources,
            incoming_resources=incoming_resources,
            value=value,
            log_prob=zero_scalar,
            entropy=zero_scalar,
            logits=score_matrix,
            concentration=None,
            global_embedding=backbone_output.global_embedding,
            node_embeddings=backbone_output.node_embeddings,
            edge_embeddings=backbone_output.edge_embeddings,
        )

    def deterministic_action(self, observation: Observation) -> PolicyOutput:
        return self.forward(observation)

    def evaluate_value(self, observation: Observation) -> Tensor:
        graph_input = self.build_graph_input(observation)
        backbone_output = self.encode_graph(graph_input)
        return self.critic(backbone_output.global_embedding)

    def sample_action(self, observation: Observation) -> PolicyOutput:
        # The current actor is deterministic by construction.
        return self.forward(observation)
