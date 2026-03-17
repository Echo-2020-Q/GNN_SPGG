from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.distributions import Dirichlet
import torch.nn.functional as F

from Project1.env import Observation


@dataclass
class GNNPolicyConfig:
    hidden_dim: int = 64
    num_message_passing_layers: int = 2
    temperature: float = 1.0
    dirichlet_concentration_scale: float = 1.0
    dirichlet_concentration_floor: float = 0.1

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.num_message_passing_layers <= 0:
            raise ValueError("num_message_passing_layers must be positive.")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        if self.dirichlet_concentration_scale <= 0.0:
            raise ValueError("dirichlet_concentration_scale must be positive.")
        if self.dirichlet_concentration_floor <= 0.0:
            raise ValueError("dirichlet_concentration_floor must be positive.")


@dataclass
class PolicyOutput:
    allocation_matrix: Tensor
    value: Tensor
    log_prob: Tensor | None = None
    entropy: Tensor | None = None
    logits: Tensor | None = None
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


class MeanMessagePassingLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_linear = nn.Linear(hidden_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, node_embeddings: Tensor, local_mask: Tensor) -> Tensor:
        mask = local_mask.float()
        neighborhood_mean = mask @ node_embeddings
        neighborhood_mean = neighborhood_mean / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

        updated = self.self_linear(node_embeddings) + self.neighbor_linear(neighborhood_mean)
        return F.gelu(self.layer_norm(updated))


class GNNAllocationPolicy(nn.Module):
    """Shared GNN encoder with dual-role heads and local allocation policy."""

    def __init__(self, config: GNNPolicyConfig):
        super().__init__()
        self.config = config
        hidden_dim = config.hidden_dim

        self.node_encoder = MLP(7, hidden_dim, hidden_dim)
        self.message_passing_layers = nn.ModuleList(
            [MeanMessagePassingLayer(hidden_dim) for _ in range(config.num_message_passing_layers)]
        )
        self.agent_head = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.pool_head = MLP(hidden_dim, hidden_dim, hidden_dim)
        self.pair_scorer = MLP((2 * hidden_dim) + 6, hidden_dim, 1)
        self.value_head = MLP(hidden_dim, hidden_dim, 1)

    def forward(self, observation: Observation) -> PolicyOutput:
        tensors = self._observation_to_tensors(observation)
        logits, value = self._compute_logits_and_value(tensors)
        allocation = self._masked_softmax(logits, tensors["local_mask"])
        return PolicyOutput(
            allocation_matrix=allocation,
            value=value,
            logits=logits,
        )

    def deterministic_action(self, observation: Observation) -> PolicyOutput:
        return self.forward(observation)

    def evaluate_value(self, observation: Observation) -> Tensor:
        tensors = self._observation_to_tensors(observation)
        _, value = self._compute_logits_and_value(tensors)
        return value

    def sample_action(self, observation: Observation) -> PolicyOutput:
        tensors = self._observation_to_tensors(observation)
        logits, value = self._compute_logits_and_value(tensors)
        local_mask = tensors["local_mask"]

        allocation = torch.zeros_like(logits)
        concentration_matrix = torch.zeros_like(logits)
        total_log_prob = torch.zeros((), device=logits.device)
        total_entropy = torch.zeros((), device=logits.device)

        scaled_logits = logits / self.config.temperature
        for pool_index in range(logits.size(0)):
            valid_indices = local_mask[pool_index].nonzero(as_tuple=False).squeeze(-1)
            if valid_indices.numel() == 1:
                allocation[pool_index, valid_indices] = 1.0
                concentration_matrix[pool_index, valid_indices] = 1.0
                continue

            row_logits = scaled_logits[pool_index, valid_indices]
            concentration = (
                F.softplus(row_logits) * self.config.dirichlet_concentration_scale
                + self.config.dirichlet_concentration_floor
            )
            distribution = Dirichlet(concentration)
            sampled_weights = distribution.rsample()

            allocation[pool_index, valid_indices] = sampled_weights
            concentration_matrix[pool_index, valid_indices] = concentration
            total_log_prob = total_log_prob + distribution.log_prob(sampled_weights)
            total_entropy = total_entropy + distribution.entropy()

        return PolicyOutput(
            allocation_matrix=allocation,
            value=value,
            log_prob=total_log_prob,
            entropy=total_entropy,
            logits=logits,
            concentration=concentration_matrix,
        )

    def _compute_logits_and_value(self, tensors: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        node_embeddings = self.node_encoder(self._build_node_features(tensors))
        for layer in self.message_passing_layers:
            node_embeddings = layer(node_embeddings, tensors["local_mask"])

        agent_embeddings = self.agent_head(node_embeddings)
        pool_embeddings = self.pool_head(node_embeddings)
        pair_features = self._build_pair_features(tensors)

        num_nodes = node_embeddings.size(0)
        pool_context = pool_embeddings[:, None, :].expand(num_nodes, num_nodes, -1)
        agent_context = agent_embeddings[None, :, :].expand(num_nodes, num_nodes, -1)
        pair_inputs = torch.cat([pool_context, agent_context, pair_features], dim=-1)

        logits = self.pair_scorer(pair_inputs).squeeze(-1)
        logits = logits.masked_fill(~tensors["local_mask"], torch.finfo(logits.dtype).min)

        global_embedding = node_embeddings.mean(dim=0)
        value = self.value_head(global_embedding).squeeze(-1)
        return logits, value

    def _build_node_features(self, tensors: dict[str, Tensor]) -> Tensor:
        return torch.stack(
            [
                tensors["x_nominal"],
                tensors["x_actual"],
                tensors["resources"],
                tensors["investment"],
                tensors["pool_raw"],
                tensors["pool_grown"],
                tensors["degrees"],
            ],
            dim=-1,
        )

    def _build_pair_features(self, tensors: dict[str, Tensor]) -> Tensor:
        num_nodes = tensors["x_nominal"].size(0)
        is_self = torch.eye(num_nodes, device=tensors["x_nominal"].device, dtype=torch.float32)
        x_nominal = tensors["x_nominal"][None, :].expand(num_nodes, num_nodes)
        x_actual = tensors["x_actual"][None, :].expand(num_nodes, num_nodes)
        resources = tensors["resources"][None, :].expand(num_nodes, num_nodes)
        unit_investment = tensors["unit_investment"][None, :].expand(num_nodes, num_nodes)
        degrees = tensors["degrees"][None, :].expand(num_nodes, num_nodes)
        return torch.stack(
            [
                is_self,
                x_nominal,
                x_actual,
                resources,
                unit_investment,
                degrees,
            ],
            dim=-1,
        )

    def _masked_softmax(self, logits: Tensor, local_mask: Tensor) -> Tensor:
        weights = torch.softmax(logits / self.config.temperature, dim=-1)
        weights = weights * local_mask.float()
        return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)

    def _observation_to_tensors(self, observation: Observation) -> dict[str, Tensor]:
        device = next(self.parameters()).device
        return {
            "x_nominal": torch.as_tensor(observation["x_nominal"], dtype=torch.float32, device=device),
            "x_actual": torch.as_tensor(observation["x_actual"], dtype=torch.float32, device=device),
            "resources": torch.as_tensor(observation["resources"], dtype=torch.float32, device=device),
            "investment": torch.as_tensor(observation["investment"], dtype=torch.float32, device=device),
            "unit_investment": torch.as_tensor(observation["unit_investment"], dtype=torch.float32, device=device),
            "pool_raw": torch.as_tensor(observation["pool_raw"], dtype=torch.float32, device=device),
            "pool_grown": torch.as_tensor(observation["pool_grown"], dtype=torch.float32, device=device),
            "degrees": torch.as_tensor(observation["degrees"], dtype=torch.float32, device=device),
            "local_mask": torch.as_tensor(observation["local_mask"], dtype=torch.bool, device=device),
        }
