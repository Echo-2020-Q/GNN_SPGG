from __future__ import annotations

import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE:
    import numpy as np

    from Project1.env import SPGGConfig, SPGGEnv, make_grid_graph
    from Project1.policies.rule_based import ProportionalContributionPolicy, UniformAllocationPolicy

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import torch

    from Project1.policies.gnn_rl import (
        GNNAllocationPolicy,
        GNNPolicyConfig,
        GraphNetBlock,
        GraphTensorState,
        _masked_edge_sum_by_receiver,
        _masked_global_edge_normalized_sum,
        extract_ego_subgraph,
    )


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is required for rule-based policy tests")
class RuleBasedPolicyTests(unittest.TestCase):
    def test_uniform_policy_matches_local_uniform_weights(self) -> None:
        local_mask = np.array([[True, True], [True, True]])
        observation = {
            "local_mask": local_mask,
            "unit_investment": np.zeros(2, dtype=np.float64),
        }
        allocation = UniformAllocationPolicy().allocate(observation)
        np.testing.assert_allclose(allocation, np.array([[0.5, 0.5], [0.5, 0.5]]))

    def test_proportional_policy_falls_back_to_uniform_when_no_investment(self) -> None:
        observation = {
            "local_mask": np.array([[True, True], [True, True]]),
            "unit_investment": np.array([0.0, 0.0], dtype=np.float64),
        }
        allocation = ProportionalContributionPolicy().allocate(observation)
        np.testing.assert_allclose(allocation, np.array([[0.5, 0.5], [0.5, 0.5]]))

    def test_proportional_policy_tracks_unit_investment(self) -> None:
        observation = {
            "local_mask": np.array([[True, True], [True, True]]),
            "unit_investment": np.array([0.25, 0.75], dtype=np.float64),
        }
        allocation = ProportionalContributionPolicy().allocate(observation)
        np.testing.assert_allclose(allocation, np.array([[0.25, 0.75], [0.25, 0.75]]))


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for GNN policy tests")
class GNNPolicyTests(unittest.TestCase):
    def test_graphnet_block_matches_concat_reference(self) -> None:
        torch.manual_seed(0)
        block = GraphNetBlock(
            global_input_dim=5,
            node_input_dim=4,
            edge_input_dim=3,
            hidden_dim=7,
            global_output_dim=6,
            node_output_dim=8,
            edge_output_dim=9,
        )

        state = GraphTensorState(
            global_features=torch.randn(2, 5),
            node_features=torch.randn(2, 4, 4),
            edge_features=torch.randn(2, 4, 4, 3),
            edge_mask=torch.tensor(
                [
                    [
                        [True, True, False, False],
                        [True, True, True, False],
                        [False, True, True, True],
                        [False, False, True, True],
                    ],
                    [
                        [True, True, True, False],
                        [True, True, False, False],
                        [True, False, True, True],
                        [False, False, True, True],
                    ],
                ],
                dtype=torch.bool,
            ),
            node_mask=torch.tensor(
                [
                    [True, True, True, False],
                    [True, True, True, True],
                ],
                dtype=torch.bool,
            ),
        )

        actual = block(state)

        global_features = state.global_features
        node_features = state.node_features
        edge_features = state.edge_features
        edge_mask = state.edge_mask
        node_mask = state.node_mask
        batch_size, num_nodes = node_features.shape[:2]

        edge_inputs = torch.cat(
            [
                edge_features,
                node_features[:, :, None, :].expand(batch_size, num_nodes, num_nodes, -1),
                node_features[:, None, :, :].expand(batch_size, num_nodes, num_nodes, -1),
                global_features[:, None, None, :].expand(batch_size, num_nodes, num_nodes, -1),
            ],
            dim=-1,
        )
        expected_edges = block.edge_model(edge_inputs)
        expected_edges = expected_edges * edge_mask.unsqueeze(-1).to(dtype=expected_edges.dtype)

        aggregated_edge_messages = _masked_edge_sum_by_receiver(expected_edges, edge_mask)
        node_inputs = torch.cat(
            [
                node_features,
                aggregated_edge_messages,
                global_features[:, None, :].expand(batch_size, num_nodes, -1),
            ],
            dim=-1,
        )
        expected_nodes = block.node_model(node_inputs)
        expected_nodes = expected_nodes * node_mask.unsqueeze(-1).to(dtype=expected_nodes.dtype)

        node_count = node_mask.sum(dim=1).clamp_min(1).to(dtype=expected_nodes.dtype).unsqueeze(-1)
        aggregated_nodes = expected_nodes.sum(dim=1) / node_count
        aggregated_edges = _masked_global_edge_normalized_sum(expected_edges, edge_mask)
        global_inputs = torch.cat(
            [
                global_features,
                aggregated_nodes,
                aggregated_edges,
            ],
            dim=-1,
        )
        expected_global = block.global_model(global_inputs)

        self.assertTrue(torch.allclose(actual.edge_features, expected_edges, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(actual.node_features, expected_nodes, atol=1e-6, rtol=1e-6))
        self.assertTrue(torch.allclose(actual.global_features, expected_global, atol=1e-6, rtol=1e-6))

    def test_ego_subgraph_extraction_keeps_induced_neighbor_edges(self) -> None:
        env = SPGGEnv(
            SPGGConfig(alpha=0.0, r=0.5, p_max=5.0, beta=1.0, episode_length=2),
            {
                0: [1, 2],
                1: [0, 2],
                2: [0, 1],
                3: [],
            },
        )
        observation = env.reset(seed=0)

        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))
        graph_input = policy.build_graph_input(observation)
        backbone_output = policy.encode_graph(graph_input)
        ego_subgraph = extract_ego_subgraph(backbone_output, center_index=0)

        self.assertEqual(set(ego_subgraph.member_indices.tolist()), {0, 1, 2})
        local_positions = {int(node_index): position for position, node_index in enumerate(ego_subgraph.member_indices.tolist())}
        self.assertTrue(ego_subgraph.local_edge_mask[local_positions[1], local_positions[2]].item())
        self.assertTrue(ego_subgraph.local_edge_mask[local_positions[2], local_positions[1]].item())

    def test_gnn_policy_outputs_valid_local_simplex(self) -> None:
        env = SPGGEnv(
            SPGGConfig(alpha=0.0, r=0.5, p_max=5.0, beta=1.0, episode_length=2),
            make_grid_graph(2, 2),
        )
        observation = env.reset(seed=0)

        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))
        deterministic = policy.deterministic_action(observation)
        sampled = policy.sample_action(observation)

        local_mask = torch.as_tensor(observation["local_mask"], dtype=torch.bool)
        pool_grown = torch.as_tensor(observation["pool_grown"], dtype=torch.float32)
        self.assertEqual(tuple(deterministic.allocation_matrix.shape), (4, 4))
        self.assertTrue(torch.allclose(deterministic.allocation_matrix[~local_mask], torch.zeros_like(deterministic.allocation_matrix[~local_mask])))
        self.assertTrue(torch.allclose(deterministic.allocation_matrix.sum(dim=1), torch.ones(4), atol=1e-5))
        self.assertTrue(torch.allclose(deterministic.transferred_resources.sum(dim=1), pool_grown, atol=1e-5))
        self.assertTrue(torch.allclose(deterministic.incoming_resources, deterministic.transferred_resources.sum(dim=0), atol=1e-5))
        self.assertTrue(torch.isfinite(deterministic.value))
        self.assertTrue(torch.allclose(sampled.allocation_matrix[~local_mask], torch.zeros_like(sampled.allocation_matrix[~local_mask])))
        self.assertTrue(torch.allclose(sampled.allocation_matrix.sum(dim=1), torch.ones(4), atol=1e-5))
        self.assertTrue(torch.isfinite(sampled.log_prob))
        self.assertTrue(torch.isfinite(sampled.entropy))

    def test_dirichlet_policy_batch_interfaces_are_valid(self) -> None:
        env = SPGGEnv(
            SPGGConfig(alpha=0.0, r=0.5, p_max=5.0, beta=1.0, episode_length=2),
            make_grid_graph(2, 2),
        )
        observation = env.reset(seed=0)
        batched_observation = {
            key: torch.stack([torch.as_tensor(value), torch.as_tensor(value)], dim=0)
            for key, value in observation.items()
        }

        policy = GNNAllocationPolicy(
            GNNPolicyConfig(
                hidden_dim=16,
                num_message_passing_layers=2,
                action_distribution="dirichlet",
            )
        )

        sampled = policy.sample_action_tensor_batch(batched_observation)
        evaluated = policy.evaluate_action_tensor_batch(batched_observation, sampled.allocation_matrix)
        local_mask = torch.stack(
            [
                torch.as_tensor(observation["local_mask"], dtype=torch.bool),
                torch.as_tensor(observation["local_mask"], dtype=torch.bool),
            ],
            dim=0,
        )

        self.assertEqual(tuple(sampled.allocation_matrix.shape), (2, 4, 4))
        self.assertTrue(torch.allclose(sampled.allocation_matrix[~local_mask], torch.zeros_like(sampled.allocation_matrix[~local_mask])))
        self.assertTrue(torch.allclose(sampled.allocation_matrix.sum(dim=-1), torch.ones((2, 4)), atol=1e-5))
        self.assertIsNotNone(sampled.log_prob)
        self.assertIsNotNone(sampled.entropy)
        self.assertIsNotNone(sampled.concentration)
        self.assertTrue(torch.isfinite(sampled.log_prob).all())
        self.assertTrue(torch.isfinite(sampled.entropy).all())
        self.assertTrue(torch.isfinite(evaluated.log_prob).all())
        self.assertTrue(torch.isfinite(evaluated.entropy).all())
        self.assertEqual(tuple(evaluated.value.shape), (2,))
