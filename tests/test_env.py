from __future__ import annotations

import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    import numpy as np

    from Project1.env import (
        RewardConfig,
        SPGGConfig,
        SPGGEnv,
        make_barabasi_albert_graph,
        make_erdos_renyi_graph,
        make_random_regular_graph,
        make_watts_strogatz_graph,
    )
    from Project1.policies.rule_based import UniformAllocationPolicy


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is required for environment tests")
class SPGGEnvTests(unittest.TestCase):
    def test_two_node_manual_dynamics_with_uniform_allocation(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=10.0,
                beta=0.0,
                episode_length=1,
                reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.0, lambda_gini=0.0),
            ),
            {0: [1], 1: [0]},
        )

        observation = env.reset(initial_resources=[3.0, 3.0], initial_strategies=[1, 1], seed=0)
        np.testing.assert_allclose(observation["x_actual"], np.array([1.0, 1.0]))
        np.testing.assert_allclose(observation["investment"], np.array([2.0, 2.0]))
        np.testing.assert_allclose(observation["unit_investment"], np.array([1.0, 1.0]))
        np.testing.assert_allclose(observation["pool_raw"], np.array([2.0, 2.0]))
        np.testing.assert_allclose(observation["pool_grown"], np.array([2.0, 2.0]))

        next_observation, reward, done, info = env.step(UniformAllocationPolicy().allocate(observation))
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 0.0)
        np.testing.assert_allclose(info["income"], np.array([2.0, 2.0]))
        np.testing.assert_allclose(info["payoff"], np.array([0.0, 0.0]))
        np.testing.assert_allclose(next_observation["resources"], np.array([3.0, 3.0]))

    def test_resource_constraint_forces_defection(self) -> None:
        env = SPGGEnv(
            SPGGConfig(alpha=0.0, r=0.0, p_max=10.0, beta=0.0, episode_length=2),
            {0: [1], 1: [0]},
        )

        observation = env.reset(initial_resources=[1.0, 3.0], initial_strategies=[1, 1], seed=0)
        np.testing.assert_allclose(observation["x_actual"], np.array([0.0, 1.0]))
        np.testing.assert_allclose(observation["investment"], np.array([0.0, 2.0]))

    def test_pool_growth_respects_pmax(self) -> None:
        env = SPGGEnv(
            SPGGConfig(alpha=0.0, r=1.0, p_max=3.0, beta=0.0, episode_length=1),
            {0: [1], 1: [0]},
        )

        observation = env.reset(initial_resources=[3.0, 3.0], initial_strategies=[1, 1], seed=0)
        np.testing.assert_allclose(observation["pool_raw"], np.array([2.0, 2.0]))
        np.testing.assert_allclose(observation["pool_grown"], np.array([3.0, 3.0]))
        np.testing.assert_allclose(observation["p_max"], np.array([3.0, 3.0]))

    def test_dynamic_pool_capacity_uses_actual_cooperators(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=10.0,
                p_mode="dynamic",
                p_max=999.0,
                p_c=10.0,
                beta=0.0,
                episode_length=1,
            ),
            {0: [1], 1: [0]},
        )

        observation = env.reset(initial_resources=[3.0, 1.0], initial_strategies=[1, 1], seed=0)
        np.testing.assert_allclose(observation["x_actual"], np.array([1.0, 0.0]))
        np.testing.assert_allclose(observation["local_actual_cooperators"], np.array([1.0, 1.0]))
        np.testing.assert_allclose(observation["p_max"], np.array([20.0, 20.0]))
        np.testing.assert_allclose(observation["pool_theoretical_max"], np.array([20.0, 20.0]))
        np.testing.assert_allclose(observation["pool_capacity"], np.array([5.0, 5.0]))
        np.testing.assert_allclose(observation["pool_grown"], np.array([5.0, 5.0]))
        np.testing.assert_allclose(observation["pool_raw_norm"], np.array([0.05, 0.05]))

    def test_dynamic_pool_capacity_scales_with_group_size_and_density(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=1.0,
                p_mode="dynamic",
                p_max=999.0,
                p_c=3.0,
                beta=0.0,
                episode_length=1,
            ),
            {0: [1], 1: [0, 2], 2: [1]},
        )

        observation = env.reset(initial_resources=[4.0, 5.0, 4.0], initial_strategies=[1, 1, 1], seed=0)
        np.testing.assert_allclose(observation["local_actual_cooperators"], np.array([2.0, 3.0, 2.0]))
        np.testing.assert_allclose(observation["p_max"], np.array([6.0, 9.0, 6.0]))
        np.testing.assert_allclose(observation["pool_theoretical_max"], np.array([6.0, 9.0, 6.0]))
        np.testing.assert_allclose(observation["pool_capacity"], np.array([6.0, 9.0, 6.0]))

    def test_dynamic_pool_raw_norm_can_exceed_one(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=1.0,
                r=0.0,
                p_mode="dynamic",
                p_max=999.0,
                p_c=1.0,
                beta=0.0,
                episode_length=1,
            ),
            {0: [1], 1: [0]},
        )

        observation = env.reset(initial_resources=[10.0, 10.0], initial_strategies=[1, 1], seed=0)
        np.testing.assert_allclose(observation["pool_theoretical_max"], np.array([2.0, 2.0]))
        np.testing.assert_allclose(observation["pool_raw"], np.array([10.0, 10.0]))
        np.testing.assert_allclose(observation["pool_raw_norm"], np.array([5.0, 5.0]))

    def test_fixed_resource_consumption_reduces_resources_but_not_payoff(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=10.0,
                resource_consumption_mode="fixed",
                resource_consumption_fixed=1.0,
                beta=0.0,
                episode_length=1,
            ),
            {0: [1], 1: [0]},
        )

        observation = env.reset(initial_resources=[3.0, 3.0], initial_strategies=[1, 1], seed=0)
        next_observation, reward, done, info = env.step(UniformAllocationPolicy().allocate(observation))
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 0.0)
        np.testing.assert_allclose(info["consumption"], np.array([1.0, 1.0]))
        np.testing.assert_allclose(info["payoff"], np.array([0.0, 0.0]))
        np.testing.assert_allclose(next_observation["resources"], np.array([2.0, 2.0]))

    def test_reward_can_include_mean_resource_term(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=10.0,
                beta=0.0,
                episode_length=1,
                reward=RewardConfig(
                    lambda_payoff=0.0,
                    lambda_cooperation=0.0,
                    lambda_total_resource=0.5,
                    lambda_gini=0.0,
                ),
            ),
            {0: [1], 1: [0]},
        )

        observation = env.reset(initial_resources=[3.0, 3.0], initial_strategies=[1, 1], seed=0)
        _, reward, done, info = env.step(UniformAllocationPolicy().allocate(observation))
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 1.5)
        self.assertAlmostEqual(info["reward_components"]["mean_resource_next"], 3.0)
        self.assertAlmostEqual(info["reward_components"]["total_resource_next"], 6.0)

    def test_proportional_resource_consumption_is_capped_by_available_resources(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                resource_consumption_mode="proportional",
                resource_consumption_rate=2.0,
                beta=0.0,
                episode_length=1,
                num_nodes=1,
            ),
            [],
        )

        observation = env.reset(initial_resources=[1.0], initial_strategies=[0], seed=0)
        next_observation, reward, done, info = env.step(np.array([[1.0]], dtype=np.float64))
        self.assertTrue(done)
        self.assertAlmostEqual(reward, 0.0)
        np.testing.assert_allclose(info["consumption"], np.array([1.0]))
        np.testing.assert_allclose(info["payoff"], np.array([0.0]))
        np.testing.assert_allclose(next_observation["resources"], np.array([0.0]))

    def test_degree_scaled_fixed_consumption_uses_node_degree(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                resource_consumption_mode="fixed",
                resource_consumption_fixed_mode="degree_scaled",
                resource_consumption_degree_multiplier=0.5,
                beta=0.0,
                episode_length=1,
            ),
            {0: [1], 1: [0, 2], 2: [1]},
        )
        nominal = env._compute_nominal_consumption(np.array([10.0, 10.0, 10.0], dtype=np.float64))
        np.testing.assert_allclose(nominal, np.array([0.5, 1.0, 0.5]))

    def test_piecewise_linear_consumption_adds_fixed_and_excess_terms(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                resource_consumption_mode="piecewise_linear",
                resource_consumption_fixed=1.0,
                resource_consumption_rate=0.2,
                resource_consumption_threshold=5.0,
                beta=0.0,
                episode_length=1,
                num_nodes=2,
            ),
            {0: [1], 1: [0]},
        )
        nominal = env._compute_nominal_consumption(np.array([2.0, 10.0], dtype=np.float64))
        np.testing.assert_allclose(nominal, np.array([1.0, 2.0]))

    def test_piecewise_linear_consumption_can_use_degree_scaled_fixed_term(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                resource_consumption_mode="piecewise_linear",
                resource_consumption_fixed_mode="degree_scaled",
                resource_consumption_degree_multiplier=0.5,
                resource_consumption_rate=0.2,
                resource_consumption_threshold=5.0,
                beta=0.0,
                episode_length=1,
            ),
            {0: [1], 1: [0, 2], 2: [1]},
        )
        nominal = env._compute_nominal_consumption(np.array([2.0, 10.0, 8.0], dtype=np.float64))
        np.testing.assert_allclose(nominal, np.array([0.5, 2.0, 1.1]))

    def test_synchronous_fermi_update_uses_previous_payoff(self) -> None:
        env = SPGGEnv(
            SPGGConfig(alpha=0.0, r=0.0, p_max=1.0, beta=100.0, episode_length=1),
            {0: [1], 1: [0]},
        )
        env.rng = np.random.default_rng(0)
        next_nominal = env._synchronous_fermi_update(
            np.array([0, 1], dtype=np.int8),
            np.array([0.0, 10.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(next_nominal, np.array([1, 1], dtype=np.int8))

    def test_imitate_best_copies_highest_payoff_strategy(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                strategy_update_rule="imitate_best",
                episode_length=1,
            ),
            {0: [1], 1: [0]},
        )
        env.rng = np.random.default_rng(0)
        next_nominal = env._imitate_best_update(
            np.array([0, 1], dtype=np.int8),
            np.array([0.0, 10.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(next_nominal, np.array([1, 1], dtype=np.int8))

    def test_q_learning_can_switch_to_other_action_after_negative_reward(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                strategy_update_rule="q_learning",
                q_learning_rate=1.0,
                q_learning_discount=0.0,
                q_learning_epsilon=0.0,
                episode_length=1,
                num_nodes=1,
            ),
            [],
        )
        env.reset(initial_resources=[5.0], initial_strategies=[1], seed=0)
        next_nominal = env._q_learning_update(
            np.array([1], dtype=np.int8),
            np.array([-1.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(next_nominal, np.array([0], dtype=np.int8))

    def test_q_learning_uses_actual_action_when_resources_force_defection(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                strategy_update_rule="q_learning",
                q_learning_rate=1.0,
                q_learning_discount=0.0,
                q_learning_epsilon=0.0,
                episode_length=1,
                num_nodes=1,
            ),
            [],
        )
        env.reset(initial_resources=[0.0], initial_strategies=[1], seed=0)
        env._q_values = np.array([[0.0, 0.0]], dtype=np.float64)

        next_nominal = env._q_learning_update(
            np.array([0], dtype=np.int8),
            np.array([3.0], dtype=np.float64),
        )

        self.assertAlmostEqual(float(env._q_values[0, 0]), 3.0)
        self.assertAlmostEqual(float(env._q_values[0, 1]), 0.0)
        np.testing.assert_array_equal(next_nominal, np.array([0], dtype=np.int8))

    def test_q_learning_2x2_updates_using_previous_action_state(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                strategy_update_rule="q_learning_2x2",
                q_learning_rate=1.0,
                q_learning_discount=1.0,
                q_learning_epsilon=0.0,
                episode_length=1,
                num_nodes=1,
            ),
            [],
        )
        env.reset(initial_resources=[5.0], initial_strategies=[1], seed=0)
        env._q_values = np.array([[[0.0, 0.0], [0.0, 2.0]]], dtype=np.float64)
        env._q_learning_previous_actions = np.array([0], dtype=np.int8)

        next_nominal = env._q_learning_2x2_update(
            np.array([1], dtype=np.int8),
            np.array([5.0], dtype=np.float64),
        )

        self.assertAlmostEqual(float(env._q_values[0, 0, 1]), 7.0)
        np.testing.assert_array_equal(next_nominal, np.array([1], dtype=np.int8))
        np.testing.assert_array_equal(env._q_learning_previous_actions, np.array([1], dtype=np.int8))

    def test_q_learning_2x2_uses_actual_action_and_previous_actual_state(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.0,
                p_max=1.0,
                strategy_update_rule="q_learning_2x2",
                q_learning_rate=1.0,
                q_learning_discount=0.0,
                q_learning_epsilon=0.0,
                episode_length=1,
                num_nodes=1,
            ),
            [],
        )
        env.reset(initial_resources=[0.0], initial_strategies=[1], seed=0)
        env._q_values = np.array([[[1.0, 0.0], [0.0, 0.0]]], dtype=np.float64)
        env._q_learning_previous_actions = np.array([1], dtype=np.int8)

        next_nominal = env._q_learning_2x2_update(
            np.array([0], dtype=np.int8),
            np.array([4.0], dtype=np.float64),
        )

        self.assertAlmostEqual(float(env._q_values[0, 1, 0]), 4.0)
        self.assertAlmostEqual(float(env._q_values[0, 1, 1]), 0.0)
        np.testing.assert_array_equal(next_nominal, np.array([0], dtype=np.int8))
        np.testing.assert_array_equal(env._q_learning_previous_actions, np.array([0], dtype=np.int8))

    def test_isolated_node_keeps_nominal_strategy(self) -> None:
        env = SPGGEnv(
            SPGGConfig(alpha=0.0, r=0.0, p_max=1.0, beta=100.0, episode_length=1, num_nodes=1),
            [],
        )
        observation = env.reset(initial_resources=[5.0], initial_strategies=[1], seed=0)
        next_observation, _, _, _ = env.step(np.array([[1.0]], dtype=np.float64))
        np.testing.assert_array_equal(observation["x_nominal"], np.array([1], dtype=np.int8))
        np.testing.assert_array_equal(next_observation["x_nominal"], np.array([1], dtype=np.int8))

    def test_random_regular_graph_has_uniform_degree(self) -> None:
        graph = make_random_regular_graph(num_nodes=10, degree=3, seed=0)
        degrees = np.array([len(neighbors) for neighbors in graph.values()])
        np.testing.assert_array_equal(degrees, np.full(10, 3))
        for node, neighbors in graph.items():
            self.assertNotIn(node, neighbors)
            for neighbor in neighbors:
                self.assertIn(node, graph[neighbor])

    def test_erdos_renyi_extremes_are_valid(self) -> None:
        empty_graph = make_erdos_renyi_graph(num_nodes=5, edge_prob=0.0, seed=0)
        complete_graph = make_erdos_renyi_graph(num_nodes=5, edge_prob=1.0, seed=0)
        self.assertTrue(all(len(neighbors) == 0 for neighbors in empty_graph.values()))
        self.assertTrue(all(len(neighbors) == 4 for neighbors in complete_graph.values()))

    def test_watts_strogatz_preserves_edge_count(self) -> None:
        num_nodes = 12
        degree = 4
        graph = make_watts_strogatz_graph(
            num_nodes=num_nodes,
            degree=degree,
            rewiring_prob=0.5,
            seed=0,
        )
        total_degree = sum(len(neighbors) for neighbors in graph.values())
        self.assertEqual(total_degree, num_nodes * degree)
        for node, neighbors in graph.items():
            self.assertNotIn(node, neighbors)
            for neighbor in neighbors:
                self.assertIn(node, graph[neighbor])

    def test_barabasi_albert_graph_edge_count(self) -> None:
        num_nodes = 8
        attachments = 2
        graph = make_barabasi_albert_graph(
            num_nodes=num_nodes,
            attachments_per_new_node=attachments,
            seed=0,
        )
        expected_edges = ((attachments + 1) * attachments // 2) + (
            attachments * (num_nodes - (attachments + 1))
        )
        actual_edges = sum(len(neighbors) for neighbors in graph.values()) // 2
        self.assertEqual(actual_edges, expected_edges)
        self.assertTrue(all(len(neighbors) > 0 for neighbors in graph.values()))
