from __future__ import annotations

import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import numpy as np

    from Project1.env import RewardConfig, SPGGConfig, SPGGEnv, make_grid_graph
    from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig
    from Project1.td3.config import DomainRandomizationConfig, GraphTD3Config, WorkerConfig
    from Project1.td3.exploration import LogitSpaceExplorer
    from Project1.td3.worker import RandomizedEnvFactory, RolloutWorker


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for worker tests")
class RolloutWorkerTeacherTests(unittest.TestCase):
    def _make_worker(self, *, teacher_takeover_enabled: bool = False) -> RolloutWorker:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.5,
                p_max=5.0,
                beta=1.0,
                episode_length=4,
                reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.5, lambda_gini=0.1),
            ),
            make_grid_graph(2, 2),
        )
        config = GraphTD3Config(
            total_updates=10,
            steps_per_update=4,
            eval_interval=1,
            eval_episodes=1,
            warmup_steps=0,
            warmup_pool_power_k=19.0,
            teacher_takeover_enabled=teacher_takeover_enabled,
            teacher_takeover_start_prob=1.0,
            teacher_takeover_end_prob=1.0,
            teacher_takeover_decay_end_fraction=1.0,
            device="cpu",
        )
        return RolloutWorker(
            actor=GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2)),
            explorer=LogitSpaceExplorer(),
            env_factory=RandomizedEnvFactory.from_env(env),
            config=WorkerConfig(worker_id=0, seed=0, rollout_steps_per_sync=4, num_envs_per_worker=1),
            train_config=config,
            device="cpu",
        )

    def test_forced_demo_collection_n_step_targets_match_rewards(self) -> None:
        worker = self._make_worker()
        result = worker.collect(
            num_steps=4,
            forced_behavior_source="pool_power_mix",
            mark_as_demo=True,
            count_env_steps=False,
            demo_return_target_mode="n_step",
            demo_return_n_step=2,
        )
        batch = result.replay_batch
        rewards = batch.reward.detach().cpu().numpy()
        dones = batch.done.detach().cpu().numpy() > 0.5
        expected = []
        for start in range(len(rewards)):
            discounted_return = 0.0
            discount = 1.0
            for offset in range(2):
                index = start + offset
                if index >= len(rewards):
                    break
                discounted_return += discount * float(rewards[index])
                if bool(dones[index]):
                    break
                discount *= float(worker.train_config.gamma)
            expected.append(discounted_return)

        np.testing.assert_allclose(
            batch.demo_return_target.detach().cpu().numpy(),
            np.asarray(expected, dtype=np.float32),
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertTrue(np.all(batch.demo_return_valid.detach().cpu().numpy()))

    def test_forced_demo_collection_mc_targets_complete_episode(self) -> None:
        worker = self._make_worker()
        result = worker.collect(
            num_steps=2,
            forced_behavior_source="pool_power_mix",
            mark_as_demo=True,
            count_env_steps=False,
            demo_return_target_mode="mc",
        )
        batch = result.replay_batch
        rewards = batch.reward.detach().cpu().numpy()
        dones = batch.done.detach().cpu().numpy() > 0.5
        self.assertTrue(bool(np.any(dones)))

        expected = np.zeros_like(rewards, dtype=np.float32)
        episode_start = 0
        while episode_start < len(rewards):
            done_positions = np.flatnonzero(dones[episode_start:])
            self.assertGreater(len(done_positions), 0)
            episode_end = episode_start + int(done_positions[0])
            for start in range(episode_start, episode_end + 1):
                discounted_return = 0.0
                discount = 1.0
                for reward_value in rewards[start : episode_end + 1]:
                    discounted_return += discount * float(reward_value)
                    discount *= float(worker.train_config.gamma)
                expected[start] = discounted_return
            episode_start = episode_end + 1

        np.testing.assert_allclose(
            batch.demo_return_target.detach().cpu().numpy(),
            expected,
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertTrue(np.all(batch.demo_return_valid.detach().cpu().numpy()))

    def test_teacher_takeover_marks_teacher_samples_as_demo(self) -> None:
        worker = self._make_worker(teacher_takeover_enabled=True)
        result = worker.collect(
            num_steps=4,
            global_env_start_step=0,
        )

        self.assertTrue(np.all(result.replay_batch.is_demo.detach().cpu().numpy()))
        self.assertTrue(np.all(result.replay_batch.pool_power_demo_flag.detach().cpu().numpy()))
        self.assertGreater(result.metrics["teacher_takeover_prob_mean"], 0.0)
        self.assertGreater(result.metrics["behavior_source_counts"].get("pool_power_mix", 0), 0)


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for worker tests")
class RandomizedEnvFactoryFixedGraphBankTests(unittest.TestCase):
    def _make_base_env(self) -> SPGGEnv:
        return SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.5,
                p_max=5.0,
                beta=1.0,
                episode_length=4,
                reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.5, lambda_gini=0.1),
            ),
            make_grid_graph(2, 2),
        )

    @staticmethod
    def _graph_signature(env: SPGGEnv) -> tuple[tuple[int, tuple[int, ...]], ...]:
        return tuple((node, tuple(neighbors)) for node, neighbors in enumerate(env.graph.neighbors))

    def test_fixed_graph_bank_uniform_uses_only_k_graphs_per_topology(self) -> None:
        base_env = self._make_base_env()
        for network_type in ("regular", "erdos_renyi", "small_world", "scale_free"):
            factory = RandomizedEnvFactory.from_env(
                base_env,
                randomization=DomainRandomizationConfig(
                    enabled=True,
                    network_types=(network_type,),
                    fixed_graph_bank_enabled=True,
                    fixed_graph_bank_size_per_type=2,
                    fixed_graph_bank_seed=12345,
                    fixed_graph_bank_sampling="uniform",
                    num_nodes_choices=(12,),
                    regular_degree_choices=(4,),
                    er_mean_degree_choices=(4.0,),
                    ws_degree_choices=(4,),
                    ws_rewiring_choices=(0.10,),
                    ba_attachment_choices=(2,),
                ),
            )
            rng = np.random.default_rng(0)
            seen_indices: set[int] = set()
            seen_graphs: set[tuple[tuple[int, tuple[int, ...]], ...]] = set()
            for _ in range(50):
                sampled_env, metadata = factory.sample_environment(rng)
                seen_indices.add(int(metadata["graph_bank_index"]))
                seen_graphs.add(self._graph_signature(sampled_env))
            self.assertLessEqual(len(seen_graphs), 2)
            self.assertEqual(seen_indices, {0, 1})

    def test_fixed_graph_bank_round_robin_cycles_graph_indices(self) -> None:
        factory = RandomizedEnvFactory.from_env(
            self._make_base_env(),
            randomization=DomainRandomizationConfig(
                enabled=True,
                network_types=("regular",),
                fixed_graph_bank_enabled=True,
                fixed_graph_bank_size_per_type=3,
                fixed_graph_bank_seed=7,
                fixed_graph_bank_sampling="round_robin",
                num_nodes_choices=(12,),
                regular_degree_choices=(4,),
            ),
        )
        rng = np.random.default_rng(0)
        sampled_indices = []
        for _ in range(7):
            _, metadata = factory.sample_environment(rng)
            sampled_indices.append(int(metadata["graph_bank_index"]))
        self.assertEqual(sampled_indices, [0, 1, 2, 0, 1, 2, 0])
