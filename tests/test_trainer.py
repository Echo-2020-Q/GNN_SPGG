from __future__ import annotations

import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import numpy as np
    import torch

    from Project1.env import RewardConfig, SPGGConfig, SPGGEnv, make_grid_graph
    from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig
    from Project1.td3.critic import GraphActionCritic, GraphActionCriticConfig
    from Project1.trainer import CentralizedActorCriticTrainer, TrainerConfig


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for trainer tests")
class TrainerSmokeTests(unittest.TestCase):
    def test_rollout_logits_match_full_policy_logits(self) -> None:
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
        observation = env.reset(seed=0)
        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))

        with torch.inference_mode():
            full_output = policy.deterministic_action(observation)
            rollout_logits = policy.rollout_logits(observation)
            rollout_logits_batch = policy.rollout_logits_batch([observation, observation])

        self.assertIsNotNone(full_output.logits)
        self.assertTrue(torch.allclose(rollout_logits, full_output.logits))
        self.assertEqual(tuple(rollout_logits_batch.shape), (2, 4, 4))
        self.assertTrue(torch.allclose(rollout_logits_batch[0], full_output.logits))
        self.assertTrue(torch.allclose(rollout_logits_batch[1], full_output.logits))

    def test_batched_policy_forward_matches_individual_forward(self) -> None:
        env_config = SPGGConfig(
            alpha=0.0,
            r=0.5,
            p_max=5.0,
            beta=1.0,
            episode_length=4,
            reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.5, lambda_gini=0.1),
        )
        env_a = SPGGEnv(env_config, make_grid_graph(2, 2))
        env_b = SPGGEnv(env_config, make_grid_graph(2, 2))
        observation_a = env_a.reset(seed=0)
        observation_b = env_b.reset(seed=1)
        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))

        with torch.inference_mode():
            output_a = policy.deterministic_action(observation_a)
            output_b = policy.deterministic_action(observation_b)
            batched_outputs = policy.deterministic_action_batch([observation_a, observation_b])

        self.assertEqual(len(batched_outputs), 2)
        self.assertTrue(torch.allclose(batched_outputs[0].logits, output_a.logits))
        self.assertTrue(torch.allclose(batched_outputs[1].logits, output_b.logits))
        self.assertTrue(torch.allclose(batched_outputs[0].allocation_matrix, output_a.allocation_matrix))
        self.assertTrue(torch.allclose(batched_outputs[1].allocation_matrix, output_b.allocation_matrix))

    def test_batched_critic_forward_matches_individual_forward(self) -> None:
        env_config = SPGGConfig(
            alpha=0.0,
            r=0.5,
            p_max=5.0,
            beta=1.0,
            episode_length=4,
            reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.5, lambda_gini=0.1),
        )
        env_a = SPGGEnv(env_config, make_grid_graph(2, 2))
        env_b = SPGGEnv(env_config, make_grid_graph(2, 2))
        observation_a = env_a.reset(seed=0)
        observation_b = env_b.reset(seed=1)
        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))
        critic = GraphActionCritic(GraphActionCriticConfig(state_hidden_dim=16))

        with torch.inference_mode():
            action_a = policy.deterministic_action(observation_a).allocation_matrix
            action_b = policy.deterministic_action(observation_b).allocation_matrix
            q_a = critic.forward(observation_a, action_a)
            q_b = critic.forward(observation_b, action_b)
            batched_q = critic.forward_batch([observation_a, observation_b], [action_a, action_b])

        self.assertEqual(tuple(batched_q.shape), (2,))
        self.assertTrue(torch.allclose(batched_q[0], q_a, atol=1e-5, rtol=1e-4))
        self.assertTrue(torch.allclose(batched_q[1], q_b, atol=1e-5, rtol=1e-4))

    def test_single_update_training_smoke(self) -> None:
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
        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))
        trainer = CentralizedActorCriticTrainer(
            env=env,
            policy=policy,
            config=TrainerConfig(
                total_updates=1,
                steps_per_update=4,
                eval_interval=1,
                eval_episodes=1,
                seed=0,
            ),
        )

        history = trainer.train(num_updates=1)
        self.assertEqual(len(history), 1)
        self.assertTrue(np.isfinite(history[0]["loss"]))
        self.assertIn("eval_return_mean", history[0])
        trainer.close()

    def test_demo_pretrain_config_rejects_unsupported_behavior_source(self) -> None:
        with self.assertRaises(ValueError):
            TrainerConfig(
                total_updates=1,
                steps_per_update=4,
                eval_interval=1,
                eval_episodes=1,
                demo_collection_behavior_source="uniform",
                seed=0,
            )

    def test_demo_pretrain_training_smoke(self) -> None:
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
        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))
        trainer = CentralizedActorCriticTrainer(
            env=env,
            policy=policy,
            config=TrainerConfig(
                total_updates=1,
                steps_per_update=4,
                eval_interval=1,
                eval_episodes=1,
                demo_pretrain_enabled=True,
                demo_collection_env_steps=4,
                actor_bc_pretrain_updates=1,
                critic_pretrain_updates=1,
                demo_pretrain_batch_size=2,
                replay_strategy="topology_stratified_mixed",
                replay_topology_names=("fixed",),
                replay_recent_fraction=0.50,
                replay_long_term_fraction=0.35,
                replay_demo_fraction=0.15,
                warmup_steps=0,
                seed=0,
            ),
        )

        try:
            history = trainer.train(num_updates=1)
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0]["loss"]))
            self.assertIsNotNone(trainer.demo_pretrain_summary)
            self.assertGreater(float(trainer.demo_pretrain_summary["demo_replay_size_after_collection"]), 0.0)
            self.assertEqual(int(trainer.demo_pretrain_summary["actor_bc_updates"]), 1)
            self.assertEqual(int(trainer.demo_pretrain_summary["critic_pretrain_updates"]), 1)
        finally:
            trainer.close()

    def test_parallel_worker_training_smoke(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.5,
                p_max=5.0,
                beta=1.0,
                episode_length=3,
                reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.5, lambda_gini=0.1),
            ),
            make_grid_graph(2, 2),
        )
        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))
        trainer = CentralizedActorCriticTrainer(
            env=env,
            policy=policy,
            config=TrainerConfig(
                total_updates=1,
                steps_per_update=3,
                eval_interval=1,
                eval_episodes=1,
                num_workers=2,
                num_envs_per_worker=2,
                rollout_inference_mode="centralized",
                rollout_device=("cpu", "cpu"),
                rollout_inference_batch_timeout_ms=0.0,
                rollout_num_threads=1,
                warmup_steps=0,
                seed=0,
            ),
        )

        try:
            history = trainer.train(num_updates=1)
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0]["loss"]))
            self.assertIn("eval_return_mean", history[0])
            self.assertIn("global_env_steps", history[0])
            self.assertEqual(int(history[0]["global_env_steps"]), 6)
            self.assertIn("profile_rollout_collect_seconds", history[0])
            self.assertIn("profile_rollout_env_step_seconds", history[0])
            self.assertIn("profile_learner_update_seconds", history[0])
            self.assertIn("profile_rollout_finish_wait_seconds", history[0])
            self.assertIn("profile_rollout_overlap_seconds", history[0])
            self.assertGreaterEqual(float(history[0]["profile_rollout_inference_batch_size_mean"]), 1.0)
        finally:
            trainer.close()

    def test_parallel_worker_overlap_training_smoke(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.5,
                p_max=5.0,
                beta=1.0,
                episode_length=3,
                reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.5, lambda_gini=0.1),
            ),
            make_grid_graph(2, 2),
        )
        policy = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))
        trainer = CentralizedActorCriticTrainer(
            env=env,
            policy=policy,
            config=TrainerConfig(
                total_updates=2,
                steps_per_update=3,
                eval_interval=10,
                eval_episodes=1,
                num_workers=2,
                num_envs_per_worker=2,
                overlap_rollout_and_update=True,
                rollout_inference_mode="local",
                rollout_device="cpu",
                rollout_num_threads=1,
                warmup_steps=0,
                seed=0,
            ),
        )

        try:
            history = trainer.train(num_updates=2)
            self.assertEqual(len(history), 2)
            self.assertGreater(float(history[1]["profile_rollout_overlap_seconds"]), 0.0)
            self.assertGreaterEqual(float(history[1]["profile_rollout_collect_seconds"]), float(history[1]["profile_rollout_finish_wait_seconds"]))
        finally:
            trainer.close()

    def test_parallel_worker_full_resume_checkpoint(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.5,
                p_max=5.0,
                beta=1.0,
                episode_length=3,
                reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.5, lambda_gini=0.1),
            ),
            make_grid_graph(2, 2),
        )
        config = TrainerConfig(
            total_updates=2,
            steps_per_update=3,
            eval_interval=10,
            eval_episodes=1,
            num_workers=2,
            warmup_steps=2,
            replay_capacity=128,
            batch_size=4,
            seed=0,
        )

        trainer = CentralizedActorCriticTrainer(
            env=env,
            policy=GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2)),
            config=config,
        )
        try:
            first_history = trainer.train(num_updates=1)
            checkpoint = trainer.build_checkpoint(
                update=1,
                metrics=first_history[-1],
                checkpoint_mode="full_resume",
            )
            replay_size_before = len(trainer.replay_buffer)
            global_env_steps_before = trainer.global_env_steps
        finally:
            trainer.close()

        resumed_trainer = CentralizedActorCriticTrainer(
            env=env,
            policy=GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2)),
            config=config,
        )
        try:
            checkpoint_mode = resumed_trainer.load_checkpoint(checkpoint)
            self.assertEqual(checkpoint_mode, "full_resume")
            self.assertEqual(len(resumed_trainer.replay_buffer), replay_size_before)
            self.assertEqual(resumed_trainer.global_env_steps, global_env_steps_before)

            resumed_history = resumed_trainer.train(num_updates=2)
            self.assertEqual(len(resumed_history), 2)
            self.assertGreater(int(resumed_history[-1]["global_env_steps"]), int(global_env_steps_before))
        finally:
            resumed_trainer.close()

    def test_parallel_worker_remote_traceback_is_forwarded(self) -> None:
        env = SPGGEnv(
            SPGGConfig(
                alpha=0.0,
                r=0.5,
                p_max=5.0,
                beta=1.0,
                episode_length=3,
                reward=RewardConfig(lambda_payoff=1.0, lambda_cooperation=0.5, lambda_gini=0.1),
            ),
            make_grid_graph(2, 2),
        )
        trainer = CentralizedActorCriticTrainer(
            env=env,
            policy=GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2)),
            config=TrainerConfig(
                total_updates=1,
                steps_per_update=3,
                eval_interval=10,
                eval_episodes=1,
                num_workers=2,
                seed=0,
            ),
        )
        try:
            with self.assertRaises(RuntimeError) as context:
                trainer.workers[0].load_state_dict({"bad_state": True})
            self.assertIn("Remote worker", str(context.exception))
            self.assertIn("Traceback", str(context.exception))
        finally:
            trainer.close()
