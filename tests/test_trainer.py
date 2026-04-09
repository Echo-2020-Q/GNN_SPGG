from __future__ import annotations

import importlib.util
import unittest
from unittest import mock


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import numpy as np
    import torch

    from Project1.env import RewardConfig, SPGGConfig, SPGGEnv, make_grid_graph
    from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig
    from Project1.policies.rule_based import UniformAllocationPolicy
    from Project1.td3.data import TensorReplayActionRecord, TensorTransition
    from Project1.td3.replay import ReplayBuffer
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
            self.assertEqual(str(trainer.demo_pretrain_summary["critic_target_mode"]), "n_step")
            self.assertIn("demo_return_target_mean", trainer.demo_pretrain_summary)
            self.assertIn("demo_return_target_std", trainer.demo_pretrain_summary)
        finally:
            trainer.close()

    def test_critic_bridge_training_smoke(self) -> None:
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
                critic_bridge_enabled=True,
                critic_bridge_env_steps=4,
                critic_bridge_updates=1,
                critic_bridge_batch_size=2,
                critic_bridge_teacher_return_aux_coef=0.5,
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
            self.assertGreaterEqual(float(trainer.demo_pretrain_summary["critic_bridge_updates"]), 1.0)
            self.assertIn("critic_bridge_val_loss_best", trainer.demo_pretrain_summary)
            self.assertIn("critic_bridge_replay_size_after_collection", trainer.demo_pretrain_summary)
            self.assertIn("critic_bridge_teacher_aux_loss_last", trainer.demo_pretrain_summary)
        finally:
            trainer.close()

    def test_critic_bridge_adaptive_teacher_aux_can_decay(self) -> None:
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
                critic_bridge_enabled=True,
                critic_bridge_env_steps=4,
                critic_bridge_updates=2,
                critic_bridge_batch_size=2,
                critic_bridge_eval_interval=1,
                critic_bridge_patience=5,
                critic_bridge_teacher_return_aux_schedule="adaptive",
                critic_bridge_teacher_return_aux_levels=(1.0, 0.0),
                critic_bridge_teacher_return_aux_required_evals=1,
                critic_bridge_teacher_return_aux_max_val_ratio=10.0,
                critic_bridge_teacher_return_aux_max_error_ratio=1_000_000.0,
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
            self.assertIsNotNone(trainer.demo_pretrain_summary)
            self.assertEqual(float(trainer.demo_pretrain_summary["critic_bridge_teacher_aux_coef"]), 0.0)
            self.assertGreaterEqual(
                float(trainer.demo_pretrain_summary["critic_bridge_teacher_aux_reduction_count"]),
                1.0,
            )
            self.assertGreaterEqual(
                float(trainer.demo_pretrain_summary["critic_bridge_teacher_aux_level_index"]),
                1.0,
            )
        finally:
            trainer.close()

    def test_preloaded_demo_replay_skips_internal_collection(self) -> None:
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
        allocation = UniformAllocationPolicy().allocate(observation).astype(np.float32, copy=False)
        next_observation, reward, done, _ = env.step(allocation)
        transition = TensorTransition.from_step(
            obs=observation,
            action=TensorReplayActionRecord(allocation=torch.as_tensor(allocation, dtype=torch.float32)),
            reward=float(reward),
            next_obs=next_observation,
            done=bool(done),
            is_demo=True,
            collapse_flag=False,
            topology_name="fixed",
            pool_power_demo_flag=True,
            demo_return_target=1.0,
            demo_return_valid=True,
        )
        replay_buffer = ReplayBuffer(
            capacity=32,
            seed=0,
            replay_strategy="topology_stratified_mixed",
            topology_names=("fixed",),
            recent_fraction=0.50,
            long_term_fraction=0.35,
            demo_fraction=0.15,
            demo_behavior_source="pool_power_mix",
        )
        replay_buffer.add(transition)

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
                demo_collection_env_steps=0,
                actor_bc_pretrain_updates=1,
                critic_pretrain_updates=1,
                demo_pretrain_batch_size=1,
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
            trainer.preload_demo_replay(
                replay_buffer,
                {
                    "enabled": True,
                    "demo_collection_env_steps": 4.0,
                    "demo_replay_size_after_collection": 1.0,
                    "actor_bc_updates": 0.0,
                    "critic_pretrain_updates": 0.0,
                    "actor_bc_loss_last": 0.0,
                    "critic_loss_last": 0.0,
                    "seconds_collection": 0.5,
                    "seconds_actor_bc": 0.0,
                    "seconds_critic": 0.0,
                    "dataset_path": None,
                    "behavior_source": "pool_power_mix",
                    "critic_target_mode": "n_step",
                    "demo_return_target_mean": 1.0,
                    "demo_return_target_std": 0.0,
                },
            )
            summary = trainer._run_demo_pretrain()
            self.assertEqual(int(summary["actor_bc_updates"]), 1)
            self.assertEqual(int(summary["critic_pretrain_updates"]), 1)
            self.assertEqual(float(summary["demo_collection_env_steps"]), 4.0)
            self.assertEqual(float(summary["demo_replay_size_after_collection"]), 1.0)
            self.assertEqual(float(summary["seconds_collection"]), 0.5)
        finally:
            trainer.close()

    def test_demo_pretrain_early_stopping_uses_validation_metrics(self) -> None:
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
        allocation = UniformAllocationPolicy().allocate(observation).astype(np.float32, copy=False)
        next_observation, reward, done, _ = env.step(allocation)
        transition = TensorTransition.from_step(
            obs=observation,
            action=TensorReplayActionRecord(allocation=torch.as_tensor(allocation, dtype=torch.float32)),
            reward=float(reward),
            next_obs=next_observation,
            done=bool(done),
            is_demo=True,
            collapse_flag=False,
            topology_name="fixed",
            pool_power_demo_flag=True,
            demo_return_target=1.0,
            demo_return_valid=True,
        )
        replay_buffer = ReplayBuffer(
            capacity=32,
            seed=0,
            replay_strategy="topology_stratified_mixed",
            topology_names=("fixed",),
            recent_fraction=0.50,
            long_term_fraction=0.35,
            demo_fraction=0.15,
            demo_behavior_source="pool_power_mix",
        )
        replay_buffer.add(transition)
        validation_batch = replay_buffer.export_demo_batch()
        assert validation_batch is not None

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
                demo_collection_env_steps=0,
                actor_bc_pretrain_updates=4,
                critic_pretrain_updates=4,
                demo_pretrain_batch_size=1,
                demo_pretrain_eval_interval=1,
                demo_pretrain_patience=1,
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
            trainer.preload_demo_replay(
                replay_buffer,
                {
                    "enabled": True,
                    "demo_collection_env_steps": 4.0,
                    "demo_replay_size_after_collection": 1.0,
                    "demo_train_replay_size_after_split": 1.0,
                    "demo_val_replay_size_after_split": 1.0,
                },
                validation_batch,
            )
            validation_sequence = iter(
                [
                    {"actor_bc_val_loss": 1.0, "quick_eval_return_mean": 1.0},
                    {"actor_bc_val_loss": 1.0, "quick_eval_return_mean": 1.0},
                    {"critic_val_loss": 2.0, "critic_q_pred_mean": 1.0, "critic_target_mean": 1.0},
                    {"critic_val_loss": 2.0, "critic_q_pred_mean": 1.0, "critic_target_mean": 1.0},
                ]
            )
            with mock.patch.object(
                trainer,
                "_run_demo_pretrain_validation",
                side_effect=lambda include_quick_eval: dict(next(validation_sequence)),
            ):
                summary = trainer._run_demo_pretrain()

            self.assertTrue(bool(summary["actor_bc_early_stopped"]))
            self.assertTrue(bool(summary["critic_pretrain_early_stopped"]))
            self.assertLess(int(summary["actor_bc_updates"]), 4)
            self.assertLess(int(summary["critic_pretrain_updates"]), 4)
            self.assertGreaterEqual(float(summary["actor_bc_eval_count"]), 2.0)
            self.assertGreaterEqual(float(summary["critic_eval_count"]), 2.0)
        finally:
            trainer.close()

    def test_adaptive_teacher_release_unlocks_after_stable_evals(self) -> None:
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
                total_updates=2,
                steps_per_update=4,
                eval_interval=1,
                eval_episodes=1,
                adaptive_teacher_release_enabled=True,
                adaptive_teacher_release_required_evals=2,
                adaptive_teacher_release_min_criteria=2,
                warmup_steps=0,
                seed=0,
            ),
        )
        try:
            trainer.demo_pretrain_summary = {
                "quick_eval_return_best": 10.0,
                "actor_bc_val_loss_best": 0.5,
                "critic_val_loss_best": 2.0,
            }
            trainer.global_env_steps = 123
            first = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.0,
                online_eval_return_mean=9.5,
                actor_bc_val_loss=0.55,
                critic_val_loss=2.1,
                behavior_frac_actor_logits=0.0,
            )
            second = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.0,
                online_eval_return_mean=9.6,
                actor_bc_val_loss=0.56,
                critic_val_loss=2.2,
                behavior_frac_actor_logits=0.0,
            )
            self.assertEqual(float(first["teacher_release_unlocked"]), 0.0)
            self.assertEqual(float(second["teacher_release_unlocked"]), 1.0)
            self.assertEqual(int(trainer.teacher_takeover_release_env_step or -1), 123)
            self.assertEqual(int(second["teacher_handoff_stage"]), 1)
        finally:
            trainer.close()

    def test_adaptive_teacher_release_can_unlock_from_eval_cooperation_threshold(self) -> None:
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
                total_updates=2,
                steps_per_update=4,
                eval_interval=1,
                eval_episodes=1,
                adaptive_teacher_release_enabled=True,
                adaptive_teacher_release_mode="eval_cooperation",
                adaptive_teacher_release_min_cooperation=0.8,
                adaptive_teacher_release_required_evals=2,
                warmup_steps=0,
                seed=0,
            ),
        )
        try:
            trainer.global_env_steps = 456
            first = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.81,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.0,
            )
            second = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.83,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.0,
            )
            self.assertEqual(float(first["teacher_release_unlocked"]), 0.0)
            self.assertEqual(float(second["teacher_release_unlocked"]), 1.0)
            self.assertEqual(int(trainer.teacher_takeover_release_env_step or -1), 456)
            self.assertEqual(int(second["teacher_handoff_stage"]), 1)
        finally:
            trainer.close()

    def test_adaptive_teacher_handoff_advances_to_full_stage_after_actor_behavior_ready(self) -> None:
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
                total_updates=2,
                steps_per_update=4,
                eval_interval=1,
                eval_episodes=1,
                adaptive_teacher_release_enabled=True,
                adaptive_teacher_release_mode="eval_cooperation",
                adaptive_teacher_release_min_cooperation=0.8,
                adaptive_teacher_release_required_evals=1,
                adaptive_teacher_handoff_min_actor_behavior=0.6,
                adaptive_teacher_handoff_required_evals=2,
                warmup_steps=0,
                seed=0,
            ),
        )
        try:
            trainer.global_env_steps = 100
            unlocked = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.85,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.2,
            )
            trainer.global_env_steps = 120
            first_full = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.86,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.7,
            )
            trainer.global_env_steps = 140
            second_full = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.88,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.75,
            )
            self.assertEqual(int(unlocked["teacher_handoff_stage"]), 1)
            self.assertEqual(int(first_full["teacher_handoff_stage"]), 1)
            self.assertEqual(int(second_full["teacher_handoff_stage"]), 2)
            self.assertEqual(int(trainer.teacher_takeover_full_release_env_step or -1), 140)
        finally:
            trainer.close()

    def test_adaptive_teacher_handoff_can_rollback_to_soft_release(self) -> None:
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
                total_updates=2,
                steps_per_update=4,
                eval_interval=1,
                eval_episodes=1,
                adaptive_teacher_release_enabled=True,
                adaptive_teacher_release_mode="eval_cooperation",
                adaptive_teacher_release_min_cooperation=0.8,
                adaptive_teacher_release_required_evals=1,
                adaptive_teacher_handoff_min_actor_behavior=0.6,
                adaptive_teacher_handoff_required_evals=1,
                adaptive_teacher_handoff_rollback_enabled=True,
                adaptive_teacher_handoff_rollback_min_actor_behavior=0.45,
                adaptive_teacher_handoff_rollback_required_evals=2,
                warmup_steps=0,
                seed=0,
            ),
        )
        try:
            trainer.global_env_steps = 100
            trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.85,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.2,
            )
            trainer.global_env_steps = 120
            trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.86,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.7,
            )
            trainer.global_env_steps = 140
            first_regression = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.82,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.3,
            )
            trainer.global_env_steps = 160
            second_regression = trainer._update_adaptive_teacher_release(
                online_eval_cooperation_mean=0.82,
                online_eval_return_mean=0.0,
                actor_bc_val_loss=None,
                critic_val_loss=None,
                behavior_frac_actor_logits=0.3,
            )
            self.assertEqual(int(first_regression["teacher_handoff_stage"]), 2)
            self.assertEqual(int(second_regression["teacher_handoff_stage"]), 1)
            self.assertEqual(float(second_regression["teacher_handoff_stage_just_regressed"]), 1.0)
            self.assertEqual(trainer.teacher_takeover_full_release_env_step, None)
            self.assertEqual(int(trainer.teacher_takeover_soft_release_env_step or -1), 160)
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
