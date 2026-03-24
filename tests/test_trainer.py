from __future__ import annotations

import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import numpy as np

    from Project1.env import RewardConfig, SPGGConfig, SPGGEnv, make_grid_graph
    from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig
    from Project1.trainer import CentralizedActorCriticTrainer, TrainerConfig


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for trainer tests")
class TrainerSmokeTests(unittest.TestCase):
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
                warmup_steps=2,
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
