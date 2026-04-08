from __future__ import annotations

import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import numpy as np

    from Project1.env import RewardConfig, SPGGConfig, SPGGEnv, make_grid_graph
    from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig
    from Project1.ppo import GraphPPOConfig, GraphPPOTrainer


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for PPO trainer tests")
class GraphPPOTrainerSmokeTests(unittest.TestCase):
    def _build_trainer(self) -> GraphPPOTrainer:
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
        policy = GNNAllocationPolicy(
            GNNPolicyConfig(
                hidden_dim=16,
                num_message_passing_layers=2,
                action_distribution="dirichlet",
            )
        )
        return GraphPPOTrainer(
            env=env,
            policy=policy,
            config=GraphPPOConfig(
                total_updates=1,
                steps_per_update=2,
                eval_interval=1,
                eval_episodes=1,
                device="cpu",
                seed=0,
            ),
        )

    def test_single_update_training_smoke(self) -> None:
        trainer = self._build_trainer()
        try:
            history = trainer.train(num_updates=1)
            self.assertEqual(len(history), 1)
            self.assertTrue(np.isfinite(history[0]["loss"]))
            self.assertIn("eval_return_mean", history[0])
            self.assertAlmostEqual(float(history[0]["behavior_frac_actor_logits"]), 1.0, places=6)
        finally:
            trainer.close()

    def test_checkpoint_round_trip_restores_progress(self) -> None:
        trainer = self._build_trainer()
        try:
            history = trainer.train(num_updates=1)
            checkpoint = trainer.build_checkpoint(update=1, metrics=history[-1], checkpoint_mode="full_resume")
        finally:
            trainer.close()

        resumed_trainer = self._build_trainer()
        try:
            checkpoint_mode = resumed_trainer.load_checkpoint(checkpoint)
            self.assertEqual(checkpoint_mode, "full_resume")
            self.assertEqual(int(resumed_trainer.completed_updates), 1)
            self.assertEqual(int(resumed_trainer.global_env_steps), int(checkpoint["global_env_steps"]))
            self.assertEqual(len(resumed_trainer.history), 1)
        finally:
            resumed_trainer.close()
