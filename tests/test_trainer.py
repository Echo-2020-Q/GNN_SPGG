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
