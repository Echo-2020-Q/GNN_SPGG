from __future__ import annotations

import copy
import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import torch

    from Project1.env import RewardConfig, SPGGConfig, SPGGEnv, make_grid_graph
    from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig
    from Project1.policies.rule_based import UniformAllocationPolicy
    from Project1.td3.config import GraphTD3Config
    from Project1.td3.critic import GraphActionCritic, GraphActionCriticConfig, TwinCritic
    from Project1.td3.data import TensorTransition
    from Project1.td3.exploration import LogitSpaceExplorer
    from Project1.td3.learner import GraphTD3Learner
    from Project1.td3.replay import ReplayBuffer


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for learner tests")
class GraphTD3LearnerPretrainTests(unittest.TestCase):
    def _make_env(self) -> SPGGEnv:
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

    def _make_learner(self) -> GraphTD3Learner:
        env = self._make_env()
        replay = ReplayBuffer(capacity=32, seed=0)
        uniform_policy = UniformAllocationPolicy()
        explorer = LogitSpaceExplorer()

        for seed in range(4):
            obs = env.reset(seed=seed)
            allocation = uniform_policy.allocate(obs)
            action = explorer.action_from_allocation(
                allocation=allocation,
                ego_mask=obs["local_mask"],
                pool_values=obs["pool_grown"],
                noise_std=0.0,
                noise_clip=0.0,
                device="cpu",
            )
            next_obs, reward, done, info = env.step(allocation)
            replay.add(
                TensorTransition.from_step(
                    obs=obs,
                    action=action,
                    reward=float(reward),
                    next_obs=next_obs,
                    done=bool(done),
                    is_demo=True,
                    collapse_flag=bool(info.get("actual_cooperation_rate", 0.0) < 0.1),
                    topology_name="regular",
                    pool_power_demo_flag=True,
                    demo_return_target=float(reward) + 1.0,
                    demo_return_valid=True,
                )
            )

        actor = GNNAllocationPolicy(GNNPolicyConfig(hidden_dim=16, num_message_passing_layers=2))
        target_actor = copy.deepcopy(actor)
        critic_config = GraphActionCriticConfig(state_hidden_dim=16)
        critics = TwinCritic(
            GraphActionCritic(critic_config),
            GraphActionCritic(critic_config),
        )
        target_critics = TwinCritic(
            GraphActionCritic(critic_config),
            GraphActionCritic(critic_config),
        )
        target_critics.load_state_dict(critics.state_dict())

        config = GraphTD3Config(
            total_updates=1,
            steps_per_update=1,
            eval_interval=1,
            eval_episodes=1,
            batch_size=2,
            demo_pretrain_batch_size=2,
            learning_rate=1e-3,
            actor_lr=1e-3,
            critic_lr=1e-3,
            warmup_actor_bc_coef=1.0,
            warmup_steps=1,
            critic_loss_type="huber",
            device="cpu",
        )
        return GraphTD3Learner(
            actor=actor,
            critics=critics,
            target_actor=target_actor,
            target_critics=target_critics,
            replay_buffer=replay,
            target_explorer=LogitSpaceExplorer(),
            config=config,
        )

    def test_actor_bc_pretrain_step_does_not_update_critics(self) -> None:
        learner = self._make_learner()
        actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in learner.critics.parameters()]

        metrics = learner.actor_bc_pretrain_step()

        actor_after = [parameter.detach().clone() for parameter in learner.actor.parameters()]
        critic_after = [parameter.detach().clone() for parameter in learner.critics.parameters()]
        self.assertGreater(float(metrics["actor_bc_loss"]), 0.0)
        self.assertIn("actor_grad_norm_pre_clip", metrics)
        self.assertIn("actor_grad_norm_post_clip", metrics)
        self.assertGreaterEqual(float(metrics["actor_grad_norm_pre_clip"]), float(metrics["actor_grad_norm_post_clip"]))
        self.assertTrue(any(not torch.allclose(before, after) for before, after in zip(actor_before, actor_after)))
        self.assertTrue(all(torch.allclose(before, after) for before, after in zip(critic_before, critic_after)))

    def test_critic_pretrain_step_does_not_update_actor(self) -> None:
        learner = self._make_learner()
        learner.actor_bc_pretrain_step()
        actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in learner.critics.parameters()]

        metrics = learner.critic_pretrain_step()

        actor_after = [parameter.detach().clone() for parameter in learner.actor.parameters()]
        critic_after = [parameter.detach().clone() for parameter in learner.critics.parameters()]
        self.assertGreaterEqual(float(metrics["critic_loss"]), 0.0)
        self.assertTrue(all(torch.allclose(before, after) for before, after in zip(actor_before, actor_after)))
        self.assertTrue(any(not torch.allclose(before, after) for before, after in zip(critic_before, critic_after)))

    def test_critic_pretrain_step_ignores_target_actor_bootstrap(self) -> None:
        learner = self._make_learner()
        for parameter in learner.target_actor.parameters():
            parameter.data.fill_(float("nan"))
        for parameter in learner.target_critics.parameters():
            parameter.data.fill_(float("nan"))

        metrics = learner.critic_pretrain_step()

        self.assertTrue(torch.isfinite(torch.tensor(float(metrics["critic_loss"]))))

    def test_demo_validation_metrics_are_finite(self) -> None:
        learner = self._make_learner()
        validation_batch = learner.replay_buffer.export_demo_batch()
        assert validation_batch is not None

        actor_metrics = learner.evaluate_actor_bc_on_demo_batch(validation_batch, batch_size=2)
        critic_metrics = learner.evaluate_critic_on_demo_return_batch(validation_batch, batch_size=2)

        self.assertGreater(float(actor_metrics["actor_bc_val_num_entries"]), 0.0)
        self.assertGreater(float(critic_metrics["critic_val_num_targets"]), 0.0)
        for metrics in (actor_metrics, critic_metrics):
            for value in metrics.values():
                self.assertTrue(torch.isfinite(torch.tensor(float(value))))

    def test_critic_td_validation_metrics_are_finite(self) -> None:
        learner = self._make_learner()
        validation_batch = learner.replay_buffer.export_demo_batch()
        assert validation_batch is not None

        critic_metrics = learner.evaluate_critic_on_td_batch(validation_batch, batch_size=2)

        self.assertGreater(float(critic_metrics["critic_val_num_targets"]), 0.0)
        for value in critic_metrics.values():
            self.assertTrue(torch.isfinite(torch.tensor(float(value))))

    def test_critic_bridge_step_updates_only_critics(self) -> None:
        learner = self._make_learner()
        actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in learner.critics.parameters()]

        metrics = learner.critic_bridge_step(learner.replay_buffer, batch_size=2)

        actor_after = [parameter.detach().clone() for parameter in learner.actor.parameters()]
        critic_after = [parameter.detach().clone() for parameter in learner.critics.parameters()]
        self.assertGreaterEqual(float(metrics["critic_loss"]), 0.0)
        self.assertTrue(all(torch.allclose(before, after) for before, after in zip(actor_before, actor_after)))
        self.assertTrue(any(not torch.allclose(before, after) for before, after in zip(critic_before, critic_after)))

    def test_critic_bridge_step_can_use_teacher_return_aux(self) -> None:
        learner = self._make_learner()
        learner.config.critic_bridge_teacher_return_aux_coef = 0.5

        metrics = learner.critic_bridge_step(learner.replay_buffer, batch_size=2)

        self.assertGreater(float(metrics["critic_bridge_teacher_aux_coef"]), 0.0)
        self.assertGreaterEqual(float(metrics["critic_bridge_teacher_aux_loss"]), 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(float(metrics["critic_loss"]))))

    def test_critic_bridge_step_honors_teacher_aux_override(self) -> None:
        learner = self._make_learner()
        learner.config.critic_bridge_teacher_return_aux_coef = 0.5

        metrics = learner.critic_bridge_step(
            learner.replay_buffer,
            batch_size=2,
            teacher_aux_coef_override=0.0,
        )

        self.assertEqual(float(metrics["critic_bridge_teacher_aux_coef"]), 0.0)
        self.assertEqual(float(metrics["critic_bridge_teacher_aux_loss"]), 0.0)

    def test_train_step_reports_actor_q_coef_schedule(self) -> None:
        learner = self._make_learner()
        learner.config.warmup_steps = 0
        learner.config.policy_delay = 1
        learner.config.total_updates = 10
        learner.config.steps_per_update = 1
        learner.config.num_workers = 1
        learner.config.online_actor_q_coef_initial = 0.2
        learner.config.online_actor_q_coef_final = 1.0
        learner.config.online_actor_q_coef_ramp_end_fraction = 0.5

        early_metrics = learner.train_step(global_env_steps=0)
        late_metrics = learner.train_step(global_env_steps=10)

        self.assertAlmostEqual(float(early_metrics["actor_q_coef"]), 0.2, places=6)
        self.assertAlmostEqual(float(late_metrics["actor_q_coef"]), 1.0, places=6)
        self.assertGreaterEqual(float(late_metrics["critic_grad_norm"]), 0.0)
        self.assertGreaterEqual(
            float(late_metrics["critic_grad_norm_pre_clip"]),
            float(late_metrics["critic_grad_norm_post_clip"]),
        )

    def test_freeze_actor_during_warmup_skips_actor_updates(self) -> None:
        learner = self._make_learner()
        learner.config.warmup_steps = 10
        learner.config.policy_delay = 1
        learner.config.freeze_actor_during_warmup = True

        actor_before = [parameter.detach().clone() for parameter in learner.actor.parameters()]
        critic_before = [parameter.detach().clone() for parameter in learner.critics.parameters()]

        metrics = learner.train_step(global_env_steps=0)

        actor_after = [parameter.detach().clone() for parameter in learner.actor.parameters()]
        critic_after = [parameter.detach().clone() for parameter in learner.critics.parameters()]
        self.assertEqual(float(metrics["actor_bc_coef"]), 0.0)
        self.assertEqual(float(metrics["actor_q_coef"]), 0.0)
        self.assertTrue(all(torch.allclose(before, after) for before, after in zip(actor_before, actor_after)))
        self.assertTrue(any(not torch.allclose(before, after) for before, after in zip(critic_before, critic_after)))

    def test_freeze_actor_q_until_teacher_release_uses_bc_only_before_release(self) -> None:
        learner = self._make_learner()
        learner.config.warmup_steps = 1
        learner.config.policy_delay = 1
        learner.config.total_updates = 100
        learner.config.steps_per_update = 1
        learner.config.batch_size = 4
        learner.config.adaptive_teacher_release_enabled = True
        learner.config.freeze_actor_q_until_teacher_release = True

        fixed_batch = learner.replay_buffer.export_demo_batch()
        assert fixed_batch is not None
        learner.replay_buffer.sample = lambda *args, **kwargs: fixed_batch  # type: ignore[method-assign]
        learner.replay_buffer.get_last_sample_stats = lambda: {}  # type: ignore[method-assign]

        locked_metrics = learner.train_step(global_env_steps=10, teacher_release_unlocked=False)
        unlocked_metrics = learner.train_step(global_env_steps=10, teacher_release_unlocked=True)

        self.assertGreater(float(locked_metrics["actor_bc_coef"]), 0.0)
        self.assertEqual(float(locked_metrics["actor_q_coef"]), 0.0)
        self.assertAlmostEqual(float(locked_metrics["actor_q_loss"]), 0.0, places=8)
        self.assertGreater(float(unlocked_metrics["actor_q_coef"]), 0.0)

    def test_release_relative_actor_schedules_restart_at_unlock(self) -> None:
        learner = self._make_learner()
        learner.config.warmup_steps = 10
        learner.config.policy_delay = 1
        learner.config.total_updates = 100
        learner.config.steps_per_update = 1
        learner.config.num_workers = 1
        learner.config.batch_size = 4
        learner.config.adaptive_teacher_release_enabled = True
        learner.config.freeze_actor_q_until_teacher_release = True
        learner.config.actor_demo_bc_coef = 0.8
        learner.config.actor_demo_bc_decay_end_fraction = 0.50
        learner.config.actor_demo_bc_decay_from_teacher_release = True
        learner.config.online_actor_q_coef_initial = 0.2
        learner.config.online_actor_q_coef_final = 1.0
        learner.config.online_actor_q_coef_ramp_end_fraction = 0.50
        learner.config.online_actor_q_ramp_from_teacher_release = True

        fixed_batch = learner.replay_buffer.export_demo_batch()
        assert fixed_batch is not None
        learner.replay_buffer.sample = lambda *args, **kwargs: fixed_batch  # type: ignore[method-assign]
        learner.replay_buffer.get_last_sample_stats = lambda: {}  # type: ignore[method-assign]

        just_released = learner.train_step(
            global_env_steps=60,
            teacher_release_unlocked=True,
            teacher_release_env_step=60,
        )
        after_release = learner.train_step(
            global_env_steps=90,
            teacher_release_unlocked=True,
            teacher_release_env_step=60,
        )

        self.assertAlmostEqual(float(just_released["actor_q_coef"]), 0.2, places=6)
        self.assertAlmostEqual(float(just_released["actor_bc_coef"]), 0.8, places=6)
        self.assertGreater(float(after_release["actor_q_coef"]), float(just_released["actor_q_coef"]))
        self.assertLess(float(after_release["actor_bc_coef"]), float(just_released["actor_bc_coef"]))

    def test_freeze_actor_q_until_teacher_release_is_noop_without_adaptive_release(self) -> None:
        learner = self._make_learner()
        learner.config.warmup_steps = 1
        learner.config.policy_delay = 1
        learner.config.total_updates = 100
        learner.config.steps_per_update = 1
        learner.config.batch_size = 4
        learner.config.adaptive_teacher_release_enabled = False
        learner.config.freeze_actor_q_until_teacher_release = True

        fixed_batch = learner.replay_buffer.export_demo_batch()
        assert fixed_batch is not None
        learner.replay_buffer.sample = lambda *args, **kwargs: fixed_batch  # type: ignore[method-assign]
        learner.replay_buffer.get_last_sample_stats = lambda: {}  # type: ignore[method-assign]

        metrics = learner.train_step(global_env_steps=10, teacher_release_unlocked=False)

        self.assertGreater(float(metrics["actor_q_coef"]), 0.0)

    def test_q_filter_can_disable_demo_bc_loss(self) -> None:
        learner = self._make_learner()
        learner.config.warmup_steps = 1
        learner.config.policy_delay = 1
        learner.config.total_updates = 100
        learner.config.steps_per_update = 1
        learner.config.batch_size = 4
        learner.config.actor_bc_q_filter_enabled = True
        learner.config.actor_bc_q_filter_margin = 1e6
        learner.config.actor_bc_q_filter_require_teacher_release = False

        fixed_batch = learner.replay_buffer.export_demo_batch()
        assert fixed_batch is not None
        learner.replay_buffer.sample = lambda *args, **kwargs: fixed_batch  # type: ignore[method-assign]
        learner.replay_buffer.get_last_sample_stats = lambda: {}  # type: ignore[method-assign]

        metrics = learner.train_step(global_env_steps=10, teacher_release_unlocked=True)

        self.assertEqual(float(metrics["q_filter_enabled"]), 1.0)
        self.assertEqual(float(metrics["q_filter_pass_frac"]), 0.0)
        self.assertAlmostEqual(float(metrics["actor_bc_loss"]), 0.0, places=8)

    def test_q_filter_can_wait_for_teacher_release(self) -> None:
        learner = self._make_learner()
        learner.config.warmup_steps = 1
        learner.config.policy_delay = 1
        learner.config.total_updates = 100
        learner.config.steps_per_update = 1
        learner.config.batch_size = 4
        learner.config.actor_bc_q_filter_enabled = True
        learner.config.actor_bc_q_filter_margin = 1e6
        learner.config.actor_bc_q_filter_require_teacher_release = True
        learner.config.adaptive_teacher_release_enabled = True

        fixed_batch = learner.replay_buffer.export_demo_batch()
        assert fixed_batch is not None
        learner.replay_buffer.sample = lambda *args, **kwargs: fixed_batch  # type: ignore[method-assign]
        learner.replay_buffer.get_last_sample_stats = lambda: {}  # type: ignore[method-assign]

        locked_metrics = learner.train_step(global_env_steps=10, teacher_release_unlocked=False)
        unlocked_metrics = learner.train_step(global_env_steps=10, teacher_release_unlocked=True)

        self.assertEqual(float(locked_metrics["q_filter_enabled"]), 0.0)
        self.assertGreater(float(locked_metrics["actor_bc_loss"]), 0.0)
        self.assertEqual(float(unlocked_metrics["q_filter_enabled"]), 1.0)
        self.assertEqual(float(unlocked_metrics["q_filter_pass_frac"]), 0.0)
