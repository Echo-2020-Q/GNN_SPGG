from __future__ import annotations

import importlib.util
import unittest
from copy import deepcopy


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None

if NUMPY_AVAILABLE:
    from run_experiment import (
        BASE_EXPERIMENT,
        PPO_BASELINE_EXPERIMENT,
        _log_tensorboard_custom_layout,
        _log_tensorboard_update_metrics,
        _tensorboard_tag_for_metric,
        build_trainer_config,
    )


class _FakeSummaryWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.texts: list[tuple[str, str, int]] = []
        self.custom_scalar_layout: dict[str, object] | None = None

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, float(value), int(step)))

    def add_text(self, tag: str, text: str, step: int) -> None:
        self.texts.append((tag, str(text), int(step)))

    def add_custom_scalars(self, layout: dict[str, object]) -> None:
        self.custom_scalar_layout = layout


@unittest.skipUnless(NUMPY_AVAILABLE, "numpy is required for run_experiment tensorboard tests")
class RunExperimentTensorboardTests(unittest.TestCase):
    def test_build_trainer_config_preserves_rollout_device_settings(self) -> None:
        spec = deepcopy(BASE_EXPERIMENT)
        spec["training"]["rollout_device"] = ["cuda:1", "cuda:2"]
        spec["training"]["rollout_inference_mode"] = "centralized"
        spec["training"]["rollout_inference_batch_timeout_ms"] = 3.5
        spec["training"]["rollout_num_threads"] = 1
        spec["training"]["num_envs_per_worker"] = 4
        spec["training"]["overlap_rollout_and_update"] = False

        trainer_config = build_trainer_config(spec)

        self.assertEqual(trainer_config.rollout_device, ("cuda:1", "cuda:2"))
        self.assertEqual(trainer_config.rollout_inference_mode, "centralized")
        self.assertEqual(trainer_config.rollout_inference_batch_timeout_ms, 3.5)
        self.assertEqual(trainer_config.rollout_num_threads, 1)
        self.assertEqual(trainer_config.num_envs_per_worker, 4)
        self.assertFalse(trainer_config.overlap_rollout_and_update)

    def test_build_trainer_config_switches_to_ppo_when_requested(self) -> None:
        spec = deepcopy(PPO_BASELINE_EXPERIMENT)
        spec["training"]["ppo_rollout_horizon"] = 64

        trainer_config = build_trainer_config(spec)

        self.assertEqual(type(trainer_config).__name__, "GraphPPOConfig")
        self.assertEqual(trainer_config.steps_per_update, 64)
        self.assertEqual(trainer_config.num_workers, 1)
        self.assertEqual(trainer_config.num_envs_per_worker, 1)

    def test_update_metrics_use_global_env_steps_as_tensorboard_step(self) -> None:
        writer = _FakeSummaryWriter()
        stage_log_state = {"last_stage_index": None}
        metrics = {
            "update": 5.0,
            "global_env_steps": 600.0,
            "loss": 1.25,
            "eval_return_mean": 3.5,
            "curriculum_stage": 1.0,
        }

        _log_tensorboard_update_metrics(
            writer,
            metrics,
            curriculum_stages=[{"label": "stage0"}, {"label": "stage1"}],
            stage_log_state=stage_log_state,
        )

        scalar_steps = {tag: step for tag, _, step in writer.scalars}
        self.assertEqual(scalar_steps["loss/loss"], 600)
        self.assertEqual(scalar_steps["eval/return_mean"], 600)
        self.assertEqual(scalar_steps["curriculum/stage_index"], 600)
        self.assertEqual(writer.texts, [("curriculum/active_stage_label", "stage1", 600)])

    def test_update_metrics_fall_back_to_update_when_global_env_steps_missing(self) -> None:
        writer = _FakeSummaryWriter()
        metrics = {
            "update": 7.0,
            "loss": 2.0,
            "behavior_frac_actor_logits": 1.0,
        }

        _log_tensorboard_update_metrics(writer, metrics)

        scalar_steps = {tag: step for tag, _, step in writer.scalars}
        self.assertEqual(scalar_steps["loss/loss"], 7)
        self.assertEqual(scalar_steps["behavior/actor_logits"], 7)

    def test_profile_metrics_map_to_profile_tensorboard_namespace(self) -> None:
        self.assertEqual(
            _tensorboard_tag_for_metric("profile_rollout_collect_seconds"),
            "profile/rollout_collect_seconds",
        )
        self.assertEqual(
            _tensorboard_tag_for_metric("profile_rollout_overlap_seconds"),
            "profile/rollout_overlap_seconds",
        )

    def test_grad_clip_metrics_map_to_grad_tensorboard_namespace(self) -> None:
        self.assertEqual(
            _tensorboard_tag_for_metric("actor_grad_norm_pre_clip"),
            "grad/actor_grad_norm_pre_clip",
        )
        self.assertEqual(
            _tensorboard_tag_for_metric("critic_grad_norm_post_clip"),
            "grad/critic_grad_norm_post_clip",
        )

    def test_custom_layout_adds_eval_only_fc_panel(self) -> None:
        writer = _FakeSummaryWriter()

        _log_tensorboard_custom_layout(writer)

        self.assertIsNotNone(writer.custom_scalar_layout)
        layout = writer.custom_scalar_layout or {}
        self.assertIn("Eval Only", layout)
        eval_only = layout["Eval Only"]
        self.assertIn("f_c", eval_only)
        self.assertEqual(
            eval_only["f_c"],
            [
                "Multiline",
                [
                    "eval/f_c",
                    "eval/f_c/regular",
                    "eval/f_c/erdos_renyi",
                    "eval/f_c/small_world",
                    "eval/f_c/scale_free",
                ],
            ],
        )
