from __future__ import annotations

import importlib.util
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import torch

    from run_experiment import (
        BASE_EXPERIMENT,
        build_env_config,
        build_graph,
        build_output_dir,
        build_trainer_config,
        run_gnn_training_mode,
    )


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for pretrain checkpoint tests")
class RunExperimentDemoPretrainCheckpointTests(unittest.TestCase):
    def _build_spec(self, root_dir: str, experiment_name: str) -> dict:
        spec = deepcopy(BASE_EXPERIMENT)
        spec["experiment_name"] = experiment_name
        spec["seed"] = 0
        spec["run_mode"] = "gnn_train"
        spec["network"]["type"] = "regular"
        spec["network"]["num_nodes"] = 10
        spec["network"]["regular_degree"] = 2
        spec["dynamics"]["episode_length"] = 3
        spec["domain_randomization"]["enabled"] = False
        spec["curriculum"]["enabled"] = False
        spec["rollout"]["post_training_eval_episodes"] = 1
        spec["visualization"]["enable_micro_snapshots"] = False
        spec["visualization"]["enable_macro_timeseries"] = False
        spec["output"]["save_micro_snapshots"] = False
        spec["output"]["save_macro_timeseries"] = False
        spec["output"]["save_results_json"] = False
        spec["output"]["save_console_log"] = False
        spec["output"]["root_dir"] = root_dir
        spec["tensorboard"]["enabled"] = False
        spec["evaluation"]["use_best_checkpoint_for_post_training_eval"] = False

        training = spec["training"]
        training["device"] = "cpu"
        training["rollout_device"] = "cpu"
        training["rollout_inference_mode"] = "local"
        training["rollout_num_threads"] = 1
        training["num_workers"] = 1
        training["num_envs_per_worker"] = 1
        training["steps_per_update"] = 2
        training["total_env_steps"] = 2
        training["warmup_env_steps"] = 0
        training["eval_interval_env_steps"] = 2
        training["batch_size"] = 2
        training["graph_batch_chunk_size"] = 2
        training["replay_capacity"] = 128
        training["save_checkpoints"] = False
        training["save_final_checkpoint"] = False
        training["save_best_checkpoint"] = True
        training["resume_from_checkpoint"] = None
        training["demo_pretrain_enabled"] = True
        training["demo_collection_runtime"] = "isolated_cpu"
        training["demo_collection_env_steps"] = 4
        training["actor_bc_pretrain_updates"] = 1
        training["critic_pretrain_updates"] = 1
        training["demo_pretrain_batch_size"] = 2
        training["demo_pretrain_validation_episodes"] = 1
        training["critic_bridge_enabled"] = True
        training["critic_bridge_env_steps"] = 4
        training["critic_bridge_updates"] = 1
        training["critic_bridge_batch_size"] = 2
        training["critic_bridge_eval_interval"] = 1
        training["critic_bridge_patience"] = 1
        training["critic_bridge_teacher_return_aux_schedule"] = "fixed"
        training["critic_bridge_teacher_return_aux_coef"] = 0.0
        training["save_demo_pretrain_checkpoint"] = True
        training["demo_pretrain_checkpoint_name"] = "demo_pretrained.pt"
        training["stop_after_demo_pretrain"] = True
        return spec

    def test_base_experiment_uses_demo_regularized_actor_only_td3_defaults(self) -> None:
        config = build_trainer_config(BASE_EXPERIMENT)

        self.assertEqual(
            BASE_EXPERIMENT["experiment_name"],
            "0415_demo_regularized_graph_td3_regular_ba_actor_only_qoff_diagnostic",
        )
        self.assertTrue(bool(config.demo_pretrain_enabled))
        self.assertFalse(bool(config.teacher_takeover_enabled))
        self.assertFalse(bool(config.adaptive_teacher_release_enabled))
        self.assertFalse(bool(config.actor_bc_q_filter_enabled))
        self.assertEqual(config.replay_strategy, "topology_stratified_mixed")
        self.assertEqual(tuple(config.replay_topology_names), ("regular", "scale_free"))
        self.assertAlmostEqual(float(config.replay_demo_fraction), 0.50)
        self.assertAlmostEqual(float(config.replay_long_term_fraction), 0.35)
        self.assertAlmostEqual(float(config.replay_recent_fraction), 0.15)
        self.assertTrue(bool(config.demo_collection_use_domain_randomization))
        self.assertEqual(tuple(config.demo_collection_network_types), ("regular", "scale_free"))
        self.assertAlmostEqual(float(config.actor_demo_bc_coef), 1.0)
        self.assertAlmostEqual(float(config.actor_demo_bc_decay_end_fraction), 1.0)
        self.assertAlmostEqual(float(config.online_actor_q_coef_initial), 0.0)
        self.assertAlmostEqual(float(config.online_actor_q_coef_final), 0.0)
        self.assertAlmostEqual(float(config.online_actor_q_coef_ramp_end_fraction), 1.0)
        self.assertAlmostEqual(float(config.actor_entropy_coef), 5e-3)
        self.assertAlmostEqual(float(config.actor_logit_l2_coef), 1e-4)
        self.assertTrue(bool(config.critic_bridge_enabled))
        self.assertEqual(config.critic_bridge_behavior_mode, "actor_only")
        self.assertAlmostEqual(float(config.critic_bridge_teacher_takeover_prob), 0.0)
        self.assertAlmostEqual(float(config.critic_bridge_teacher_return_aux_coef), 0.0)
        self.assertEqual(config.critic_bridge_teacher_return_aux_schedule, "fixed")
        self.assertEqual(int(config.critic_bridge_updates), 1000)
        self.assertEqual(int(config.demo_pretrain_validation_episodes), 5)

    def test_demo_pretrain_checkpoint_save_and_stop(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._build_spec(tmpdir, "unit_demo_pretrain_stop")
            graph = build_graph(spec)
            env_config = build_env_config(spec, graph)
            output_dir = build_output_dir(spec)

            results = run_gnn_training_mode(spec, graph, env_config, output_dir)

            self.assertTrue(bool(results["stopped_after_demo_pretrain"]))
            checkpoint_path = Path(str(results["demo_pretrain_checkpoint_path"]))
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(results["post_training_eval_model_source"], "demo_pretrain_checkpoint")
            self.assertIsNotNone(results["demo_pretrain_eval_summary"])
            self.assertEqual(results["history"], [])

            try:
                checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                checkpoint_payload = torch.load(checkpoint_path, map_location="cpu")
            self.assertTrue(bool(checkpoint_payload["demo_pretrain_completed"]))
            self.assertTrue(bool(checkpoint_payload["is_demo_pretrain_checkpoint"]))
            self.assertIn("replay_buffer_state", checkpoint_payload)
            self.assertIn("demo_pretrain_eval_summary", checkpoint_payload)
            self.assertAlmostEqual(
                float(checkpoint_payload["best_eval_return_so_far"]),
                float(checkpoint_payload["demo_pretrain_eval_summary"]["return_mean"]),
                places=6,
            )

    def test_resume_from_demo_pretrain_checkpoint_skips_pretrain_and_trains_online(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pretrain_spec = self._build_spec(tmpdir, "unit_demo_pretrain_resume_seed")
            graph = build_graph(pretrain_spec)
            env_config = build_env_config(pretrain_spec, graph)
            output_dir = build_output_dir(pretrain_spec)
            pretrain_results = run_gnn_training_mode(pretrain_spec, graph, env_config, output_dir)
            checkpoint_path = Path(str(pretrain_results["demo_pretrain_checkpoint_path"]))
            self.assertTrue(checkpoint_path.exists())

            resume_spec = self._build_spec(tmpdir, "unit_demo_pretrain_resume_online")
            resume_spec["training"]["resume_from_checkpoint"] = str(checkpoint_path)
            resume_spec["training"]["stop_after_demo_pretrain"] = False
            resume_spec["training"]["save_demo_pretrain_checkpoint"] = False
            resume_spec["training"]["demo_collection_env_steps"] = 4
            resume_spec["rollout"]["post_training_eval_episodes"] = 1
            resume_graph = build_graph(resume_spec)
            resume_env_config = build_env_config(resume_spec, resume_graph)
            resume_output_dir = build_output_dir(resume_spec)

            resumed_results = run_gnn_training_mode(resume_spec, resume_graph, resume_env_config, resume_output_dir)

            self.assertFalse(bool(resumed_results["stopped_after_demo_pretrain"]))
            self.assertEqual(len(resumed_results["history"]), 1)
            self.assertEqual(str(resumed_results["demo_pretrain_checkpoint_path"]), str(checkpoint_path))
            self.assertIsNotNone(resumed_results["demo_pretrain_summary"])
            self.assertIsNotNone(resumed_results["demo_pretrain_eval_summary"])
            self.assertEqual(int(resumed_results["history"][0]["update"]), 1)
