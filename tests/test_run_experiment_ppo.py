from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    from run_experiment import (
        PPO_BASELINE_EXPERIMENT,
        build_env_config,
        build_graph,
        build_output_dir,
        run_gnn_training_mode,
    )


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for PPO run_experiment tests")
class RunExperimentPPOSmokeTests(unittest.TestCase):
    def _build_spec(self, root_dir: str, experiment_name: str) -> dict:
        spec = deepcopy(PPO_BASELINE_EXPERIMENT)
        spec["experiment_name"] = experiment_name
        spec["seed"] = 0
        spec["run_mode"] = "gnn_train"
        spec["network"]["type"] = "regular"
        spec["network"]["num_nodes"] = 10
        spec["network"]["regular_degree"] = 2
        spec["dynamics"]["episode_length"] = 3
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
        training["total_env_steps"] = 4
        training["ppo_rollout_horizon"] = 2
        training["eval_interval_env_steps"] = 2
        training["save_checkpoints"] = True
        training["checkpoint_interval"] = 1
        training["save_final_checkpoint"] = True
        training["save_best_checkpoint"] = True
        training["resume_from_checkpoint"] = None
        return spec

    def test_run_gnn_training_mode_with_ppo_smoke(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            spec = self._build_spec(tmpdir, "unit_ppo_smoke")
            graph = build_graph(spec)
            env_config = build_env_config(spec, graph)
            output_dir = build_output_dir(spec)

            results = run_gnn_training_mode(spec, graph, env_config, output_dir)

            self.assertFalse(bool(results["stopped_after_demo_pretrain"]))
            self.assertGreaterEqual(len(results["history"]), 1)
            self.assertIn("ppo_policy_loss", results["history"][-1])
            self.assertIn("eval_return_mean", results["history"][-1])
            self.assertEqual(results["post_training_eval_model_source"], "final_policy")
            self.assertEqual(len(results["post_training_evaluation"]), 1)
            self.assertTrue((output_dir / "checkpoints" / "latest.pt").exists())
            self.assertTrue((output_dir / "checkpoints" / "final.pt").exists())
            top_k_manifest_path = output_dir / "checkpoints" / "top_k_manifest.json"
            self.assertTrue(top_k_manifest_path.exists())
            top_k_manifest = json.loads(top_k_manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(top_k_manifest["metric"], "eval_return_mean")
            self.assertGreaterEqual(len(top_k_manifest["checkpoints"]), 1)
            self.assertLessEqual(len(top_k_manifest["checkpoints"]), spec["training"]["top_k_checkpoints"])
            self.assertEqual(len(results["top_k_checkpoints"]), len(top_k_manifest["checkpoints"]))
            self.assertTrue(Path(top_k_manifest["checkpoints"][0]["path"]).exists())

    def test_resume_from_ppo_checkpoint_continues_training(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            seed_spec = self._build_spec(tmpdir, "unit_ppo_resume_seed")
            graph = build_graph(seed_spec)
            env_config = build_env_config(seed_spec, graph)
            output_dir = build_output_dir(seed_spec)
            seed_results = run_gnn_training_mode(seed_spec, graph, env_config, output_dir)
            checkpoint_path = Path(output_dir / "checkpoints" / "latest.pt")
            self.assertTrue(checkpoint_path.exists())
            self.assertGreaterEqual(len(seed_results["history"]), 1)

            resume_spec = self._build_spec(tmpdir, "unit_ppo_resume_online")
            resume_spec["training"]["resume_from_checkpoint"] = str(checkpoint_path)
            resume_spec["training"]["total_env_steps"] = 8
            resume_graph = build_graph(resume_spec)
            resume_env_config = build_env_config(resume_spec, resume_graph)
            resume_output_dir = build_output_dir(resume_spec)

            resumed_results = run_gnn_training_mode(resume_spec, resume_graph, resume_env_config, resume_output_dir)

            self.assertGreaterEqual(len(resumed_results["history"]), 2)
            self.assertGreaterEqual(int(resumed_results["history"][-1]["update"]), 2)
            self.assertGreaterEqual(int(resumed_results["final_metrics"]["global_env_steps"]), 4)
