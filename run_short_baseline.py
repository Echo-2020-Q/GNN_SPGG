from __future__ import annotations

"""
短基线 profiling 入口。

目标：
1. 复用 run_experiment.py 里的正式训练逻辑；
2. 只通过 spec override 跑一个隔离的短实验；
3. 把关键 profile 指标汇总成 baseline_summary.json，便于回传分析。

示例：
    python run_short_baseline.py
    python run_short_baseline.py --total-updates 8 --warmup-updates 1 --eval-interval-updates 4
    python run_short_baseline.py --device cuda:0 --rollout-device cpu --num-workers 4 --num-envs-per-worker 4
    python run_short_baseline.py --patch-json '{"training":{"batch_size":128}}'
"""

from copy import deepcopy
from datetime import datetime
from collections.abc import Iterable, Mapping
import argparse
import json
import math
from pathlib import Path
from statistics import mean
import sys
from time import perf_counter
from typing import Any, Dict, Optional


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_experiment import (  # noqa: E402
    BASE_EXPERIMENT,
    _configure_stdio,
    _resolve_effective_steps_per_update,
    _resolve_training_schedule,
    build_output_dir,
    deep_update,
    run_one_experiment,
)


PROFILE_KEYS = [
    "profile_rollout_collect_seconds",
    "profile_learner_update_seconds",
    "profile_actor_sync_seconds",
    "profile_rollout_env_step_seconds",
    "profile_rollout_transition_encode_seconds",
    "profile_replay_extend_seconds",
    "profile_replay_sample_seconds",
    "profile_batch_to_device_seconds",
    "profile_eval_seconds",
    "profile_on_update_seconds",
    "profile_rollout_action_to_numpy_seconds",
    "profile_rollout_stack_transitions_seconds",
    "profile_rollout_shared_memory_serialize_seconds",
    "profile_rollout_shared_memory_deserialize_seconds",
    "profile_rollout_collect_worker_seconds",
    "profile_rollout_inference_wait_seconds",
    "profile_rollout_inference_request_build_seconds",
    "profile_rollout_local_policy_forward_seconds",
    "profile_rollout_finish_wait_seconds",
    "profile_rollout_overlap_seconds",
    "profile_rollout_inference_batch_size_mean",
    "profile_rollout_inference_batch_size_max",
    "profile_critic_update_seconds",
    "profile_actor_update_seconds",
    "profile_target_soft_update_seconds",
    "profile_rollout_steps_per_second",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a short isolated profiling baseline.")
    parser.add_argument("--experiment-name", type=str, default=None, help="Explicit experiment name.")
    parser.add_argument("--tag", type=str, default=None, help="Extra suffix for the experiment name.")
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs/perf_baseline",
        help="Isolated output root for the short baseline run.",
    )
    parser.add_argument("--total-updates", type=int, default=4, help="Total training updates to run.")
    parser.add_argument(
        "--warmup-updates",
        type=int,
        default=1,
        help="Warm-up expressed in update units. Converted to global env steps internally.",
    )
    parser.add_argument(
        "--eval-interval-updates",
        type=int,
        default=4,
        help="Periodic evaluation interval expressed in update units.",
    )
    parser.add_argument("--eval-episodes", type=int, default=2, help="Episodes per periodic evaluation.")
    parser.add_argument(
        "--post-train-eval-episodes",
        type=int,
        default=0,
        help="Episodes for post-training evaluation. Default 0 to keep the baseline short.",
    )
    parser.add_argument("--num-workers", type=int, default=None, help="Override training.num_workers.")
    parser.add_argument(
        "--num-envs-per-worker",
        type=int,
        default=None,
        help="Override training.num_envs_per_worker.",
    )
    parser.add_argument(
        "--steps-per-update",
        type=int,
        default=None,
        help="Override training.steps_per_update.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Override training.batch_size.")
    parser.add_argument(
        "--graph-batch-chunk-size",
        type=int,
        default=None,
        help="Override training.graph_batch_chunk_size.",
    )
    parser.add_argument("--gradient-steps-per-update", type=int, default=None, help="Override training.gradient_steps_per_update.")
    parser.add_argument("--device", type=str, default=None, help="Override learner training device.")
    parser.add_argument("--rollout-device", type=str, default=None, help="Override rollout inference device.")
    parser.add_argument(
        "--rollout-inference-mode",
        type=str,
        choices=("local", "centralized"),
        default=None,
        help="Override rollout inference mode.",
    )
    parser.add_argument(
        "--disable-overlap",
        action="store_true",
        help="Disable overlap_rollout_and_update for A/B profiling.",
    )
    parser.add_argument(
        "--disable-domain-randomization",
        action="store_true",
        help="Disable domain randomization for a cleaner pipeline-only baseline.",
    )
    parser.add_argument(
        "--disable-curriculum",
        action="store_true",
        help="Disable curriculum for a cleaner pipeline-only baseline.",
    )
    parser.add_argument(
        "--disable-custom-eval-families",
        action="store_true",
        help="Use only the base eval env instead of the configured evaluation env families.",
    )
    parser.add_argument(
        "--enable-tensorboard",
        action="store_true",
        help="Keep TensorBoard logging enabled. Disabled by default for a lighter baseline.",
    )
    parser.add_argument(
        "--save-console-log",
        action="store_true",
        help="Save stdout/stderr to train.log under the isolated output directory.",
    )
    parser.add_argument(
        "--console-log-interval",
        type=int,
        default=None,
        help="Override tensorboard.console_log_interval.",
    )
    parser.add_argument(
        "--console-progress-interval",
        type=int,
        default=None,
        help="Override tensorboard.console_progress_interval.",
    )
    parser.add_argument(
        "--recent-window-updates",
        type=int,
        default=5,
        help="Window size used when summarizing the last few updates.",
    )
    parser.add_argument(
        "--patch-json",
        type=str,
        default=None,
        help="Extra JSON overrides applied last.",
    )
    parser.add_argument(
        "--patch-file",
        type=str,
        default=None,
        help="Path to a JSON file containing extra overrides applied last.",
    )
    return parser.parse_args()


def _load_json_patch(path_text: Optional[str], inline_json: Optional[str]) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}
    if path_text:
        patch_path = Path(path_text).expanduser()
        if not patch_path.is_absolute():
            patch_path = Path.cwd() / patch_path
        patch = json.loads(patch_path.read_text(encoding="utf-8"))
    if inline_json:
        patch = deep_update(patch, json.loads(inline_json))
    return patch


def _experiment_name(args: argparse.Namespace) -> str:
    if args.experiment_name:
        return args.experiment_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parts = [str(BASE_EXPERIMENT["experiment_name"]), "baseline", timestamp]
    if args.tag:
        parts.append(str(args.tag))
    return "__".join(parts)


def _finite_values(history: Iterable[Mapping[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in history:
        value = record.get(key)
        if value is None:
            continue
        number = float(value)
        if math.isfinite(number):
            values.append(number)
    return values


def _metric_stats(history: list[Mapping[str, Any]], key: str, recent_window: int) -> Optional[Dict[str, float]]:
    values = _finite_values(history, key)
    if not values:
        return None
    recent_count = max(1, min(recent_window, len(values)))
    recent_values = values[-recent_count:]
    return {
        "count": float(len(values)),
        "last": float(values[-1]),
        "mean": float(mean(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "recent_mean": float(mean(recent_values)),
        "recent_count": float(recent_count),
    }


def _safe_mean(metric_stats: Mapping[str, Mapping[str, float]], key: str) -> Optional[float]:
    entry = metric_stats.get(key)
    if entry is None:
        return None
    return float(entry["mean"])


def _safe_ratio(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None:
        return None
    if abs(denominator) <= 1e-12:
        return None
    return float(numerator / denominator)


def _build_summary(
    spec: Mapping[str, Any],
    results: Mapping[str, Any],
    output_dir: Path,
    recent_window: int,
    wall_seconds: float,
) -> Dict[str, Any]:
    history = list(results.get("history", []))
    metric_stats: Dict[str, Dict[str, float]] = {}
    for key in PROFILE_KEYS:
        stats = _metric_stats(history, key, recent_window=recent_window)
        if stats is not None:
            metric_stats[key] = stats

    rollout_collect_mean = _safe_mean(metric_stats, "profile_rollout_collect_seconds")
    learner_update_mean = _safe_mean(metric_stats, "profile_learner_update_seconds")
    env_step_mean = _safe_mean(metric_stats, "profile_rollout_env_step_seconds")
    transition_encode_mean = _safe_mean(metric_stats, "profile_rollout_transition_encode_seconds")
    replay_extend_mean = _safe_mean(metric_stats, "profile_replay_extend_seconds")
    replay_sample_mean = _safe_mean(metric_stats, "profile_replay_sample_seconds")
    batch_to_device_mean = _safe_mean(metric_stats, "profile_batch_to_device_seconds")
    eval_mean = _safe_mean(metric_stats, "profile_eval_seconds")
    callback_mean = _safe_mean(metric_stats, "profile_on_update_seconds")

    training = spec["training"]
    derived = {
        "rollout_collect_over_learner_update": _safe_ratio(rollout_collect_mean, learner_update_mean),
        "env_step_share_of_rollout_collect": _safe_ratio(env_step_mean, rollout_collect_mean),
        "transition_encode_share_of_rollout_collect": _safe_ratio(transition_encode_mean, rollout_collect_mean),
        "replay_extend_share_of_rollout_collect": _safe_ratio(replay_extend_mean, rollout_collect_mean),
        "replay_sample_share_of_learner_update": _safe_ratio(replay_sample_mean, learner_update_mean),
        "batch_to_device_share_of_learner_update": _safe_ratio(batch_to_device_mean, learner_update_mean),
        "eval_share_of_on_update": _safe_ratio(eval_mean, callback_mean),
    }

    schedule = _resolve_training_schedule(spec)
    final_metrics = dict(results.get("final_metrics", {}))
    config_summary = {
        "experiment_name": spec["experiment_name"],
        "output_dir": str(output_dir),
        "output_root": str(spec["output"]["root_dir"]),
        "total_updates": int(schedule["total_updates"]),
        "global_env_steps_per_update": int(schedule["global_env_steps_per_update"]),
        "total_env_steps_effective": int(schedule["total_env_steps_effective"]),
        "warmup_env_steps": int(schedule["warmup_env_steps"]),
        "eval_interval_updates": int(schedule["eval_interval_updates"]),
        "num_workers": int(training["num_workers"]),
        "num_envs_per_worker": int(training["num_envs_per_worker"]),
        "steps_per_update": int(training["steps_per_update"]),
        "gradient_steps_per_update": int(training["gradient_steps_per_update"]),
        "batch_size": int(training["batch_size"]),
        "graph_batch_chunk_size": int(training["graph_batch_chunk_size"]),
        "learner_device": str(training["device"]),
        "rollout_device": training["rollout_device"],
        "rollout_inference_mode": str(training["rollout_inference_mode"]),
        "overlap_rollout_and_update": bool(training["overlap_rollout_and_update"]),
        "demo_pretrain_enabled": bool(training.get("demo_pretrain_enabled", False)),
        "demo_collection_env_steps": int(training.get("demo_collection_env_steps", 0)),
        "actor_bc_pretrain_updates": int(training.get("actor_bc_pretrain_updates", 0)),
        "critic_pretrain_updates": int(training.get("critic_pretrain_updates", 0)),
        "teacher_takeover_enabled": bool(training.get("teacher_takeover_enabled", False)),
        "replay_strategy": str(training.get("replay_strategy", "fifo")),
        "domain_randomization_enabled": bool(spec["domain_randomization"]["enabled"]),
        "curriculum_enabled": bool(spec["curriculum"]["enabled"]),
        "custom_eval_families_enabled": bool(spec["evaluation"]["use_custom_env_families"]),
        "eval_episodes": int(training["eval_episodes"]),
        "post_training_eval_episodes": int(spec["rollout"]["post_training_eval_episodes"]),
    }

    return {
        "config": config_summary,
        "wall_seconds": float(wall_seconds),
        "history_length": int(len(history)),
        "stopped_after_demo_pretrain": bool(results.get("stopped_after_demo_pretrain", False)),
        "demo_pretrain_summary": (
            dict(results["demo_pretrain_summary"])
            if isinstance(results.get("demo_pretrain_summary"), Mapping)
            else None
        ),
        "demo_pretrain_eval_summary": (
            dict(results["demo_pretrain_eval_summary"])
            if isinstance(results.get("demo_pretrain_eval_summary"), Mapping)
            else None
        ),
        "selected_profile_metrics": metric_stats,
        "derived_ratios": derived,
        "final_metrics_subset": {
            key: float(final_metrics[key])
            for key in [
                "update",
                "global_env_steps",
                "rollout_reward_mean",
                "critic_loss",
                "actor_loss",
                "eval_return_mean",
                "eval_cooperation_mean",
            ]
            if key in final_metrics and final_metrics[key] is not None
        },
    }


def _build_spec(args: argparse.Namespace) -> Dict[str, Any]:
    spec = deepcopy(BASE_EXPERIMENT)
    training = spec["training"]

    if args.num_workers is not None:
        training["num_workers"] = int(args.num_workers)
    if args.num_envs_per_worker is not None:
        training["num_envs_per_worker"] = int(args.num_envs_per_worker)
    if args.steps_per_update is not None:
        training["steps_per_update"] = int(args.steps_per_update)
    if args.batch_size is not None:
        training["batch_size"] = int(args.batch_size)
    if args.graph_batch_chunk_size is not None:
        training["graph_batch_chunk_size"] = int(args.graph_batch_chunk_size)
    if args.gradient_steps_per_update is not None:
        training["gradient_steps_per_update"] = int(args.gradient_steps_per_update)
    if args.device is not None:
        training["device"] = str(args.device)
    if args.rollout_device is not None:
        training["rollout_device"] = str(args.rollout_device)
    if args.rollout_inference_mode is not None:
        training["rollout_inference_mode"] = str(args.rollout_inference_mode)
    if args.disable_overlap:
        training["overlap_rollout_and_update"] = False

    effective_steps_per_update = _resolve_effective_steps_per_update(spec)
    global_env_steps_per_update = int(training["num_workers"]) * int(effective_steps_per_update)
    warmup_env_steps = max(0, int(args.warmup_updates)) * global_env_steps_per_update

    deep_update(
        spec,
        {
            "experiment_name": _experiment_name(args),
            "training": {
                "total_env_steps": None,
                "total_updates": int(args.total_updates),
                "warmup_env_steps": int(warmup_env_steps),
                "eval_interval_env_steps": None,
                "eval_interval": int(args.eval_interval_updates),
                "eval_episodes": int(args.eval_episodes),
                "save_checkpoints": False,
                "checkpoint_interval": 0,
                "save_final_checkpoint": False,
                "save_best_checkpoint": False,
                "resume_from_checkpoint": None,
            },
            "rollout": {
                "post_training_eval_episodes": int(args.post_train_eval_episodes),
            },
            "visualization": {
                "enable_micro_snapshots": False,
                "enable_macro_timeseries": False,
            },
            "tensorboard": {
                "enabled": bool(args.enable_tensorboard),
            },
            "output": {
                "root_dir": args.output_root,
                "save_micro_snapshots": False,
                "save_macro_timeseries": False,
                "save_console_log": bool(args.save_console_log),
                "save_results_json": True,
            },
        },
    )

    if args.disable_domain_randomization:
        spec["domain_randomization"]["enabled"] = False
    if args.disable_curriculum:
        spec["curriculum"]["enabled"] = False
    if args.disable_custom_eval_families:
        spec["evaluation"]["use_custom_env_families"] = False
    if args.console_log_interval is not None:
        spec["tensorboard"]["console_log_interval"] = int(args.console_log_interval)
    if args.console_progress_interval is not None:
        spec["tensorboard"]["console_progress_interval"] = int(args.console_progress_interval)

    patch = _load_json_patch(args.patch_file, args.patch_json)
    if patch:
        deep_update(spec, patch)

    patch_training = patch.get("training", {}) if isinstance(patch.get("training"), Mapping) else {}
    if "warmup_env_steps" not in patch_training and "warmup_steps" not in patch_training:
        effective_steps_per_update = _resolve_effective_steps_per_update(spec)
        global_env_steps_per_update = int(spec["training"]["num_workers"]) * int(effective_steps_per_update)
        spec["training"]["warmup_env_steps"] = max(0, int(args.warmup_updates)) * global_env_steps_per_update

    spec["tensorboard"]["console_recent_window_updates"] = max(1, int(args.recent_window_updates))
    return spec


def main() -> None:
    _configure_stdio()
    args = _parse_args()
    spec = _build_spec(args)
    output_dir = build_output_dir(spec)

    print("Short baseline config:")
    print(json.dumps(
        {
            "experiment_name": spec["experiment_name"],
            "output_dir": str(output_dir),
            "output_root": spec["output"]["root_dir"],
            "total_updates": spec["training"]["total_updates"],
            "warmup_env_steps": spec["training"]["warmup_env_steps"],
            "eval_interval": spec["training"]["eval_interval"],
            "num_workers": spec["training"]["num_workers"],
            "num_envs_per_worker": spec["training"]["num_envs_per_worker"],
            "steps_per_update": spec["training"]["steps_per_update"],
            "batch_size": spec["training"]["batch_size"],
            "device": spec["training"]["device"],
            "rollout_device": spec["training"]["rollout_device"],
            "overlap_rollout_and_update": spec["training"]["overlap_rollout_and_update"],
        },
        ensure_ascii=False,
        indent=2,
    ))

    spec_path = output_dir / "baseline_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Baseline spec saved to: {0}".format(spec_path))

    run_start = perf_counter()
    results = run_one_experiment(spec)
    wall_seconds = float(perf_counter() - run_start)
    summary = _build_summary(
        spec=spec,
        results=results,
        output_dir=output_dir,
        recent_window=max(1, int(args.recent_window_updates)),
        wall_seconds=wall_seconds,
    )
    summary_path = output_dir / "baseline_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Baseline summary saved to: {0}".format(summary_path))
    print("Baseline summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
