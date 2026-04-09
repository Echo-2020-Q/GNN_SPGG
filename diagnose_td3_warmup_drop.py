from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


NUMERIC_TYPES = (int, float, bool)


@dataclass
class RunData:
    source_kind: str
    source_path: Path
    history: list[dict[str, Any]]
    trainer_config: dict[str, Any]
    demo_pretrain_summary: dict[str, Any] | None
    experiment_spec: dict[str, Any] | None
    checkpoint_mode: str | None


def _load_torch_payload(path: Path) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on user env
        raise RuntimeError(
            "Loading .pt checkpoints requires PyTorch. Use a results.json input or install torch."
        ) from exc

    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("Checkpoint payload must be a dict.")
    return dict(payload)


def _load_results_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("results.json payload must be a dict.")
    return payload


def _resolve_input_path(raw_path: str) -> tuple[str, Path, Path | None]:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    path = path.resolve()

    if path.is_dir():
        checkpoint_candidates = [
            path / "checkpoints" / "latest.pt",
            path / "checkpoints" / "final.pt",
            path / "checkpoints" / "best_eval.pt",
            path / "checkpoints" / "demo_pretrained.pt",
        ]
        for candidate in checkpoint_candidates:
            if candidate.exists():
                results_path = path / "results.json"
                return "checkpoint", candidate, results_path if results_path.exists() else None
        results_path = path / "results.json"
        if results_path.exists():
            return "results", results_path, None
        raise FileNotFoundError(
            "No supported input found. Expected one of: checkpoints/latest.pt, final.pt, best_eval.pt, "
            "demo_pretrained.pt, or results.json inside {0}".format(path)
        )

    if not path.exists():
        raise FileNotFoundError("Input path does not exist: {0}".format(path))

    if path.suffix == ".json":
        return "results", path, None
    if path.suffix in {".pt", ".pth"}:
        results_path = path.parent.parent / "results.json"
        return "checkpoint", path, results_path if results_path.exists() else None
    raise ValueError("Unsupported input path: {0}".format(path))


def _normalize_history(history: Any) -> list[dict[str, Any]]:
    if history is None:
        return []
    if not isinstance(history, list):
        raise TypeError("history must be a list.")
    normalized: list[dict[str, Any]] = []
    for item in history:
        if not isinstance(item, Mapping):
            continue
        normalized.append(dict(item))
    normalized.sort(key=lambda item: (int(item.get("update", 0)), int(item.get("global_env_steps", 0))))
    return normalized


def _load_run_data(path_str: str) -> RunData:
    source_kind, source_path, sidecar_results_path = _resolve_input_path(path_str)

    experiment_spec: dict[str, Any] | None = None
    if source_kind == "checkpoint":
        checkpoint = _load_torch_payload(source_path)
        history = _normalize_history(checkpoint.get("history"))
        trainer_config_raw = checkpoint.get("trainer_config", {})
        if not isinstance(trainer_config_raw, Mapping):
            raise TypeError("trainer_config must be a mapping inside checkpoint.")
        demo_pretrain_summary_raw = checkpoint.get("demo_pretrain_summary")
        demo_pretrain_summary = (
            dict(demo_pretrain_summary_raw)
            if isinstance(demo_pretrain_summary_raw, Mapping)
            else None
        )
        if sidecar_results_path is not None and sidecar_results_path.exists():
            results_payload = _load_results_payload(sidecar_results_path)
            experiment_raw = results_payload.get("experiment")
            if isinstance(experiment_raw, Mapping):
                experiment_spec = dict(experiment_raw)
        return RunData(
            source_kind="checkpoint",
            source_path=source_path,
            history=history,
            trainer_config=dict(trainer_config_raw),
            demo_pretrain_summary=demo_pretrain_summary,
            experiment_spec=experiment_spec,
            checkpoint_mode=str(checkpoint.get("checkpoint_mode")) if "checkpoint_mode" in checkpoint else None,
        )

    results_payload = _load_results_payload(source_path)
    results_raw = results_payload.get("results", {})
    if not isinstance(results_raw, Mapping):
        raise TypeError("results.json must contain a mapping at payload['results'].")
    trainer_config_raw = results_raw.get("trainer_config", {})
    if not isinstance(trainer_config_raw, Mapping):
        raise TypeError("results.trainer_config must be a mapping.")
    demo_pretrain_summary_raw = results_raw.get("demo_pretrain_summary")
    demo_pretrain_summary = (
        dict(demo_pretrain_summary_raw)
        if isinstance(demo_pretrain_summary_raw, Mapping)
        else None
    )
    experiment_raw = results_payload.get("experiment")
    if isinstance(experiment_raw, Mapping):
        experiment_spec = dict(experiment_raw)
    return RunData(
        source_kind="results",
        source_path=source_path,
        history=_normalize_history(results_raw.get("history")),
        trainer_config=dict(trainer_config_raw),
        demo_pretrain_summary=demo_pretrain_summary,
        experiment_spec=experiment_spec,
        checkpoint_mode=None,
    )


def _mean_metric(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record and isinstance(record[key], NUMERIC_TYPES)]
    if not values:
        return None
    return float(statistics.fmean(values))


def _first_record_with_metric(records: Sequence[Mapping[str, Any]], key: str) -> dict[str, Any] | None:
    for record in records:
        if key in record and isinstance(record[key], NUMERIC_TYPES):
            return dict(record)
    return None


def _format_float(value: float | None, *, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    return "{0:.{1}f}".format(float(value), digits)


def _format_int(value: int | None) -> str:
    if value is None:
        return "n/a"
    return str(int(value))


def _delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def _record_by_update(history: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, Any]]:
    resolved: dict[int, dict[str, Any]] = {}
    for record in history:
        if "update" not in record or not isinstance(record["update"], NUMERIC_TYPES):
            continue
        resolved[int(record["update"])] = dict(record)
    return resolved


def _extract_metric_snapshot(record: Mapping[str, Any] | None, keys: Sequence[str]) -> dict[str, float]:
    if record is None:
        return {}
    snapshot: dict[str, float] = {}
    for key in keys:
        value = record.get(key)
        if isinstance(value, NUMERIC_TYPES):
            snapshot[key] = float(value)
    return snapshot


def _global_steps_per_update(config: Mapping[str, Any]) -> int:
    return max(1, int(config.get("steps_per_update", 1))) * max(1, int(config.get("num_workers", 1)))


def _estimate_actor_updates(warmup_rollout_updates: int, gradient_steps_per_update: int, policy_delay: int) -> int:
    learner_steps = max(0, int(warmup_rollout_updates)) * max(1, int(gradient_steps_per_update))
    if learner_steps <= 0:
        return 0
    delay = max(1, int(policy_delay))
    return 1 + int((learner_steps - 1) // delay)


def _window_records_by_update(
    history: Sequence[Mapping[str, Any]],
    *,
    start_update: int,
    end_update: int,
) -> list[dict[str, Any]]:
    return [
        dict(record)
        for record in history
        if "update" in record
        and isinstance(record["update"], NUMERIC_TYPES)
        and int(start_update) <= int(record["update"]) <= int(end_update)
    ]


def _leading_records_from_update(
    history: Sequence[Mapping[str, Any]],
    *,
    start_update: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for record in history:
        if "update" not in record or not isinstance(record["update"], NUMERIC_TYPES):
            continue
        if int(record["update"]) < int(start_update):
            continue
        selected.append(dict(record))
        if len(selected) >= int(limit):
            break
    return selected


def _trailing_records_to_update(
    history: Sequence[Mapping[str, Any]],
    *,
    end_update: int,
    limit: int,
) -> list[dict[str, Any]]:
    selected = _window_records_by_update(history, start_update=1, end_update=end_update)
    if len(selected) <= int(limit):
        return selected
    return selected[-int(limit):]


def _append_finding(findings: list[str], text: str) -> None:
    if text not in findings:
        findings.append(text)


def analyze_run_data(run_data: RunData, window_updates: int) -> dict[str, Any]:
    config = dict(run_data.trainer_config)
    history = list(run_data.history)

    warmup_steps = max(0, int(config.get("warmup_steps", 0)))
    global_steps_per_update = _global_steps_per_update(config)
    warmup_full_updates = warmup_steps // global_steps_per_update if warmup_steps > 0 else 0
    warmup_remainder = warmup_steps % global_steps_per_update if warmup_steps > 0 else 0
    mixed_transition_update = (warmup_full_updates + 1) if warmup_remainder > 0 else None
    last_pure_warmup_update = warmup_full_updates if warmup_full_updates > 0 else None
    if warmup_steps <= 0:
        first_pure_post_update = 1 if history else None
    else:
        first_pure_post_update = warmup_full_updates + (2 if warmup_remainder > 0 else 1)

    record_lookup = _record_by_update(history)
    pure_before_window = (
        _trailing_records_to_update(history, end_update=last_pure_warmup_update, limit=window_updates)
        if last_pure_warmup_update is not None and last_pure_warmup_update >= 1
        else []
    )
    pure_after_window = (
        _leading_records_from_update(history, start_update=first_pure_post_update, limit=window_updates)
        if first_pure_post_update is not None
        else []
    )

    key_metrics = [
        "mean_rollout_reward",
        "rollout_f_c",
        "rollout_R_mean",
        "rollout_gini",
        "behavior_frac_pool_power_mix",
        "behavior_frac_actor_logits",
        "behavior_frac_random_logits",
        "behavior_frac_uniform",
        "behavior_frac_proportional",
        "behavior_frac_constant_mix",
        "teacher_takeover_prob",
        "actor_bc_coef",
        "actor_q_coef",
        "actor_bc_loss",
        "actor_q_loss",
        "critic_loss",
        "replay_demo_frac",
        "replay_source_frac_demo",
        "replay_source_frac_recent",
        "replay_source_frac_long_term",
        "eval_return_mean",
        "eval_return_per_step_mean",
        "online_actor_bc_val_loss",
        "online_critic_val_loss",
    ]

    before_means = {metric: _mean_metric(pure_before_window, metric) for metric in key_metrics}
    after_means = {metric: _mean_metric(pure_after_window, metric) for metric in key_metrics}
    deltas = {metric: _delta(after_means[metric], before_means[metric]) for metric in key_metrics}

    eval_after_warmup = (
        _first_record_with_metric(
            [record for record in history if int(record.get("update", 0)) >= int(first_pure_post_update or 1)],
            "eval_return_mean",
        )
        if history
        else None
    )

    demo_pretrain_summary = run_data.demo_pretrain_summary or {}
    pretrain_quick_eval_best = (
        float(demo_pretrain_summary["quick_eval_return_best"])
        if "quick_eval_return_best" in demo_pretrain_summary
        and isinstance(demo_pretrain_summary["quick_eval_return_best"], NUMERIC_TYPES)
        else None
    )

    eval_interval_updates = max(1, int(config.get("eval_interval", 1)))
    adaptive_release_required_evals = max(1, int(config.get("adaptive_teacher_release_required_evals", 1)))
    adaptive_release_enabled = bool(config.get("adaptive_teacher_release_enabled", False))
    earliest_release_update = (
        eval_interval_updates * adaptive_release_required_evals if adaptive_release_enabled else None
    )
    earliest_release_global_step = (
        earliest_release_update * global_steps_per_update if earliest_release_update is not None else None
    )

    warmup_rollout_updates = int(math.ceil(float(warmup_steps) / float(global_steps_per_update))) if warmup_steps > 0 else 0
    estimated_warmup_actor_updates = _estimate_actor_updates(
        warmup_rollout_updates=warmup_rollout_updates,
        gradient_steps_per_update=int(config.get("gradient_steps_per_update", 1)),
        policy_delay=int(config.get("policy_delay", 1)),
    )

    findings: list[str] = []
    before_actor_frac = before_means.get("behavior_frac_actor_logits")
    after_actor_frac = after_means.get("behavior_frac_actor_logits")
    before_reward = before_means.get("mean_rollout_reward")
    after_reward = after_means.get("mean_rollout_reward")

    if before_actor_frac is not None and after_actor_frac is not None and after_actor_frac >= before_actor_frac + 0.10:
        _append_finding(
            findings,
            "Warmup 结束后 rollout 行为不再是纯启发式，actor 行为占比从 {0} 升到 {1}。".format(
                _format_float(before_actor_frac, digits=3),
                _format_float(after_actor_frac, digits=3),
            ),
        )
    if before_reward is not None and after_reward is not None and after_reward < before_reward:
        _append_finding(
            findings,
            "纯 warmup 窗口到纯 post-warmup 窗口的 `mean_rollout_reward` 出现下降，delta={0}。".format(
                _format_float(after_reward - before_reward, digits=6)
            ),
        )

    warmup_behavior_sources = {
        "uniform": float(config.get("warmup_uniform_prob", 0.0)),
        "proportional": float(config.get("warmup_proportional_prob", 0.0)),
        "constant_mix": float(config.get("warmup_constant_mix_prob", 0.0)),
        "pool_power_mix": float(config.get("warmup_pool_power_mix_prob", 0.0)),
        "random_logits": float(config.get("warmup_random_logits_prob", 0.0)),
    }
    non_pool_warmup_mass = (
        warmup_behavior_sources["uniform"]
        + warmup_behavior_sources["proportional"]
        + warmup_behavior_sources["constant_mix"]
        + warmup_behavior_sources["random_logits"]
    )
    if bool(config.get("demo_pretrain_enabled", False)) and str(config.get("demo_collection_behavior_source")) == "pool_power_mix":
        if non_pool_warmup_mass > 1e-8:
            _append_finding(
                findings,
                "Pretrain demo 只来自 `pool_power_mix`，但 warmup BC 使用了混合行为分布，额外非 `pool_power_mix` 权重为 {0}。".format(
                    _format_float(non_pool_warmup_mass, digits=3)
                ),
            )

    if bool(config.get("freeze_actor_q_during_warmup", False)) and float(config.get("warmup_actor_bc_coef", 0.0)) > 0.0:
        if estimated_warmup_actor_updates > 0:
            _append_finding(
                findings,
                "Warmup 期间虽然关闭了 actor 的 Q 项，但仍会发生约 {0} 次 actor BC 更新。".format(
                    estimated_warmup_actor_updates
                ),
            )

    if adaptive_release_enabled and earliest_release_global_step is not None:
        if earliest_release_global_step > warmup_steps:
            _append_finding(
                findings,
                "Adaptive teacher release 最早也要到 global_env_steps≈{0} 才可能解锁，明显晚于 warmup_end={1}。".format(
                    earliest_release_global_step,
                    warmup_steps,
                ),
            )

    rollout_noise_std = float(config.get("rollout_logit_noise_std", 0.0))
    warmup_noise_std = float(config.get("warmup_logit_noise_std", 0.0))
    if rollout_noise_std > warmup_noise_std + 1e-8:
        _append_finding(
            findings,
            "Actor rollout 的 logits 噪声标准差 ({0}) 高于 warmup 启发式噪声 ({1})。".format(
                _format_float(rollout_noise_std, digits=3),
                _format_float(warmup_noise_std, digits=3),
            ),
        )

    if eval_after_warmup is None:
        _append_finding(
            findings,
            "warmup 切换点之后还没有落到一次 `eval_return_mean`，切换点附近看到的下降主要是 rollout 混合行为指标。"
        )
    elif pretrain_quick_eval_best is not None:
        first_eval_return = float(eval_after_warmup.get("eval_return_mean", 0.0))
        ratio = first_eval_return / pretrain_quick_eval_best if abs(pretrain_quick_eval_best) > 1e-8 else None
        if ratio is not None:
            _append_finding(
                findings,
                "warmup 后第一次 eval 的 `eval_return_mean / pretrain quick_eval_return_best = {0}`。".format(
                    _format_float(ratio, digits=3)
                ),
            )

    report = {
        "source_kind": run_data.source_kind,
        "source_path": str(run_data.source_path),
        "checkpoint_mode": run_data.checkpoint_mode,
        "history_length": len(history),
        "warmup": {
            "warmup_steps": warmup_steps,
            "global_steps_per_update": global_steps_per_update,
            "warmup_rollout_updates_estimate": warmup_rollout_updates,
            "warmup_full_updates": warmup_full_updates,
            "warmup_remainder_steps": warmup_remainder,
            "last_pure_warmup_update": last_pure_warmup_update,
            "mixed_transition_update": mixed_transition_update,
            "first_pure_post_update": first_pure_post_update,
            "estimated_warmup_actor_updates": estimated_warmup_actor_updates,
        },
        "config_summary": {
            "freeze_actor_q_during_warmup": bool(config.get("freeze_actor_q_during_warmup", False)),
            "warmup_actor_bc_coef": float(config.get("warmup_actor_bc_coef", 0.0)),
            "actor_demo_bc_coef": float(config.get("actor_demo_bc_coef", 0.0)),
            "online_actor_q_coef_initial": float(config.get("online_actor_q_coef_initial", 0.0)),
            "online_actor_q_coef_final": float(config.get("online_actor_q_coef_final", 0.0)),
            "teacher_takeover_enabled": bool(config.get("teacher_takeover_enabled", False)),
            "teacher_takeover_start_prob": float(config.get("teacher_takeover_start_prob", 0.0)),
            "teacher_takeover_end_prob": float(config.get("teacher_takeover_end_prob", 0.0)),
            "adaptive_teacher_release_enabled": adaptive_release_enabled,
            "adaptive_teacher_release_required_evals": adaptive_release_required_evals,
            "eval_interval_updates": eval_interval_updates,
            "eval_interval_global_steps": eval_interval_updates * global_steps_per_update,
            "gradient_steps_per_update": int(config.get("gradient_steps_per_update", 1)),
            "policy_delay": int(config.get("policy_delay", 1)),
            "rollout_logit_noise_std": rollout_noise_std,
            "warmup_logit_noise_std": warmup_noise_std,
            "replay_recent_fraction": float(config.get("replay_recent_fraction", 0.0)),
            "replay_long_term_fraction": float(config.get("replay_long_term_fraction", 0.0)),
            "replay_demo_fraction": float(config.get("replay_demo_fraction", 0.0)),
            "demo_collection_behavior_source": str(config.get("demo_collection_behavior_source", "")),
            "warmup_behavior_weights": warmup_behavior_sources,
        },
        "snapshots": {
            "last_pure_warmup": _extract_metric_snapshot(
                record_lookup.get(last_pure_warmup_update or -1),
                key_metrics,
            ),
            "mixed_transition": _extract_metric_snapshot(
                record_lookup.get(mixed_transition_update or -1),
                key_metrics,
            ),
            "first_pure_post_warmup": _extract_metric_snapshot(
                record_lookup.get(first_pure_post_update or -1),
                key_metrics,
            ),
            "first_eval_after_warmup": _extract_metric_snapshot(
                eval_after_warmup,
                key_metrics,
            ),
        },
        "window_means": {
            "pure_warmup_before": before_means,
            "pure_post_warmup_after": after_means,
            "delta_after_minus_before": deltas,
        },
        "pretrain": {
            "quick_eval_return_best": pretrain_quick_eval_best,
        },
        "findings": findings,
    }
    return report


def _print_metric_table(
    *,
    title: str,
    before: Mapping[str, float | None],
    after: Mapping[str, float | None],
    delta: Mapping[str, float | None],
    metrics: Sequence[str],
) -> None:
    print(title)
    print("  {0:<32} {1:>14} {2:>14} {3:>14}".format("metric", "before", "after", "delta"))
    for metric in metrics:
        print(
            "  {0:<32} {1:>14} {2:>14} {3:>14}".format(
                metric,
                _format_float(before.get(metric)),
                _format_float(after.get(metric)),
                _format_float(delta.get(metric)),
            )
        )


def _print_snapshot(title: str, snapshot: Mapping[str, float]) -> None:
    print(title)
    if not snapshot:
        print("  n/a")
        return
    for key in (
        "mean_rollout_reward",
        "rollout_f_c",
        "rollout_R_mean",
        "rollout_gini",
        "behavior_frac_pool_power_mix",
        "behavior_frac_actor_logits",
        "teacher_takeover_prob",
        "actor_bc_coef",
        "actor_q_coef",
        "actor_bc_loss",
        "actor_q_loss",
        "critic_loss",
        "eval_return_mean",
        "eval_return_per_step_mean",
    ):
        if key in snapshot:
            print("  {0}: {1}".format(key, _format_float(snapshot[key])))


def _render_report(report: Mapping[str, Any], window_updates: int) -> None:
    warmup = dict(report["warmup"])
    config_summary = dict(report["config_summary"])
    window_means = dict(report["window_means"])
    snapshots = dict(report["snapshots"])
    findings = list(report["findings"])
    pretrain = dict(report["pretrain"])

    print("TD3 Warmup Drop Diagnosis")
    print("source: {0}".format(report["source_path"]))
    print("source_kind: {0}".format(report["source_kind"]))
    print("history_length: {0}".format(report["history_length"]))
    if report.get("checkpoint_mode") is not None:
        print("checkpoint_mode: {0}".format(report["checkpoint_mode"]))
    print()

    print("Boundary")
    print("  warmup_steps: {0}".format(_format_int(warmup.get("warmup_steps"))))
    print("  global_steps_per_update: {0}".format(_format_int(warmup.get("global_steps_per_update"))))
    print("  warmup_rollout_updates_estimate: {0}".format(_format_int(warmup.get("warmup_rollout_updates_estimate"))))
    print("  warmup_full_updates: {0}".format(_format_int(warmup.get("warmup_full_updates"))))
    print("  warmup_remainder_steps: {0}".format(_format_int(warmup.get("warmup_remainder_steps"))))
    print("  last_pure_warmup_update: {0}".format(_format_int(warmup.get("last_pure_warmup_update"))))
    print("  mixed_transition_update: {0}".format(_format_int(warmup.get("mixed_transition_update"))))
    print("  first_pure_post_update: {0}".format(_format_int(warmup.get("first_pure_post_update"))))
    print("  estimated_warmup_actor_updates: {0}".format(_format_int(warmup.get("estimated_warmup_actor_updates"))))
    print()

    print("Config")
    print("  freeze_actor_q_during_warmup: {0}".format(config_summary["freeze_actor_q_during_warmup"]))
    print("  warmup_actor_bc_coef: {0}".format(_format_float(config_summary["warmup_actor_bc_coef"])))
    print("  actor_demo_bc_coef: {0}".format(_format_float(config_summary["actor_demo_bc_coef"])))
    print(
        "  online_actor_q_coef: {0} -> {1}".format(
            _format_float(config_summary["online_actor_q_coef_initial"]),
            _format_float(config_summary["online_actor_q_coef_final"]),
        )
    )
    print(
        "  teacher_takeover: enabled={0}, start_prob={1}, end_prob={2}".format(
            config_summary["teacher_takeover_enabled"],
            _format_float(config_summary["teacher_takeover_start_prob"]),
            _format_float(config_summary["teacher_takeover_end_prob"]),
        )
    )
    print(
        "  adaptive_teacher_release: enabled={0}, required_evals={1}, eval_interval_updates={2}, eval_interval_steps={3}".format(
            config_summary["adaptive_teacher_release_enabled"],
            _format_int(config_summary["adaptive_teacher_release_required_evals"]),
            _format_int(config_summary["eval_interval_updates"]),
            _format_int(config_summary["eval_interval_global_steps"]),
        )
    )
    print(
        "  learner cadence: gradient_steps_per_update={0}, policy_delay={1}".format(
            _format_int(config_summary["gradient_steps_per_update"]),
            _format_int(config_summary["policy_delay"]),
        )
    )
    print(
        "  replay mix: recent={0}, long_term={1}, demo={2}".format(
            _format_float(config_summary["replay_recent_fraction"], digits=3),
            _format_float(config_summary["replay_long_term_fraction"], digits=3),
            _format_float(config_summary["replay_demo_fraction"], digits=3),
        )
    )
    print(
        "  rollout vs warmup noise std: {0} vs {1}".format(
            _format_float(config_summary["rollout_logit_noise_std"], digits=3),
            _format_float(config_summary["warmup_logit_noise_std"], digits=3),
        )
    )
    print("  warmup_behavior_weights:")
    for key, value in dict(config_summary["warmup_behavior_weights"]).items():
        print("    {0}: {1}".format(key, _format_float(value, digits=3)))
    print()

    if pretrain.get("quick_eval_return_best") is not None:
        print(
            "Pretrain\n  quick_eval_return_best: {0}\n".format(
                _format_float(pretrain["quick_eval_return_best"])
            )
        )

    _print_metric_table(
        title="Window Means (last {0} pure-warmup updates vs first {0} pure-post-warmup updates)".format(window_updates),
        before=window_means["pure_warmup_before"],
        after=window_means["pure_post_warmup_after"],
        delta=window_means["delta_after_minus_before"],
        metrics=(
            "mean_rollout_reward",
            "rollout_f_c",
            "rollout_R_mean",
            "rollout_gini",
            "behavior_frac_pool_power_mix",
            "behavior_frac_actor_logits",
            "behavior_frac_random_logits",
            "teacher_takeover_prob",
            "actor_bc_coef",
            "actor_q_coef",
            "actor_bc_loss",
            "actor_q_loss",
            "critic_loss",
            "replay_demo_frac",
            "replay_source_frac_demo",
            "replay_source_frac_recent",
            "replay_source_frac_long_term",
            "eval_return_mean",
            "eval_return_per_step_mean",
            "online_actor_bc_val_loss",
            "online_critic_val_loss",
        ),
    )
    print()

    _print_snapshot("Snapshot: last pure warmup update", snapshots["last_pure_warmup"])
    _print_snapshot("Snapshot: mixed transition update", snapshots["mixed_transition"])
    _print_snapshot("Snapshot: first pure post-warmup update", snapshots["first_pure_post_warmup"])
    _print_snapshot("Snapshot: first eval after warmup", snapshots["first_eval_after_warmup"])
    print()

    print("Findings")
    if not findings:
        print("  No strong heuristic findings from the available history.")
    else:
        for finding in findings:
            print("  - {0}".format(finding))
    print()
    print("Note")
    print("  `mean_rollout_reward` is the mixed behavior actually used during collection.")
    print("  `eval_return_mean` is the deterministic actor-only evaluation metric.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose the TD3 warmup-to-online performance drop from a checkpoint, output directory, or results.json."
    )
    parser.add_argument(
        "input_path",
        help="Path to a checkpoint (.pt), an experiment output directory, or a results.json file.",
    )
    parser.add_argument(
        "--window-updates",
        type=int,
        default=5,
        help="Number of pure-warmup and pure-post-warmup updates to average on each side of the boundary.",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="Optional path to save the diagnostic report as JSON.",
    )
    args = parser.parse_args()

    if int(args.window_updates) <= 0:
        raise ValueError("--window-updates must be positive.")

    run_data = _load_run_data(args.input_path)
    report = analyze_run_data(run_data, window_updates=int(args.window_updates))
    _render_report(report, window_updates=int(args.window_updates))

    if args.json_out:
        output_path = Path(args.json_out).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print("\nJSON report saved to: {0}".format(output_path))


if __name__ == "__main__":
    main()
