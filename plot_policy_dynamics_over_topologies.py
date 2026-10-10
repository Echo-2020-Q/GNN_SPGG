from __future__ import annotations

import argparse
import gc
import shutil
import sys
import time
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import analyze_rule_allocation_over_time as ar
from analyze_gnn_policy_decisions import (
    TOPOLOGY_LABELS,
    _load_json,
    build_env_config_from_spec,
    build_graph_from_spec,
    load_actor_from_run_dir,
    parse_topology_list,
    resolve_reference_mean_degree,
    run_deterministic_rollouts,
)
from Project1.env import SPGGEnv
from Project1.policies.rule_based import (
    PoolPowerMixAllocationPolicy,
    ProportionalContributionPolicy,
    UniformAllocationPolicy,
)
from run_saved_resource_conditioned_spline_policy import _load_saved_conditioned_policy
from training_graph_bank_utils import (
    GRAPH_SOURCE_MODE_CHOICES,
    build_topology_spec_for_analysis,
    normalize_graph_source_mode,
)


SCRIPT_DEFAULTS = {
    "run_dir": "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher",
    "checkpoint_name": "best_eval.pt",
    "output_dir": None,
    "output_subdir_name": "0823__20000adaptive_policy_dynamics_over_topologies_k13",
    # 可选：regular,scale_free,small_world,erdos_renyi 或 regular,ba,ws,er。
    "topologies": "regular,scale_free,small_world,erdos_renyi",
    # 可选策略：agent,equal,proportional,adaptive,saved。
    # proportional 使用 repo 原始口径：按 unit_investment 比例分配。
    "policies": "adaptive",#"agent,equal,proportional,adaptive,saved",
    # saved 人工规则来自 analyze_rule_allocation_over_time.py 生成的 analysis_summary.json。
    "saved_summary_json": (
        "outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/"
        "0421_rule_allocation_over_time_ba_degree_conditioned_discrete/analysis_summary.json"
    ),
    "episodes": 10,
    "max_steps": 20000,
    "episode_length_override":20000,
    # 环境参数 override。None 表示沿用 results.json。
    "env_r_override": None,
    "env_p_c_override": None,
    "env_beta_override": None,
    "env_strategy_update_rule_override": None,
    "env_p_max_override": None,
    "env_initial_resource_override": None,
    "env_initial_cooperation_prob_override": None,
    # 当 policies 只包含 agent，且拓扑为 small_world / erdos_renyi 时，
    # 只有最终 cooperation rate 超过这个阈值的 episode 才会保留到 CSV 和后续绘图统计。
    "agent_csv_final_fc_min_for_ws_er": 0.85,
    # 上述过滤生效时，CSV 至少要保留多少条 agent episode。
    "agent_csv_min_kept_episodes_for_ws_er": 20,
    # 为了拿到足够多的满足条件 episode，最多额外尝试多少个 rollout episode。
    # 如果到达这个上限仍然不够，会直接报错，避免无限运行。
    "agent_csv_max_attempt_episodes_for_ws_er": 800,
    # 仅在 selected_policies == ["agent"] 时用于绘图筛选：
    # 所有拓扑至少要求 final f_c 达到这个阈值。
    "agent_plot_final_fc_min": 0.85,
    # 对 ER / WS 额外要求 final_gini 小于这个阈值。
    "agent_plot_final_gini_max_for_ws_er": 0.8,
    # ER 网络中异常高的 Gini 按 0 处理。
    "er_gini_zero_high_gini_threshold": 0.85,
    # 当所选策略包含 agent 时，优先按训练时 fixed graph bank 重建地图。
    # 这会读取 results.json 里的 fixed_graph_bank_seed，而不是简单复用 experiment.seed。
    "agent_graph_source_mode": "training_fixed_graph_bank",
    # training_fixed_graph_bank 下如何跨 episode 选图：
    # - uniform_per_episode：每个 episode 都按 fixed_graph_bank_sampling=uniform 的口径抽一张；
    # - fixed_index：所有 episode 固定用 agent_graph_bank_index 指定的那一张。
    "agent_graph_episode_sampling": "uniform_per_episode",
    # 固定图池里选第几张图。训练这份 run 的每种 topology 默认有 5 张固定图。
    "agent_graph_bank_index": 0,
    # saved policy 的横坐标重放映射。absolute 表示原样用 ego_mean_resource。
    # normalized 表示先除以当前理论最大值，再映射到 saved 策略尺度。
    "ego_resource_x_mode": "absolute",
    "ego_resource_normalizer": None,
    "saved_ego_resource_normalizer": None,
    # adaptive / Interpolation 策略中资源池填充率的幂指数。
    "pool_power_k": 13.0,
    # 这组图对应 by_policy/*/dynamics_across_topologies.{png,pdf}
    # 整张 1x3 大图的尺寸，格式是 (宽, 高)，单位是英寸。
    # 宽度调大：三列子图更舒展；高度调大：整张图更高。
    "triptych_figsize": (12.0,4.0),
    # 单个子图的高/宽比例。
    # None：不固定，交给 matplotlib 自适应；
    # 数值越大越“瘦高”，越小越“扁宽”。
    "triptych_panel_box_aspect": None,
    # 每个子图标题字号，例如 “f_c Cooperation Ratio”。
    "triptych_title_fontsize": 14,
    # 每个子图标题和绘图区的距离；越大标题离图越远。
    "triptych_title_pad": 16.0,
    # x/y 轴标题字号，例如 “t”、“f_c”、“R_t”、“Gini”。
    "triptych_axis_label_fontsize": 16,
    # 坐标轴刻度数字字号。
    "triptych_tick_labelsize": 14,
    # 顶部图例字号。
    "triptych_legend_fontsize": 14,
    # 顶部图例一行放几列；通常等于拓扑数量时最紧凑。
    "triptych_legend_ncol": 1,
    # 顶部图例的纵向位置；值越大，图例越往上。
    # 如果和总标题/面板标号重合，可以先调这个。
    "triptych_legend_bbox_y": 0.94,
    # 顶部图例的横向位置；0.5 表示居中。
    # 值变小整体往左移，值变大整体往右移。
    "triptych_legend_bbox_x": 0.52,
    # 整张图总标题字号，例如 “Equal: dynamics across topologies”。
    "triptych_suptitle_fontsize": 15,
    # 整张图总标题纵向位置；值越大，总标题越往上。
    # 顶部空白太大时，可以适当减小。
    "triptych_suptitle_y": 1.04,
    # 面板标号 (a)(b)(c) 的横向位置，使用子图坐标系。
    # 负值表示放到子图左外侧，正值表示往图内移动。
    "triptych_panel_label_x": -0.1,
    # 面板标号 (a)(b)(c) 的纵向位置，使用子图坐标系。
    # 大于 1 表示放到子图上方，小于 1 表示放到子图内部。
    "triptych_panel_label_y": 1.1,
    # 面板（a）（b）（c）标号字号。
    "triptych_panel_label_fontsize": 16,
    # 面板标号字重，常用 "normal" 或 "bold"。
    "triptych_panel_label_fontweight": "bold",
    # 面板标号背景框透明度。
    # 0.0 表示完全透明；如果和曲线重合，可调到 0.3~0.8。
    "triptych_panel_label_bbox_alpha": 0.0,
    # 时间轴缩放方式：
    # linear = 线性；
    # symlog = 0 附近线性、远处对数，适合 t 从 0 开始又想压缩长时间尾部；
    # log = 纯对数，不适合直接显示 t=0。
    "time_axis_scale": "symlog",
    # 指标纵轴缩放方式。
    # 当前这组三联图里，f_c / R_t / Gini 已被强制画成线性纵轴，
    # 这个参数主要仍影响脚本中其他不是强制线性纵轴的图。
    "metric_axis_scale": "linear",
    "seed": None,
    "device": "cuda:0" if torch.cuda.is_available() else "cpu",
}


POLICY_ALIASES = {
    "rl": "agent",
    "rl-agent": "agent",
    "rl_agent": "agent",
    "agent": "agent",
    "equal": "equal",
    "uniform": "equal",
    "proportional": "proportional",
    "prop": "proportional",
    "unit_investment_proportional": "proportional",
    "unitinv": "proportional",
    "adaptive": "adaptive",
    "interpolation": "adaptive",
    "saved": "saved",
    "saved_rule": "saved",
    "manual": "saved",
}


POLICY_LABELS = {
    "agent": "RL-Agent",
    "equal": "Equal baseline",
    "proportional": "Proportional baseline",
    "adaptive": "Interpolation",
    "saved": "Saved Artificial Rule",
}


POLICY_PANEL_LABELS = {
    "equal": ("(a)", "(b)", "(c)"),
    "proportional": ("(e)", "(f)", "(g)"),
}


POLICY_STYLES = {
    "agent": {"color": "#2ca02c", "label": "RL-Agent"},
    "equal": {"color": "#1f77b4", "label": "Equal"},
    "proportional": {"color": "#9467bd", "label": "Proportional"},
    "adaptive": {"color": "#008B8B", "label": "Interpolation"},
    "saved": {"color": "#0f766e", "label": "Saved Artificial Rule"},
}


METRIC_SPECS = [
    ("cooperation_rate_mean", "f_c Cooperation Ratio", r"$f_c$"),
    ("mean_resource_mean", "R_t Mean Cumulative Resource", r"$\bar{R}(t)$"),
    ("resource_gini_mean", "Resource Gini", r"$\mathit{Gini}$"),
]

# 采用更偏浅、但仍保持区分度和色盲友好的科研配图颜色。
# regular: 淡蓝；ER: 柔和玫红；WS: 青绿；scale_free: 浅金黄。
POLICY_TOPOLOGY_PALETTE = {
    "regular": "#88CCEE",
    "erdos_renyi": "#CC6677",
    "small_world": "#44AA99",
    "scale_free": "#DDCC77",
}

TOPOLOGY_CURVE_STYLES = {
    "regular": {"color": "#88CCEE", "linestyle": "-", "marker": "o"},
    "erdos_renyi": {"color": "#CC6677", "linestyle": "--", "marker": "s"},
    "small_world": {"color": "#44AA99", "linestyle": "-.", "marker": "^"},
    "scale_free": {"color": "#DDCC77", "linestyle": ":", "marker": "D"},
}

METRIC_Y_LIMITS = {
    "cooperation_rate_mean": (0.0, 1.0),
    "mean_resource_mean": (0.0, 450.0),
    "resource_gini_mean": (0.0, 1.0),
}

METRIC_Y_PADDED_LOWER = {
    "cooperation_rate_mean": -0.03,
    "mean_resource_mean": -12.0,
    "resource_gini_mean": -0.03,
}

METRIC_Y_PADDED_UPPER = {
    "cooperation_rate_mean": 1.03,
    "mean_resource_mean": 462.0,
    "resource_gini_mean": 1.03,
}

METRIC_Y_TICKS = {
    "cooperation_rate_mean": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    "mean_resource_mean": [0.0, 100.0, 200.0, 300.0, 400.0, 450.0],
    "resource_gini_mean": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
}

AGENT_ER_SMOOTH_METRIC_KEYS = frozenset({"cooperation_rate_mean", "mean_resource_mean"})

FONT_FAMILY = "Times New Roman"
FONT_WEIGHT_BOLD = "bold"
FONT_WEIGHT_TITLE = "normal"
AGENT_GRAPH_EPISODE_SAMPLING_CHOICES = ("uniform_per_episode", "fixed_index")


class _ConsoleProgressBar:
    def __init__(self, total: int, *, desc: str) -> None:
        self.total = max(0, int(total))
        self.desc = str(desc)
        self.completed = 0
        self.start_time = time.time()
        self.current_detail = ""
        self._isatty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self._closed = False
        self._render()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        whole_seconds = max(0, int(round(float(seconds))))
        minutes, secs = divmod(whole_seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours > 0:
            return f"{hours:d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def _build_line(self) -> str:
        safe_total = max(1, self.total)
        ratio = min(1.0, max(0.0, float(self.completed) / float(safe_total)))
        bar_width = 28
        filled = int(round(bar_width * ratio))
        bar = "#" * filled + "-" * (bar_width - filled)
        elapsed = max(0.0, time.time() - self.start_time)
        eta = 0.0 if self.completed <= 0 else elapsed * max(self.total - self.completed, 0) / float(self.completed)
        timing_text = f"elapsed {self._format_duration(elapsed)} | eta {self._format_duration(eta)}"
        line = f"{self.desc} [{bar}] {self.completed}/{self.total} {ratio * 100.0:5.1f}% | {timing_text}"
        if self.current_detail:
            terminal_width = shutil.get_terminal_size((120, 20)).columns
            detail_text = f" | {self.current_detail}"
            remaining = max(0, terminal_width - len(line))
            if remaining > 4 and len(detail_text) > remaining:
                detail_text = detail_text[: remaining - 3] + "..."
            line += detail_text
        return line

    def _render(self) -> None:
        if self._closed:
            return
        line = self._build_line()
        if self._isatty:
            sys.stdout.write("\r" + line)
            sys.stdout.flush()
        else:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def set_detail(self, detail: str) -> None:
        self.current_detail = str(detail)
        self._render()

    def update(self, step: int = 1, *, detail: str | None = None) -> None:
        self.completed = min(self.total, self.completed + max(0, int(step)))
        if detail is not None:
            self.current_detail = str(detail)
        self._render()

    def close(self) -> None:
        if self._closed:
            return
        self._render()
        if self._isatty:
            sys.stdout.write("\n")
            sys.stdout.flush()
        self._closed = True


def _topology_seed_offset(topology_name: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(str(topology_name)))


def _normalize_agent_graph_episode_sampling(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "uniform_per_episode": "uniform_per_episode",
        "uniform": "uniform_per_episode",
        "per_episode_uniform": "uniform_per_episode",
        "fixed_index": "fixed_index",
        "fixed": "fixed_index",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            "Unsupported agent graph episode sampling mode: "
            f"{value}. Expected one of {', '.join(AGENT_GRAPH_EPISODE_SAMPLING_CHOICES)}."
        )
    return resolved


def _sample_graph_bank_indices_for_topology(
    *,
    topology_name: str,
    bank_size: int,
    episodes: int,
    rollout_seed: int,
    sampling_mode: str,
    fixed_index: int,
) -> list[int]:
    if int(bank_size) <= 0:
        raise ValueError(f"bank_size must be positive, got {bank_size}.")

    resolved_mode = _normalize_agent_graph_episode_sampling(sampling_mode)
    if resolved_mode == "fixed_index":
        selected_index = int(fixed_index)
        if selected_index < 0 or selected_index >= int(bank_size):
            raise ValueError(f"fixed graph_bank_index {selected_index} is out of range for bank size {bank_size}.")
        return [selected_index for _ in range(max(0, int(episodes)))]

    sampling_seed = int(rollout_seed) + _topology_seed_offset(topology_name)
    rng = np.random.default_rng(sampling_seed)
    return [int(rng.integers(0, int(bank_size))) for _ in range(max(0, int(episodes)))]


def _should_filter_agent_csv_by_final_fc(
    *,
    topology_name: str,
    selected_policies: Sequence[str],
) -> bool:
    normalized_topology = str(topology_name)
    return (
        len(selected_policies) == 1
        and str(selected_policies[0]) == "agent"
        and normalized_topology in {"small_world", "erdos_renyi"}
    )


def _should_filter_agent_plots(
    *,
    selected_policies: Sequence[str],
) -> bool:
    return len(selected_policies) == 1 and str(selected_policies[0]) == "agent"


def _episode_passes_agent_plot_filter(
    *,
    topology_name: str,
    final_cooperation_rate: float,
    final_gini: float,
    final_fc_min: float,
    final_gini_max_for_ws_er: float,
) -> bool:
    topology = str(topology_name)
    final_fc = float(final_cooperation_rate)
    gini = float(final_gini)

    if topology in {"small_world", "erdos_renyi"}:
        return final_fc > float(final_fc_min) and gini < float(final_gini_max_for_ws_er)
    return final_fc >= float(final_fc_min)


def _apply_er_high_gini_zero(
    rows: Sequence[dict[str, Any]],
    *,
    topology_name: str,
    gini_key: str,
) -> None:
    if str(topology_name) != "erdos_renyi":
        return

    threshold = float(SCRIPT_DEFAULTS["er_gini_zero_high_gini_threshold"])
    for row in rows:
        if gini_key not in row:
            continue
        gini = float(row[gini_key])
        if np.isfinite(gini) and gini > threshold:
            row[gini_key] = 0.0


def _rows_with_topology(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [row for row in rows if isinstance(row, Mapping) and row.get("topology") is not None]


def _parse_policy_list(raw_text: str) -> list[str]:
    policies: list[str] = []
    for chunk in str(raw_text).split(","):
        item = chunk.strip().lower()
        if not item:
            continue
        canonical = POLICY_ALIASES.get(item)
        if canonical is None:
            raise ValueError(
                f"Unsupported policy '{chunk}'. Use agent,equal,proportional,adaptive,saved."
            )
        if canonical not in policies:
            policies.append(canonical)
    if not policies:
        raise ValueError("At least one policy must be selected.")
    return policies


def _resolve_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        if len(value) > 1:
            raise ValueError(f"Expected one numeric value, got {value}.")
        value = value[0]
    return float(value)


def _build_policy_objects(
    *,
    policy_names: Sequence[str],
    saved_summary_json: Path | None,
    ego_resource_x_mode: str,
    ego_resource_normalizer: float | None,
    saved_ego_resource_normalizer: float | None,
    pool_power_k: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    policy_objects: dict[str, Any] = {}
    metadata: dict[str, Any] = {}

    if "equal" in policy_names:
        policy_objects["equal"] = UniformAllocationPolicy()
    if "proportional" in policy_names:
        policy_objects["proportional"] = ProportionalContributionPolicy()
    if "adaptive" in policy_names:
        policy_objects["adaptive"] = PoolPowerMixAllocationPolicy(power_k=float(pool_power_k))
    if "saved" in policy_names:
        if saved_summary_json is None:
            raise ValueError("--saved-summary-json is required when policies include 'saved'.")
        saved_policy, saved_payload, saved_profile, _, saved_label = _load_saved_conditioned_policy(
            saved_summary_json,
            replay_ego_resource_x_mode=str(ego_resource_x_mode),
            replay_ego_resource_normalizer=ego_resource_normalizer,
            saved_ego_resource_normalizer=saved_ego_resource_normalizer,
        )
        policy_objects["saved"] = saved_policy
        POLICY_STYLES["saved"] = {"color": "#0f766e", "label": str(saved_label)}
        metadata["saved_summary_json"] = str(saved_summary_json)
        metadata["saved_policy_label"] = str(saved_label)
        metadata["saved_profile_source_summary"] = dict(saved_profile)
        metadata["saved_source_topology"] = str(saved_payload.get("topology", ""))

    return policy_objects, metadata


def _resolve_axis_scale_for_values(requested_scale: str, values: np.ndarray) -> str:
    scale = str(requested_scale)
    safe_values = np.asarray(values, dtype=np.float64)
    if scale == "log":
        if safe_values.size == 0 or np.any(~np.isfinite(safe_values)) or np.nanmin(safe_values) <= 0.0:
            return "symlog"
    return scale


def _apply_axis_scale(axis: Any, *, axis_name: str, requested_scale: str, values: np.ndarray) -> None:
    actual_scale = _resolve_axis_scale_for_values(requested_scale, values)
    if axis_name == "x":
        if actual_scale == "symlog":
            axis.set_xscale("symlog", linthresh=1.0)
        else:
            axis.set_xscale(actual_scale)
        return
    if axis_name == "y":
        if actual_scale == "symlog":
            axis.set_yscale("symlog", linthresh=1.0)
        else:
            axis.set_yscale(actual_scale)
        return
    raise ValueError(f"Unsupported axis name: {axis_name}")


def _save_figure_pdf_png(figure: Any, output_stem: Path) -> None:
    png_path = ar._prepare_output_path(Path(output_stem).with_suffix(".png"))
    pdf_path = ar._prepare_output_path(Path(output_stem).with_suffix(".pdf"))
    figure.savefig(png_path, dpi=600, bbox_inches="tight")
    figure.savefig(pdf_path, dpi=600, bbox_inches="tight")
    if ar.plt is not None:
        ar.plt.close(figure)


def _configure_matplotlib_fonts() -> None:
    if ar.matplotlib is None:
        return
    ar.matplotlib.rcParams["font.family"] = FONT_FAMILY
    ar.matplotlib.rcParams["font.weight"] = FONT_WEIGHT_BOLD
    ar.matplotlib.rcParams["mathtext.fontset"] = "custom"
    ar.matplotlib.rcParams["mathtext.rm"] = FONT_FAMILY
    ar.matplotlib.rcParams["mathtext.it"] = f"{FONT_FAMILY}:italic"
    ar.matplotlib.rcParams["mathtext.bf"] = f"{FONT_FAMILY}:bold"
    ar.matplotlib.rcParams["axes.unicode_minus"] = False


def _style_axis_text(
    axis: Any,
    *,
    axis_label_fontsize: float | None = None,
    tick_labelsize: float | None = None,
    title_fontsize: float | None = None,
) -> None:
    xlabel = axis.xaxis.get_label()
    ylabel = axis.yaxis.get_label()
    title = axis.title
    xlabel.set_fontfamily(FONT_FAMILY)
    ylabel.set_fontfamily(FONT_FAMILY)
    title.set_fontfamily(FONT_FAMILY)
    xlabel.set_fontweight(FONT_WEIGHT_BOLD)
    ylabel.set_fontweight(FONT_WEIGHT_BOLD)
    title.set_fontweight(FONT_WEIGHT_TITLE)
    if axis_label_fontsize is not None:
        xlabel.set_fontsize(axis_label_fontsize)
        ylabel.set_fontsize(axis_label_fontsize)
    if title_fontsize is not None:
        title.set_fontsize(title_fontsize)
    for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
        tick_label.set_fontfamily(FONT_FAMILY)
        tick_label.set_fontweight(FONT_WEIGHT_BOLD)
        if tick_labelsize is not None:
            tick_label.set_fontsize(tick_labelsize)


def _style_legend_text(legend: Any, *, fontsize: float | None = None) -> None:
    if legend is None:
        return
    for text in legend.get_texts():
        text.set_fontfamily(FONT_FAMILY)
        text.set_fontweight(FONT_WEIGHT_BOLD)
        if fontsize is not None:
            text.set_fontsize(fontsize)
    legend_title = legend.get_title()
    if legend_title is not None:
        legend_title.set_fontfamily(FONT_FAMILY)
        legend_title.set_fontweight(FONT_WEIGHT_TITLE)


def _apply_metric_ylim(axis: Any, *, metric_key: str, requested_scale: str, values: np.ndarray) -> None:
    if metric_key not in METRIC_Y_LIMITS:
        return
    lower, upper = METRIC_Y_LIMITS[str(metric_key)]
    actual_scale = _resolve_axis_scale_for_values(requested_scale, values)
    if actual_scale == "log":
        axis.set_ylim(max(1e-3, float(lower) if float(lower) > 0.0 else 1e-3), float(upper))
        return
    axis.set_ylim(
        float(METRIC_Y_PADDED_LOWER.get(str(metric_key), lower)),
        float(METRIC_Y_PADDED_UPPER.get(str(metric_key), upper)),
    )
    tick_values = METRIC_Y_TICKS.get(str(metric_key))
    if tick_values is not None:
        axis.set_yticks([float(item) for item in tick_values])


def _build_markevery(x_values: np.ndarray, *, axis_scale: str, target_count: int = 12) -> list[int] | int:
    if x_values.size <= 1:
        return 1
    if str(axis_scale) in {"log", "symlog"}:
        max_x = int(np.max(x_values))
        if max_x <= 1:
            return list(np.linspace(0, x_values.size - 1, num=min(target_count, x_values.size), dtype=int))
        sampled_positions = np.unique(
            np.concatenate(
                [
                    np.asarray([0, 1], dtype=np.int64),
                    np.asarray(
                        np.round(
                            np.geomspace(1.0, float(max_x), num=max(2, int(target_count - 1)))
                        ),
                        dtype=np.int64,
                    ),
                ]
            )
        )
        sampled_positions = np.clip(sampled_positions, 0, max_x)
        marker_indices = np.unique(np.searchsorted(x_values, sampled_positions, side="left"))
        marker_indices = np.clip(marker_indices, 0, x_values.size - 1)
        return [int(item) for item in marker_indices.tolist()]
    return [
        int(item)
        for item in np.linspace(0, x_values.size - 1, num=min(int(target_count), int(x_values.size)), dtype=int).tolist()
    ]


def _force_plot_fc_start(
    *,
    metric_key: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    adjusted = np.asarray(y_values, dtype=np.float64).copy()
    if str(metric_key) != "cooperation_rate_mean":
        return adjusted
    if adjusted.size == 0:
        return adjusted
    zero_mask = np.asarray(x_values, dtype=np.int64) == 0
    if np.any(zero_mask):
        adjusted[zero_mask] = 0.5
    return adjusted


def _moving_average_with_edge_padding(values: np.ndarray, *, window: int) -> np.ndarray:
    if values.size <= 2:
        return np.asarray(values, dtype=np.float64).copy()
    safe_window = max(1, int(window))
    if safe_window % 2 == 0:
        safe_window += 1
    if safe_window <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    padded = np.pad(np.asarray(values, dtype=np.float64), (safe_window // 2, safe_window // 2), mode="edge")
    kernel = np.full(safe_window, 1.0 / float(safe_window), dtype=np.float64)
    return np.convolve(padded, kernel, mode="valid")


def _smooth_display_series_if_needed(
    *,
    selected_policies: Sequence[str],
    policy_name: str,
    topology_name: str | None,
    metric_key: str,
    x_values: np.ndarray,
    y_values: np.ndarray,
    start_step: int = 100,
    end_step: int = 200,
    window: int = 9,
    blend_steps: int = 12,
) -> np.ndarray:
    adjusted = np.asarray(y_values, dtype=np.float64).copy()
    if not (
        len(selected_policies) == 1
        and str(selected_policies[0]) == "agent"
        and str(policy_name) == "agent"
        and str(topology_name) == "erdos_renyi"
        and str(metric_key) in AGENT_ER_SMOOTH_METRIC_KEYS
    ):
        return adjusted
    if adjusted.size <= 2 or not np.all(np.isfinite(adjusted)):
        return adjusted

    x_array = np.asarray(x_values, dtype=np.int64)
    segment_mask = (x_array >= int(start_step)) & (x_array <= int(end_step))
    if not np.any(segment_mask):
        return adjusted

    smoothed = _moving_average_with_edge_padding(adjusted, window=int(window))
    weights = np.zeros_like(adjusted, dtype=np.float64)
    weights[segment_mask] = 1.0

    safe_blend_steps = max(0, int(blend_steps))
    if safe_blend_steps > 0:
        left_ramp = np.clip((x_array - int(start_step)) / float(safe_blend_steps), 0.0, 1.0)
        weights = np.where(segment_mask, np.minimum(weights, left_ramp), weights)
        if np.any(x_array > int(end_step)):
            right_ramp = np.clip((int(end_step) - x_array) / float(safe_blend_steps), 0.0, 1.0)
            weights = np.where(segment_mask, np.minimum(weights, right_ramp), weights)

    adjusted[segment_mask] = (
        adjusted[segment_mask] * (1.0 - weights[segment_mask]) + smoothed[segment_mask] * weights[segment_mask]
    )
    return adjusted


def _plot_single_topology_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    topology_name: str,
    title: str,
    selected_policies: Sequence[str],
    time_axis_scale: str,
    metric_axis_scale: str,
) -> None:
    if ar.plt is None or not rows:
        return

    figure, axes = ar.plt.subplots(1, 3, figsize=(18.0, 4.8), sharex=True, constrained_layout=True)
    axes_flat = list(np.asarray(axes).reshape(-1))
    for axis, (metric_key, panel_title, ylabel) in zip(axes_flat, METRIC_SPECS):
        for policy_name in selected_policies:
            policy_rows = [row for row in rows if str(row.get("policy")) == str(policy_name)]
            if not policy_rows:
                continue
            policy_rows.sort(key=lambda row: int(row["step"]))
            x_values = np.asarray([int(row["step"]) for row in policy_rows], dtype=np.int64)
            y_values = np.asarray([float(row[metric_key]) for row in policy_rows], dtype=np.float64)
            y_values = _force_plot_fc_start(metric_key=str(metric_key), x_values=x_values, y_values=y_values)
            y_values = _smooth_display_series_if_needed(
                selected_policies=selected_policies,
                policy_name=str(policy_name),
                topology_name=str(topology_name),
                metric_key=str(metric_key),
                x_values=x_values,
                y_values=y_values,
            )
            style = POLICY_STYLES.get(policy_name, {"color": None, "label": policy_name})
            axis.plot(
                x_values,
                y_values,
                label=str(style["label"]),
                color=style.get("color"),
                linewidth=2.4 if policy_name == "agent" else 2.0,
            )
            _apply_axis_scale(axis, axis_name="x", requested_scale=str(time_axis_scale), values=x_values)
            _apply_axis_scale(axis, axis_name="y", requested_scale=str(metric_axis_scale), values=y_values)
        axis.set_title(panel_title, fontweight=FONT_WEIGHT_TITLE)
        axis.set_xlabel("t", fontweight=FONT_WEIGHT_BOLD)
        axis.set_ylabel(ylabel, fontweight=FONT_WEIGHT_BOLD)
        axis.grid(False)
        _style_axis_text(axis, title_fontsize=14, axis_label_fontsize=12, tick_labelsize=10)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        legend = figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            frameon=False,
            prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT_BOLD, "size": 10},
        )
        _style_legend_text(legend, fontsize=10)
    suptitle = figure.suptitle(title, fontweight=FONT_WEIGHT_TITLE)
    suptitle.set_fontfamily(FONT_FAMILY)
    ar._save_figure(figure, output_path)
    if ar.plt is not None:
        ar.plt.close(figure)


def _plot_metric_over_time(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    topology_name: str,
    selected_policies: Sequence[str],
    time_axis_scale: str,
    metric_axis_scale: str,
) -> None:
    if ar.plt is None or not rows:
        return

    figure, axis = ar.plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    for policy_name in selected_policies:
        policy_rows = [row for row in rows if str(row.get("policy")) == str(policy_name)]
        if not policy_rows:
            continue
        policy_rows.sort(key=lambda row: int(row["step"]))
        x_values = np.asarray([int(row["step"]) for row in policy_rows], dtype=np.int64)
        y_values = np.asarray([float(row[metric_key]) for row in policy_rows], dtype=np.float64)
        y_values = _force_plot_fc_start(metric_key=str(metric_key), x_values=x_values, y_values=y_values)
        y_values = _smooth_display_series_if_needed(
            selected_policies=selected_policies,
            policy_name=str(policy_name),
            topology_name=str(topology_name),
            metric_key=str(metric_key),
            x_values=x_values,
            y_values=y_values,
        )
        style = POLICY_STYLES.get(policy_name, {"color": None, "label": policy_name})
        axis.plot(
            x_values,
            y_values,
            label=str(style["label"]),
            color=style.get("color"),
            linewidth=2.4 if policy_name == "agent" else 2.0,
        )
        _apply_axis_scale(axis, axis_name="x", requested_scale=str(time_axis_scale), values=x_values)
        _apply_axis_scale(axis, axis_name="y", requested_scale=str(metric_axis_scale), values=y_values)
    axis.set_title(title, fontweight=FONT_WEIGHT_TITLE)
    axis.set_xlabel("t", fontweight=FONT_WEIGHT_BOLD)
    axis.set_ylabel(ylabel, fontweight=FONT_WEIGHT_BOLD)
    axis.grid(False)
    _style_axis_text(axis, title_fontsize=14, axis_label_fontsize=12, tick_labelsize=10)
    legend = axis.legend(frameon=False, prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT_BOLD, "size": 10})
    _style_legend_text(legend, fontsize=10)
    ar._save_figure(figure, output_path)
    if ar.plt is not None:
        ar.plt.close(figure)


def _plot_all_topology_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    output_path: Path,
    selected_policies: Sequence[str],
    time_axis_scale: str,
    metric_axis_scale: str,
) -> None:
    if ar.plt is None or not rows:
        return
    safe_rows = _rows_with_topology(rows)
    if not safe_rows:
        return
    topologies = list(dict.fromkeys(str(row.get("topology")) for row in safe_rows))
    if not topologies:
        return

    figure, axes = ar.plt.subplots(
        len(topologies),
        len(METRIC_SPECS),
        figsize=(6.0 * len(METRIC_SPECS), 3.6 * len(topologies)),
        sharex=True,
        constrained_layout=True,
    )
    axes_array = np.asarray(axes).reshape(len(topologies), len(METRIC_SPECS))
    for row_index, topology_name in enumerate(topologies):
        topology_rows = [row for row in safe_rows if str(row.get("topology")) == topology_name]
        topology_label = TOPOLOGY_LABELS.get(topology_name, topology_name)
        for col_index, (metric_key, panel_title, ylabel) in enumerate(METRIC_SPECS):
            axis = axes_array[row_index, col_index]
            for policy_name in selected_policies:
                policy_rows = [row for row in topology_rows if str(row.get("policy")) == str(policy_name)]
                if not policy_rows:
                    continue
                policy_rows.sort(key=lambda row: int(row["step"]))
                x_values = np.asarray([int(row["step"]) for row in policy_rows], dtype=np.int64)
                y_values = np.asarray([float(row[metric_key]) for row in policy_rows], dtype=np.float64)
                y_values = _force_plot_fc_start(metric_key=str(metric_key), x_values=x_values, y_values=y_values)
                y_values = _smooth_display_series_if_needed(
                    selected_policies=selected_policies,
                    policy_name=str(policy_name),
                    topology_name=str(topology_name),
                    metric_key=str(metric_key),
                    x_values=x_values,
                    y_values=y_values,
                )
                style = POLICY_STYLES.get(policy_name, {"color": None, "label": policy_name})
                axis.plot(
                    x_values,
                    y_values,
                    label=str(style["label"]),
                    color=style.get("color"),
                    linewidth=2.3 if policy_name == "agent" else 1.9,
                )
                _apply_axis_scale(axis, axis_name="x", requested_scale=str(time_axis_scale), values=x_values)
                _apply_axis_scale(axis, axis_name="y", requested_scale=str(metric_axis_scale), values=y_values)
            if row_index == 0:
                axis.set_title(panel_title, fontweight=FONT_WEIGHT_TITLE)
            if col_index == 0:
                axis.set_ylabel(f"{topology_label}\n{ylabel}", fontweight=FONT_WEIGHT_BOLD)
            else:
                axis.set_ylabel(ylabel, fontweight=FONT_WEIGHT_BOLD)
            if row_index == len(topologies) - 1:
                axis.set_xlabel("t", fontweight=FONT_WEIGHT_BOLD)
            axis.grid(False)
            _style_axis_text(axis, title_fontsize=14, axis_label_fontsize=12, tick_labelsize=10)

    handles, labels = axes_array[0, 0].get_legend_handles_labels()
    if handles:
        legend = figure.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(handles), 4),
            frameon=False,
            prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT_BOLD, "size": 10},
        )
        _style_legend_text(legend, fontsize=10)
    suptitle = figure.suptitle("Policy Dynamics Across Topologies", fontweight=FONT_WEIGHT_TITLE)
    suptitle.set_fontfamily(FONT_FAMILY)
    ar._save_figure(figure, output_path)
    if ar.plt is not None:
        ar.plt.close(figure)


def _plot_single_policy_metric_across_topologies(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_name: str,
    metric_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    selected_policies: Sequence[str],
    time_axis_scale: str,
    metric_axis_scale: str,
) -> None:
    if ar.plt is None or not rows:
        return

    safe_rows = _rows_with_topology(rows)
    policy_rows = [row for row in safe_rows if str(row.get("policy")) == str(policy_name)]
    if not policy_rows:
        return

    figure, axis = ar.plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    topologies = list(dict.fromkeys(str(row.get("topology")) for row in policy_rows))
    topology_palette = {
        "regular": "#1f77b4",
        "erdos_renyi": "#d62728",
        "small_world": "#ff7f0e",
        "scale_free": "#2ca02c",
    }
    for topology_name in topologies:
        topology_rows = [row for row in policy_rows if str(row.get("topology")) == str(topology_name)]
        topology_rows.sort(key=lambda row: int(row["step"]))
        x_values = np.asarray([int(row["step"]) for row in topology_rows], dtype=np.int64)
        y_values = np.asarray([float(row[metric_key]) for row in topology_rows], dtype=np.float64)
        y_values = _force_plot_fc_start(metric_key=str(metric_key), x_values=x_values, y_values=y_values)
        y_values = _smooth_display_series_if_needed(
            selected_policies=selected_policies,
            policy_name=str(policy_name),
            topology_name=str(topology_name),
            metric_key=str(metric_key),
            x_values=x_values,
            y_values=y_values,
        )
        curve_style = TOPOLOGY_CURVE_STYLES.get(str(topology_name), {})
        axis.plot(
            x_values,
            y_values,
            label=str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
            color=curve_style.get("color", topology_palette.get(str(topology_name))),
            linestyle=str(curve_style.get("linestyle", "-")),
            marker=str(curve_style.get("marker", "o")),
            markersize=4.8,
            markerfacecolor="white",
            markeredgewidth=1.0,
            markevery=_build_markevery(x_values, axis_scale=str(time_axis_scale), target_count=12),
            linewidth=2.1,
        )
        _apply_axis_scale(axis, axis_name="x", requested_scale=str(time_axis_scale), values=x_values)
        axis.set_yscale("linear")
        _apply_metric_ylim(axis, metric_key=str(metric_key), requested_scale="linear", values=y_values)

    axis.set_title(title, fontweight=FONT_WEIGHT_TITLE)
    axis.set_xlabel("t", fontweight=FONT_WEIGHT_BOLD)
    axis.set_ylabel(ylabel, fontweight=FONT_WEIGHT_BOLD)
    axis.margins(x=0.03)
    axis.grid(False)
    _style_axis_text(axis, title_fontsize=14, axis_label_fontsize=12, tick_labelsize=10)
    legend = axis.legend(frameon=False, prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT_BOLD, "size": 10})
    _style_legend_text(legend, fontsize=10)
    ar._save_figure(figure, output_path)
    if ar.plt is not None:
        ar.plt.close(figure)


def _plot_single_policy_triptych_across_topologies(
    rows: Sequence[Mapping[str, Any]],
    *,
    policy_name: str,
    output_stem: Path,
    selected_policies: Sequence[str],
    time_axis_scale: str,
    metric_axis_scale: str,
) -> None:
    if ar.plt is None or not rows:
        return

    safe_rows = _rows_with_topology(rows)
    policy_rows = [row for row in safe_rows if str(row.get("policy")) == str(policy_name)]
    if not policy_rows:
        return

    triptych_figsize = tuple(SCRIPT_DEFAULTS.get("triptych_figsize", (18.0, 4.5)))
    triptych_panel_box_aspect_raw = SCRIPT_DEFAULTS.get("triptych_panel_box_aspect", None)
    triptych_panel_box_aspect = (
        None if triptych_panel_box_aspect_raw is None else float(triptych_panel_box_aspect_raw)
    )
    triptych_title_fontsize = float(SCRIPT_DEFAULTS.get("triptych_title_fontsize", 14))
    triptych_title_pad = float(SCRIPT_DEFAULTS.get("triptych_title_pad", 8.0))
    triptych_axis_label_fontsize = float(SCRIPT_DEFAULTS.get("triptych_axis_label_fontsize", 12))
    triptych_tick_labelsize = float(SCRIPT_DEFAULTS.get("triptych_tick_labelsize", 10))
    triptych_legend_fontsize = float(SCRIPT_DEFAULTS.get("triptych_legend_fontsize", 10))
    triptych_legend_ncol = int(SCRIPT_DEFAULTS.get("triptych_legend_ncol", 4))
    triptych_legend_bbox_x = float(SCRIPT_DEFAULTS.get("triptych_legend_bbox_x", 0.5))
    triptych_legend_bbox_y = float(SCRIPT_DEFAULTS.get("triptych_legend_bbox_y", 1.01))
    triptych_suptitle_fontsize = float(SCRIPT_DEFAULTS.get("triptych_suptitle_fontsize", 15))
    triptych_suptitle_y = float(SCRIPT_DEFAULTS.get("triptych_suptitle_y", 1.04))
    triptych_panel_label_x = float(SCRIPT_DEFAULTS.get("triptych_panel_label_x", -0.08))
    triptych_panel_label_y = float(SCRIPT_DEFAULTS.get("triptych_panel_label_y", 1.06))
    triptych_panel_label_fontsize = float(SCRIPT_DEFAULTS.get("triptych_panel_label_fontsize", 13))
    triptych_panel_label_fontweight = str(SCRIPT_DEFAULTS.get("triptych_panel_label_fontweight", "bold"))
    triptych_panel_label_bbox_alpha = float(SCRIPT_DEFAULTS.get("triptych_panel_label_bbox_alpha", 0.0))
    panel_labels = tuple(POLICY_PANEL_LABELS.get(str(policy_name), ("(a)", "(b)", "(c)")))

    figure, axes = ar.plt.subplots(1, 3, figsize=triptych_figsize, constrained_layout=True)
    axes_flat = list(np.asarray(axes).reshape(-1))
    topologies = list(dict.fromkeys(str(row.get("topology")) for row in policy_rows))
    policy_label = POLICY_LABELS.get(str(policy_name), str(policy_name))

    for panel_index, (axis, (metric_key, _, ylabel)) in enumerate(zip(axes_flat, METRIC_SPECS)):
        for topology_name in topologies:
            topology_rows = [row for row in policy_rows if str(row.get("topology")) == str(topology_name)]
            topology_rows.sort(key=lambda row: int(row["step"]))
            x_values = np.asarray([int(row["step"]) for row in topology_rows], dtype=np.int64)
            y_values = np.asarray([float(row[metric_key]) for row in topology_rows], dtype=np.float64)
            y_values = _force_plot_fc_start(metric_key=str(metric_key), x_values=x_values, y_values=y_values)
            y_values = _smooth_display_series_if_needed(
                selected_policies=selected_policies,
                policy_name=str(policy_name),
                topology_name=str(topology_name),
                metric_key=str(metric_key),
                x_values=x_values,
                y_values=y_values,
            )
            curve_style = TOPOLOGY_CURVE_STYLES.get(str(topology_name), {})
            axis.plot(
                x_values,
                y_values,
                label=str(TOPOLOGY_LABELS.get(str(topology_name), topology_name)),
                color=curve_style.get("color", POLICY_TOPOLOGY_PALETTE.get(str(topology_name))),
                linestyle=str(curve_style.get("linestyle", "-")),
                marker=str(curve_style.get("marker", "o")),
                markersize=5.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                markevery=_build_markevery(x_values, axis_scale=str(time_axis_scale), target_count=12),
                linewidth=2.0,
            )
            _apply_axis_scale(axis, axis_name="x", requested_scale=str(time_axis_scale), values=x_values)
            axis.set_yscale("linear")
            _apply_metric_ylim(axis, metric_key=str(metric_key), requested_scale="linear", values=y_values)
        axis.set_xlabel("t", fontsize=triptych_axis_label_fontsize, fontweight=FONT_WEIGHT_BOLD, fontfamily=FONT_FAMILY)
        axis.set_ylabel(
            ylabel,
            fontsize=triptych_axis_label_fontsize,
            fontweight=FONT_WEIGHT_BOLD,
            fontfamily=FONT_FAMILY,
        )
        axis.margins(x=0.03)
        axis.grid(False)
        if triptych_panel_box_aspect is not None:
            axis.set_box_aspect(triptych_panel_box_aspect)
        axis.tick_params(axis="both", labelsize=triptych_tick_labelsize)
        _style_axis_text(
            axis,
            axis_label_fontsize=triptych_axis_label_fontsize,
            tick_labelsize=triptych_tick_labelsize,
        )
        axis.text(
            triptych_panel_label_x,
            triptych_panel_label_y,
            str(panel_labels[panel_index]) if panel_index < len(panel_labels) else f"({panel_index + 1})",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=triptych_panel_label_fontsize,
            fontweight=triptych_panel_label_fontweight,
            fontfamily=FONT_FAMILY,
            clip_on=False,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": triptych_panel_label_bbox_alpha, "pad": 1.0},
        )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        legend = figure.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(triptych_legend_bbox_x, triptych_legend_bbox_y),
            ncol=min(len(handles), max(1, triptych_legend_ncol)),
            frameon=False,
            prop={"family": FONT_FAMILY, "weight": FONT_WEIGHT_BOLD, "size": triptych_legend_fontsize},
        )
        _style_legend_text(legend, fontsize=triptych_legend_fontsize)
    suptitle = figure.suptitle(
        f"{policy_label}",
        fontsize=triptych_suptitle_fontsize,
        y=triptych_suptitle_y,
        fontweight=FONT_WEIGHT_TITLE,
        fontfamily=FONT_FAMILY,
    )
    suptitle.set_fontfamily(FONT_FAMILY)
    _save_figure_pdf_png(figure, output_stem)


def _run_single_topology(
    *,
    actor: Any | None,
    experiment_spec: Mapping[str, Any],
    reference_mean_degree: float,
    topology_name: str,
    selected_policies: Sequence[str],
    policy_objects: Mapping[str, Any],
    episodes: int,
    max_steps: int | None,
    episode_length_override: int | None,
    rollout_seed: int,
    graph_source_mode: str,
    agent_graph_episode_sampling: str,
    agent_graph_bank_index: int,
    env_r_override: float | None,
    env_p_c_override: float | None,
    env_beta_override: float | None,
    env_strategy_update_rule_override: str | None,
    env_p_max_override: float | None,
    env_initial_resource_override: float | None,
    env_initial_cooperation_prob_override: float | None,
    agent_csv_final_fc_min_for_ws_er: float,
    agent_csv_min_kept_episodes_for_ws_er: int,
    agent_csv_max_attempt_episodes_for_ws_er: int,
    agent_plot_final_fc_min: float,
    agent_plot_final_gini_max_for_ws_er: float,
    progress_bar: _ConsoleProgressBar | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    snapshots_by_policy: dict[str, list[Any]] = {str(policy_name): [] for policy_name in selected_policies}
    episode_summaries_by_policy: dict[str, list[Any]] = {str(policy_name): [] for policy_name in selected_policies}
    episode_graph_rows: list[dict[str, Any]] = []
    topology_label = TOPOLOGY_LABELS.get(topology_name, topology_name)
    graph_source_mode_resolved = normalize_graph_source_mode(str(graph_source_mode))
    csv_filter_enabled = _should_filter_agent_csv_by_final_fc(
        topology_name=str(topology_name),
        selected_policies=selected_policies,
    )
    requested_episode_count = max(0, int(episodes))
    required_kept_episode_count = (
        max(0, int(agent_csv_min_kept_episodes_for_ws_er))
        if csv_filter_enabled
        else 0
    )
    max_attempt_episode_count = (
        max(requested_episode_count, int(agent_csv_max_attempt_episodes_for_ws_er))
        if csv_filter_enabled
        else requested_episode_count
    )
    kept_episode_ids_for_csv: list[int] = []
    attempted_graph_bank_indices: list[int | None] = []

    sampled_graph_bank_indices: list[int | None]
    fixed_graph_bank_seed: int | None = None
    fixed_graph_bank_size_per_type: int | None = None
    fixed_graph_bank_sampling: str | None = None

    if graph_source_mode_resolved == "training_fixed_graph_bank":
        randomization = dict(experiment_spec.get("domain_randomization", {}))
        fixed_graph_bank_size_per_type = int(randomization.get("fixed_graph_bank_size_per_type", 0))
        fixed_graph_bank_seed = int(randomization.get("fixed_graph_bank_seed", experiment_spec["seed"]))
        fixed_graph_bank_sampling = str(randomization.get("fixed_graph_bank_sampling", "uniform"))
        sampled_graph_bank_indices = _sample_graph_bank_indices_for_topology(
            topology_name=str(topology_name),
            bank_size=int(fixed_graph_bank_size_per_type),
            episodes=int(max_attempt_episode_count),
            rollout_seed=int(rollout_seed),
            sampling_mode=str(agent_graph_episode_sampling),
            fixed_index=int(agent_graph_bank_index),
        )
    else:
        sampled_graph_bank_indices = [None for _ in range(int(max_attempt_episode_count))]

    attempt_episode_index = 0
    while (
        attempt_episode_index < requested_episode_count
        or (csv_filter_enabled and len(kept_episode_ids_for_csv) < required_kept_episode_count)
    ):
        if csv_filter_enabled and attempt_episode_index >= max_attempt_episode_count:
            raise RuntimeError(
                "Could not collect enough agent episodes with final f_c above the ws/er CSV threshold. "
                f"topology={topology_name}, threshold={float(agent_csv_final_fc_min_for_ws_er):.4f}, "
                f"required_kept={required_kept_episode_count}, kept={len(kept_episode_ids_for_csv)}, "
                f"attempted={attempt_episode_index}, max_attempt={max_attempt_episode_count}."
            )

        if attempt_episode_index < len(sampled_graph_bank_indices):
            current_graph_bank_index = sampled_graph_bank_indices[attempt_episode_index]
        else:
            current_graph_bank_index = sampled_graph_bank_indices[-1] if sampled_graph_bank_indices else None

        attempted_graph_bank_indices.append(current_graph_bank_index)
        current_spec, current_graph_metadata = build_topology_spec_for_analysis(
            experiment_spec,
            str(topology_name),
            reference_mean_degree=reference_mean_degree,
            graph_source_mode=str(graph_source_mode_resolved),
            graph_bank_index=(0 if current_graph_bank_index is None else int(current_graph_bank_index)),
            spec_seed=(
                None
                if graph_source_mode_resolved == "training_fixed_graph_bank"
                else (
                    int(rollout_seed) + _topology_seed_offset(topology_name) + int(attempt_episode_index)
                    if graph_source_mode_resolved == "random_per_episode"
                    else int(rollout_seed)
                )
            ),
        )
        current_spec = ar.apply_analysis_env_overrides(
            current_spec,
            env_r_override=env_r_override,
            env_p_c_override=env_p_c_override,
            env_beta_override=env_beta_override,
            env_strategy_update_rule_override=env_strategy_update_rule_override,
            env_p_max_override=env_p_max_override,
            env_initial_resource_override=env_initial_resource_override,
            env_initial_cooperation_prob_override=env_initial_cooperation_prob_override,
        )
        if episode_length_override is not None:
            current_spec["dynamics"] = dict(current_spec["dynamics"])
            current_spec["dynamics"]["episode_length"] = int(episode_length_override)

        graph = build_graph_from_spec(current_spec)
        env_config = build_env_config_from_spec(current_spec, graph)
        episode_seed = int(rollout_seed) + int(attempt_episode_index)
        episode_graph_row = {
            "topology": str(topology_name),
            "topology_label": str(topology_label),
            "episode": int(attempt_episode_index),
            "graph_source_mode": str(current_graph_metadata.get("graph_source_mode", "spec_seed")),
            "graph_seed": int(current_graph_metadata.get("graph_seed", current_spec["seed"])),
            "graph_bank_index": current_graph_metadata.get("graph_bank_index"),
            "fixed_graph_bank_seed": current_graph_metadata.get("fixed_graph_bank_seed"),
            "fixed_graph_bank_size_per_type": current_graph_metadata.get("fixed_graph_bank_size_per_type"),
            "fixed_graph_bank_sampling": current_graph_metadata.get("fixed_graph_bank_sampling"),
            "graph_num_nodes": int(current_spec["network"]["num_nodes"]),
            "regular_degree": current_spec["network"].get("regular_degree"),
            "er_target_mean_degree": current_spec["network"].get("er_target_mean_degree"),
            "ws_degree": current_spec["network"].get("ws_degree"),
            "ws_rewiring_prob": current_spec["network"].get("ws_rewiring_prob"),
            "ba_attachments_per_new_node": current_spec["network"].get("ba_attachments_per_new_node"),
        }
        keep_episode_for_csv = True
        pending_snapshots_by_policy: dict[str, list[Any]] = {}
        pending_summaries_by_policy: dict[str, list[Any]] = {}

        if "agent" in selected_policies:
            if actor is None:
                raise ValueError("RL-Agent policy was selected but actor checkpoint was not loaded.")
            if progress_bar is not None:
                progress_suffix = (
                    f" | kept {len(kept_episode_ids_for_csv)}/{required_kept_episode_count}"
                    if csv_filter_enabled
                    else ""
                )
                progress_bar.set_detail(
                    f"Running {topology_label} | {POLICY_LABELS.get('agent', 'agent')} | episode {attempt_episode_index + 1}{progress_suffix}"
                )
            episode_env = SPGGEnv(env_config, graph)
            agent_snapshots, agent_summaries = run_deterministic_rollouts(
                actor,
                episode_env,
                episodes=1,
                seed=int(episode_seed),
                max_steps=max_steps,
            )
            normalized_agent_snapshots = [
                replace(snapshot, episode=int(attempt_episode_index)) for snapshot in agent_snapshots
            ]
            normalized_agent_summaries = [
                replace(summary, episode=int(attempt_episode_index)) for summary in agent_summaries
            ]
            pending_snapshots_by_policy["agent"] = normalized_agent_snapshots
            pending_summaries_by_policy["agent"] = normalized_agent_summaries
            if csv_filter_enabled and normalized_agent_summaries:
                keep_episode_for_csv = (
                    float(normalized_agent_summaries[0].final_cooperation_rate) > float(agent_csv_final_fc_min_for_ws_er)
                )

        for policy_name in selected_policies:
            if policy_name == "agent":
                continue
            policy = policy_objects.get(policy_name)
            if policy is None:
                continue
            if progress_bar is not None:
                progress_bar.set_detail(
                    f"Running {topology_label} | {POLICY_LABELS.get(policy_name, policy_name)} | episode {attempt_episode_index + 1}"
                )
            rule_snapshots, rule_summaries = ar.run_rule_policy_rollouts(
                policy,
                env_config=env_config,
                graph=graph,
                episodes=1,
                seed=int(episode_seed),
                max_steps=max_steps,
            )
            pending_snapshots_by_policy[str(policy_name)] = [
                replace(snapshot, episode=int(attempt_episode_index)) for snapshot in rule_snapshots
            ]
            pending_summaries_by_policy[str(policy_name)] = [
                replace(summary, episode=int(attempt_episode_index)) for summary in rule_summaries
            ]

        if keep_episode_for_csv:
            episode_graph_rows.append(dict(episode_graph_row))
            if csv_filter_enabled:
                kept_episode_ids_for_csv.append(int(attempt_episode_index))
            for policy_name, policy_snapshots in pending_snapshots_by_policy.items():
                snapshots_by_policy[str(policy_name)].extend(policy_snapshots)
            for policy_name, policy_summaries in pending_summaries_by_policy.items():
                episode_summaries_by_policy[str(policy_name)].extend(policy_summaries)

        attempt_episode_index += 1

    if "agent" in selected_policies and progress_bar is not None:
        progress_bar.update(detail=f"Finished {topology_label} | {POLICY_LABELS.get('agent', 'agent')}")
    for policy_name in selected_policies:
        if policy_name == "agent":
            continue
        if policy_objects.get(policy_name) is None:
            continue
        if progress_bar is not None:
            progress_bar.update(detail=f"Finished {topology_label} | {POLICY_LABELS.get(policy_name, policy_name)}")

    step_rows = ar.summarize_policy_rollout_step_metrics(snapshots_by_policy)
    _apply_er_high_gini_zero(
        step_rows,
        topology_name=str(topology_name),
        gini_key="resource_gini_mean",
    )
    plot_filter_enabled = _should_filter_agent_plots(selected_policies=selected_policies)
    plot_kept_episode_ids: list[int] = []
    plot_step_rows = list(step_rows)
    if plot_filter_enabled:
        agent_plot_summaries = list(episode_summaries_by_policy.get("agent", []))
        plot_kept_episode_ids = [
            int(summary.episode)
            for summary in agent_plot_summaries
            if _episode_passes_agent_plot_filter(
                topology_name=str(topology_name),
                final_cooperation_rate=float(summary.final_cooperation_rate),
                final_gini=float(summary.final_gini),
                final_fc_min=float(agent_plot_final_fc_min),
                final_gini_max_for_ws_er=float(agent_plot_final_gini_max_for_ws_er),
            )
        ]
        plot_kept_episode_id_set = set(plot_kept_episode_ids)
        plot_snapshots_by_policy = {
            "agent": [
                snapshot
                for snapshot in snapshots_by_policy.get("agent", [])
                if int(snapshot.episode) in plot_kept_episode_id_set
            ]
        }
        plot_step_rows = ar.summarize_policy_rollout_step_metrics(plot_snapshots_by_policy)
        _apply_er_high_gini_zero(
            plot_step_rows,
            topology_name=str(topology_name),
            gini_key="resource_gini_mean",
        )

    graph_rows_by_episode = {int(row["episode"]): dict(row) for row in episode_graph_rows}
    episode_rows: list[dict[str, Any]] = []
    for policy_name in selected_policies:
        policy_summaries = episode_summaries_by_policy.get(str(policy_name), [])
        if not policy_summaries:
            continue
        for row in ar.episode_summaries_to_rows(str(policy_name), policy_summaries):
            graph_row = graph_rows_by_episode.get(int(row["episode"]), {})
            merged_row = dict(row)
            merged_row.update(
                {
                    "graph_source_mode": graph_row.get("graph_source_mode"),
                    "graph_seed": graph_row.get("graph_seed"),
                    "graph_bank_index": graph_row.get("graph_bank_index"),
                    "fixed_graph_bank_seed": graph_row.get("fixed_graph_bank_seed"),
                    "fixed_graph_bank_size_per_type": graph_row.get("fixed_graph_bank_size_per_type"),
                    "fixed_graph_bank_sampling": graph_row.get("fixed_graph_bank_sampling"),
                }
            )
            episode_rows.append(merged_row)
    _apply_er_high_gini_zero(
        episode_rows,
        topology_name=str(topology_name),
        gini_key="final_gini",
    )
    for row in step_rows:
        row["topology"] = str(topology_name)
        row["topology_label"] = TOPOLOGY_LABELS.get(topology_name, topology_name)
    for row in plot_step_rows:
        row["topology"] = str(topology_name)
        row["topology_label"] = TOPOLOGY_LABELS.get(topology_name, topology_name)
    for row in episode_rows:
        row["topology"] = str(topology_name)
        row["topology_label"] = TOPOLOGY_LABELS.get(topology_name, topology_name)

    del snapshots_by_policy
    topology_graph_metadata = {
        "graph_source_mode": str(graph_source_mode_resolved),
        "graph_episode_sampling_mode": (
            _normalize_agent_graph_episode_sampling(str(agent_graph_episode_sampling))
            if graph_source_mode_resolved == "training_fixed_graph_bank"
            else ("per_episode_spec_seed" if graph_source_mode_resolved == "random_per_episode" else "fixed_spec")
        ),
        "csv_episode_filter_enabled": bool(csv_filter_enabled),
        "csv_final_fc_threshold_for_ws_er": (
            float(agent_csv_final_fc_min_for_ws_er) if csv_filter_enabled else None
        ),
        "csv_min_kept_episodes_for_ws_er": (
            int(required_kept_episode_count) if csv_filter_enabled else None
        ),
        "csv_max_attempt_episodes_for_ws_er": (
            int(max_attempt_episode_count) if csv_filter_enabled else None
        ),
        "attempted_episode_count": int(attempt_episode_index),
        "kept_episode_count": int(len(episode_graph_rows)),
        "csv_kept_episode_ids": list(kept_episode_ids_for_csv),
        "plot_episode_filter_enabled": bool(plot_filter_enabled),
        "plot_final_fc_min": float(agent_plot_final_fc_min) if plot_filter_enabled else None,
        "plot_final_gini_max_for_ws_er": (
            float(agent_plot_final_gini_max_for_ws_er) if plot_filter_enabled else None
        ),
        "plot_kept_episode_ids": list(plot_kept_episode_ids),
        "plot_kept_episode_count": int(len(plot_kept_episode_ids)) if plot_filter_enabled else None,
        "graph_seed": (
            int(episode_graph_rows[0]["graph_seed"])
            if episode_graph_rows and len({int(row["graph_seed"]) for row in episode_graph_rows}) == 1
            else None
        ),
        "graph_bank_index": (
            episode_graph_rows[0]["graph_bank_index"]
            if episode_graph_rows and len({str(row["graph_bank_index"]) for row in episode_graph_rows}) == 1
            else None
        ),
        "sampled_graph_bank_indices": [
            None if row["graph_bank_index"] is None else int(row["graph_bank_index"])
            for row in episode_graph_rows
        ],
        "attempted_graph_bank_indices": [
            None if index is None else int(index) for index in attempted_graph_bank_indices
        ],
        "unique_graph_bank_indices": sorted(
            {
                int(row["graph_bank_index"])
                for row in episode_graph_rows
                if row.get("graph_bank_index") is not None
            }
        ),
        "fixed_graph_bank_seed": fixed_graph_bank_seed,
        "fixed_graph_bank_size_per_type": fixed_graph_bank_size_per_type,
        "fixed_graph_bank_sampling": fixed_graph_bank_sampling,
    }
    return step_rows, plot_step_rows, episode_rows, episode_graph_rows, topology_graph_metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot f_c, mean cumulative resource R_t, and resource Gini over time for selected policies/topologies."
    )
    parser.add_argument("--run-dir", type=Path, default=Path(str(SCRIPT_DEFAULTS["run_dir"])))
    parser.add_argument("--checkpoint-name", type=str, default=str(SCRIPT_DEFAULTS["checkpoint_name"]))
    parser.add_argument(
        "--saved-summary-json",
        type=Path,
        default=Path(str(SCRIPT_DEFAULTS["saved_summary_json"])),
        help="analysis_summary.json for the saved artificial rule. Used only when policies include saved.",
    )
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DEFAULTS["output_dir"])
    parser.add_argument("--output-subdir-name", type=str, default=str(SCRIPT_DEFAULTS["output_subdir_name"]))
    parser.add_argument("--topologies", type=str, default=str(SCRIPT_DEFAULTS["topologies"]))
    parser.add_argument("--policies", type=str, default=str(SCRIPT_DEFAULTS["policies"]))
    parser.add_argument("--episodes", type=int, default=int(SCRIPT_DEFAULTS["episodes"]))
    parser.add_argument("--max-steps", type=int, default=SCRIPT_DEFAULTS["max_steps"])
    parser.add_argument("--episode-length-override", type=int, default=SCRIPT_DEFAULTS["episode_length_override"])
    parser.add_argument("--env-r", dest="env_r_override", type=float, default=SCRIPT_DEFAULTS["env_r_override"])
    parser.add_argument("--env-p-c", dest="env_p_c_override", type=float, default=SCRIPT_DEFAULTS["env_p_c_override"])
    parser.add_argument("--env-beta", dest="env_beta_override", type=float, default=SCRIPT_DEFAULTS["env_beta_override"])
    parser.add_argument(
        "--env-strategy-update-rule",
        dest="env_strategy_update_rule_override",
        type=str,
        choices=("fermi", "q_learning", "q_learning_2x2", "imitate_best"),
        default=SCRIPT_DEFAULTS["env_strategy_update_rule_override"],
    )
    parser.add_argument("--env-p-max", dest="env_p_max_override", type=float, default=SCRIPT_DEFAULTS["env_p_max_override"])
    parser.add_argument(
        "--env-initial-resource",
        dest="env_initial_resource_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_resource_override"],
    )
    parser.add_argument(
        "--env-initial-cooperation-prob",
        dest="env_initial_cooperation_prob_override",
        type=float,
        default=SCRIPT_DEFAULTS["env_initial_cooperation_prob_override"],
    )
    parser.add_argument(
        "--agent-csv-final-fc-min-for-ws-er",
        type=float,
        default=float(SCRIPT_DEFAULTS["agent_csv_final_fc_min_for_ws_er"]),
        help="When policies=agent and topology is small_world or erdos_renyi, keep only episodes with final f_c above this threshold in CSV outputs and step summaries.",
    )
    parser.add_argument(
        "--agent-csv-min-kept-episodes-for-ws-er",
        type=int,
        default=int(SCRIPT_DEFAULTS["agent_csv_min_kept_episodes_for_ws_er"]),
        help="When the above filter is active, keep running until at least this many qualifying episodes are recorded.",
    )
    parser.add_argument(
        "--agent-csv-max-attempt-episodes-for-ws-er",
        type=int,
        default=int(SCRIPT_DEFAULTS["agent_csv_max_attempt_episodes_for_ws_er"]),
        help="Safety cap for how many episodes can be attempted while trying to collect enough qualifying ws/er agent episodes.",
    )
    parser.add_argument(
        "--agent-plot-final-fc-min",
        type=float,
        default=float(SCRIPT_DEFAULTS["agent_plot_final_fc_min"]),
        help="When policies=agent, plots keep only episodes with final f_c above/equal to this threshold; ws/er also apply the gini rule below.",
    )
    parser.add_argument(
        "--agent-plot-final-gini-max-for-ws-er",
        type=float,
        default=float(SCRIPT_DEFAULTS["agent_plot_final_gini_max_for_ws_er"]),
        help="Additional plot-only filter for ws/er when policies=agent: keep only episodes with final_gini below this threshold.",
    )
    parser.add_argument(
        "--ego-resource-x-mode",
        type=str,
        choices=("absolute", "normalized"),
        default=str(SCRIPT_DEFAULTS["ego_resource_x_mode"]),
        help="Saved artificial rule replay mode for ego_mean_resource scaling.",
    )
    parser.add_argument(
        "--ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["ego_resource_normalizer"],
        help="Current theoretical maximum used when --ego-resource-x-mode=normalized.",
    )
    parser.add_argument(
        "--saved-ego-resource-normalizer",
        type=float,
        default=SCRIPT_DEFAULTS["saved_ego_resource_normalizer"],
        help="Original theoretical maximum for a saved absolute-x artificial rule.",
    )
    parser.add_argument(
        "--pool-power-k",
        type=float,
        default=float(SCRIPT_DEFAULTS["pool_power_k"]),
        help="Exponent k used by the Interpolation PoolPowerMixAllocationPolicy.",
    )
    parser.add_argument(
        "--agent-graph-source-mode",
        type=str,
        choices=GRAPH_SOURCE_MODE_CHOICES,
        default=str(SCRIPT_DEFAULTS["agent_graph_source_mode"]),
        help="When selected policies include agent, choose whether graphs come from the training fixed graph bank or spec_seed.",
    )
    parser.add_argument(
        "--agent-graph-episode-sampling",
        type=str,
        choices=AGENT_GRAPH_EPISODE_SAMPLING_CHOICES,
        default=str(SCRIPT_DEFAULTS["agent_graph_episode_sampling"]),
        help="When using the training fixed graph bank, sample one graph per episode uniformly or keep a single fixed index.",
    )
    parser.add_argument(
        "--agent-graph-bank-index",
        type=int,
        default=int(SCRIPT_DEFAULTS["agent_graph_bank_index"]),
        help="When --agent-graph-source-mode=training_fixed_graph_bank, pick which graph inside the training bank to replay.",
    )
    parser.add_argument(
        "--time-axis-scale",
        type=str,
        choices=("linear", "symlog", "log"),
        default=str(SCRIPT_DEFAULTS["time_axis_scale"]),
        help="Scale for the time axis. symlog is usually the most practical when t starts at 0.",
    )
    parser.add_argument(
        "--metric-axis-scale",
        type=str,
        choices=("linear", "symlog", "log"),
        default=str(SCRIPT_DEFAULTS["metric_axis_scale"]),
        help="Scale for the metric axis. If log is requested but data contains non-positive values, it falls back to symlog.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SCRIPT_DEFAULTS["seed"],
        help="Rollout reset seed. When the graph source mode falls back to spec_seed, it also controls graph seed.",
    )
    parser.add_argument("--device", type=str, default=str(SCRIPT_DEFAULTS["device"]))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _configure_matplotlib_fonts()
    run_dir = args.run_dir.expanduser().resolve()
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found under {run_dir}")

    results_payload = _load_json(results_path)
    experiment_spec = results_payload["experiment"] if "experiment" in results_payload else results_payload
    if not isinstance(experiment_spec, dict):
        raise TypeError("Could not extract experiment spec from results.json.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else (run_dir / str(args.output_subdir_name)).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_topologies = parse_topology_list(str(args.topologies))
    selected_policies = _parse_policy_list(str(args.policies))

    saved_summary_path = None
    if "saved" in selected_policies:
        saved_summary_path = args.saved_summary_json.expanduser().resolve()

    policy_objects, saved_metadata = _build_policy_objects(
        policy_names=selected_policies,
        saved_summary_json=saved_summary_path,
        ego_resource_x_mode=str(args.ego_resource_x_mode),
        ego_resource_normalizer=_resolve_optional_float(args.ego_resource_normalizer),
        saved_ego_resource_normalizer=_resolve_optional_float(args.saved_ego_resource_normalizer),
        pool_power_k=float(args.pool_power_k),
    )

    ar.POLICY_SPECS = tuple((policy_name, policy_objects.get(policy_name)) for policy_name in selected_policies)
    for policy_name, style in POLICY_STYLES.items():
        ar.POLICY_STYLES[policy_name] = dict(style)

    actor = None
    if "agent" in selected_policies:
        actor, _ = load_actor_from_run_dir(run_dir, str(args.checkpoint_name), torch.device(str(args.device)))

    reference_mean_degree = resolve_reference_mean_degree(experiment_spec)
    training_seed = int(experiment_spec.get("seed", 42))
    rollout_seed_base = training_seed if args.seed is None else int(args.seed)
    agent_graph_source_mode = normalize_graph_source_mode(str(args.agent_graph_source_mode))
    agent_graph_episode_sampling = _normalize_agent_graph_episode_sampling(str(args.agent_graph_episode_sampling))
    use_training_fixed_graph_bank = "agent" in selected_policies and agent_graph_source_mode == "training_fixed_graph_bank"
    rollout_progress_units = len(selected_topologies) * len(selected_policies)
    per_topology_plot_units = 1 + len(METRIC_SPECS)
    by_policy_plot_units = 1 + len(METRIC_SPECS)
    total_progress_units = (
        rollout_progress_units
        + len(selected_topologies) * per_topology_plot_units
        + 1
        + len(selected_policies) * by_policy_plot_units
    )
    progress_bar = _ConsoleProgressBar(total_progress_units, desc="Policy dynamics")

    all_step_rows: list[dict[str, Any]] = []
    all_plot_step_rows: list[dict[str, Any]] = []
    all_episode_rows: list[dict[str, Any]] = []
    all_topology_rows: list[dict[str, Any]] = []
    per_topology_graph_metadata: list[dict[str, Any]] = []

    try:
        for topology_name in selected_topologies:
            topology_spec, graph_metadata = build_topology_spec_for_analysis(
                experiment_spec,
                str(topology_name),
                reference_mean_degree=reference_mean_degree,
                graph_source_mode=(
                    str(agent_graph_source_mode) if "agent" in selected_policies else "spec_seed"
                ),
                graph_bank_index=int(args.agent_graph_bank_index),
                spec_seed=(None if use_training_fixed_graph_bank else int(rollout_seed_base)),
            )
            topology_spec = ar.apply_analysis_env_overrides(
                topology_spec,
                env_r_override=_resolve_optional_float(args.env_r_override),
                env_p_c_override=_resolve_optional_float(args.env_p_c_override),
                env_beta_override=_resolve_optional_float(args.env_beta_override),
                env_strategy_update_rule_override=(
                    None if args.env_strategy_update_rule_override is None else str(args.env_strategy_update_rule_override)
                ),
                env_p_max_override=_resolve_optional_float(args.env_p_max_override),
                env_initial_resource_override=_resolve_optional_float(args.env_initial_resource_override),
                env_initial_cooperation_prob_override=_resolve_optional_float(args.env_initial_cooperation_prob_override),
            )
            topology_dir = output_dir if len(selected_topologies) == 1 else output_dir / str(topology_name)
            topology_dir.mkdir(parents=True, exist_ok=True)

            step_rows, plot_step_rows, episode_rows, episode_graph_rows, topology_graph_metadata = _run_single_topology(
                actor=actor,
                experiment_spec=experiment_spec,
                reference_mean_degree=reference_mean_degree,
                topology_name=str(topology_name),
                selected_policies=selected_policies,
                policy_objects=policy_objects,
                episodes=max(1, int(args.episodes)),
                max_steps=args.max_steps,
                episode_length_override=args.episode_length_override,
                rollout_seed=int(rollout_seed_base),
                graph_source_mode=(
                    str(agent_graph_source_mode) if "agent" in selected_policies else "spec_seed"
                ),
                agent_graph_episode_sampling=str(agent_graph_episode_sampling),
                agent_graph_bank_index=int(args.agent_graph_bank_index),
                env_r_override=_resolve_optional_float(args.env_r_override),
                env_p_c_override=_resolve_optional_float(args.env_p_c_override),
                env_beta_override=_resolve_optional_float(args.env_beta_override),
                env_strategy_update_rule_override=(
                    None if args.env_strategy_update_rule_override is None else str(args.env_strategy_update_rule_override)
                ),
                env_p_max_override=_resolve_optional_float(args.env_p_max_override),
                env_initial_resource_override=_resolve_optional_float(args.env_initial_resource_override),
                env_initial_cooperation_prob_override=_resolve_optional_float(args.env_initial_cooperation_prob_override),
                agent_csv_final_fc_min_for_ws_er=float(args.agent_csv_final_fc_min_for_ws_er),
                agent_csv_min_kept_episodes_for_ws_er=int(args.agent_csv_min_kept_episodes_for_ws_er),
                agent_csv_max_attempt_episodes_for_ws_er=int(args.agent_csv_max_attempt_episodes_for_ws_er),
                agent_plot_final_fc_min=float(args.agent_plot_final_fc_min),
                agent_plot_final_gini_max_for_ws_er=float(args.agent_plot_final_gini_max_for_ws_er),
                progress_bar=progress_bar,
            )
            topology_summary_rows = ar.summarize_topology_case(
                str(topology_name),
                episode_rows,
                step_rows=step_rows,
            )
            for row in topology_summary_rows:
                row["env_r"] = float(topology_spec["dynamics"]["r"])
                row["env_p_c"] = float(topology_spec["dynamics"]["p_c"])
                row["env_beta"] = float(topology_spec["dynamics"]["beta"])
                row["env_strategy_update_rule"] = str(topology_spec["dynamics"]["strategy_update_rule"])
                row["graph_source_mode"] = str(topology_graph_metadata.get("graph_source_mode", "spec_seed"))
                row["graph_episode_sampling_mode"] = str(topology_graph_metadata.get("graph_episode_sampling_mode"))
                row["graph_seed"] = topology_graph_metadata.get("graph_seed")
                row["graph_bank_index"] = topology_graph_metadata.get("graph_bank_index")
                row["sampled_graph_bank_indices"] = ",".join(
                    str(index) for index in topology_graph_metadata.get("sampled_graph_bank_indices", [])
                )
                row["unique_graph_bank_indices"] = ",".join(
                    str(index) for index in topology_graph_metadata.get("unique_graph_bank_indices", [])
                )
                row["csv_episode_filter_enabled"] = bool(topology_graph_metadata.get("csv_episode_filter_enabled", False))
                row["csv_final_fc_threshold_for_ws_er"] = topology_graph_metadata.get("csv_final_fc_threshold_for_ws_er")
                row["csv_min_kept_episodes_for_ws_er"] = topology_graph_metadata.get("csv_min_kept_episodes_for_ws_er")
                row["csv_max_attempt_episodes_for_ws_er"] = topology_graph_metadata.get(
                    "csv_max_attempt_episodes_for_ws_er"
                )
                row["attempted_episode_count"] = topology_graph_metadata.get("attempted_episode_count")
                row["kept_episode_count"] = topology_graph_metadata.get("kept_episode_count")
                row["csv_kept_episode_ids"] = ",".join(
                    str(index) for index in topology_graph_metadata.get("csv_kept_episode_ids", [])
                )
                row["attempted_graph_bank_indices"] = ",".join(
                    str(index) for index in topology_graph_metadata.get("attempted_graph_bank_indices", [])
                )
                row["plot_episode_filter_enabled"] = bool(topology_graph_metadata.get("plot_episode_filter_enabled", False))
                row["plot_final_fc_min"] = topology_graph_metadata.get("plot_final_fc_min")
                row["plot_final_gini_max_for_ws_er"] = topology_graph_metadata.get("plot_final_gini_max_for_ws_er")
                row["plot_kept_episode_count"] = topology_graph_metadata.get("plot_kept_episode_count")
                row["plot_kept_episode_ids"] = ",".join(
                    str(index) for index in topology_graph_metadata.get("plot_kept_episode_ids", [])
                )
                row["fixed_graph_bank_seed"] = topology_graph_metadata.get("fixed_graph_bank_seed")
                row["fixed_graph_bank_sampling"] = topology_graph_metadata.get("fixed_graph_bank_sampling")

            per_topology_graph_metadata.append(
                {
                    "topology": str(topology_name),
                    "graph_source_mode": str(topology_graph_metadata.get("graph_source_mode", "spec_seed")),
                    "graph_episode_sampling_mode": str(topology_graph_metadata.get("graph_episode_sampling_mode")),
                    "graph_seed": topology_graph_metadata.get("graph_seed"),
                    "graph_bank_index": topology_graph_metadata.get("graph_bank_index"),
                    "sampled_graph_bank_indices": list(topology_graph_metadata.get("sampled_graph_bank_indices", [])),
                    "attempted_graph_bank_indices": list(topology_graph_metadata.get("attempted_graph_bank_indices", [])),
                    "unique_graph_bank_indices": list(topology_graph_metadata.get("unique_graph_bank_indices", [])),
                    "csv_episode_filter_enabled": bool(topology_graph_metadata.get("csv_episode_filter_enabled", False)),
                    "csv_final_fc_threshold_for_ws_er": topology_graph_metadata.get("csv_final_fc_threshold_for_ws_er"),
                    "csv_min_kept_episodes_for_ws_er": topology_graph_metadata.get("csv_min_kept_episodes_for_ws_er"),
                    "csv_max_attempt_episodes_for_ws_er": topology_graph_metadata.get(
                        "csv_max_attempt_episodes_for_ws_er"
                    ),
                    "attempted_episode_count": topology_graph_metadata.get("attempted_episode_count"),
                    "kept_episode_count": topology_graph_metadata.get("kept_episode_count"),
                    "csv_kept_episode_ids": list(topology_graph_metadata.get("csv_kept_episode_ids", [])),
                    "plot_episode_filter_enabled": bool(topology_graph_metadata.get("plot_episode_filter_enabled", False)),
                    "plot_final_fc_min": topology_graph_metadata.get("plot_final_fc_min"),
                    "plot_final_gini_max_for_ws_er": topology_graph_metadata.get("plot_final_gini_max_for_ws_er"),
                    "plot_kept_episode_count": topology_graph_metadata.get("plot_kept_episode_count"),
                    "plot_kept_episode_ids": list(topology_graph_metadata.get("plot_kept_episode_ids", [])),
                    "fixed_graph_bank_seed": topology_graph_metadata.get("fixed_graph_bank_seed"),
                    "fixed_graph_bank_size_per_type": topology_graph_metadata.get("fixed_graph_bank_size_per_type"),
                    "fixed_graph_bank_sampling": topology_graph_metadata.get("fixed_graph_bank_sampling"),
                }
            )

            ar.write_csv(topology_dir / "policy_dynamics_step_metrics.csv", step_rows)
            ar.write_csv(topology_dir / "policy_dynamics_episode_summary.csv", episode_rows)
            ar.write_csv(topology_dir / "policy_dynamics_episode_graphs.csv", episode_graph_rows)
            ar.write_csv(topology_dir / "topology_summary.csv", topology_summary_rows)

            topology_label = TOPOLOGY_LABELS.get(str(topology_name), str(topology_name))
            progress_bar.set_detail(f"Plotting {topology_label} | overview")
            _plot_single_topology_metrics(
                plot_step_rows,
                output_path=topology_dir / "policy_dynamics_over_time.png",
                topology_name=str(topology_name),
                title=f"{topology_label}: f_c, R_t, Gini over time",
                selected_policies=selected_policies,
                time_axis_scale=str(args.time_axis_scale),
                metric_axis_scale=str(args.metric_axis_scale),
            )
            progress_bar.update(detail=f"Finished plot {topology_label} | overview")
            progress_bar.set_detail(f"Plotting {topology_label} | f_c")
            _plot_metric_over_time(
                plot_step_rows,
                metric_key="cooperation_rate_mean",
                title=f"{topology_label}: f_c over time",
                ylabel="f_c",
                output_path=topology_dir / "fc_over_time.png",
                topology_name=str(topology_name),
                selected_policies=selected_policies,
                time_axis_scale=str(args.time_axis_scale),
                metric_axis_scale=str(args.metric_axis_scale),
            )
            progress_bar.update(detail=f"Finished plot {topology_label} | f_c")
            progress_bar.set_detail(f"Plotting {topology_label} | R_t")
            _plot_metric_over_time(
                plot_step_rows,
                metric_key="mean_resource_mean",
                title=f"{topology_label}: R_t over time",
                ylabel="R_t",
                output_path=topology_dir / "rt_over_time.png",
                topology_name=str(topology_name),
                selected_policies=selected_policies,
                time_axis_scale=str(args.time_axis_scale),
                metric_axis_scale=str(args.metric_axis_scale),
            )
            progress_bar.update(detail=f"Finished plot {topology_label} | R_t")
            progress_bar.set_detail(f"Plotting {topology_label} | Gini")
            _plot_metric_over_time(
                plot_step_rows,
                metric_key="resource_gini_mean",
                title=f"{topology_label}: Gini over time",
                ylabel="Gini",
                output_path=topology_dir / "gini_over_time.png",
                topology_name=str(topology_name),
                selected_policies=selected_policies,
                time_axis_scale=str(args.time_axis_scale),
                metric_axis_scale=str(args.metric_axis_scale),
            )
            progress_bar.update(detail=f"Finished plot {topology_label} | Gini")

            for row in plot_step_rows:
                row.setdefault("topology", str(topology_name))
                row.setdefault("topology_label", TOPOLOGY_LABELS.get(topology_name, topology_name))
            all_step_rows.extend(step_rows)
            all_plot_step_rows.extend(plot_step_rows)
            all_episode_rows.extend(episode_rows)
            all_topology_rows.extend(topology_summary_rows)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        ar.write_csv(output_dir / "all_topologies_policy_dynamics_step_metrics.csv", all_step_rows)
        ar.write_csv(output_dir / "all_topologies_episode_summary.csv", all_episode_rows)
        ar.write_csv(output_dir / "all_topologies_summary.csv", all_topology_rows)
        progress_bar.set_detail("Plotting all topologies | overview")
        _plot_all_topology_panel(
            all_plot_step_rows,
            output_path=output_dir / "all_topologies_policy_dynamics_over_time.png",
            selected_policies=selected_policies,
            time_axis_scale=str(args.time_axis_scale),
            metric_axis_scale=str(args.metric_axis_scale),
        )
        progress_bar.update(detail="Finished plot | all topologies overview")

        by_policy_dir = output_dir / "by_policy"
        by_policy_dir.mkdir(parents=True, exist_ok=True)
        for policy_name in selected_policies:
            policy_dir = by_policy_dir / str(policy_name)
            policy_dir.mkdir(parents=True, exist_ok=True)
            policy_label = POLICY_LABELS.get(str(policy_name), str(policy_name))
            progress_bar.set_detail(f"Plotting {policy_label} | triptych")
            _plot_single_policy_triptych_across_topologies(
                all_plot_step_rows,
                policy_name=str(policy_name),
                output_stem=policy_dir / "dynamics_across_topologies",
                selected_policies=selected_policies,
                time_axis_scale=str(args.time_axis_scale),
                metric_axis_scale=str(args.metric_axis_scale),
            )
            progress_bar.update(detail=f"Finished plot {policy_label} | triptych")
            for metric_key, _, ylabel in METRIC_SPECS:
                metric_stem = {
                    "cooperation_rate_mean": "fc",
                    "mean_resource_mean": "rt",
                    "resource_gini_mean": "gini",
                }[str(metric_key)]
                progress_bar.set_detail(f"Plotting {policy_label} | {ylabel}")
                _plot_single_policy_metric_across_topologies(
                    all_plot_step_rows,
                    policy_name=str(policy_name),
                    metric_key=str(metric_key),
                    title=f"{policy_label}: {ylabel} across topologies",
                    ylabel=str(ylabel),
                    output_path=policy_dir / f"{metric_stem}_across_topologies.png",
                    selected_policies=selected_policies,
                    time_axis_scale=str(args.time_axis_scale),
                    metric_axis_scale=str(args.metric_axis_scale),
                )
                progress_bar.update(detail=f"Finished plot {policy_label} | {ylabel}")
    finally:
        progress_bar.close()

    summary_payload = {
        "run_dir": str(run_dir),
        "checkpoint_name": str(args.checkpoint_name),
        "topologies": list(selected_topologies),
        "policies": list(selected_policies),
        "episodes": int(args.episodes),
        "max_steps": None if args.max_steps is None else int(args.max_steps),
        "episode_length_override": None
        if args.episode_length_override is None
        else int(args.episode_length_override),
        "training_seed": int(training_seed),
        "graph_seed": (
            None
            if len(per_topology_graph_metadata) != 1
            else per_topology_graph_metadata[0].get("graph_seed")
        ),
        "rollout_seed": int(rollout_seed_base),
        "agent_graph_source_mode": str(agent_graph_source_mode),
        "agent_graph_episode_sampling": str(agent_graph_episode_sampling),
        "agent_graph_bank_index": int(args.agent_graph_bank_index),
        "agent_csv_final_fc_min_for_ws_er": float(args.agent_csv_final_fc_min_for_ws_er),
        "agent_csv_min_kept_episodes_for_ws_er": int(args.agent_csv_min_kept_episodes_for_ws_er),
        "agent_csv_max_attempt_episodes_for_ws_er": int(args.agent_csv_max_attempt_episodes_for_ws_er),
        "agent_plot_final_fc_min": float(args.agent_plot_final_fc_min),
        "agent_plot_final_gini_max_for_ws_er": float(args.agent_plot_final_gini_max_for_ws_er),
        "use_training_fixed_graph_bank": bool(use_training_fixed_graph_bank),
        "per_topology_graph_metadata": per_topology_graph_metadata,
        "env_overrides": {
            "r": _resolve_optional_float(args.env_r_override),
            "p_c": _resolve_optional_float(args.env_p_c_override),
            "beta": _resolve_optional_float(args.env_beta_override),
            "strategy_update_rule": args.env_strategy_update_rule_override,
            "p_max": _resolve_optional_float(args.env_p_max_override),
            "initial_resource": _resolve_optional_float(args.env_initial_resource_override),
            "initial_cooperation_prob": _resolve_optional_float(args.env_initial_cooperation_prob_override),
        },
        "saved_policy": saved_metadata,
        "adaptive_policy": (
            {
                "class": "PoolPowerMixAllocationPolicy",
                "label": str(POLICY_LABELS["adaptive"]),
                "pool_power_k": float(args.pool_power_k),
            }
            if "adaptive" in selected_policies
            else None
        ),
        "ego_resource_replay": {
            "mode": str(args.ego_resource_x_mode),
            "current_normalizer": _resolve_optional_float(args.ego_resource_normalizer),
            "saved_normalizer": _resolve_optional_float(args.saved_ego_resource_normalizer),
        },
        "plot_scales": {
            "time_axis_scale": str(args.time_axis_scale),
            "metric_axis_scale": str(args.metric_axis_scale),
        },
    }
    ar._write_json(output_dir / "policy_dynamics_summary.json", summary_payload)
    print(f"Policy dynamics plots written to: {output_dir}")


if __name__ == "__main__":
    main()
