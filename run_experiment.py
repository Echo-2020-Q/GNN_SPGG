from __future__ import annotations

"""
单文件实验入口。

使用方式：
1. 直接在这个文件顶部改参数。
2. 运行：
   python run_experiment.py

设计目标：
- 把论文实验常改的参数集中在一个文件中。
- 支持单次实验，也支持一次跑多组不同网络/模型参数。
- 支持两类可视化：
  1. 微观图：每个时间步一张网络快照图，观察节点、边、策略和资源状态。
  2. 宏观图：同一实验过程中统计量随时间的变化曲线。
"""

from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from collections import deque
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict, replace
from datetime import datetime
import json
from math import ceil
from multiprocessing.connection import wait
import os
from pathlib import Path
from statistics import mean
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from Project1 import (
    RewardConfig,
    SPGGConfig,
    SPGGEnv,
    gini_coefficient,
    make_barabasi_albert_graph,
    make_erdos_renyi_graph,
    make_grid_graph,
    make_random_regular_graph,
    make_watts_strogatz_graph,
)
from Project1.policies.rule_based import (
    ConstantMixAllocationPolicy,
    PoolPowerMixAllocationPolicy,
    ProportionalContributionPolicy,
    UniformAllocationPolicy,
)

PROJECT_ROOT = Path(__file__).resolve().parent


class _TeeStream:
    def __init__(self, *streams: Any):
        self._streams = streams
        self.encoding = getattr(streams[0], "encoding", "utf-8") if streams else "utf-8"

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def fileno(self) -> int:
        return self._streams[0].fileno()


@contextmanager
def experiment_console_log_context(spec: Mapping[str, Any], output_dir: Path):
    output = spec.get("output", {})
    if not bool(output.get("save_console_log", True)):
        yield None
        return

    log_filename = str(output.get("console_log_filename", "train.log"))
    log_path = output_dir / log_filename
    with log_path.open("a", encoding="utf-8", buffering=1) as handle:
        tee_stream = _TeeStream(sys.stdout, handle)
        tee_error_stream = _TeeStream(sys.stderr, handle)
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(tee_stream))
            stack.enter_context(redirect_stderr(tee_error_stream))
            yield log_path


# =============================================================================
# 基础实验配置：如果你只想跑一组实验，通常只改这里
# =============================================================================

BASE_EXPERIMENT = {
    # 这次实验的名字。
    # 它会决定输出目录名、结果 JSON 中的实验名，也方便你区分不同实验。
    "experiment_name": "spgg_GNN_12workers_100length_Fermi_Teacher-Return_Critic",

    # 全局随机种子。
    # 用来控制网络生成、环境初始化、批量实验中的随机性。
    # 如果你想复现结果，就固定成一个整数。
    "seed": 42,

    # 运行模式：
    # - "uniform"      ：人工规则，均匀分配
    # - "proportional" ：人工规则，按贡献比例分配
    # - "constant_mix" ：人工规则，常数混合分配
    # - "pool_power_mix"：人工规则，pool 驱动混合分配
    # - "gnn_train"    ：训练 GNN-RL 分配器，再做训练后评估
    "run_mode": "gnn_train",

    # ---------------------------
    # 网络配置
    # ---------------------------
    "network": {
        # 网络类型：
        # - "regular"     ：随机正则网络
        # - "erdos_renyi" ：ER 随机网络
        # - "small_world" ：WS 小世界网络
        # - "scale_free"  ：BA 无标度网络
        # - "grid"        ：二维格子，仅保留做对照
        "type": "regular",

        # 节点总数 n。
        # 对 regular / erdos_renyi / small_world / scale_free 都使用这个值。
        # 对 grid 网络，这个值会被 grid_rows * grid_cols 取代。
        "num_nodes": 50,

        # regular 网络参数：每个节点的度。
        # 例如 degree=4 表示每个节点都恰好连 4 条边。
        "regular_degree": 4,

        # ER 网络参数：目标平均度。
        # 如果这里不是 None，程序会自动按
        #   p = er_target_mean_degree / (num_nodes - 1)
        # 计算 ER 连边概率。
        # 例如目标平均度为 4：
        # - n=100  时，p≈4/99   ≈ 0.0404
        # - n=2500 时，p≈4/2499 ≈ 0.0016
        "er_target_mean_degree": 4.0,

        # ER 网络参数：手动指定任意两点之间连边的概率。
        # 仅当 er_target_mean_degree 为 None 时使用。
        "er_edge_prob": None,

        # WS 小世界网络参数：初始规则环网络的度。
        # 当前实现要求是偶数。
        "ws_degree": 4,

        # WS 小世界网络参数：重连概率。
        # 越大，随机捷径越多。
        "ws_rewiring_prob": 0.10,

        # BA 无标度网络参数：每个新节点接到几个已有节点上。
        # 一般用 2、3、4 这类小整数。
        "ba_attachments_per_new_node": 2,

        # grid 网络参数：只有 type == "grid" 时使用。
        "grid_rows": 10,
        "grid_cols": 10,
        "grid_periodic": False,
    },

    # ---------------------------
    # 环境动力学参数
    # ---------------------------
    "dynamics": {
        # alpha：控制合作投入随资源增长的线性变化强度。
        # 对应：
        #   e_i,t = min(R_i,t, max(0, (d_i+1) + alpha * (R_i,t - (d_i+1))))
        "alpha": 0.5,

        # r：资源池放大参数。
        # 对应：
        #   G_i,t = min((1 + r) * P_i,t, P_max)
        "r": 0.75,

        # 公共池容量上限模式：
        # - "constant" ：固定常数上限，使用 p_max
        # - "dynamic"  ：动态上限，使用
        #                P_max,i,t = p_c * N_c,i,t * rho_c,i,t
        #                          = p_c * N_c,i,t^2 / (d_i + 1)
        "p_mode": "dynamic" ,

        # P_max：资源池增长后的饱和上限。
        # 仅当 p_mode == "constant" 时使用。
        "p_max": 250.0,#40

        # 动态公共池容量系数 p_c。
        # 仅当 p_mode == "dynamic" 时使用。
        # 例如在规则四邻居图中，若希望全合作时动态容量上限约等于 100，
        # 可取 p_c = 20，因为此时 d_i + 1 = 5，故 P_max = 20 * 5 = 100。
        "p_c": 50.0,

        # 资源自然消耗模式：
        # - "fixed"            ：每轮固定消耗
        # - "proportional"     ：按当前资源按比例消耗
        # - "piecewise_linear" ：先固定消耗，超过阈值后再按比例增加
        "resource_consumption_mode": "piecewise_linear",

        # 固定消耗项的计算模式：
        # - "constant"      ：所有节点用同一个固定常数
        # - "degree_scaled" ：固定项 = 倍数 × 节点度 d_i
        "resource_consumption_fixed_mode": "degree_scaled",

        # 固定消耗constant项。
        # - fixed            ：每轮直接消耗这个值
        # - piecewise_linear ：作为基础固定消耗
        # 仅当 resource_consumption_fixed_mode == "constant" 时生效。
        "resource_consumption_fixed": 4, #暂时设置等于平均度

        # 度比例固定消耗倍数。
        # 当 resource_consumption_fixed_mode == "degree_scaled" 时：
        # fixed_term_i = resource_consumption_degree_multiplier * degree_i
        "resource_consumption_degree_multiplier": 1.0,

        # 比例消耗系数。
        # - proportional     ：consumption = rate * resources
        # - piecewise_linear ：consumption = fixed + rate * max(resources - threshold, 0)
        "resource_consumption_rate": 0.05, #0.1

        # 分段线性消耗阈值。
        # 仅当 resource_consumption_mode == "piecewise_linear" 时使用。
        "resource_consumption_threshold": 4.0, #50

        # 个体策略更新规则：
        # - "fermi"        ：同步 Fermi 更新
        # - "q_learning"   ：每个节点使用二动作无状态 Q-learning 更新
        # - "q_learning_2x2"：每个节点使用 2状态×2动作 Q-learning，
        #                     状态=自己上一轮动作，动作=本轮选 C/D
        # - "imitate_best" ：最优邻居模仿 / Best-takes-over
        "strategy_update_rule": "fermi",

        # beta：同步 Fermi 更新的选择强度。
        # 仅当 strategy_update_rule == "fermi" 时使用。
        # 越大，节点越偏向模仿高收益邻居。
        "beta": 1.0,

        # 以下参数仅当 strategy_update_rule == "q_learning"
        # 或 "q_learning_2x2" 时使用。
        "q_learning_rate": 0.1,
        "q_learning_discount": 0.1,
        "q_learning_epsilon": 0.05,
        "q_learning_initial_value": 0.0,

        # 每个 episode 的时间步上限。
        # 到达这个步数后，本 episode 结束。
        "episode_length": 100, #150 10000

        # 所有节点统一的初始资源。
        "initial_resource": 20.0,#10

        # 初始名义合作概率。
        # 如果不单独指定节点策略向量，reset 时每个节点按这个概率初始化为合作。
        "initial_cooperation_prob": 0.5,
    },

    # ---------------------------
    # planner 奖励参数
    # ---------------------------
    "reward": {
        # 环境返回给 RL planner 的标量奖励：
        # reward =
        #   lambda_payoff * mean(payoff)
        #   + lambda_cooperation * mean(next_actual_cooperation)
        #   + lambda_total_resource * mean(next_resources)
        #   - lambda_collapse * mean(next_resources < degree + 1)
        #   - lambda_gini * gini(next_resources)
        # 当前这组默认系数下，实际 reward = mean(payoff)。
        #   # 为了方便正则化，我们将这些标量奖励规范化，每项为10，规范化的分母为对应项的理论稳态最大值，

        # 平均净收益项的权重。
        "lambda_payoff":0.0,

        # 下一时刻实际合作比例项的权重。
        "lambda_cooperation": 5.0,

        # 下一时刻全局平均资源项的权重。
        # 单步使用 mean(next_resources)；跨时间平均后对应评估里的 mean_resource 口径。
        #Pc=50，Pmax=250,α=0.5，τ=0.1，\bar_{d}=4, R_M=370.83
        "lambda_total_resource": 15/371.0,

        # 下一时刻“低资源/塌缩个体比例”惩罚项的权重。
        # 口径：mean(next_resources < degree + 1)。
        "lambda_collapse": 1,

        # Gini 不平等惩罚项的权重。
        "lambda_gini": 5.0,

        # Gini 分母的极小修正项，通常不需要改。
        "epsilon": 1e-8,
    },

    # ---------------------------
    # GNN 策略参数
    # ---------------------------
    "gnn": {
        # 节点嵌入隐藏维度。
        # 这是全图两层 GraphNet backbone 的统一隐藏维度。
        "hidden_dim":128,

        # ego-local tiny GraphNet 的隐藏维度。
        # 设为 None 时，自动回退到 hidden_dim。
        "local_hidden_dim": 64,

        # 局部 score readout MLP 的隐藏维度。
        # 设为 None 时，自动回退到 local_hidden_dim。
        "score_hidden_dim":64,

        # actor 内部全局 value head 的隐藏维度。
        # 这个头当前在 TD3 主训练中不是核心训练对象，但模型结构里仍然保留。
        # 设为 None 时，自动回退到 hidden_dim。
        "critic_hidden_dim": None,

        # 消息传递层数。
        # 当前实现固定为 2，不建议改成其他值，否则会触发配置校验错误。
        "num_message_passing_layers": 2,

        # 局部 softmax 温度参数 tau。
        # 当前已真实接入 actor 的局部 softmax：alpha_i = softmax(score_i / temperature)。
        # 越小分配越尖锐，越大分配越平滑。
        "temperature": 1,
    },

    # ---------------------------
    # 训练参数
    # ---------------------------
    "training": {
        # 训练总环境步数（不含 warm-up）。
        # 这是推荐直接调的“训练阶段全局总步数”入口，单位与 warmup / eval_interval 保持一致。
        # 程序内部会按：
        #   total_updates = ceil((warmup_env_steps + total_env_steps) / (num_workers * effective_steps_per_update))
        # 自动换算成 update 次数。
        "total_env_steps": 20_000_000,#episode需要用总的steps除以episode_length=150

        # warm-up 总环境步数。
        # 这是所有 worker 共享的“全局 warm-up 总步数”，不会再乘 num_workers。
        # 它不计入上面的 total_env_steps，会额外附加在训练前面。
        # 设为 0 时，不仅没有 warm-up 行为采样，Actor 的 BC 约束也会一起关闭。
        "warmup_env_steps": 307_200,

        # 每隔多少个全局环境步做一次评估。
        # 程序内部会自动换算成 update 间隔。
        "eval_interval_env_steps": 2_000_000,

        # 兼容旧配置的回退项：只有 total_env_steps 为 None 时才使用。
        "total_updates": None,

        # 每个 worker 在每次训练迭代中收集多少个环境步。
        # 这是“每个 worker 每个 update”的采样粒度，不是全局总步数。
        "steps_per_update": 400,  # episode_length=100

        # 是否强制把每次训练迭代的采样长度设为一个完整 episode。
        # 为 True 时，会忽略上面的 steps_per_update，改为使用 dynamics.episode_length。
        # 对当前演化博弈设定，这意味着每个 worker 每个 update 都会先采完整个演化过程。
        # 这里不需要你手动再乘 num_workers，程序会自动根据：
        #   global_env_steps_per_update = num_workers * effective_steps_per_update
        # 去换算 total_updates / eval_interval_updates。
        "use_episode_length_as_steps_per_update": False,

        # 折扣因子 gamma。
        "gamma": 0.99,

        # 共享学习率。
        # 如果 actor_lr / critic_lr 为 None，就回退到这里。
        # 当前默认作为指数退火的初始学习率。
        "learning_rate": 1e-4,

        # Actor 学习率。
        "actor_lr": None,

        # Critic 学习率。
        "critic_lr": None,

        # 学习率调度类型：
        # - "constant"          ：固定学习率
        # - "exponential_decay" ：指数退火，lr = max(lr_final, lr_init * decay_rate^(step / decay_steps))
        "lr_schedule_type": "exponential_decay",

        # 指数退火的最小学习率下界。
        "lr_final": 1e-5,

        # 指数退火的 decay_rate。
        "lr_decay_rate": 0.05,

        # 指数退火的 decay_steps。
        # 当 learner 的训练步数增加到这个量级时，学习率会衰减一个 decay_rate 的量级。
        "lr_decay_steps": 10_000,

        # 是否在训练过程中保存 checkpoint。
        "save_checkpoints": True,

        # checkpoint 保存间隔。
        # 例如 100 表示每 100 个 update 保存一次。
        "checkpoint_interval": 500,

        # 是否额外保存最终 checkpoint。
        "save_final_checkpoint": True,

        # 是否按 eval_return_mean 额外保存一份当前最佳 checkpoint。
        # 只有在该 update 触发评估时才会比较和更新。
        "save_best_checkpoint": True,

        # checkpoint 模式：
        # - "lightweight" ：只保存 learner / optimizer / 历史 / update 进度，文件小很多，
        #                   但恢复时不会找回 replay buffer 和 worker 当前环境状态。
        # - "full_resume" ：额外保存 replay buffer / worker / env 运行态，可做真正的无损续训，
        #                   但文件会很大，尤其 replay_capacity 较大时。
        "checkpoint_mode": "lightweight",

        # 从已有 checkpoint 恢复训练。
        # 设为 None 表示从头开始训练；
        # 设为字符串路径时，会恢复 actor / critics / optimizer / 训练历史 / update 进度。
        # 若 checkpoint_mode == "full_resume" 保存出的文件，还会同时恢复 replay buffer /
        # workers / 当前环境状态，实现更接近无损的续训。
        "resume_from_checkpoint": None,

        # Actor optimizer 的 L2 weight decay。
        "actor_weight_decay": 0.0,

        # Critic optimizer 的 L2 weight decay。
        "critic_weight_decay": 0.0,

        # Actor loss 里的分配熵正则权重。
        # 这是训练目标里的辅助项，不是环境 reward。
        # > 0 会鼓励分配更平滑、更不那么尖锐。
        "actor_entropy_coef":  5e-4,

        # Actor loss 里的 valid logits L2 正则权重。
        # > 0 会抑制 logits 绝对值过大，减轻策略过尖。
        "actor_logit_l2_coef": 5e-5,

        # TD3 twin critics 的状态编码器隐藏维度。
        # 对应 GraphActionCritic 里 state encoder 的 hidden_dim。
        # 设为 None 时，自动回退到 gnn.hidden_dim。
        "critic_state_hidden_dim": 64,

        # TD3 twin critics 的局部动作编码器隐藏维度。
        # 对应每个 pool 的 local action encoder MLP 宽度。
        # 设为 None 时，自动回退到 gnn.hidden_dim。
        "critic_action_hidden_dim":64,

        # TD3 twin critics 的 pool token 编码器隐藏维度。
        # 对应 pool-level encoder MLP 宽度。
        # 设为 None 时，自动回退到 gnn.hidden_dim。
        "critic_pool_hidden_dim": 64,

        # TD3 twin critics 的最终 Q head 隐藏维度。
        # 对应输出标量 Q(s, a) 前的最后一个 MLP 宽度。
        # 设为 None 时，自动回退到 gnn.hidden_dim。
        "critic_q_hidden_dim": 64,

        # replay buffer 容量。
        "replay_capacity": 1_000_000, # 300k 步=0.3M 步

        # replay 采样策略：
        # - "fifo"                     ：原始单一 FIFO replay
        # - "topology_stratified_mixed": 按拓扑分层，并混合 recent / long_term / demo 三类样本
        "replay_strategy": "topology_stratified_mixed",

        # replay 中允许显式分层的拓扑类型集合。
        # 设为 None 时，自动继承 domain_randomization.network_types；
        # 若 domain_randomization 未开启，则自动退回到 ["fixed"]。
        "replay_topology_names": None,

        # topology_stratified_mixed 下，recent 缓冲区容量占比。
        # recent 用普通 FIFO，负责跟踪当前训练分布。
        "replay_recent_fraction": 0.50,

        # topology_stratified_mixed 下，long_term 缓冲区容量占比。
        # long_term 用 reservoir 保留全历史代表样本，负责减轻遗忘。
        "replay_long_term_fraction": 0.35,

        # topology_stratified_mixed 下，demo 缓冲区容量占比。
        # 当前 demo 缓冲区专门保留 pool_power_mix 产生的样本。
        "replay_demo_fraction": 0.15,

        # demo 缓冲区收集哪类 demo 行为源。
        # 当前只支持 "pool_power_mix"。
        "replay_demo_behavior_source": "pool_power_mix",

        # learner 每次更新采样的 batch 大小。
        "batch_size": 256,

        # learner 内部做图张量化时的微批大小。
        # 这是为了避免把整个 replay batch 一次性展开成 dense [B, N, N, H] 图张量后显存占用过高。
        # 它不改变 replay sample 的 batch_size，只影响 actor / critic 在 GPU 上分几小块做前向与反向。
        "graph_batch_chunk_size": 48,

        # 兼容旧配置的回退项：只有 warmup_env_steps 为 None 时才使用。
        "warmup_steps": None,

        # warm-up 行为模式：
        # - "random_only"   ：只用随机 logits + softmax
        # - "heuristic_mix" ：在启发式与随机 logits 之间按权重混合
        "warmup_behavior_mode": "heuristic_mix",

        # warm-up 行为源的采样粒度：
        # - "per_episode" ：每个 episode 固定选一种 warm-up 行为
        # - "per_step"    ：每一步都重新采样 warm-up 行为
        "warmup_selection_granularity": "per_episode",

        # warm-up 中均匀分配启发式的采样权重。
        "warmup_uniform_prob": 0.05,

        # warm-up 中按局部贡献比例分配启发式的采样权重。
        "warmup_proportional_prob": 0.05,

        # warm-up 中常数混合启发式的采样权重。
        # 行为形式：omega * uniform + (1 - omega) * proportional。
        "warmup_constant_mix_prob": 0.05,

        # warm-up 中 pool 驱动混合启发式的采样权重。
        # 行为形式：omega_i * uniform + (1 - omega_i) * proportional，
        # 其中 omega_i = (clip(pool_raw_i, 0, p_max) / p_max) ^ k。
        "warmup_pool_power_mix_prob": 0.8,

        # warm-up 中随机 logits 行为的采样权重。
        "warmup_random_logits_prob": 0.05,

        # 常数混合启发式中的 omega。
        "warmup_constant_mix_omega": 0.5,

        # pool 驱动混合启发式中的幂指数 k。
        "warmup_pool_power_k": 19.0,

        # 启发式 warm-up 动作在 logits 空间追加的噪声标准差。
        # 只对 uniform / proportional / mixed 这类启发式行为生效。
        "warmup_logit_noise_std": 0.15,

        # 启发式 warm-up logits 噪声的截断范围。
        "warmup_logit_noise_clip": 0.25,

        # warm-up 期间 actor 不做 max-Q 更新，只做 demo 行为克隆。
        "freeze_actor_q_during_warmup": True,

        # warm-up 期间 actor 的行为克隆损失系数。
        "warmup_actor_bc_coef": 1.0,

        # warm-up 结束后，继续在 demo 样本上保留一个较轻的行为克隆锚点。
        "actor_demo_bc_coef": 0.6,

        # warm-up 结束后，demo BC 系数线性衰减到 0 的总 rollout 步数比例。
        # 例如 0.50 表示到总 rollout 步数的 20% 时衰减到 0。
        "actor_demo_bc_decay_end_fraction": 0.50,

        # 是否在正式 online TD3 训练前，先执行 demo 预训练三阶段：
        # 1) 固定专家轨迹收集
        # 2) actor BC 预训练
        # 3) critic 预训练
        "demo_pretrain_enabled": True,

        # demo 预收集的总环境步数。
        # 这些步数不会计入 online warm-up，也不会计入 total_env_steps。
        "demo_collection_env_steps": 100_000,

        # demo 预收集使用的行为源。
        # 当前 v1 只支持 "pool_power_mix"。
        "demo_collection_behavior_source": "pool_power_mix",

        # demo 预收集时是否启用 domain randomization。
        # True 时默认覆盖 domain_randomization.network_types 的全域拓扑；
        # False 时退回到当前 base env 的固定图。
        "demo_collection_use_domain_randomization": True,

        # demo 预收集允许覆盖的拓扑类型子集。
        # 设为 None 时，自动继承 domain_randomization.network_types。
        "demo_collection_network_types": None,

        # demo 预收集运行时：
        # - "parallel_cpu"  : 单独启动并行 CPU teacher workers，保留多进程吞吐，避开 local CUDA rollout worker
        # - "isolated_cpu"  : 单进程 CPU teacher collector，最稳但最慢
        # - "reuse_workers" : 复用当前 rollout workers，速度最快，但会重新走本地 CUDA/并行 rollout 路径
        "demo_collection_runtime": "parallel_cpu",

        # demo 预收集结束后，纯 BC 预训练 actor 的更新次数。
        "actor_bc_pretrain_updates": 3000,

        # actor BC 预训练完成后，critic 仅用 demo transition 做 TD 回归的更新次数。
        "critic_pretrain_updates": 3000,

        # demo 预训练的 batch 大小。
        # 设为 None 时，回退到 training.batch_size。
        "demo_pretrain_batch_size": 512,

        # 可选：把 demo 预收集得到的 replay batch 落盘成一个 .pt 文件。
        # 设为 None 时，只放进 replay，不额外保存到磁盘。
        "demo_dataset_save_path": None,

        # 是否在 demo pretrain 完成后，额外保存一个可直接续训的专用 checkpoint。
        # 这个 checkpoint 始终按 full_resume 口径保存，包含 replay buffer。
        "save_demo_pretrain_checkpoint": False,

        # demo pretrain 专用 checkpoint 的文件名。
        "demo_pretrain_checkpoint_name": "demo_pretrained.pt",

        # 是否在 demo pretrain 完成并保存 checkpoint 后直接退出，不进入 online training。
        "stop_after_demo_pretrain": False,

        # demo critic 预训练使用的目标类型：
        # - "n_step" ：teacher 轨迹自身的纯 n-step return
        # - "mc"     ：teacher 轨迹自身的整局 Monte Carlo return
        "demo_critic_pretrain_target_mode": "n_step",

        # 当 demo_critic_pretrain_target_mode="n_step" 时使用的步长 n。
        "demo_critic_pretrain_n_step": 10,

        # warm-up 结束后，是否让 teacher 先和 actor 混合接管，而不是立刻纯 actor。
        "teacher_takeover_enabled": True,

        # teacher takeover 阶段使用的启发式行为源。
        # 当前只支持 "pool_power_mix"。
        "teacher_takeover_behavior_source": "pool_power_mix",

        # teacher takeover 起始概率。
        # 例如 0.8 表示刚脱离 warm-up 时，80% 概率仍由 teacher 接管。
        "teacher_takeover_start_prob": 0.8,

        # teacher takeover 结束概率。
        "teacher_takeover_end_prob": 0.0,

        # teacher takeover 线性衰减到 end_prob 的总 rollout 步数比例。
        "teacher_takeover_decay_end_fraction": 0.30,

        # online 阶段 actor 的 Q 项初始系数。
        # 早期让 actor loss 以 BC 为主，Q 为辅。
        "online_actor_q_coef_initial": 0.2,

        # online 阶段 actor 的 Q 项最终系数。
        "online_actor_q_coef_final": 1.0,

        # actor Q 系数从 initial 线性升到 final 的总 rollout 步数比例。
        "online_actor_q_coef_ramp_end_fraction": 0.30,

        # critic 损失类型：
        # - "mse"
        # - "huber"
        "critic_loss_type": "huber",

        # Huber critic loss 的 delta。
        "critic_huber_delta": 1.0,

        # Actor / Critic 的梯度裁剪范数。
        # 设为 None 表示关闭裁剪。
        "actor_grad_clip_norm": 5.0,
        "critic_grad_clip_norm": 5.0,

        # 每隔多少个外层训练迭代做一次 learner 更新。
        "train_every": 1,

        # 每个外层训练迭代做多少次梯度更新。
        "gradient_steps_per_update": 24,

        # TD3 delayed policy update 频率。
        "policy_delay": 4,

        # 将实际合作率过低的 transition 视为塌缩样本。
        "replay_collapse_fc_threshold": 0.10,

        # replay 采样时允许的塌缩样本最大占比。
        # 在 topology_stratified_mixed 模式下，这个上限会落实到“每个拓扑单独采样时”。
        "replay_max_collapse_sample_ratio": 0.20,

        # target network soft update 系数 tau。
        "tau": 0.005,

        # rollout 时在 logits 空间加噪声的标准差。
        "rollout_logit_noise_std": 0.30,

        # rollout 时 logits 噪声的截断范围。
        "rollout_logit_noise_clip": 0.50,

        # rollout 噪声衰减系数。
        "rollout_noise_decay": 0.9995,

        # target policy smoothing 使用的 logits 噪声标准差。
        "target_logit_noise_std": 0.10,

        # target policy smoothing 的 logits 噪声截断范围。
        "target_logit_noise_clip": 0.25,

        # 真实并行的 rollout worker 进程数。
        # num_workers=1 表示单进程采样；num_workers>1 会启动多进程并行采样。
        # learner 仍然在主进程单点更新。
        "num_workers": 12,

        # 每个 worker 内同时维护多少个环境实例。
        # 这些环境会在 worker 内做 batched actor forward，但总环境步数语义保持不变：
        # steps_per_update 仍然表示“每个 worker 每个 update 一共采多少步”，而不是每个 env 各采多少步。
        # 固定 num_nodes 时，4 或 8 往往比继续堆 worker 更值得先试。
        "num_envs_per_worker": 4,

        # learner 参数同步到 worker 的间隔。
        "worker_sync_interval": 1,

        # 是否把下一轮 rollout collect 和当前轮 learner update 做第一阶段双缓冲重叠。
        # 只在 num_workers > 1 时生效。
        "overlap_rollout_and_update": True,

        # 并行 worker RPC 的超时时间（秒）。
        # 包括 actor 参数同步、collect 回传、state_dict/load_state_dict 等控制消息。
        "worker_rpc_timeout_seconds": 12000.0,

        # rollout actor 的推理设备。
        # 默认建议先用 "cpu"，让 worker 侧保持最简单、最稳定的本地 CPU rollout。
        # 如果后面要专门做 rollout 推理加速，再改成某张 GPU 或多张 GPU 列表。
        # learner 训练设备仍由下面的 device 单独控制。
        "rollout_device": "cuda:2",

        # rollout 推理模式：
        # - "local"       ：每个 worker 自己持有 actor，并在本地前向
        # - "centralized" ：worker 只做环境推进，把 observation 发给集中式 batched
        #                   inference server，由 rollout_device 上的 actor 批量前向
        # 默认先用 local，避免多进程 worker 全部去排一个中央 inference server。
        "rollout_inference_mode": "local",

        # 中央 rollout inference server 为了攒 batch 最多额外等待多少毫秒。
        # 设为 0 表示只处理当前已到达的请求，不额外等待。
        "rollout_inference_batch_timeout_ms": 2.0,

        # learner 训练设备：cpu / cuda / cuda:0 等。
        # 这只控制主进程里的 actor/critic 训练，不控制 rollout worker 的推理设备。
        "device": "cuda:3",

        # 并行 rollout worker 进程内的 PyTorch CPU 线程数。
        # 仅对多进程 worker 生效，用于减少小图前向时的线程争抢。
        # 设为 1 通常更稳；设为 None 则保持 PyTorch 默认值。
        "rollout_num_threads": 1,

        # 评估时资源低于该阈值视为 collapse。
        "collapse_resource_threshold": 1e-6,

        # 兼容旧配置的回退项：只有 eval_interval_env_steps 为 None 时才使用。
        "eval_interval": None,

        # 每次评估多少个 episode。
        "eval_episodes": 8,
    },

    # ---------------------------
    # GNN-RL 训练时的环境随机化参数
    # ---------------------------
    "domain_randomization": {
        # 是否启用 domain randomization。
        # 关闭时，所有 worker 都使用当前 spec 对应的固定图和固定环境参数。
        "enabled": True,

        # worker 采样时允许出现的网络类型集合。
        "network_types": ["regular", "erdos_renyi", "small_world", "scale_free"],

        # 与上面 network_types 一一对应的采样权重。
        # 设为 None 时，默认对这些网络类型均匀采样。
        "network_type_weights": None,

        # 允许采样的节点数集合。
        "num_nodes_choices": [50],

        # regular 图可采样的度集合。
        "regular_degree_choices": [4],

        # ER 图可采样的目标平均度集合。
        "er_mean_degree_choices": [4.0],

        # WS 图可采样的度集合。
        "ws_degree_choices": [4],

        # WS 图可采样的重连概率集合。
        "ws_rewiring_choices": [0.10],

        # BA 图可采样的新节点连接数集合。
        "ba_attachment_choices": [2],

        # 初始资源随机区间 [low, high]。
        # 设为 None 时，固定使用 dynamics.initial_resource。
        "initial_resource_range": None,

        # 初始合作概率随机区间 [low, high]。
        # 设为 None 时，固定使用 dynamics.initial_cooperation_prob。
        "initial_cooperation_prob_range": None,

        # alpha 随机区间 [low, high]。
        # 设为 None 时，固定使用 dynamics.alpha。
        "alpha_range": None,

        # r 随机区间 [low, high]。
        # 设为 None 时，固定使用 dynamics.r。
        "r_range": None,

        # p_max 随机区间 [low, high]。
        # 设为 None 时，固定使用 dynamics.p_max。
        "p_max_range": None,
    },

    # ---------------------------
    # 周期评估环境配置
    # ---------------------------
    "evaluation": {
        # 是否使用自定义评估环境族。
        # 关闭时，训练过程中的周期评估使用当前 spec 对应的固定 eval_env。
        "use_custom_env_families": True,

        # 训练结束后，是否先自动加载 checkpoints/best_eval.pt，
        # 再执行 post_training_evaluation。
        # 关闭时，post_training_evaluation 使用训练结束时内存中的最终模型。
        "use_best_checkpoint_for_post_training_eval": True,

        # 每个条目定义一个评估环境族。
        # 对随机图模型，同一条目下不同评估 episode 会用不同图 seed 重新采样图，
        # 但网络类型和超参数保持固定。
        # 周期评估时，每个环境族都会跑 training.eval_episodes 个 episode。
        "env_families": [
            {
                "network_type": "regular",
                "num_nodes": 50,
                "regular_degree": 4,
            },
            {
                "network_type": "erdos_renyi",
                "num_nodes": 50,
                "er_target_mean_degree": 4.0,
            },
            {
                "network_type": "small_world",
                "num_nodes": 50,
                "ws_degree": 4,
                "ws_rewiring_prob": 0.10,
            },
            {
                "network_type": "scale_free",
                "num_nodes": 50,
                "ba_attachments_per_new_node": 2,
            },
        ],
    },

    # ---------------------------
    # GNN-RL 训练课程学习配置
    # ---------------------------
    "curriculum": {
        # 是否启用按训练进度逐步扩展网络类型的 curriculum。
        "enabled": True,

        # 当前先只支持按训练 update 进度切阶段。
        # 后续如果需要，再加按 f_c / eval 指标收敛触发的模式。
        "mode": "update_steps",

        # 各阶段按 total_updates 的比例切分。
        # 下面这组默认含义是：
        # - 前 40%：只训练 / 评估 regular
        # - 接着 30%：训练 / 评估 regular + scale_free
        # - 最后 30%：训练 / 评估 regular + scale_free + erdos_renyi + small_world
        "stages": [
            {
                "label": "regular__small_world",
                "portion": 0.15,
                "train_network_types": ["regular","small_world","erdos_renyi"],
                "train_network_type_weights": [0.4, 0.4,0.2],
                "eval_network_types": ["regular", "small_world","erdos_renyi"],
            },
            {
                "label": "regular__scale_free__erdos_renyi",
                "portion": 0.40,
                "train_network_types": ["regular", "small_world", "erdos_renyi","scale_free"],
                "train_network_type_weights": [0.25, 0.25, 0.25,0.25],
                "eval_network_types": ["regular", "small_world", "erdos_renyi","scale_free"],
            },
            {
                "label": "all_topologies",
                "portion": 0.45,
                "train_network_types": ["regular", "scale_free", "erdos_renyi", "small_world"],
                "train_network_type_weights": [0.2, 0.5, 0.15, 0.15],
                "eval_network_types": ["regular", "scale_free", "erdos_renyi", "small_world"],
            },
        ],
    },

    # ---------------------------
    # 规则模式 / 评估模式运行长度
    # ---------------------------
    "rollout": {
        # 在人工规则模式下，一共跑多少个 episode。
        # 例如 5 表示从头 reset 5 次，每次都跑到 episode_length。
        "episodes": 5,

        # 在 gnn_train 模式下，训练结束后用训练好的策略再评估多少个 episode。
        "post_training_eval_episodes": 5,
    },

    # ---------------------------
    # 可视化参数
    # ---------------------------
    "visualization": {
        # 是否生成微观网络快照图。
        # 开启后，每个选中的 episode 都会在每个时间步保存一张图。
        "enable_micro_snapshots": False,

        # 是否生成宏观时间序列图。
        # 开启后，每个选中的 episode 都会保存一张统计量随时间变化的折线图。
        "enable_macro_timeseries": True,

        # 想要可视化哪些 episode。
        # 用 1-based 编号，例如 [1, 3] 表示只可视化第 1 和第 3 个 episode。
        # 设为 [] 表示所有 episode 都可视化。
        "episodes_to_visualize": [1],

        # 微观图每隔多少步保存一张。
        # 1 表示每步都存。
        # 5 表示 t=0,5,10,15,... 才存，能显著减少图片数量。
        "frame_stride": 1,

        # 网络布局方式：
        # - "spring"   ：力导向布局，适合一般网络
        # - "circular" ：圆形布局，简单稳定
        # - "grid"     ：只有 grid 网络建议用，其他网络不要用
        "layout": "spring",

        # spring 布局迭代次数。
        # 越大布局通常越舒展，但也越慢。
        "spring_iterations": 80,

        # 微观图里节点颜色映射到哪个节点级参数。
        # 可选典型值：
        # - "x_actual"   ：实际执行策略（合作/背叛）
        # - "x_nominal"  ：名义策略
        # - "resources"  ：累计资源
        # - "pool_raw"   ：原始池子资源
        # - "pool_grown" ：增长后池子资源
        # - "investment" ：个体投入
        # - "income"     ：个体收入
        # - "consumption": 个体消耗
        # - "payoff"     ：个体净收益
        "node_color_metric": "x_actual",

        # 微观图里节点大小映射到哪个节点级参数。
        # 常见搭配：
        # - 颜色看策略，大小看资源
        # - 颜色看资源，大小看资源
        "node_size_metric": "resources",

        # 节点中心显示哪个节点级参数。
        # 设为 None、"" 时表示不在节点中心显示数值。
        # 常见值：
        # - "resources"  ：显示当前个体资源
        # - "investment" ：显示当前投入
        # - "income"     ：显示当前收入
        # - "consumption": 显示当前消耗
        "node_value_metric": "resources",

        # 节点中心数值保留的小数位数。
        "node_value_decimals": 1,

        # 是否在微观图中标出节点编号。
        # 如果同时显示中心数值，编号会绘制在节点上方。
        # 节点很多时建议关掉，否则会很乱。
        "label_nodes": True,

        # 宏观图要画哪些统计量。
        # 横轴统一是时间 t，纵轴是这些统计量的值。
        "macro_metrics": [
            "actual_cooperation_rate",
            "mean_resource",
            "mean_pool_grown",
            "mean_consumption",
            "mean_payoff",
            "gini",
        ],

        # 画图保存分辨率。
        "save_dpi": 160,
    },

    # ---------------------------
    # TensorBoard 日志配置
    # ---------------------------
    "tensorboard": {
        # 是否启用 TensorBoard 标量日志。
        "enabled": True,

        # TensorBoard 事件文件相对于实验输出目录的子目录名。
        # 实际运行时会在该目录下自动再加一层 MMDD_HHMMSS 时间戳子目录。
        "subdir": "tensorboard",

        # SummaryWriter 的 flush_secs。
        "flush_secs": 30,

        # 是否把完整实验配置写成文本到 TensorBoard。
        "write_config_text": True,

        # 是否把静态图/环境/模型参数写成 step=0 的标量。
        "write_static_scalars": True,

        # 是否在控制台输出低频进度日志。
        # 打印当前环境步数、总环境步数、耗时和预估剩余时间。
        "console_progress_logs": True,

        # 控制台进度日志的 update 间隔。
        "console_progress_interval": 1,

        # 是否在控制台输出低频的最近训练统计块。
        # 开启后，会按最近若干个 update 的窗口均值打印 loss / reward / lr / 行为占比等。
        "console_training_logs": True,

        # 控制台最近训练统计块的 update 间隔。
        "console_log_interval": 1,

        # 控制台最近训练统计块使用的滑动窗口大小。
        "console_recent_window_updates": 10,
    },

    # ---------------------------
    # 输出与保存参数
    # ---------------------------
    "output": {
        # 所有输出结果的根目录。
        # 每个实验会在下面单独建一个子目录。
        "root_dir": "outputs/Pool_dynamic",#"outputs",

        # 是否保存结果 JSON。
        "save_results_json": True,

        # 是否把微观快照图真正落盘。
        # 如果 enable_micro_snapshots=True 但这里 False，则不保存文件。
        "save_micro_snapshots": True,

        # 是否把宏观时间序列图真正落盘。
        "save_macro_timeseries": True,

        # 是否把控制台 stdout/stderr 同步写入实验目录下的日志文件。
        "save_console_log": True,

        # 控制台日志文件名。
        "console_log_filename": "train.log",
    },
}


# =============================================================================
# 批量实验配置：如果你想“一次跑多组参数”，就在这里加条目
# =============================================================================
#
# 使用方式：
# 1. 保持 BASE_EXPERIMENT 作为共同基础配置
# 2. 在下面每个 dict 里只写你想覆盖的字段
# 3. 程序会自动把它们逐个和 BASE_EXPERIMENT 合并，然后依次运行
#
# 覆盖规则示例：
# {
#     "experiment_name": "regular_d4_alpha05",
#     "network": {"type": "regular", "regular_degree": 4},
#     "dynamics": {"alpha": 0.5},
#     "visualization": {"enable_macro_timeseries": True},
# }
#
BATCH_EXPERIMENTS = [
    # {
    #     "experiment_name": "regular_d4_prop_r15_q_learning",
    #     "network": {"type": "regular", "regular_degree": 4},
    #     "run_mode": "proportional",
    #     "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    # },
    #     {
    #     "experiment_name": "regular_d4_uniform_r15_q_learning",
    #     "network": {"type": "regular", "regular_degree": 4},
    #     "run_mode": "uniform",
    #     "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    # },
    #     {
    #     "experiment_name": "ba_m2_proportional_r15_q_learning",
    #     "network": {"type": "scale_free", "ba_attachments_per_new_node": 2},
    #     "run_mode": "proportional",
    #     "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    # },
    #     {
    #     "experiment_name": "ba_m2_uniform_r15_q_learning",
    #     "network": {"type": "scale_free", "ba_attachments_per_new_node": 2},
    #     "run_mode": "uniform",
    #     "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    # },
    #     {
    #     "experiment_name": "er_k4_proportional_r15_q_learning",
    #     "network": {"type": "erdos_renyi", "er_target_mean_degree": 4.0},
    #     "run_mode": "proportional",
    #     "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    # },
    #     {
    #     "experiment_name": "er_k4_uniform_r15_q_learning",
    #     "network": {"type": "erdos_renyi", "er_target_mean_degree": 4.0},
    #     "run_mode": "uniform",
    #     "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    # },
    #     {
    #     "experiment_name": "ws_k4_p01_proportional_r15_q_learning",
    #     "network": {"type": "small_world", "ws_degree": 4, "ws_rewiring_prob": 0.1},
    #     "run_mode": "proportional",
    #     "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    # },
    #     {
    #     "experiment_name": "ws_k4_p01_uniform_r15_q_learning",
    #     "network": {"type": "small_world", "ws_degree": 4, "ws_rewiring_prob": 0.1},
    #     "run_mode": "uniform",
    #     "dynamics": {"r": 1,"strategy_update_rule": "q_learning"},
    # },
]


# =============================================================================
# 参数扫描配置：启用后会忽略上面的 BATCH_EXPERIMENTS，自动生成扫描实验
# =============================================================================
SCAN_EXPERIMENT = {
    "enabled": False, #是否启用参数扫描实验，启用后会忽略上面的 BATCH_EXPERIMENTS，自动生成扫描实验
    "name": "3_31_dynamics_scan_mixed", #扫描实验的名字，会在output_root_dir下面生成一个同名子目录
    "output_root_dir": "outputs/Pool_dynamic_scan_mixed",#"outputs/scan",
    "parallel": True,
    "max_workers": 32,#自己的电脑为16核
    "r_values": [0.25, 0.75, 1.25, 1.75, 2.25],
    "resource_consumption_rate": [0.05],
    "num_nodes": [50],
    "network_types": ["regular", "erdos_renyi", "small_world", "scale_free"],
    "resource_consumption_modes": ["piecewise_linear"],#["fixed", "proportional", "piecewise_linear"],
    "resource_consumption_fixed_modes": ["constant", "degree_scaled"],
    "strategy_update_rules": ["q_learning", "fermi"],
    "warmup_constant_mix_omega": [0.5],
    "warmup_pool_power_k": [15 ,17.0, 18.0, 19.0, 20, 22, 25],
    "run_mode": ["proportional", "uniform","constant_mix","pool_power_mix"],#["proportional", "uniform"],
}

def deep_update(base: Dict[str, Any], overrides: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _format_float_token(value: float) -> str:
    token = format(float(value), "g")
    return token.replace("-", "m").replace(".", "p")


def _resolve_er_edge_prob(network: Mapping[str, Any]) -> float:
    num_nodes = int(network["num_nodes"])
    target_mean_degree = network.get("er_target_mean_degree")
    if target_mean_degree is not None:
        target_mean_degree = float(target_mean_degree)
        if target_mean_degree < 0.0:
            raise ValueError("er_target_mean_degree must be non-negative.")
        max_mean_degree = max(num_nodes - 1, 0)
        if target_mean_degree > max_mean_degree:
            raise ValueError(
                "er_target_mean_degree must be <= num_nodes - 1."
            )
        if num_nodes <= 1:
            return 0.0
        return target_mean_degree / float(num_nodes - 1)

    edge_prob = network.get("er_edge_prob")
    if edge_prob is None:
        raise ValueError("ER network requires er_target_mean_degree or er_edge_prob.")
    edge_prob = float(edge_prob)
    if not 0.0 <= edge_prob <= 1.0:
        raise ValueError("er_edge_prob must be in [0, 1].")
    return edge_prob


def _er_parameter_summary(network: Mapping[str, Any]) -> str:
    edge_prob = _resolve_er_edge_prob(network)
    target_mean_degree = network.get("er_target_mean_degree")
    if target_mean_degree is not None:
        return "target_mean_degree={0}, edge_prob={1:.6g}".format(
            float(target_mean_degree),
            edge_prob,
        )
    expected_mean_degree = edge_prob * max(int(network["num_nodes"]) - 1, 0)
    return "edge_prob={0:.6g}, expected_mean_degree={1:.6g}".format(
        edge_prob,
        expected_mean_degree,
    )


def _network_variant_label(network: Mapping[str, Any]) -> str:
    network_type = network["type"]
    if network_type == "regular":
        return "regular_d{0}".format(network["regular_degree"])
    if network_type == "erdos_renyi":
        if network.get("er_target_mean_degree") is not None:
            return "er_k{0}".format(_format_float_token(network["er_target_mean_degree"]))
        return "er_p{0}".format(_format_float_token(_resolve_er_edge_prob(network)))
    if network_type == "small_world":
        return "ws_k{0}_p{1}".format(
            network["ws_degree"],
            _format_float_token(network["ws_rewiring_prob"]),
        )
    if network_type == "scale_free":
        return "ba_m{0}".format(network["ba_attachments_per_new_node"])
    if network_type == "grid":
        return "grid_{0}x{1}".format(network["grid_rows"], network["grid_cols"])
    raise ValueError("Unsupported network type: {0}".format(network_type))


def _network_override_for_type(base_network: Mapping[str, Any], network_type: str) -> Dict[str, Any]:
    override: Dict[str, Any] = {"type": network_type, "num_nodes": base_network["num_nodes"]}
    if network_type == "regular":
        override["regular_degree"] = base_network["regular_degree"]
    elif network_type == "erdos_renyi":
        if "er_target_mean_degree" in base_network:
            override["er_target_mean_degree"] = base_network["er_target_mean_degree"]
        if "er_edge_prob" in base_network:
            override["er_edge_prob"] = base_network["er_edge_prob"]
    elif network_type == "small_world":
        override["ws_degree"] = base_network["ws_degree"]
        override["ws_rewiring_prob"] = base_network["ws_rewiring_prob"]
    elif network_type == "scale_free":
        override["ba_attachments_per_new_node"] = base_network["ba_attachments_per_new_node"]
    elif network_type == "grid":
        override["grid_rows"] = base_network["grid_rows"]
        override["grid_cols"] = base_network["grid_cols"]
        override["grid_periodic"] = base_network["grid_periodic"]
    else:
        raise ValueError("Unsupported network type in scan: {0}".format(network_type))
    return override


def _consumption_variants(scan_config: Mapping[str, Any]) -> List[tuple[str, Optional[str]]]:
    variants: List[tuple[str, Optional[str]]] = []
    for mode in scan_config["resource_consumption_modes"]:
        if mode == "proportional":
            variants.append((mode, None))
            continue
        for fixed_mode in scan_config["resource_consumption_fixed_modes"]:
            variants.append((mode, fixed_mode))
    return variants


def _consumption_variant_label(mode: str, fixed_mode: Optional[str]) -> str:
    if mode == "proportional":
        return "proportional"
    if fixed_mode is None:
        raise ValueError("fixed_mode is required for mode {0}".format(mode))
    return "{0}_{1}".format(mode, fixed_mode)


def _scan_run_modes(scan_config: Mapping[str, Any], base_run_mode: str) -> List[str]:
    configured_run_mode = scan_config.get("run_mode", base_run_mode)
    if isinstance(configured_run_mode, str):
        run_modes = [configured_run_mode]
    else:
        run_modes = [str(item) for item in configured_run_mode]

    if not run_modes:
        raise ValueError("SCAN_EXPERIMENT['run_mode'] must contain at least one run mode.")
    valid_run_modes = {"uniform", "proportional", "constant_mix", "pool_power_mix"}
    invalid_run_modes = [item for item in run_modes if item not in valid_run_modes]
    if invalid_run_modes:
        raise ValueError(
            "Scan mode currently supports only rule-based run_mode values {0}. Invalid values: {1}".format(
                sorted(valid_run_modes),
                invalid_run_modes,
            )
        )
    return run_modes


def _scan_num_nodes(scan_config: Mapping[str, Any], base_num_nodes: int) -> List[int]:
    configured_num_nodes = scan_config.get("num_nodes", base_num_nodes)
    if isinstance(configured_num_nodes, int):
        num_nodes_values = [configured_num_nodes]
    else:
        num_nodes_values = [int(item) for item in configured_num_nodes]

    if not num_nodes_values:
        raise ValueError("SCAN_EXPERIMENT['num_nodes'] must contain at least one value.")
    if any(item <= 0 for item in num_nodes_values):
        raise ValueError("SCAN_EXPERIMENT['num_nodes'] values must be positive.")
    return num_nodes_values


def _scan_constant_mix_omegas(scan_config: Mapping[str, Any], base_omega: float) -> List[float]:
    configured_values = scan_config.get("warmup_constant_mix_omega", base_omega)
    if isinstance(configured_values, (int, float)):
        omega_values = [float(configured_values)]
    else:
        omega_values = [float(item) for item in configured_values]

    if not omega_values:
        raise ValueError("SCAN_EXPERIMENT['warmup_constant_mix_omega'] must contain at least one value.")
    if any((item < 0.0) or (item > 1.0) for item in omega_values):
        raise ValueError("SCAN_EXPERIMENT['warmup_constant_mix_omega'] values must be in [0, 1].")
    return omega_values


def _scan_pool_power_ks(scan_config: Mapping[str, Any], base_k: float) -> List[float]:
    configured_values = scan_config.get("warmup_pool_power_k", base_k)
    if isinstance(configured_values, (int, float)):
        power_k_values = [float(configured_values)]
    else:
        power_k_values = [float(item) for item in configured_values]

    if not power_k_values:
        raise ValueError("SCAN_EXPERIMENT['warmup_pool_power_k'] must contain at least one value.")
    if any(item < 0.0 for item in power_k_values):
        raise ValueError("SCAN_EXPERIMENT['warmup_pool_power_k'] values must be non-negative.")
    return power_k_values


def _scan_run_mode_parameter_variants(
    scan_config: Mapping[str, Any],
    base_training: Mapping[str, Any],
    run_mode: str,
) -> List[Dict[str, Any]]:
    if run_mode == "constant_mix":
        variants: List[Dict[str, Any]] = []
        for omega in _scan_constant_mix_omegas(scan_config, float(base_training["warmup_constant_mix_omega"])):
            variants.append(
                {
                    "training_overrides": {
                        "warmup_constant_mix_omega": float(omega),
                    },
                    "scan_tags": {
                        "warmup_constant_mix_omega": float(omega),
                        "warmup_pool_power_k": None,
                        "run_mode_param_name": "warmup_constant_mix_omega",
                        "run_mode_param_value": float(omega),
                        "run_mode_param_label": "omega{0}".format(_format_float_token(omega)),
                    },
                }
            )
        return variants

    if run_mode == "pool_power_mix":
        variants = []
        for power_k in _scan_pool_power_ks(scan_config, float(base_training["warmup_pool_power_k"])):
            variants.append(
                {
                    "training_overrides": {
                        "warmup_pool_power_k": float(power_k),
                    },
                    "scan_tags": {
                        "warmup_constant_mix_omega": None,
                        "warmup_pool_power_k": float(power_k),
                        "run_mode_param_name": "warmup_pool_power_k",
                        "run_mode_param_value": float(power_k),
                        "run_mode_param_label": "k{0}".format(_format_float_token(power_k)),
                    },
                }
            )
        return variants

    return [
        {
            "training_overrides": {},
            "scan_tags": {
                "warmup_constant_mix_omega": None,
                "warmup_pool_power_k": None,
                "run_mode_param_name": None,
                "run_mode_param_value": None,
                "run_mode_param_label": None,
            },
        }
    ]


def build_scan_experiment_specs() -> List[Dict[str, Any]]:
    base = deepcopy(BASE_EXPERIMENT)
    scan = SCAN_EXPERIMENT
    run_modes = _scan_run_modes(scan, str(base["run_mode"]))
    num_nodes_values = _scan_num_nodes(scan, int(base["network"]["num_nodes"]))
    specs: List[Dict[str, Any]] = []

    for run_mode in run_modes:
        for run_mode_variant in _scan_run_mode_parameter_variants(scan, base["training"], run_mode):
            run_mode_param_label = run_mode_variant["scan_tags"]["run_mode_param_label"]
            run_mode_label = (
                "{0}_{1}".format(run_mode, run_mode_param_label)
                if run_mode_param_label is not None
                else run_mode
            )
            for strategy_update_rule in scan["strategy_update_rules"]:
                for resource_consumption_mode, resource_consumption_fixed_mode in _consumption_variants(scan):
                    consumption_label = _consumption_variant_label(
                        resource_consumption_mode,
                        resource_consumption_fixed_mode,
                    )
                    for num_nodes in num_nodes_values:
                        base_network = deepcopy(base["network"])
                        base_network["num_nodes"] = int(num_nodes)
                        for network_type in scan["network_types"]:
                            network_override = _network_override_for_type(base_network, network_type)
                            network_label = "n{0}_{1}".format(num_nodes, _network_variant_label(network_override))
                            for r_value in scan["r_values"]:
                                experiment_name = "{0}__{1}__{2}__{3}__r{4}".format(
                                    network_label,
                                    run_mode_label,
                                    consumption_label,
                                    strategy_update_rule,
                                    _format_float_token(r_value),
                                )
                                spec = deep_update(
                                    deepcopy(base),
                                    {
                                        "experiment_name": experiment_name,
                                        "run_mode": run_mode,
                                        "network": network_override,
                                        "training": dict(run_mode_variant["training_overrides"]),
                                        "dynamics": {
                                            "r": float(r_value),
                                            "resource_consumption_mode": resource_consumption_mode,
                                            "strategy_update_rule": strategy_update_rule,
                                        },
                                        "output": {
                                            "root_dir": scan["output_root_dir"],
                                        },
                                    },
                                )
                                if resource_consumption_fixed_mode is not None:
                                    spec["dynamics"]["resource_consumption_fixed_mode"] = resource_consumption_fixed_mode
                                spec["scan_tags"] = {
                                    "scan_name": scan["name"],
                                    "run_mode": run_mode,
                                    "run_mode_param_name": run_mode_variant["scan_tags"]["run_mode_param_name"],
                                    "run_mode_param_value": run_mode_variant["scan_tags"]["run_mode_param_value"],
                                    "run_mode_param_label": run_mode_param_label,
                                    "warmup_constant_mix_omega": run_mode_variant["scan_tags"]["warmup_constant_mix_omega"],
                                    "warmup_pool_power_k": run_mode_variant["scan_tags"]["warmup_pool_power_k"],
                                    "num_nodes": int(num_nodes),
                                    "network_label": network_label,
                                    "consumption_label": consumption_label,
                                    "resource_consumption_mode": resource_consumption_mode,
                                    "resource_consumption_fixed_mode": resource_consumption_fixed_mode,
                                    "strategy_update_rule": strategy_update_rule,
                                    "r": float(r_value),
                                }
                                spec["_runtime"] = {
                                    "quiet_console": True,
                                    "scan_mode": True,
                                }
                                specs.append(spec)

    return specs


def build_experiment_specs() -> List[Dict[str, Any]]:
    if SCAN_EXPERIMENT["enabled"]:
        return build_scan_experiment_specs()

    base = deepcopy(BASE_EXPERIMENT)
    if not BATCH_EXPERIMENTS:
        return [base]

    specs: List[Dict[str, Any]] = []
    for override in BATCH_EXPERIMENTS:
        specs.append(deep_update(deepcopy(base), override))
    return specs


def build_graph(spec: Mapping[str, Any]) -> Dict[int, List[int]]:
    network = spec["network"]
    network_type = network["type"]
    seed = spec["seed"]

    if network_type == "regular":
        return make_random_regular_graph(
            num_nodes=network["num_nodes"],
            degree=network["regular_degree"],
            seed=seed,
        )
    if network_type == "erdos_renyi":
        return make_erdos_renyi_graph(
            num_nodes=network["num_nodes"],
            edge_prob=_resolve_er_edge_prob(network),
            seed=seed,
        )
    if network_type == "small_world":
        return make_watts_strogatz_graph(
            num_nodes=network["num_nodes"],
            degree=network["ws_degree"],
            rewiring_prob=network["ws_rewiring_prob"],
            seed=seed,
        )
    if network_type == "scale_free":
        return make_barabasi_albert_graph(
            num_nodes=network["num_nodes"],
            attachments_per_new_node=network["ba_attachments_per_new_node"],
            seed=seed,
        )
    if network_type == "grid":
        return make_grid_graph(
            rows=network["grid_rows"],
            cols=network["grid_cols"],
            periodic=network["grid_periodic"],
        )
    raise ValueError("Unsupported network type: {0}".format(network_type))


def _resolve_target_mean_degree_for_env(
    spec: Mapping[str, Any],
    graph: Mapping[int, Sequence[int]],
) -> float:
    network = spec["network"]
    network_type = network["type"]

    if network_type == "regular":
        return float(network["regular_degree"])
    if network_type == "erdos_renyi":
        target_mean_degree = network.get("er_target_mean_degree")
        if target_mean_degree is not None:
            return float(target_mean_degree)
        num_nodes = int(network["num_nodes"])
        return float(_resolve_er_edge_prob(network) * max(num_nodes - 1, 0))
    if network_type == "small_world":
        return float(network["ws_degree"])
    if network_type == "scale_free":
        num_nodes = int(network["num_nodes"])
        attachments = float(network["ba_attachments_per_new_node"])
        return float((2.0 * attachments) - ((attachments * (attachments + 1.0)) / max(num_nodes, 1)))
    if network_type == "grid":
        degrees = [len(neighbors) for neighbors in graph.values()]
        return float(mean(degrees) if degrees else 0.0)
    raise ValueError("Unsupported network type: {0}".format(network_type))


def build_env_config(spec: Mapping[str, Any], graph: Mapping[int, Sequence[int]]) -> SPGGConfig:
    dynamics = spec["dynamics"]
    reward = spec["reward"]
    return SPGGConfig(
        alpha=dynamics["alpha"],
        r=dynamics["r"],
        p_mode=dynamics.get("p_mode", "constant"),
        p_max=dynamics["p_max"],
        p_c=dynamics.get("p_c", 1.0),
        resource_consumption_mode=dynamics.get("resource_consumption_mode", "fixed"),
        resource_consumption_fixed_mode=dynamics.get("resource_consumption_fixed_mode", "constant"),
        resource_consumption_fixed=dynamics.get("resource_consumption_fixed", 0.0),
        resource_consumption_degree_multiplier=dynamics.get("resource_consumption_degree_multiplier", 0.0),
        resource_consumption_rate=dynamics.get("resource_consumption_rate", 0.0),
        resource_consumption_threshold=dynamics.get("resource_consumption_threshold", 0.0),
        strategy_update_rule=dynamics.get("strategy_update_rule", "fermi"),
        beta=dynamics["beta"],
        q_learning_rate=dynamics.get("q_learning_rate", 0.1),
        q_learning_discount=dynamics.get("q_learning_discount", 0.95),
        q_learning_epsilon=dynamics.get("q_learning_epsilon", 0.05),
        q_learning_initial_value=dynamics.get("q_learning_initial_value", 0.0),
        episode_length=dynamics["episode_length"],
        initial_resource=dynamics["initial_resource"],
        initial_cooperation_prob=dynamics["initial_cooperation_prob"],
        target_mean_degree=_resolve_target_mean_degree_for_env(spec, graph),
        reward=RewardConfig(
            lambda_payoff=reward["lambda_payoff"],
            lambda_cooperation=reward["lambda_cooperation"],
            lambda_total_resource=reward.get("lambda_total_resource", 0.0),
            lambda_collapse=reward.get("lambda_collapse", 0.0),
            lambda_gini=reward["lambda_gini"],
            epsilon=reward["epsilon"],
        ),
    )


def build_gnn_policy(spec: Mapping[str, Any]) -> Any:
    from Project1.policies.gnn_rl import GNNAllocationPolicy, GNNPolicyConfig

    gnn = spec["gnn"]
    return GNNAllocationPolicy(
        GNNPolicyConfig(
            hidden_dim=gnn["hidden_dim"],
            local_hidden_dim=gnn.get("local_hidden_dim"),
            score_hidden_dim=gnn.get("score_hidden_dim"),
            critic_hidden_dim=gnn.get("critic_hidden_dim"),
            num_message_passing_layers=gnn["num_message_passing_layers"],
            temperature=gnn["temperature"],
        )
    )


def _resolve_effective_steps_per_update(spec: Mapping[str, Any]) -> int:
    training = spec["training"]
    if training.get("use_episode_length_as_steps_per_update", False):
        return int(spec["dynamics"]["episode_length"])
    return int(training["steps_per_update"])


def _resolve_training_schedule(spec: Mapping[str, Any]) -> Dict[str, Any]:
    training = spec["training"]
    effective_steps_per_update = _resolve_effective_steps_per_update(spec)
    num_workers = int(training["num_workers"])
    global_env_steps_per_update = effective_steps_per_update * num_workers

    warmup_env_steps_config = training.get("warmup_env_steps")
    if warmup_env_steps_config is not None:
        warmup_env_steps = int(warmup_env_steps_config)
        if warmup_env_steps < 0:
            raise ValueError("training.warmup_env_steps must be non-negative when provided.")
        warmup_steps_source = "training.warmup_env_steps"
    else:
        warmup_steps_raw = training.get("warmup_steps")
        if warmup_steps_raw is None:
            raise ValueError("Either training.warmup_env_steps or training.warmup_steps must be provided.")
        warmup_env_steps = int(warmup_steps_raw)
        if warmup_env_steps < 0:
            raise ValueError("training.warmup_steps must be non-negative when provided.")
        warmup_steps_source = "training.warmup_steps"

    total_env_steps_config = training.get("total_env_steps")
    if total_env_steps_config is not None:
        total_env_steps_requested = int(total_env_steps_config)
        if total_env_steps_requested <= 0:
            raise ValueError("training.total_env_steps must be positive when provided.")
        total_rollout_env_steps_requested = total_env_steps_requested + warmup_env_steps
        total_updates = max(1, int(ceil(total_rollout_env_steps_requested / float(global_env_steps_per_update))))
        total_updates_source = "training.total_env_steps"
        total_rollout_env_steps_effective = total_updates * global_env_steps_per_update
        total_env_steps_effective = max(total_rollout_env_steps_effective - warmup_env_steps, 0)
    else:
        total_updates_raw = training.get("total_updates")
        if total_updates_raw is None:
            raise ValueError("Either training.total_env_steps or training.total_updates must be provided.")
        total_updates = int(total_updates_raw)
        if total_updates <= 0:
            raise ValueError("training.total_updates must be positive when provided.")
        total_rollout_env_steps_requested = total_updates * global_env_steps_per_update
        total_rollout_env_steps_effective = total_rollout_env_steps_requested
        total_env_steps_requested = max(total_rollout_env_steps_requested - warmup_env_steps, 0)
        total_env_steps_effective = max(total_rollout_env_steps_effective - warmup_env_steps, 0)
        total_updates_source = "training.total_updates"

    eval_interval_env_steps_config = training.get("eval_interval_env_steps")
    if eval_interval_env_steps_config is not None:
        eval_interval_env_steps = int(eval_interval_env_steps_config)
        if eval_interval_env_steps <= 0:
            raise ValueError("training.eval_interval_env_steps must be positive when provided.")
        eval_interval_updates = max(1, int(ceil(eval_interval_env_steps / float(global_env_steps_per_update))))
        eval_interval_source = "training.eval_interval_env_steps"
    else:
        eval_interval_raw = training.get("eval_interval")
        if eval_interval_raw is None:
            raise ValueError("Either training.eval_interval_env_steps or training.eval_interval must be provided.")
        eval_interval_updates = int(eval_interval_raw)
        if eval_interval_updates <= 0:
            raise ValueError("training.eval_interval must be positive when provided.")
        eval_interval_env_steps = eval_interval_updates * global_env_steps_per_update
        eval_interval_source = "training.eval_interval"

    return {
        "effective_steps_per_update": effective_steps_per_update,
        "global_env_steps_per_update": global_env_steps_per_update,
        "total_updates": total_updates,
        "total_env_steps_requested": total_env_steps_requested,
        "total_env_steps_effective": total_env_steps_effective,
        "total_rollout_env_steps_requested": total_rollout_env_steps_requested,
        "total_rollout_env_steps_effective": total_rollout_env_steps_effective,
        "total_updates_source": total_updates_source,
        "warmup_env_steps": warmup_env_steps,
        "warmup_steps_source": warmup_steps_source,
        "eval_interval_updates": eval_interval_updates,
        "eval_interval_env_steps": eval_interval_env_steps,
        "eval_interval_source": eval_interval_source,
    }


def build_trainer_config(spec: Mapping[str, Any]) -> Any:
    from Project1.trainer import TrainerConfig

    training = spec["training"]
    training_schedule = _resolve_training_schedule(spec)
    replay_topology_names_override = training.get("replay_topology_names")
    if replay_topology_names_override is None:
        randomization = spec.get("domain_randomization", {})
        if bool(randomization.get("enabled", False)):
            replay_topology_names = tuple(str(item) for item in randomization.get("network_types", ("fixed",)))
        else:
            replay_topology_names = ("fixed",)
    else:
        replay_topology_names = tuple(str(item) for item in replay_topology_names_override)
    return TrainerConfig(
        total_updates=training_schedule["total_updates"],
        steps_per_update=training_schedule["effective_steps_per_update"],
        gamma=training["gamma"],
        learning_rate=training["learning_rate"],
        actor_lr=training.get("actor_lr"),
        critic_lr=training.get("critic_lr"),
        lr_schedule_type=training["lr_schedule_type"],
        lr_final=training["lr_final"],
        lr_decay_rate=training["lr_decay_rate"],
        lr_decay_steps=training["lr_decay_steps"],
        actor_weight_decay=training.get("actor_weight_decay", 0.0),
        critic_weight_decay=training.get("critic_weight_decay", 0.0),
        actor_entropy_coef=training.get("actor_entropy_coef", 0.0),
        actor_logit_l2_coef=training.get("actor_logit_l2_coef", 0.0),
        critic_state_hidden_dim=training.get("critic_state_hidden_dim"),
        critic_action_hidden_dim=training.get("critic_action_hidden_dim"),
        critic_pool_hidden_dim=training.get("critic_pool_hidden_dim"),
        critic_q_hidden_dim=training.get("critic_q_hidden_dim"),
        replay_capacity=training["replay_capacity"],
        replay_strategy=training.get("replay_strategy", "fifo"),
        replay_topology_names=replay_topology_names,
        replay_recent_fraction=training.get("replay_recent_fraction", 0.50),
        replay_long_term_fraction=training.get("replay_long_term_fraction", 0.35),
        replay_demo_fraction=training.get("replay_demo_fraction", 0.15),
        replay_demo_behavior_source=training.get("replay_demo_behavior_source", "pool_power_mix"),
        batch_size=training["batch_size"],
        graph_batch_chunk_size=training.get("graph_batch_chunk_size", 16),
        warmup_steps=training_schedule["warmup_env_steps"],
        warmup_behavior_mode=training["warmup_behavior_mode"],
        warmup_selection_granularity=training["warmup_selection_granularity"],
        warmup_uniform_prob=training["warmup_uniform_prob"],
        warmup_proportional_prob=training["warmup_proportional_prob"],
        warmup_constant_mix_prob=training["warmup_constant_mix_prob"],
        warmup_pool_power_mix_prob=training["warmup_pool_power_mix_prob"],
        warmup_random_logits_prob=training["warmup_random_logits_prob"],
        warmup_constant_mix_omega=training["warmup_constant_mix_omega"],
        warmup_pool_power_k=training["warmup_pool_power_k"],
        warmup_logit_noise_std=training["warmup_logit_noise_std"],
        warmup_logit_noise_clip=training["warmup_logit_noise_clip"],
        freeze_actor_q_during_warmup=training.get("freeze_actor_q_during_warmup", True),
        warmup_actor_bc_coef=training.get("warmup_actor_bc_coef", 1.0),
        actor_demo_bc_coef=training.get("actor_demo_bc_coef", 0.25),
        actor_demo_bc_decay_end_fraction=training.get("actor_demo_bc_decay_end_fraction", 0.50),
        demo_pretrain_enabled=training.get("demo_pretrain_enabled", False),
        demo_collection_env_steps=training.get("demo_collection_env_steps", 0),
        demo_collection_behavior_source=training.get("demo_collection_behavior_source", "pool_power_mix"),
        demo_collection_use_domain_randomization=training.get("demo_collection_use_domain_randomization", True),
        demo_collection_network_types=tuple(str(item) for item in (training.get("demo_collection_network_types") or ())),
        demo_collection_runtime=str(training.get("demo_collection_runtime", "parallel_cpu")),
        actor_bc_pretrain_updates=training.get("actor_bc_pretrain_updates", 0),
        critic_pretrain_updates=training.get("critic_pretrain_updates", 0),
        demo_pretrain_batch_size=training.get("demo_pretrain_batch_size"),
        demo_dataset_save_path=training.get("demo_dataset_save_path"),
        demo_critic_pretrain_target_mode=training.get("demo_critic_pretrain_target_mode", "n_step"),
        demo_critic_pretrain_n_step=training.get("demo_critic_pretrain_n_step", 20),
        teacher_takeover_enabled=training.get("teacher_takeover_enabled", True),
        teacher_takeover_behavior_source=training.get("teacher_takeover_behavior_source", "pool_power_mix"),
        teacher_takeover_start_prob=training.get("teacher_takeover_start_prob", 0.8),
        teacher_takeover_end_prob=training.get("teacher_takeover_end_prob", 0.0),
        teacher_takeover_decay_end_fraction=training.get("teacher_takeover_decay_end_fraction", 0.30),
        online_actor_q_coef_initial=training.get("online_actor_q_coef_initial", 0.2),
        online_actor_q_coef_final=training.get("online_actor_q_coef_final", 1.0),
        online_actor_q_coef_ramp_end_fraction=training.get("online_actor_q_coef_ramp_end_fraction", 0.30),
        critic_loss_type=training.get("critic_loss_type", "huber"),
        critic_huber_delta=training.get("critic_huber_delta", 1.0),
        actor_grad_clip_norm=training.get("actor_grad_clip_norm", 5.0),
        critic_grad_clip_norm=training.get("critic_grad_clip_norm", 5.0),
        train_every=training["train_every"],
        gradient_steps_per_update=training["gradient_steps_per_update"],
        policy_delay=training["policy_delay"],
        replay_collapse_fc_threshold=training.get("replay_collapse_fc_threshold", 0.10),
        replay_max_collapse_sample_ratio=training.get("replay_max_collapse_sample_ratio", 0.20),
        tau=training["tau"],
        rollout_logit_noise_std=training["rollout_logit_noise_std"],
        rollout_logit_noise_clip=training["rollout_logit_noise_clip"],
        rollout_noise_decay=training["rollout_noise_decay"],
        target_logit_noise_std=training["target_logit_noise_std"],
        target_logit_noise_clip=training["target_logit_noise_clip"],
        num_workers=training["num_workers"],
        num_envs_per_worker=training.get("num_envs_per_worker", 1),
        worker_sync_interval=training["worker_sync_interval"],
        overlap_rollout_and_update=training.get("overlap_rollout_and_update", True),
        worker_rpc_timeout_seconds=training.get("worker_rpc_timeout_seconds", 300.0),
        rollout_device=training.get("rollout_device", "cpu"),
        rollout_inference_mode=training.get("rollout_inference_mode", "local"),
        rollout_inference_batch_timeout_ms=training.get("rollout_inference_batch_timeout_ms", 2.0),
        rollout_num_threads=training.get("rollout_num_threads"),
        collapse_resource_threshold=training["collapse_resource_threshold"],
        eval_interval=training_schedule["eval_interval_updates"],
        eval_episodes=training["eval_episodes"],
        device=training["device"],
        seed=spec["seed"],
    )


def build_domain_randomization_config(spec: Mapping[str, Any]) -> Any:
    from Project1.td3 import DomainRandomizationConfig

    randomization = spec.get("domain_randomization", {})
    if not randomization:
        return DomainRandomizationConfig(enabled=False)

    default_er_mean_degree = spec["network"].get("er_target_mean_degree")
    if default_er_mean_degree is None:
        default_er_mean_degree = 4.0

    def _optional_range(key: str) -> tuple[float, float] | None:
        values = randomization.get(key)
        if values is None:
            return None
        if len(values) != 2:
            raise ValueError("{0} must be a length-2 range or None.".format(key))
        return (float(values[0]), float(values[1]))

    network_types = tuple(str(item) for item in randomization.get("network_types", ("regular",)))
    network_type_weights_raw = randomization.get("network_type_weights")
    network_type_weights = None
    if network_type_weights_raw is not None:
        network_type_weights = tuple(float(item) for item in network_type_weights_raw)
        if len(network_type_weights) != len(network_types):
            raise ValueError("domain_randomization.network_type_weights must align with network_types.")

    return DomainRandomizationConfig(
        enabled=bool(randomization.get("enabled", False)),
        network_types=network_types,
        network_type_weights=network_type_weights,
        num_nodes_choices=tuple(int(item) for item in randomization.get("num_nodes_choices", (int(spec["network"]["num_nodes"]),))),
        regular_degree_choices=tuple(int(item) for item in randomization.get("regular_degree_choices", (int(spec["network"]["regular_degree"]),))),
        er_mean_degree_choices=tuple(
            float(item)
            for item in randomization.get(
                "er_mean_degree_choices",
                (float(default_er_mean_degree),),
            )
        ),
        ws_degree_choices=tuple(int(item) for item in randomization.get("ws_degree_choices", (int(spec["network"]["ws_degree"]),))),
        ws_rewiring_choices=tuple(
            float(item) for item in randomization.get("ws_rewiring_choices", (float(spec["network"]["ws_rewiring_prob"]),))
        ),
        ba_attachment_choices=tuple(
            int(item)
            for item in randomization.get(
                "ba_attachment_choices",
                (int(spec["network"]["ba_attachments_per_new_node"]),),
            )
        ),
        initial_resource_range=_optional_range("initial_resource_range"),
        initial_cooperation_prob_range=_optional_range("initial_cooperation_prob_range"),
        alpha_range=_optional_range("alpha_range"),
        r_range=_optional_range("r_range"),
        p_max_range=_optional_range("p_max_range"),
    )


def _build_network_from_eval_family(
    base_network: Mapping[str, Any],
    family: Mapping[str, Any],
) -> Dict[str, Any]:
    network = deepcopy(base_network)
    network_type = str(family.get("network_type", family.get("type", network["type"])))
    network["type"] = network_type

    if "num_nodes" in family:
        network["num_nodes"] = int(family["num_nodes"])

    if network_type == "regular":
        network["regular_degree"] = int(family.get("regular_degree", network["regular_degree"]))
    elif network_type == "erdos_renyi":
        if family.get("er_target_mean_degree") is not None:
            network["er_target_mean_degree"] = float(family["er_target_mean_degree"])
            network["er_edge_prob"] = None
        elif family.get("er_edge_prob") is not None:
            network["er_target_mean_degree"] = None
            network["er_edge_prob"] = float(family["er_edge_prob"])
    elif network_type == "small_world":
        network["ws_degree"] = int(family.get("ws_degree", network["ws_degree"]))
        network["ws_rewiring_prob"] = float(family.get("ws_rewiring_prob", network["ws_rewiring_prob"]))
    elif network_type == "scale_free":
        network["ba_attachments_per_new_node"] = int(
            family.get("ba_attachments_per_new_node", network["ba_attachments_per_new_node"])
        )
    elif network_type == "grid":
        network["grid_rows"] = int(family.get("grid_rows", network["grid_rows"]))
        network["grid_cols"] = int(family.get("grid_cols", network["grid_cols"]))
        network["grid_periodic"] = bool(family.get("grid_periodic", network["grid_periodic"]))
        network["num_nodes"] = int(network["grid_rows"]) * int(network["grid_cols"])
    else:
        raise ValueError("Unsupported evaluation network_type: {0}".format(network_type))

    return network


def _build_singleton_randomization_for_network(network: Mapping[str, Any]) -> Any:
    from Project1.td3 import DomainRandomizationConfig

    network_type = str(network["type"])
    num_nodes = int(network["num_nodes"])
    if network_type == "grid":
        return DomainRandomizationConfig(enabled=False)

    er_mean_degree = network.get("er_target_mean_degree")
    if er_mean_degree is None:
        er_mean_degree = float(_resolve_er_edge_prob(network) * max(num_nodes - 1, 0))

    return DomainRandomizationConfig(
        enabled=True,
        network_types=(network_type,),
        num_nodes_choices=(num_nodes,),
        regular_degree_choices=(int(network.get("regular_degree", 4)),),
        er_mean_degree_choices=(float(er_mean_degree),),
        ws_degree_choices=(int(network.get("ws_degree", 4)),),
        ws_rewiring_choices=(float(network.get("ws_rewiring_prob", 0.10)),),
        ba_attachment_choices=(int(network.get("ba_attachments_per_new_node", 2)),),
    )


def build_evaluation_env_factories(spec: Mapping[str, Any]) -> Optional[List[Any]]:
    from Project1.td3.worker import RandomizedEnvFactory

    evaluation = spec.get("evaluation", {})
    if not evaluation or not bool(evaluation.get("use_custom_env_families", False)):
        return None

    env_families = evaluation.get("env_families", [])
    if not env_families:
        return None

    factories: List[Any] = []
    base_seed = int(spec["seed"])
    for family_index, family in enumerate(env_families):
        family_network = _build_network_from_eval_family(spec["network"], family)
        family_spec = deepcopy(spec)
        family_spec["network"] = family_network
        family_spec["seed"] = base_seed + 1_000 + family_index

        graph = build_graph(family_spec)
        env_config = build_env_config(family_spec, graph)
        env = SPGGEnv(env_config, graph)
        randomization = _build_singleton_randomization_for_network(family_network)
        factories.append(RandomizedEnvFactory.from_env(env, randomization=randomization))

    return factories


def _build_stage_evaluation_env_factories(
    spec: Mapping[str, Any],
    network_types: Sequence[str],
) -> Optional[List[Any]]:
    evaluation = spec.get("evaluation", {})
    if not evaluation or not bool(evaluation.get("use_custom_env_families", False)):
        return None

    allowed_types = {str(item) for item in network_types}
    filtered_families = [
        family
        for family in evaluation.get("env_families", [])
        if str(family.get("network_type", family.get("type", spec["network"]["type"]))) in allowed_types
    ]
    if not filtered_families:
        raise ValueError(
            "Curriculum eval_network_types {0} do not match any evaluation.env_families.".format(
                sorted(allowed_types),
            )
        )

    stage_spec = deepcopy(spec)
    stage_spec["evaluation"] = {
        **deepcopy(evaluation),
        "env_families": filtered_families,
    }
    return build_evaluation_env_factories(stage_spec)


def build_training_curriculum(spec: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    from Project1.td3 import DomainRandomizationConfig

    curriculum = spec.get("curriculum", {})
    if not curriculum or not bool(curriculum.get("enabled", False)):
        return None

    mode = str(curriculum.get("mode", "update_steps"))
    if mode != "update_steps":
        raise ValueError("curriculum.mode must currently be 'update_steps'.")

    stage_specs = list(curriculum.get("stages", []))
    if not stage_specs:
        raise ValueError("curriculum.stages must contain at least one stage when curriculum is enabled.")

    base_randomization = build_domain_randomization_config(spec)
    supported_network_types = {str(item) for item in base_randomization.network_types}
    training_schedule = _resolve_training_schedule(spec)
    total_updates = int(training_schedule["total_updates"])

    portion_sum = 0.0
    cumulative_portion = 0.0
    stages: List[Dict[str, Any]] = []
    for stage_index, stage_spec in enumerate(stage_specs):
        portion = float(stage_spec["portion"])
        if portion <= 0.0:
            raise ValueError("curriculum stage portion must be positive.")
        portion_sum += portion

        train_network_types = tuple(str(item) for item in stage_spec.get("train_network_types", ()))
        if not train_network_types:
            raise ValueError("Each curriculum stage must define at least one train_network_types entry.")
        unsupported_train_types = [item for item in train_network_types if item not in supported_network_types]
        if unsupported_train_types:
            raise ValueError(
                "Curriculum train_network_types contain unsupported values: {0}".format(unsupported_train_types)
            )
        label = str(stage_spec.get("label", "stage_{0}".format(stage_index)))

        train_network_type_weights_raw = stage_spec.get("train_network_type_weights")
        if train_network_type_weights_raw is None:
            train_network_type_weights = tuple(1.0 for _ in train_network_types)
        else:
            train_network_type_weights = tuple(float(item) for item in train_network_type_weights_raw)
            if len(train_network_type_weights) != len(train_network_types):
                raise ValueError(
                    "curriculum train_network_type_weights must align with train_network_types in stage '{0}'.".format(
                        label,
                    )
                )
        if any(weight < 0.0 for weight in train_network_type_weights):
            raise ValueError("curriculum train_network_type_weights must be non-negative.")
        if sum(train_network_type_weights) <= 0.0:
            raise ValueError("curriculum train_network_type_weights must sum to a positive value.")

        eval_network_types = tuple(str(item) for item in stage_spec.get("eval_network_types", train_network_types))
        activate_at_update = 1 if stage_index == 0 else (int(total_updates * cumulative_portion) + 1)
        cumulative_portion += portion

        train_randomization = replace(
            base_randomization,
            enabled=True,
            network_types=train_network_types,
            network_type_weights=train_network_type_weights,
        )
        if not isinstance(train_randomization, DomainRandomizationConfig):
            raise RuntimeError("Failed to build curriculum DomainRandomizationConfig.")

        stages.append(
            {
                "stage_index": stage_index,
                "label": label,
                "portion": portion,
                "activate_at_update": activate_at_update,
                "train_network_types": train_network_types,
                "train_network_type_weights": train_network_type_weights,
                "eval_network_types": eval_network_types,
                "train_randomization": train_randomization,
                "eval_env_factories": _build_stage_evaluation_env_factories(spec, eval_network_types),
            }
        )

    if abs(portion_sum - 1.0) > 1e-6:
        raise ValueError("curriculum stage portions must sum to 1.0.")

    return stages


def _should_use_external_demo_collection(
    trainer_config: Any,
    *,
    resume_from_checkpoint: Any,
) -> bool:
    return (
        bool(getattr(trainer_config, "demo_pretrain_enabled", False))
        and int(getattr(trainer_config, "demo_collection_env_steps", 0)) > 0
        and str(getattr(trainer_config, "demo_collection_runtime", "parallel_cpu")) in {"parallel_cpu", "isolated_cpu"}
        and not bool(resume_from_checkpoint)
    )


def _progress_interval(total: int) -> int:
    return max(1, int(ceil(float(max(1, total)) / 20.0)))


def _format_progress_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return "{0}h {1:02d}m {2:02d}s".format(hours, minutes, secs)
    if minutes > 0:
        return "{0}m {1:02d}s".format(minutes, secs)
    return "{0}s".format(secs)


def _print_demo_progress(stage_label: str, completed: int, total: int, started_at: float) -> None:
    if total <= 0:
        return
    elapsed = float(time.perf_counter() - started_at)
    progress = min(max(float(completed) / float(total), 0.0), 1.0)
    eta_seconds = None
    if completed > 0 and progress > 0.0:
        eta_seconds = max(0.0, elapsed * (1.0 - progress) / progress)
    eta_text = _format_progress_duration(eta_seconds) if eta_seconds is not None else "unavailable"
    print(
        "Demo Pretrain | {0} progress | {1}/{2} ({3:.1f}%) | ETA={4} | elapsed={5}".format(
            stage_label,
            int(completed),
            int(total),
            progress * 100.0,
            eta_text,
            _format_progress_duration(elapsed),
        )
    )


def _resolve_external_demo_collection_factory(
    env_config: SPGGConfig,
    graph: Mapping[int, Sequence[int]],
    randomization_config: Any,
    trainer_config: Any,
):
    from Project1.td3 import DomainRandomizationConfig
    from Project1.td3.worker import RandomizedEnvFactory

    base_randomization = randomization_config
    if not isinstance(base_randomization, DomainRandomizationConfig):
        raise TypeError("randomization_config must be a DomainRandomizationConfig.")

    if not bool(trainer_config.demo_collection_use_domain_randomization) or not bool(base_randomization.enabled):
        demo_randomization = replace(base_randomization, enabled=False)
        return RandomizedEnvFactory(env_config, graph, randomization=demo_randomization)

    configured_network_types = tuple(
        str(item) for item in getattr(trainer_config, "demo_collection_network_types", ()) if str(item)
    )
    if configured_network_types:
        supported_types = set(str(item) for item in base_randomization.network_types)
        selected_network_types = tuple(item for item in configured_network_types if item in supported_types)
        if not selected_network_types:
            raise ValueError("demo_collection_network_types must overlap domain_randomization.network_types.")
    else:
        selected_network_types = tuple(str(item) for item in base_randomization.network_types)

    selected_weights = None
    if base_randomization.network_type_weights is not None:
        weight_by_type = {
            str(network_type): float(weight)
            for network_type, weight in zip(base_randomization.network_types, base_randomization.network_type_weights)
        }
        selected_weights = tuple(weight_by_type[item] for item in selected_network_types)

    demo_randomization = replace(
        base_randomization,
        enabled=True,
        network_types=selected_network_types,
        network_type_weights=selected_weights,
    )
    return RandomizedEnvFactory(env_config, graph, randomization=demo_randomization)


def _global_step_allocations_for_workers(workers: Sequence[Any], total_steps: int) -> List[int]:
    remaining_steps = max(0, int(total_steps))
    if remaining_steps <= 0:
        return [0 for _ in workers]

    allocations: List[int] = []
    remaining_workers = len(workers)
    for worker in workers:
        per_worker_steps = int(worker.config.rollout_steps_per_sync)
        if remaining_workers <= 1:
            allocation = min(per_worker_steps, remaining_steps)
        else:
            fair_share = int(ceil(float(remaining_steps) / float(remaining_workers)))
            allocation = min(per_worker_steps, fair_share, remaining_steps)
        allocations.append(allocation)
        remaining_steps -= allocation
        remaining_workers -= 1
    return allocations


def _save_demo_dataset_external(replay_batches: Sequence[Any], demo_dataset_save_path: Optional[str]) -> Optional[str]:
    if not demo_dataset_save_path:
        return None

    import torch

    dataset_path = Path(str(demo_dataset_save_path)).expanduser()
    if not dataset_path.is_absolute():
        dataset_path = Path.cwd() / dataset_path
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "source": "pool_power_mix_demo_collection_external",
            "num_batches": int(len(replay_batches)),
            "replay_batches": [batch.clone() for batch in replay_batches],
        },
        dataset_path,
    )
    return str(dataset_path)


def run_external_demo_collection(
    spec: Mapping[str, Any],
    graph: Mapping[int, Sequence[int]],
    env_config: SPGGConfig,
    trainer_config: Any,
    randomization_config: Any,
) -> tuple[Any, Dict[str, Any]]:
    from Project1.policies.gnn_rl import GNNAllocationPolicy
    from Project1.td3.replay import ReplayBuffer
    from Project1.td3.worker import ParallelRolloutWorker, RolloutWorker, WorkerConfig
    from Project1.td3.exploration import LogitSpaceExplorer

    demo_collection_steps = max(0, int(trainer_config.demo_collection_env_steps))
    if demo_collection_steps <= 0:
        raise ValueError("External demo collection requires demo_collection_env_steps > 0.")

    runtime = str(trainer_config.demo_collection_runtime)
    if runtime == "parallel_cpu" and int(trainer_config.num_workers) <= 1:
        runtime = "isolated_cpu"

    print(
        "Demo Pretrain | external collection start | env_steps={0} | behavior={1}".format(
            demo_collection_steps,
            trainer_config.demo_collection_behavior_source,
        )
    )

    replay_buffer = ReplayBuffer(
        int(trainer_config.replay_capacity),
        seed=int(trainer_config.seed or 0),
        replay_strategy=str(trainer_config.replay_strategy),
        topology_names=tuple(str(item) for item in trainer_config.replay_topology_names),
        recent_fraction=float(trainer_config.replay_recent_fraction),
        long_term_fraction=float(trainer_config.replay_long_term_fraction),
        demo_fraction=float(trainer_config.replay_demo_fraction),
        demo_behavior_source=str(trainer_config.replay_demo_behavior_source),
    )
    demo_factory = _resolve_external_demo_collection_factory(
        env_config=env_config,
        graph=graph,
        randomization_config=randomization_config,
        trainer_config=trainer_config,
    )
    actor_template = build_gnn_policy(spec)
    demo_batches_to_save: List[Any] = []
    demo_return_targets: List[np.ndarray] = []
    started_at = time.perf_counter()
    log_interval = _progress_interval(demo_collection_steps)
    next_log_at = min(demo_collection_steps, log_interval)
    last_logged_completed = 0

    def _record_result(result: Any) -> int:
        nonlocal next_log_at, last_logged_completed
        collected_now = len(result.replay_batch)
        if collected_now <= 0:
            raise RuntimeError("Demo collection produced zero transitions.")
        replay_buffer.extend(result.replay_batch)
        if trainer_config.demo_dataset_save_path:
            demo_batches_to_save.append(result.replay_batch.clone())
        valid_mask = result.replay_batch.demo_return_valid.detach().cpu().numpy().astype(np.bool_, copy=False)
        if np.any(valid_mask):
            demo_return_targets.append(
                result.replay_batch.demo_return_target.detach().cpu().numpy().astype(np.float32, copy=False)[valid_mask]
            )
        return collected_now

    if runtime == "parallel_cpu":
        print(
            "Demo Pretrain | runtime=parallel_cpu | envs={0} | steps_per_sync={1}".format(
                max(1, int(trainer_config.num_workers)) * max(1, int(trainer_config.num_envs_per_worker)),
                max(1, int(trainer_config.steps_per_update)),
            )
        )
        demo_workers: List[Any] = []
        try:
            for worker_id in range(int(trainer_config.num_workers)):
                worker_config = WorkerConfig(
                    worker_id=worker_id,
                    seed=(int(trainer_config.seed or 0) + 2_000_000 + worker_id),
                    rollout_steps_per_sync=int(trainer_config.steps_per_update),
                    num_envs_per_worker=int(trainer_config.num_envs_per_worker),
                )
                demo_workers.append(
                    ParallelRolloutWorker(
                        actor=GNNAllocationPolicy(deepcopy(actor_template.config)),
                        env_factory=demo_factory,
                        config=worker_config,
                        train_config=trainer_config,
                        device="cpu",
                    )
                )
            remaining_steps = int(demo_collection_steps)
            while remaining_steps > 0:
                step_allocations = _global_step_allocations_for_workers(demo_workers, remaining_steps)
                positive_requests = [
                    (worker, int(num_steps))
                    for worker, num_steps in zip(demo_workers, step_allocations)
                    if int(num_steps) > 0
                ]
                if not positive_requests:
                    break
                started_workers: List[Any] = []
                pending_workers: Dict[Any, Any] = {}
                rollout_results: List[Any] = []
                try:
                    for worker, num_steps in positive_requests:
                        worker.start_collect(
                            num_steps=num_steps,
                            global_warmup_steps=0,
                            forced_behavior_source=str(trainer_config.demo_collection_behavior_source),
                            mark_as_demo=True,
                            count_env_steps=False,
                            global_env_start_step=0,
                            demo_return_target_mode=str(trainer_config.demo_critic_pretrain_target_mode),
                            demo_return_n_step=int(trainer_config.demo_critic_pretrain_n_step),
                        )
                        started_workers.append(worker)
                        pending_workers[worker.connection] = worker

                    while pending_workers:
                        ready_connections = wait(
                            list(pending_workers.keys()),
                            timeout=float(trainer_config.worker_rpc_timeout_seconds),
                        )
                        if not ready_connections:
                            raise TimeoutError(
                                "Timed out waiting for external demo workers: {0}".format(
                                    [worker.config.worker_id for worker in pending_workers.values()]
                                )
                            )
                        for ready_connection in ready_connections:
                            worker = pending_workers.pop(ready_connection)
                            rollout_results.append(worker.finish_collect_ready())
                except Exception:
                    for worker in started_workers:
                        if getattr(worker, "_collect_inflight", False):
                            try:
                                worker.finish_collect()
                            except Exception:
                                pass
                    raise

                try:
                    collected_now = 0
                    for result in rollout_results:
                        collected_now += _record_result(result)
                    remaining_steps -= collected_now
                    completed_steps = int(demo_collection_steps) - int(remaining_steps)
                    if completed_steps >= next_log_at:
                        _print_demo_progress("collection", completed_steps, int(demo_collection_steps), started_at)
                        last_logged_completed = completed_steps
                        next_log_at += log_interval
                finally:
                    for result in rollout_results:
                        result.release_shared_memory()
        finally:
            for worker in demo_workers:
                worker.close()
    else:
        total_parallel_envs = max(1, int(trainer_config.num_workers)) * max(1, int(trainer_config.num_envs_per_worker))
        total_parallel_steps = max(1, int(trainer_config.num_workers)) * max(1, int(trainer_config.steps_per_update))
        print(
            "Demo Pretrain | runtime=isolated_cpu | envs={0} | steps_per_sync={1}".format(
                total_parallel_envs,
                total_parallel_steps,
            )
        )
        worker_config = WorkerConfig(
            worker_id=-1,
            seed=int(trainer_config.seed or 0) + 1_000_000,
            rollout_steps_per_sync=total_parallel_steps,
            num_envs_per_worker=total_parallel_envs,
        )
        demo_worker = RolloutWorker(
            actor=GNNAllocationPolicy(deepcopy(actor_template.config)),
            explorer=LogitSpaceExplorer(),
            env_factory=demo_factory,
            config=worker_config,
            train_config=trainer_config,
            device="cpu",
        )
        try:
            remaining_steps = int(demo_collection_steps)
            while remaining_steps > 0:
                batch_steps = min(int(demo_worker.config.rollout_steps_per_sync), remaining_steps)
                result = demo_worker.collect(
                    num_steps=batch_steps,
                    global_warmup_steps=0,
                    forced_behavior_source=str(trainer_config.demo_collection_behavior_source),
                    mark_as_demo=True,
                    count_env_steps=False,
                    global_env_start_step=0,
                    demo_return_target_mode=str(trainer_config.demo_critic_pretrain_target_mode),
                    demo_return_n_step=int(trainer_config.demo_critic_pretrain_n_step),
                )
                try:
                    collected_now = _record_result(result)
                    remaining_steps -= collected_now
                    completed_steps = int(demo_collection_steps) - int(remaining_steps)
                    if completed_steps >= next_log_at:
                        _print_demo_progress("collection", completed_steps, int(demo_collection_steps), started_at)
                        last_logged_completed = completed_steps
                        next_log_at += log_interval
                finally:
                    result.release_shared_memory()
        finally:
            if hasattr(demo_worker, "close"):
                demo_worker.close()

    if last_logged_completed < int(demo_collection_steps):
        _print_demo_progress("collection", int(demo_collection_steps), int(demo_collection_steps), started_at)

    summary: Dict[str, Any] = {
        "enabled": True,
        "demo_collection_env_steps": float(demo_collection_steps),
        "demo_replay_size_after_collection": float(replay_buffer.demo_size()),
        "actor_bc_updates": 0.0,
        "critic_pretrain_updates": 0.0,
        "actor_bc_loss_last": 0.0,
        "critic_loss_last": 0.0,
        "seconds_collection": float(time.perf_counter() - started_at),
        "seconds_actor_bc": 0.0,
        "seconds_critic": 0.0,
        "dataset_path": _save_demo_dataset_external(demo_batches_to_save, trainer_config.demo_dataset_save_path),
        "behavior_source": str(trainer_config.demo_collection_behavior_source),
        "critic_target_mode": str(trainer_config.demo_critic_pretrain_target_mode),
        "demo_return_target_mean": 0.0,
        "demo_return_target_std": 0.0,
    }
    if demo_return_targets:
        concatenated_demo_returns = np.concatenate(demo_return_targets, axis=0)
        summary["demo_return_target_mean"] = float(np.mean(concatenated_demo_returns))
        summary["demo_return_target_std"] = float(np.std(concatenated_demo_returns))
    print(
        "Demo Pretrain | external collection done | demo_replay_size={0:.0f} | return_mode={1} | target_mean={2:.6f} | target_std={3:.6f} | seconds={4:.3f}".format(
            float(summary["demo_replay_size_after_collection"]),
            str(summary["critic_target_mode"]),
            float(summary["demo_return_target_mean"]),
            float(summary["demo_return_target_std"]),
            float(summary["seconds_collection"]),
        )
    )
    return replay_buffer, summary


def graph_summary(graph: Mapping[int, Sequence[int]]) -> Dict[str, float]:
    degrees = [len(neighbors) for neighbors in graph.values()]
    num_edges = sum(degrees) // 2
    return {
        "num_nodes": float(len(graph)),
        "num_edges": float(num_edges),
        "degree_min": float(min(degrees) if degrees else 0),
        "degree_max": float(max(degrees) if degrees else 0),
        "degree_mean": float(mean(degrees) if degrees else 0.0),
    }


def print_header(spec: Mapping[str, Any], graph: Mapping[int, Sequence[int]], env_config: SPGGConfig) -> None:
    summary = graph_summary(graph)
    print("=" * 80)
    print("Experiment: {0}".format(spec["experiment_name"]))
    print("Run mode  : {0}".format(spec["run_mode"]))
    print("Network   : {0}".format(spec["network"]["type"]))
    if spec["network"]["type"] == "erdos_renyi":
        print("ER Params : {0}".format(_er_parameter_summary(spec["network"])))
    print(
        "Graph     : nodes={0}, edges={1}, degree_min={2:.0f}, degree_max={3:.0f}, degree_mean={4:.3f}".format(
            int(summary["num_nodes"]),
            int(summary["num_edges"]),
            summary["degree_min"],
            summary["degree_max"],
            summary["degree_mean"],
        )
    )
    if env_config.p_mode == "dynamic":
        dynamics_summary = "Dynamics  : alpha={0}, r={1}, p_mode=dynamic, p_c={2}, episode_length={3}".format(
            env_config.alpha,
            env_config.r,
            env_config.p_c,
            env_config.episode_length,
        )
    else:
        dynamics_summary = "Dynamics  : alpha={0}, r={1}, p_mode=constant, p_max={2}, episode_length={3}".format(
            env_config.alpha,
            env_config.r,
            env_config.p_max,
            env_config.episode_length,
        )
    print(dynamics_summary)
    if env_config.resource_consumption_mode == "fixed":
        if env_config.resource_consumption_fixed_mode == "degree_scaled":
            print(
                "Consume   : mode=fixed, fixed_mode=degree_scaled, degree_multiplier={0}".format(
                    env_config.resource_consumption_degree_multiplier
                )
            )
        else:
            print(
                "Consume   : mode=fixed, fixed_mode=constant, fixed={0}".format(
                    env_config.resource_consumption_fixed
                )
            )
    elif env_config.resource_consumption_mode == "proportional":
        print("Consume   : mode=proportional, rate={0}".format(env_config.resource_consumption_rate))
    else:
        if env_config.resource_consumption_fixed_mode == "degree_scaled":
            print(
                "Consume   : mode=piecewise_linear, fixed_mode=degree_scaled, degree_multiplier={0}, rate={1}, threshold={2}".format(
                    env_config.resource_consumption_degree_multiplier,
                    env_config.resource_consumption_rate,
                    env_config.resource_consumption_threshold,
                )
            )
        else:
            print(
                "Consume   : mode=piecewise_linear, fixed_mode=constant, fixed={0}, rate={1}, threshold={2}".format(
                    env_config.resource_consumption_fixed,
                    env_config.resource_consumption_rate,
                    env_config.resource_consumption_threshold,
                )
            )
    if env_config.strategy_update_rule == "fermi":
        print("Strategy  : rule=fermi, beta={0}".format(env_config.beta))
    elif env_config.strategy_update_rule == "q_learning":
        print(
            "Strategy  : rule=q_learning, state=none, lr={0}, gamma={1}, epsilon={2}, q0={3}".format(
                env_config.q_learning_rate,
                env_config.q_learning_discount,
                env_config.q_learning_epsilon,
                env_config.q_learning_initial_value,
            )
        )
    elif env_config.strategy_update_rule == "q_learning_2x2":
        print(
            "Strategy  : rule=q_learning_2x2, state=last_action, lr={0}, gamma={1}, epsilon={2}, q0={3}".format(
                env_config.q_learning_rate,
                env_config.q_learning_discount,
                env_config.q_learning_epsilon,
                env_config.q_learning_initial_value,
            )
        )
    else:
        print("Strategy  : rule=imitate_best")
    print(
        "Reward    : lambda_payoff={0}, lambda_cooperation={1}, lambda_total_resource={2}, lambda_collapse={3}, lambda_gini={4}".format(
            env_config.reward.lambda_payoff,
            env_config.reward.lambda_cooperation,
            env_config.reward.lambda_total_resource,
            env_config.reward.lambda_collapse,
            env_config.reward.lambda_gini,
        )
    )
    training = spec["training"]
    print(
        "Reg       : actor_wd={0}, critic_wd={1}, actor_entropy_coef={2}, actor_logit_l2_coef={3}".format(
            training.get("actor_weight_decay", 0.0),
            training.get("critic_weight_decay", 0.0),
            training.get("actor_entropy_coef", 0.0),
            training.get("actor_logit_l2_coef", 0.0),
        )
    )
    print("=" * 80)


def build_output_dir(spec: Mapping[str, Any]) -> Path:
    root_dir = Path(spec["output"]["root_dir"]).expanduser()
    if not root_dir.is_absolute():
        root_dir = PROJECT_ROOT / root_dir
    experiment_dir = root_dir / spec["experiment_name"]
    experiment_dir.mkdir(parents=True, exist_ok=True)
    return experiment_dir


def should_visualize_episode(spec: Mapping[str, Any], episode_index: int) -> bool:
    selected = spec["visualization"]["episodes_to_visualize"]
    if not selected:
        return True
    return episode_index in selected


def capture_record(
    time_index: int,
    observation: Mapping[str, np.ndarray],
    reward: float = 0.0,
    info: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    resources = observation["resources"].copy()
    zeros = np.zeros_like(resources, dtype=np.float64)
    income = np.asarray(info["income"], dtype=np.float64).copy() if info is not None else zeros.copy()
    consumption = np.asarray(info["consumption"], dtype=np.float64).copy() if info is not None else zeros.copy()
    payoff = np.asarray(info["payoff"], dtype=np.float64).copy() if info is not None else zeros.copy()

    record = {
        "time": int(time_index),
        "x_nominal": observation["x_nominal"].copy(),
        "x_actual": observation["x_actual"].copy(),
        "resources": resources,
        "investment": observation["investment"].copy(),
        "pool_raw": observation["pool_raw"].copy(),
        "pool_grown": observation["pool_grown"].copy(),
        "income": income,
        "consumption": consumption,
        "payoff": payoff,
        "reward": float(reward),
        "gini": float(info["gini"]) if info is not None else gini_coefficient(resources),
        "actual_cooperation_rate": float(np.mean(observation["x_actual"])),
        "nominal_cooperation_rate": float(np.mean(observation["x_nominal"])),
        "mean_resource": float(np.mean(resources)),
        "mean_pool_raw": float(np.mean(observation["pool_raw"])),
        "mean_pool_grown": float(np.mean(observation["pool_grown"])),
        "mean_investment": float(np.mean(observation["investment"])),
        "mean_income": float(np.mean(income)),
        "mean_consumption": float(np.mean(consumption)),
        "mean_payoff": float(np.mean(payoff)),
    }
    return record


def save_visualizations_for_history(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    history: Sequence[Dict[str, Any]],
    output_dir: Path,
    episode_index: int,
    phase_name: str,
) -> None:
    visualization = spec["visualization"]
    output = spec["output"]
    if not visualization["enable_micro_snapshots"] and not visualization["enable_macro_timeseries"]:
        return
    if not should_visualize_episode(spec, episode_index):
        return

    from Project1.visualization import build_layout, save_macro_timeseries, save_network_snapshots

    if visualization["enable_micro_snapshots"] and output["save_micro_snapshots"]:
        network = spec["network"]
        grid_shape = None
        if network["type"] == "grid":
            grid_shape = (network["grid_rows"], network["grid_cols"])

        layout = build_layout(
            graph,
            layout_name=visualization["layout"],
            seed=spec["seed"],
            spring_iterations=visualization["spring_iterations"],
            grid_shape=grid_shape,
        )
        micro_dir = output_dir / "{0}_episode_{1:03d}_micro".format(phase_name, episode_index)
        save_network_snapshots(
            graph=graph,
            history=history,
            output_dir=micro_dir,
            positions=layout,
            node_color_metric=visualization["node_color_metric"],
            node_size_metric=visualization["node_size_metric"],
            node_value_metric=visualization.get("node_value_metric"),
            node_value_decimals=int(visualization.get("node_value_decimals", 1)),
            frame_stride=visualization["frame_stride"],
            dpi=visualization["save_dpi"],
            label_nodes=visualization["label_nodes"],
            title_prefix="{0} | {1} | ep={2:03d} | ".format(spec["experiment_name"], phase_name, episode_index),
        )

    if visualization["enable_macro_timeseries"] and output["save_macro_timeseries"]:
        macro_path = output_dir / "{0}_episode_{1:03d}_macro.png".format(phase_name, episode_index)
        save_macro_timeseries(
            history=history,
            output_path=macro_path,
            metrics=visualization["macro_metrics"],
            title="{0} | {1} | episode {2:03d}".format(spec["experiment_name"], phase_name, episode_index),
            dpi=visualization["save_dpi"],
        )


def summarize_episode(history: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    final_record = history[-1]
    return {
        "final_actual_cooperation": float(final_record["actual_cooperation_rate"]),
        "final_mean_resource": float(final_record["mean_resource"]),
        "final_mean_pool_grown": float(final_record["mean_pool_grown"]),
        "final_mean_consumption": float(final_record["mean_consumption"]),
        "final_mean_payoff": float(final_record["mean_payoff"]),
        "final_gini": float(final_record["gini"]),
    }


def summarize_rule_based_episodes(
    episode_summaries: Sequence[Mapping[str, float]],
    episode_returns: Sequence[float],
) -> Dict[str, float]:
    final_actual_cooperation_mean = float(np.mean([item["final_actual_cooperation"] for item in episode_summaries]))
    final_mean_resource_mean = float(np.mean([item["final_mean_resource"] for item in episode_summaries]))
    final_mean_pool_grown_mean = float(np.mean([item["final_mean_pool_grown"] for item in episode_summaries]))
    final_mean_consumption_mean = float(np.mean([item["final_mean_consumption"] for item in episode_summaries]))
    final_mean_payoff_mean = float(np.mean([item["final_mean_payoff"] for item in episode_summaries]))
    final_gini_mean = float(np.mean([item["final_gini"] for item in episode_summaries]))

    return {
        "return_mean": float(np.mean(episode_returns)),
        "return_std": float(np.std(episode_returns)),
        "final_cooperation_mean": final_actual_cooperation_mean,
        "final_actual_cooperation_mean": final_actual_cooperation_mean,
        "final_mean_resource_mean": final_mean_resource_mean,
        "final_mean_pool_grown_mean": final_mean_pool_grown_mean,
        "final_consumption_mean": final_mean_consumption_mean,
        "final_mean_consumption_mean": final_mean_consumption_mean,
        "final_mean_payoff_mean": final_mean_payoff_mean,
        "final_gini_mean": final_gini_mean,
    }


def _format_training_update_summary(item: Mapping[str, float]) -> str:
    summary_text = (
        "[Update {0:03d}] loss={1:.6f}, policy_loss={2:.6f}, value_loss={3:.6f}, entropy={4:.6f}, mean_rollout_reward={5:.6f}, actor_lr={6:.6g}, critic_lr={7:.6g}".format(
            int(item["update"]),
            float(item["loss"]),
            float(item["policy_loss"]),
            float(item["value_loss"]),
            float(item["entropy"]),
            float(item["mean_rollout_reward"]),
            float(item["actor_lr"]),
            float(item["critic_lr"]),
        )
    )
    behavior_terms = []
    for source in (
        "uniform",
        "proportional",
        "constant_mix",
        "pool_power_mix",
        "random_logits",
        "actor_logits",
    ):
        key = "behavior_frac_{0}".format(source)
        if key in item:
            behavior_terms.append("{0}={1:.2f}".format(source, float(item[key])))
    if behavior_terms:
        summary_text += ", behavior_mix={0}".format("/".join(behavior_terms))
    if "eval_return_mean" in item:
        summary_text += ", eval_return_mean={0:.6f}, eval_cooperation_mean={1:.6f}, eval_gini_mean={2:.6f}".format(
            float(item["eval_return_mean"]),
            float(item["eval_cooperation_mean"]),
            float(item["eval_gini_mean"]),
        )
    return summary_text


def _console_info(message: str, phase: str = "train") -> str:
    return "[INFO {0}] {1} {2}".format(datetime.now().strftime("%H:%M:%S"), phase, message)


def _runtime_quiet_console(spec: Mapping[str, Any]) -> bool:
    runtime_flags = spec.get("_runtime", {})
    return bool(runtime_flags.get("quiet_console", False))


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(float(seconds))), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return "{0} hours, {1} minutes, {2} seconds".format(hours, minutes, secs)
    if minutes > 0:
        return "{0} minutes, {1} seconds".format(minutes, secs)
    return "{0} seconds".format(secs)


def _format_metric_rows(
    items: Sequence[tuple[str, float]],
    columns: int = 3,
    label_width: int = 24,
    value_width: int = 12,
) -> List[str]:
    if columns <= 0:
        columns = 1
    rows: List[str] = []
    current_row: List[str] = []
    for label, value in items:
        current_row.append("{0:<{1}} {2:>{3}.4f}".format(label + ":", label_width, float(value), value_width))
        if len(current_row) == columns:
            rows.append("  ".join(current_row))
            current_row = []
    if current_row:
        rows.append("  ".join(current_row))
    return rows


def _mean_of_recent_metrics(
    recent_metrics: Sequence[Mapping[str, float]],
    key: str,
) -> Optional[float]:
    values = [float(item[key]) for item in recent_metrics if key in item]
    if not values:
        return None
    return float(np.mean(values))


def _format_console_progress_lines(
    update: int,
    total_updates: int,
    env_steps: int,
    total_env_steps: int,
    start_time: float,
    eta_env_steps: Optional[int] = None,
    eta_total_env_steps: Optional[int] = None,
) -> List[str]:
    now = time.time()
    elapsed = max(now - start_time, 0.0)
    progress_completed = int(env_steps if eta_env_steps is None else eta_env_steps)
    progress_total = int(total_env_steps if eta_total_env_steps is None else eta_total_env_steps)
    if progress_total > 0:
        progress_ratio = min(max(float(progress_completed) / float(progress_total), 0.0), 1.0)
    else:
        progress_ratio = min(max(float(update) / max(total_updates, 1), 0.0), 1.0)

    if progress_ratio > 0.0 and progress_ratio < 1.0:
        estimated_total = elapsed / progress_ratio
        eta = max(estimated_total - elapsed, 0.0)
    elif progress_ratio >= 1.0:
        eta = 0.0
    else:
        eta = float("inf")

    lines = [
        _console_info(
            "t_env: {0} / {1} | update: {2} / {3}".format(
                env_steps,
                total_env_steps,
                update,
                total_updates,
            )
        )
    ]
    if np.isfinite(eta):
        lines.append(
            _console_info(
                "Estimated time left: {0}. Time passed: {1}".format(
                    _format_duration(eta),
                    _format_duration(elapsed),
                )
            )
        )
    else:
        lines.append(
            _console_info(
                "Estimated time left: unavailable. Time passed: {0}".format(
                    _format_duration(elapsed)
                )
            )
        )
    return lines


def _format_rule_based_progress_lines(
    episode_index: int,
    total_episodes: int,
    start_time: float,
) -> List[str]:
    now = time.time()
    elapsed = max(now - start_time, 0.0)
    if total_episodes > 0:
        progress_ratio = min(max(float(episode_index) / float(total_episodes), 0.0), 1.0)
    else:
        progress_ratio = 1.0

    if 0.0 < progress_ratio < 1.0:
        estimated_total = elapsed / progress_ratio
        eta = max(estimated_total - elapsed, 0.0)
    elif progress_ratio >= 1.0:
        eta = 0.0
    else:
        eta = float("inf")

    lines = [
        _console_info(
            "episode: {0} / {1}".format(
                episode_index,
                total_episodes,
            ),
            phase="rollout",
        )
    ]
    if np.isfinite(eta):
        lines.append(
            _console_info(
                "Estimated time left: {0}. Time passed: {1}".format(
                    _format_duration(eta),
                    _format_duration(elapsed),
                ),
                phase="rollout",
            )
        )
    else:
        lines.append(
            _console_info(
                "Estimated time left: unavailable. Time passed: {0}".format(
                    _format_duration(elapsed)
                ),
                phase="rollout",
            )
        )
    return lines


def _format_scan_progress_lines(
    completed_count: int,
    total_count: int,
    start_time: float,
    latest_experiment_name: Optional[str] = None,
) -> List[str]:
    now = time.time()
    elapsed = max(now - start_time, 0.0)
    if total_count > 0:
        progress_ratio = min(max(float(completed_count) / float(total_count), 0.0), 1.0)
    else:
        progress_ratio = 1.0

    if 0.0 < progress_ratio < 1.0:
        estimated_total = elapsed / progress_ratio
        eta = max(estimated_total - elapsed, 0.0)
    elif progress_ratio >= 1.0:
        eta = 0.0
    else:
        eta = float("inf")

    progress_message = "experiments: {0} / {1}".format(completed_count, total_count)
    if latest_experiment_name:
        progress_message += " | latest: {0}".format(latest_experiment_name)

    lines = [_console_info(progress_message, phase="scan")]
    if np.isfinite(eta):
        lines.append(
            _console_info(
                "Estimated time left: {0}. Time passed: {1}".format(
                    _format_duration(eta),
                    _format_duration(elapsed),
                ),
                phase="scan",
            )
        )
    else:
        lines.append(
            _console_info(
                "Estimated time left: unavailable. Time passed: {0}".format(
                    _format_duration(elapsed)
                ),
                phase="scan",
            )
        )
    return lines


def _format_console_recent_stats_lines(
    recent_metrics: Sequence[Mapping[str, float]],
    latest_metrics: Mapping[str, float],
    update: int,
    total_updates: int,
    env_steps: int,
    total_env_steps: int,
    stage_label: Optional[str],
) -> List[str]:
    header = "Recent Stats | update: {0} / {1} | t_env: {2} / {3}".format(
        update,
        total_updates,
        env_steps,
        total_env_steps,
    )
    if stage_label:
        header += " | stage: {0}".format(stage_label)

    ordered_keys = [
        ("rollout_f_c", "train_f_c"),
        ("rollout_gini", "train_gini"),
        ("rollout_R_mean", "train_R_mean"),
        ("rollout_payoff_mean", "train_payoff_mean"),
        ("rollout_pool_grown_mean", "train_pool_grown"),
        ("rollout_pool_mean", "train_pool"),
        ("eval_cooperation_mean", "f_c"),
        ("eval_gini_mean", "gini"),
        ("eval_return_mean", "return_mean"),
        ("eval_mean_resource", "R_mean"),
        ("eval_mean_payoff", "payoff_mean"),
        ("eval_mean_pool_grown", "pool_grown_mean"),
        ("eval_mean_pool_raw", "pool_mean"),
        ("eval_mean_total_resource", "R_total_mean"),
        ("eval_collapse_rate", "collapse_rate"),
        ("loss", "loss"),
        ("policy_loss", "policy_loss"),
        ("value_loss", "value_loss"),
        ("mean_rollout_reward", "mean_rollout_reward"),
        ("entropy", "entropy"),
        ("actor_logit_l2", "actor_logit_l2"),
        ("actor_bc_loss", "actor_bc_loss"),
        ("actor_bc_coef", "actor_bc_coef"),
        ("actor_q_coef", "actor_q_coef"),
        ("actor_grad_norm", "actor_grad_norm"),
        ("critic_grad_norm", "critic_grad_norm"),
        ("actor_lr", "actor_lr"),
        ("critic_lr", "critic_lr"),
        ("replay_size", "replay_size"),
        ("replay_demo_frac", "replay_demo"),
        ("replay_pool_power_demo_frac", "replay_pool_demo"),
        ("replay_teacher_frac", "replay_teacher"),
        ("replay_collapse_frac", "replay_collapse"),
        ("teacher_takeover_prob", "teacher_takeover"),
        ("profile_rollout_steps_per_second", "rollout_sps"),
        ("profile_rollout_collect_seconds", "rollout_collect_s"),
        ("profile_rollout_collect_worker_seconds", "worker_collect_s"),
        ("profile_rollout_env_step_seconds", "env_step_s"),
        ("profile_rollout_inference_wait_seconds", "inference_wait_s"),
        ("profile_rollout_inference_request_build_seconds", "infer_build_s"),
        ("profile_rollout_local_policy_forward_seconds", "local_forward_s"),
        ("profile_rollout_action_to_numpy_seconds", "action_to_numpy_s"),
        ("profile_rollout_transition_encode_seconds", "transition_encode_s"),
        ("profile_rollout_stack_transitions_seconds", "stack_transitions_s"),
        ("profile_rollout_shared_memory_serialize_seconds", "shm_serialize_s"),
        ("profile_rollout_shared_memory_deserialize_seconds", "shm_deserialize_s"),
        ("profile_rollout_finish_wait_seconds", "rollout_wait_s"),
        ("profile_rollout_overlap_seconds", "overlap_s"),
        ("profile_actor_sync_seconds", "actor_sync_s"),
        ("profile_actor_publish_seconds", "actor_publish_s"),
        ("profile_actor_sync_inference_server_seconds", "actor_sync_server_s"),
        ("profile_actor_sync_worker_rpc_seconds", "actor_sync_worker_s"),
        ("profile_replay_extend_seconds", "replay_extend_s"),
        ("profile_replay_sample_seconds", "replay_sample_s"),
        ("profile_batch_to_device_seconds", "to_device_s"),
        ("profile_critic_update_seconds", "critic_update_s"),
        ("profile_actor_update_seconds", "actor_update_s"),
        ("profile_target_soft_update_seconds", "target_update_s"),
        ("profile_learner_update_seconds", "learner_update_s"),
        ("profile_eval_seconds", "eval_s"),
        ("profile_on_update_seconds", "callback_s"),
        ("profile_rollout_inference_batch_size_mean", "infer_batch_mean"),
        ("profile_rollout_inference_batch_size_max", "infer_batch_max"),
        ("behavior_frac_uniform", "uniform"),
        ("behavior_frac_proportional", "proportional"),
        ("behavior_frac_constant_mix", "constant_mix"),
        ("behavior_frac_pool_power_mix", "pool_power_mix"),
        ("behavior_frac_random_logits", "random_logits"),
        ("behavior_frac_actor_logits", "actor_logits"),
    ]
    metric_rows: List[tuple[str, float]] = []
    for metric_key, label in ordered_keys:
        value = _mean_of_recent_metrics(recent_metrics, metric_key)
        if value is None:
            continue
        metric_rows.append((label, value))
    dynamic_replay_keys = sorted(
        {
            str(key)
            for metrics in recent_metrics
            for key in metrics.keys()
            if str(key).startswith("replay_source_frac_") or str(key).startswith("replay_topology_frac_")
        }
    )
    for metric_key in dynamic_replay_keys:
        value = _mean_of_recent_metrics(recent_metrics, metric_key)
        if value is None:
            continue
        if metric_key.startswith("replay_source_frac_"):
            label = "src_{0}".format(metric_key[len("replay_source_frac_"):])
        else:
            label = "topo_{0}".format(metric_key[len("replay_topology_frac_"):])
        metric_rows.append((label, value))

    lines = [_console_info(header)]
    if not metric_rows:
        lines.append("No recent numeric metrics available.")
        return lines

    for row in _format_metric_rows(metric_rows, columns=3):
        lines.append(row)

    if "eval_return_mean" in latest_metrics:
        lines.append(
            _console_info(
                "Latest eval | f_c={0:.4f}, gini={1:.4f}, return={2:.4f}, R={3:.4f}, payoff={4:.4f}, pool_grown={5:.4f}, pool={6:.4f}, collapse_rate={7:.4f}".format(
                    float(latest_metrics["eval_cooperation_mean"]),
                    float(latest_metrics["eval_gini_mean"]),
                    float(latest_metrics["eval_return_mean"]),
                    float(latest_metrics.get("eval_mean_resource", 0.0)),
                    float(latest_metrics.get("eval_mean_payoff", 0.0)),
                    float(latest_metrics.get("eval_mean_pool_grown", 0.0)),
                    float(latest_metrics.get("eval_mean_pool_raw", 0.0)),
                    float(latest_metrics.get("eval_collapse_rate", 0.0)),
                )
            )
        )
    return lines


def _tensorboard_tag_for_metric(metric_name: str) -> Optional[str]:
    if metric_name == "update":
        return None
    if metric_name.startswith("behavior_frac_"):
        return "behavior/{0}".format(metric_name[len("behavior_frac_"):])
    if metric_name.startswith("replay_source_frac_"):
        return "replay/source_frac/{0}".format(metric_name[len("replay_source_frac_"):])
    if metric_name.startswith("replay_topology_frac_"):
        return "replay/topology_frac/{0}".format(metric_name[len("replay_topology_frac_"):])
    if metric_name in {
        "replay_demo_frac",
        "replay_pool_power_demo_frac",
        "replay_teacher_frac",
        "replay_collapse_frac",
        "replay_sample_size",
    }:
        return "replay/{0}".format(metric_name)
    if metric_name.startswith("rollout_"):
        return "train_global/{0}".format(metric_name[len("rollout_"):])
    if metric_name.startswith("profile_"):
        return "profile/{0}".format(metric_name[len("profile_"):])
    if metric_name.startswith("eval_"):
        eval_key = metric_name[len("eval_"):]
        if "/" in eval_key:
            base_key, suffix = eval_key.split("/", 1)
        else:
            base_key, suffix = eval_key, None
        eval_name_mapping = {
            "cooperation_mean": "f_c",
            "gini_mean": "gini",
            "mean_resource": "R_mean",
            "mean_total_resource": "R_total_mean",
            "mean_payoff": "payoff_mean",
            "mean_pool_grown": "pool_grown_mean",
            "mean_pool_raw": "pool_mean",
            "return_mean": "return_mean",
            "collapse_rate": "collapse_rate",
            "sustainability_rate": "sustainability_rate",
        }
        mapped_key = eval_name_mapping.get(base_key, base_key)
        if suffix is not None:
            return "eval/{0}/{1}".format(mapped_key, suffix)
        return "eval/{0}".format(mapped_key)
    if metric_name in {"actor_lr", "critic_lr"}:
        return "optim/{0}".format(metric_name)
    if metric_name == "replay_size":
        return "replay/size"
    if metric_name == "curriculum_stage":
        return "curriculum/stage_index"
    if metric_name in {
        "loss",
        "policy_loss",
        "value_loss",
        "critic1_loss",
        "critic2_loss",
        "critic_loss",
        "actor_loss",
        "actor_q_loss",
        "actor_reg_loss",
    }:
        return "loss/{0}".format(metric_name)
    return "train/{0}".format(metric_name)


def _log_tensorboard_static_metadata(
    writer: Any,
    spec: Mapping[str, Any],
    graph: Mapping[int, Sequence[int]],
    env_config: SPGGConfig,
) -> None:
    summary = graph_summary(graph)
    training_schedule = _resolve_training_schedule(spec)
    static_scalars = {
        "static/graph/num_nodes": float(summary["num_nodes"]),
        "static/graph/num_edges": float(summary["num_edges"]),
        "static/graph/degree_min": float(summary["degree_min"]),
        "static/graph/degree_max": float(summary["degree_max"]),
        "static/graph/degree_mean": float(summary["degree_mean"]),
        "static/dynamics/alpha": float(env_config.alpha),
        "static/dynamics/r": float(env_config.r),
        "static/dynamics/p_mode_is_dynamic": float(env_config.p_mode == "dynamic"),
        "static/dynamics/p_max": float(env_config.p_max),
        "static/dynamics/p_c": float(env_config.p_c),
        "static/dynamics/episode_length": float(env_config.episode_length),
        "static/reward/lambda_payoff": float(env_config.reward.lambda_payoff),
        "static/reward/lambda_cooperation": float(env_config.reward.lambda_cooperation),
        "static/reward/lambda_total_resource": float(env_config.reward.lambda_total_resource),
        "static/reward/lambda_collapse": float(env_config.reward.lambda_collapse),
        "static/reward/lambda_gini": float(env_config.reward.lambda_gini),
        "static/gnn/hidden_dim": float(spec["gnn"]["hidden_dim"]),
        "static/gnn/local_hidden_dim": float(spec["gnn"].get("local_hidden_dim") or spec["gnn"]["hidden_dim"]),
        "static/gnn/score_hidden_dim": float(
            spec["gnn"].get("score_hidden_dim")
            or spec["gnn"].get("local_hidden_dim")
            or spec["gnn"]["hidden_dim"]
        ),
        "static/gnn/temperature": float(spec["gnn"]["temperature"]),
        "static/training/steps_per_update": float(
            training_schedule["effective_steps_per_update"]
        ),
        "static/training/global_env_steps_per_update": float(training_schedule["global_env_steps_per_update"]),
        "static/training/total_env_steps": float(training_schedule["total_env_steps_effective"]),
        "static/training/total_rollout_env_steps": float(training_schedule["total_rollout_env_steps_effective"]),
        "static/training/warmup_env_steps": float(training_schedule["warmup_env_steps"]),
        "static/training/eval_interval_env_steps": float(training_schedule["eval_interval_env_steps"]),
        "static/training/num_workers": float(spec["training"]["num_workers"]),
        "static/training/num_envs_per_worker": float(spec["training"].get("num_envs_per_worker", 1)),
        "static/training/overlap_rollout_and_update": float(
            1.0 if spec["training"].get("overlap_rollout_and_update", True) else 0.0
        ),
        "static/training/batch_size": float(spec["training"]["batch_size"]),
        "static/training/graph_batch_chunk_size": float(spec["training"].get("graph_batch_chunk_size", 16)),
        "static/training/replay_capacity": float(spec["training"]["replay_capacity"]),
        "static/training/replay_recent_fraction": float(spec["training"].get("replay_recent_fraction", 0.50)),
        "static/training/replay_long_term_fraction": float(spec["training"].get("replay_long_term_fraction", 0.35)),
        "static/training/replay_demo_fraction": float(spec["training"].get("replay_demo_fraction", 0.15)),
        "static/training/replay_max_collapse_sample_ratio": float(
            spec["training"].get("replay_max_collapse_sample_ratio", 0.20)
        ),
        "static/training/demo_pretrain_enabled": float(
            1.0 if spec["training"].get("demo_pretrain_enabled", False) else 0.0
        ),
        "static/training/demo_collection_env_steps": float(spec["training"].get("demo_collection_env_steps", 0)),
        "static/training/actor_bc_pretrain_updates": float(spec["training"].get("actor_bc_pretrain_updates", 0)),
        "static/training/critic_pretrain_updates": float(spec["training"].get("critic_pretrain_updates", 0)),
        "static/training/demo_critic_pretrain_n_step": float(spec["training"].get("demo_critic_pretrain_n_step", 20)),
        "static/training/save_demo_pretrain_checkpoint": float(
            1.0 if spec["training"].get("save_demo_pretrain_checkpoint", False) else 0.0
        ),
        "static/training/stop_after_demo_pretrain": float(
            1.0 if spec["training"].get("stop_after_demo_pretrain", False) else 0.0
        ),
        "static/training/teacher_takeover_enabled": float(
            1.0 if spec["training"].get("teacher_takeover_enabled", True) else 0.0
        ),
        "static/training/teacher_takeover_start_prob": float(spec["training"].get("teacher_takeover_start_prob", 0.8)),
        "static/training/teacher_takeover_end_prob": float(spec["training"].get("teacher_takeover_end_prob", 0.0)),
        "static/training/teacher_takeover_decay_end_fraction": float(
            spec["training"].get("teacher_takeover_decay_end_fraction", 0.30)
        ),
        "static/training/online_actor_q_coef_initial": float(
            spec["training"].get("online_actor_q_coef_initial", 0.2)
        ),
        "static/training/online_actor_q_coef_final": float(
            spec["training"].get("online_actor_q_coef_final", 1.0)
        ),
        "static/training/online_actor_q_coef_ramp_end_fraction": float(
            spec["training"].get("online_actor_q_coef_ramp_end_fraction", 0.30)
        ),
        "static/training/critic_huber_delta": float(spec["training"].get("critic_huber_delta", 1.0)),
        "static/training/rollout_inference_batch_timeout_ms": float(
            spec["training"].get("rollout_inference_batch_timeout_ms", 2.0)
        ),
    }
    rollout_num_threads = spec["training"].get("rollout_num_threads")
    if rollout_num_threads is not None:
        static_scalars["static/training/rollout_num_threads"] = float(rollout_num_threads)
    for tag, value in static_scalars.items():
        writer.add_scalar(tag, value, 0)
    writer.add_text("static/training/learner_device", str(spec["training"]["device"]), 0)
    rollout_device = spec["training"].get("rollout_device", "cpu")
    if isinstance(rollout_device, (list, tuple)):
        rollout_device_text = ",".join(str(item) for item in rollout_device)
    else:
        rollout_device_text = str(rollout_device)
    writer.add_text("static/training/rollout_device", rollout_device_text, 0)
    writer.add_text(
        "static/training/rollout_inference_mode",
        str(spec["training"].get("rollout_inference_mode", "local")),
        0,
    )
    writer.add_text("static/training/replay_strategy", str(spec["training"].get("replay_strategy", "fifo")), 0)
    writer.add_text(
        "static/training/replay_demo_behavior_source",
        str(spec["training"].get("replay_demo_behavior_source", "pool_power_mix")),
        0,
    )
    writer.add_text(
        "static/training/demo_collection_behavior_source",
        str(spec["training"].get("demo_collection_behavior_source", "pool_power_mix")),
        0,
    )
    writer.add_text(
        "static/training/demo_critic_pretrain_target_mode",
        str(spec["training"].get("demo_critic_pretrain_target_mode", "n_step")),
        0,
    )
    writer.add_text(
        "static/training/demo_pretrain_checkpoint_name",
        str(spec["training"].get("demo_pretrain_checkpoint_name", "demo_pretrained.pt")),
        0,
    )
    writer.add_text(
        "static/training/teacher_takeover_behavior_source",
        str(spec["training"].get("teacher_takeover_behavior_source", "pool_power_mix")),
        0,
    )
    writer.add_text(
        "static/training/critic_loss_type",
        str(spec["training"].get("critic_loss_type", "huber")),
        0,
    )
    replay_topology_names_value = spec["training"].get("replay_topology_names")
    if replay_topology_names_value is None:
        randomization = spec.get("domain_randomization", {})
        if bool(randomization.get("enabled", False)):
            replay_topology_names_text = ",".join(str(item) for item in randomization.get("network_types", ("fixed",)))
        else:
            replay_topology_names_text = "fixed"
    else:
        replay_topology_names_text = ",".join(str(item) for item in replay_topology_names_value)
    writer.add_text("static/training/replay_topology_names", replay_topology_names_text, 0)
    demo_collection_network_types_value = spec["training"].get("demo_collection_network_types")
    if demo_collection_network_types_value is None:
        randomization = spec.get("domain_randomization", {})
        if bool(spec["training"].get("demo_collection_use_domain_randomization", True)) and bool(
            randomization.get("enabled", False)
        ):
            demo_collection_network_types_text = ",".join(
                str(item) for item in randomization.get("network_types", ("fixed",))
            )
        else:
            demo_collection_network_types_text = "fixed"
    else:
        demo_collection_network_types_text = ",".join(str(item) for item in demo_collection_network_types_value)
    writer.add_text("static/training/demo_collection_network_types", demo_collection_network_types_text, 0)


def _log_tensorboard_custom_layout(writer: Any) -> None:
    writer.add_custom_scalars(
        {
            "Train vs Eval": {
                "f_c": ["Multiline", ["train_global/f_c", "eval/f_c"]],
                "R_mean": ["Multiline", ["train_global/R_mean", "eval/R_mean"]],
                "gini": ["Multiline", ["train_global/gini", "eval/gini"]],
                "payoff_mean": ["Multiline", ["train_global/payoff_mean", "eval/payoff_mean"]],
                "pool_grown_mean": ["Multiline", ["train_global/pool_grown_mean", "eval/pool_grown_mean"]],
                "pool_mean": ["Multiline", ["train_global/pool_mean", "eval/pool_mean"]],
            }
        }
    )


def _log_tensorboard_demo_pretrain_summary(
    writer: Any,
    summary: Mapping[str, Any],
) -> None:
    scalar_keys = {
        "demo_collection_env_steps": "demo_pretrain/demo_collection_env_steps",
        "demo_replay_size_after_collection": "demo_pretrain/demo_replay_size_after_collection",
        "actor_bc_updates": "demo_pretrain/actor_bc_updates",
        "critic_pretrain_updates": "demo_pretrain/critic_pretrain_updates",
        "actor_bc_loss_last": "demo_pretrain/actor_bc_loss_last",
        "critic_loss_last": "demo_pretrain/critic_loss_last",
        "demo_return_target_mean": "demo_pretrain/demo_return_target_mean",
        "demo_return_target_std": "demo_pretrain/demo_return_target_std",
        "seconds_collection": "demo_pretrain/seconds_collection",
        "seconds_actor_bc": "demo_pretrain/seconds_actor_bc",
        "seconds_critic": "demo_pretrain/seconds_critic",
    }
    for key, tag in scalar_keys.items():
        if key in summary and summary[key] is not None:
            writer.add_scalar(tag, float(summary[key]), 0)
    if "behavior_source" in summary and summary["behavior_source"] is not None:
        writer.add_text("demo_pretrain/behavior_source", str(summary["behavior_source"]), 0)
    if "critic_target_mode" in summary and summary["critic_target_mode"] is not None:
        writer.add_text("demo_pretrain/critic_target_mode", str(summary["critic_target_mode"]), 0)
    if "dataset_path" in summary and summary["dataset_path"] is not None:
        writer.add_text("demo_pretrain/dataset_path", str(summary["dataset_path"]), 0)


def _log_tensorboard_demo_pretrain_eval_summary(
    writer: Any,
    summary: Mapping[str, Any],
) -> None:
    if "num_episodes" in summary and summary["num_episodes"] is not None:
        writer.add_scalar("demo_pretrain/eval_num_episodes", float(summary["num_episodes"]), 0)
    for key, value in summary.items():
        if key in {"num_episodes", "checkpoint_path"} or not isinstance(value, (int, float)):
            continue
        writer.add_scalar("demo_pretrain/eval_{0}".format(key), float(value), 0)
    if "checkpoint_eval_return_mean" in summary and summary["checkpoint_eval_return_mean"] is not None:
        writer.add_scalar(
            "demo_pretrain/checkpoint_eval_return_mean",
            float(summary["checkpoint_eval_return_mean"]),
            0,
        )
    if "checkpoint_path" in summary and summary["checkpoint_path"] is not None:
        writer.add_text("demo_pretrain/checkpoint_path", str(summary["checkpoint_path"]), 0)


def _log_tensorboard_update_metrics(
    writer: Any,
    metrics: Mapping[str, float],
    curriculum_stages: Optional[Sequence[Mapping[str, Any]]] = None,
    stage_log_state: Optional[Dict[str, Any]] = None,
) -> None:
    update = int(metrics["update"])
    tensorboard_step = int(metrics.get("global_env_steps", update))
    for key, value in metrics.items():
        if not isinstance(value, (int, float)):
            continue
        tag = _tensorboard_tag_for_metric(key)
        if tag is None:
            continue
        writer.add_scalar(tag, float(value), tensorboard_step)

    if curriculum_stages is not None and stage_log_state is not None and "curriculum_stage" in metrics:
        stage_index = int(metrics["curriculum_stage"])
        if stage_log_state.get("last_stage_index") != stage_index:
            stage_label = str(stage_index)
            if 0 <= stage_index < len(curriculum_stages):
                stage_label = str(curriculum_stages[stage_index].get("label", stage_label))
            writer.add_text("curriculum/active_stage_label", stage_label, tensorboard_step)
            stage_log_state["last_stage_index"] = stage_index


def _log_tensorboard_post_training_evaluation(
    writer: Any,
    episode_summaries: Sequence[Mapping[str, float]],
    final_env_steps: int,
) -> None:
    if not episode_summaries:
        return

    for summary in episode_summaries:
        episode_index = int(summary["episode_index"])
        writer.add_scalar("post_eval/episode_return", float(summary["episode_return"]), episode_index)
        writer.add_scalar(
            "post_eval/final_actual_cooperation",
            float(summary["final_actual_cooperation"]),
            episode_index,
        )
        writer.add_scalar("post_eval/final_mean_resource", float(summary["final_mean_resource"]), episode_index)
        writer.add_scalar("post_eval/final_mean_pool_grown", float(summary["final_mean_pool_grown"]), episode_index)
        writer.add_scalar("post_eval/final_mean_consumption", float(summary["final_mean_consumption"]), episode_index)
        writer.add_scalar("post_eval/final_mean_payoff", float(summary["final_mean_payoff"]), episode_index)
        writer.add_scalar("post_eval/final_gini", float(summary["final_gini"]), episode_index)

    writer.add_scalar(
        "post_eval/return_mean",
        float(np.mean([summary["episode_return"] for summary in episode_summaries])),
        final_env_steps,
    )
    writer.add_scalar(
        "post_eval/final_actual_cooperation_mean",
        float(np.mean([summary["final_actual_cooperation"] for summary in episode_summaries])),
        final_env_steps,
    )
    writer.add_scalar(
        "post_eval/final_mean_resource_mean",
        float(np.mean([summary["final_mean_resource"] for summary in episode_summaries])),
        final_env_steps,
    )
    writer.add_scalar(
        "post_eval/final_mean_pool_grown_mean",
        float(np.mean([summary["final_mean_pool_grown"] for summary in episode_summaries])),
        final_env_steps,
    )
    writer.add_scalar(
        "post_eval/final_mean_consumption_mean",
        float(np.mean([summary["final_mean_consumption"] for summary in episode_summaries])),
        final_env_steps,
    )
    writer.add_scalar(
        "post_eval/final_mean_payoff_mean",
        float(np.mean([summary["final_mean_payoff"] for summary in episode_summaries])),
        final_env_steps,
    )
    writer.add_scalar(
        "post_eval/final_gini_mean",
        float(np.mean([summary["final_gini"] for summary in episode_summaries])),
        final_env_steps,
    )


def run_rule_based_mode(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    env_config: SPGGConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    env = SPGGEnv(env_config, graph)
    run_mode = spec["run_mode"]
    rollout = spec["rollout"]
    training = spec["training"]
    quiet_console = _runtime_quiet_console(spec)
    tensorboard = spec.get("tensorboard", {})
    console_progress_logs = bool(tensorboard.get("console_progress_logs", True))
    console_progress_interval = int(tensorboard.get("console_progress_interval", 1))
    total_episodes = int(rollout["episodes"])

    if run_mode == "uniform":
        policy = UniformAllocationPolicy()
    elif run_mode == "proportional":
        policy = ProportionalContributionPolicy()
    elif run_mode == "constant_mix":
        policy = ConstantMixAllocationPolicy(float(training["warmup_constant_mix_omega"]))
    elif run_mode == "pool_power_mix":
        policy = PoolPowerMixAllocationPolicy(float(training["warmup_pool_power_k"]))
    else:
        raise ValueError("Unsupported rule-based run_mode: {0}".format(run_mode))

    episode_summaries: List[Dict[str, float]] = []
    episode_returns: List[float] = []
    rollout_start_time = time.time()

    for episode_index in range(1, total_episodes + 1):
        observation = env.reset(seed=spec["seed"] + episode_index - 1)
        done = False
        episode_return = 0.0
        time_index = 0
        history = [capture_record(time_index, observation)]

        while not done:
            allocation = policy.allocate(observation)
            observation, reward, done, info = env.step(allocation)
            time_index += 1
            episode_return += reward
            history.append(capture_record(time_index, observation, reward=reward, info=info))

        summary = summarize_episode(history)
        summary["episode_return"] = float(episode_return)
        summary["episode_index"] = float(episode_index)

        episode_summaries.append(summary)
        episode_returns.append(episode_return)

        if not quiet_console:
            print(
                "[Episode {0:03d}] return={1:.6f}, final_actual_cooperation={2:.6f}, final_gini={3:.6f}".format(
                    episode_index,
                    episode_return,
                    summary["final_actual_cooperation"],
                    summary["final_gini"],
                )
            )
            should_log_progress = console_progress_logs and (
                episode_index == 1
                or episode_index == total_episodes
                or episode_index % max(console_progress_interval, 1) == 0
            )
            if should_log_progress:
                for line in _format_rule_based_progress_lines(
                    episode_index=episode_index,
                    total_episodes=total_episodes,
                    start_time=rollout_start_time,
                ):
                    print(line)

        save_visualizations_for_history(
            spec=spec,
            graph=graph,
            history=history,
            output_dir=output_dir,
            episode_index=episode_index,
            phase_name="rollout",
        )

    return {
        "experiment_name": spec["experiment_name"],
        "run_mode": run_mode,
        "network_type": spec["network"]["type"],
        "episode_summaries": episode_summaries,
        "summary": summarize_rule_based_episodes(episode_summaries, episode_returns),
    }


def run_trained_policy_evaluation(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    env_config: SPGGConfig,
    policy: Any,
    output_dir: Path,
) -> List[Dict[str, float]]:
    import torch

    eval_env = SPGGEnv(env_config, graph)
    episode_summaries: List[Dict[str, float]] = []
    rollout = spec["rollout"]

    for episode_index in range(1, rollout["post_training_eval_episodes"] + 1):
        observation = eval_env.reset(seed=spec["seed"] + 10_000 + episode_index)
        done = False
        episode_return = 0.0
        time_index = 0
        history = [capture_record(time_index, observation)]

        while not done:
            with torch.no_grad():
                action_output = policy.deterministic_action(observation)
            allocation = action_output.allocation_matrix.detach().cpu().numpy()
            observation, reward, done, info = eval_env.step(allocation)
            time_index += 1
            episode_return += reward
            history.append(capture_record(time_index, observation, reward=reward, info=info))

        summary = summarize_episode(history)
        summary["episode_return"] = float(episode_return)
        summary["episode_index"] = float(episode_index)
        episode_summaries.append(summary)

        print(
            "[Post-Train Eval {0:03d}] return={1:.6f}, final_actual_cooperation={2:.6f}, final_gini={3:.6f}".format(
                episode_index,
                episode_return,
                summary["final_actual_cooperation"],
                summary["final_gini"],
            )
        )

        save_visualizations_for_history(
            spec=spec,
            graph=graph,
            history=history,
            output_dir=output_dir,
            episode_index=episode_index,
            phase_name="post_train_eval",
        )

    return episode_summaries


def run_gnn_training_mode(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    env_config: SPGGConfig,
    output_dir: Path,
) -> Dict[str, Any]:
    from Project1.trainer import CentralizedActorCriticTrainer
    import torch

    env = SPGGEnv(env_config, graph)
    eval_env = SPGGEnv(env_config, graph)
    trainer_config = build_trainer_config(spec)
    training_schedule = _resolve_training_schedule(spec)
    randomization_config = build_domain_randomization_config(spec)
    evaluation = spec.get("evaluation", {})
    eval_env_factories = build_evaluation_env_factories(spec)
    curriculum_stages = build_training_curriculum(spec)
    training = spec["training"]
    tensorboard = spec.get("tensorboard", {})
    resume_from_checkpoint = training.get("resume_from_checkpoint")
    external_demo_replay = None
    external_demo_summary = None
    effective_trainer_config = trainer_config
    policy: Any = None
    trainer: Any = None
    steps_source = (
        "dynamics.episode_length"
        if training.get("use_episode_length_as_steps_per_update", False)
        else "training.steps_per_update"
    )
    writer: Any = None
    try:
        rollout_device = trainer_config.rollout_device
        rollout_device_text = (
            ",".join(str(item) for item in rollout_device)
            if isinstance(rollout_device, tuple)
            else str(rollout_device)
        )
        print(
            "Train CFG : steps_per_update={0}, source={1}, total_env_steps={2} ({3}, excludes warmup), warmup_env_steps={4} ({5}), total_rollout_env_steps={6}, total_updates={7}".format(
                trainer_config.steps_per_update,
                steps_source,
                training_schedule["total_env_steps_effective"],
                training_schedule["total_updates_source"],
                training_schedule["warmup_env_steps"],
                training_schedule["warmup_steps_source"],
                training_schedule["total_rollout_env_steps_effective"],
                trainer_config.total_updates,
            )
        )
        print(
            "Train DEV : learner_device={0}, rollout_device={1}, rollout_mode={2}, rollout_batch_timeout_ms={3}, rollout_num_threads={4}, num_envs_per_worker={5}, overlap_rollout_update={6}".format(
                trainer_config.device,
                rollout_device_text,
                trainer_config.rollout_inference_mode,
                trainer_config.rollout_inference_batch_timeout_ms,
                trainer_config.rollout_num_threads if trainer_config.rollout_num_threads is not None else "default",
                trainer_config.num_envs_per_worker,
                trainer_config.overlap_rollout_and_update,
            )
        )
        print(
            "Replay CFG: strategy={0}, capacity={1}, topology_names={2}, recent={3:.2f}, long_term={4:.2f}, demo={5:.2f}, demo_source={6}, collapse_cap_per_topology={7:.2f}".format(
                trainer_config.replay_strategy,
                trainer_config.replay_capacity,
                ",".join(trainer_config.replay_topology_names),
                trainer_config.replay_recent_fraction,
                trainer_config.replay_long_term_fraction,
                trainer_config.replay_demo_fraction,
                trainer_config.replay_demo_behavior_source,
                trainer_config.replay_max_collapse_sample_ratio,
            )
        )
        if bool(trainer_config.demo_pretrain_enabled):
            demo_collection_network_types = (
                ",".join(trainer_config.demo_collection_network_types)
                if trainer_config.demo_collection_network_types
                else (
                    ",".join(randomization_config.network_types)
                    if bool(trainer_config.demo_collection_use_domain_randomization) and bool(randomization_config.enabled)
                    else "fixed"
                )
            )
            print(
                "Demo CFG : collection_steps={0}, behavior={1}, use_domain_randomization={2}, network_types={3}, runtime={4}, actor_bc_updates={5}, critic_pretrain_updates={6}, critic_target={7}, n_step={8}, batch_size={9}, dataset_path={10}, save_ckpt={11}, ckpt_name={12}, stop_after={13}".format(
                    trainer_config.demo_collection_env_steps,
                    trainer_config.demo_collection_behavior_source,
                    trainer_config.demo_collection_use_domain_randomization,
                    demo_collection_network_types,
                    trainer_config.demo_collection_runtime,
                    trainer_config.actor_bc_pretrain_updates,
                    trainer_config.critic_pretrain_updates,
                    trainer_config.demo_critic_pretrain_target_mode,
                    trainer_config.demo_critic_pretrain_n_step,
                    trainer_config.demo_pretrain_batch_size
                    if trainer_config.demo_pretrain_batch_size is not None
                    else trainer_config.batch_size,
                    trainer_config.demo_dataset_save_path or "None",
                    bool(training.get("save_demo_pretrain_checkpoint", False)),
                    str(training.get("demo_pretrain_checkpoint_name", "demo_pretrained.pt")),
                    bool(training.get("stop_after_demo_pretrain", False)),
                )
            )
        print(
            "Stab CFG : teacher_takeover={0}({1}->{2}, end@{3}), actor_q_coef={4}->{5} end@{6}, critic_loss={7}, huber_delta={8}, grad_clip(actor={9}, critic={10})".format(
                trainer_config.teacher_takeover_enabled,
                trainer_config.teacher_takeover_start_prob,
                trainer_config.teacher_takeover_end_prob,
                trainer_config.teacher_takeover_decay_end_fraction,
                trainer_config.online_actor_q_coef_initial,
                trainer_config.online_actor_q_coef_final,
                trainer_config.online_actor_q_coef_ramp_end_fraction,
                trainer_config.critic_loss_type,
                trainer_config.critic_huber_delta,
                trainer_config.actor_grad_clip_norm,
                trainer_config.critic_grad_clip_norm,
            )
        )
        print(
            "Eval CFG  : mode={0}, periodic_eval_episodes={1}, eval_interval_env_steps={2} ({3}), eval_interval_updates={4}".format(
                "custom_env_families({0})".format(len(eval_env_factories))
                if eval_env_factories is not None
                else "fixed_base_env",
                trainer_config.eval_episodes,
                training_schedule["eval_interval_env_steps"],
                training_schedule["eval_interval_source"],
                trainer_config.eval_interval,
            )
        )
        if curriculum_stages:
            stage_parts = [
                "{0}@{1}[{2}]".format(
                    stage["label"],
                    stage["activate_at_update"],
                    ",".join(
                        "{0}:{1:.2f}".format(network_type, float(weight))
                        for network_type, weight in zip(
                            stage["train_network_types"],
                            stage["train_network_type_weights"],
                        )
                    ),
                )
                for stage in curriculum_stages
            ]
            print("Curriculum: {0}".format(" | ".join(stage_parts)))

        if _should_use_external_demo_collection(
            trainer_config,
            resume_from_checkpoint=resume_from_checkpoint,
        ):
            external_demo_replay, external_demo_summary = run_external_demo_collection(
                spec=spec,
                graph=graph,
                env_config=env_config,
                trainer_config=trainer_config,
                randomization_config=randomization_config,
            )
            effective_trainer_config = replace(trainer_config, demo_collection_env_steps=0)

        policy = build_gnn_policy(spec)
        trainer = CentralizedActorCriticTrainer(
            env=env,
            policy=policy,
            eval_env=eval_env,
            config=effective_trainer_config,
            randomization=randomization_config,
            eval_env_factories=eval_env_factories,
            curriculum_stages=curriculum_stages,
        )
        if external_demo_replay is not None:
            trainer.preload_demo_replay(external_demo_replay, external_demo_summary)

        checkpoint_dir = output_dir / "checkpoints"
        should_save_checkpoints = bool(training.get("save_checkpoints", False))
        save_final_checkpoint = bool(training.get("save_final_checkpoint", True))
        save_best_checkpoint = bool(training.get("save_best_checkpoint", True))
        save_demo_pretrain_checkpoint = bool(training.get("save_demo_pretrain_checkpoint", False))
        demo_pretrain_checkpoint_name = str(training.get("demo_pretrain_checkpoint_name", "demo_pretrained.pt"))
        stop_after_demo_pretrain = bool(training.get("stop_after_demo_pretrain", False))
        if stop_after_demo_pretrain and not save_demo_pretrain_checkpoint:
            raise ValueError(
                "training.stop_after_demo_pretrain=True requires "
                "training.save_demo_pretrain_checkpoint=True."
            )
        checkpoint_interval = int(training.get("checkpoint_interval", 0))
        checkpoint_mode = str(training.get("checkpoint_mode", "lightweight"))
        if checkpoint_mode not in {"lightweight", "full_resume"}:
            raise ValueError("training.checkpoint_mode must be one of {'lightweight', 'full_resume'}.")
        best_eval_return = float("-inf")
        demo_pretrain_checkpoint_path: str | None = None
        demo_pretrain_eval_summary: dict[str, Any] | None = None
        resumed_update = 0
        tensorboard_enabled = bool(tensorboard.get("enabled", False))
        console_progress_logs = bool(tensorboard.get("console_progress_logs", True))
        console_progress_interval = int(tensorboard.get("console_progress_interval", 50))
        console_training_logs = bool(tensorboard.get("console_training_logs", False))
        console_log_interval = int(tensorboard.get("console_log_interval", 50))
        console_recent_window_updates = int(tensorboard.get("console_recent_window_updates", 50))
        tensorboard_stage_log_state: Dict[str, Any] = {"last_stage_index": None}
        training_start_time = time.time()
        effective_steps_per_update = int(trainer_config.steps_per_update) * int(trainer_config.num_workers)
        total_env_steps = int(training_schedule["total_env_steps_effective"])
        warmup_env_steps = int(training_schedule["warmup_env_steps"])
        recent_metrics: deque[dict[str, float]] = deque(maxlen=max(console_recent_window_updates, 1))
        resumed_global_env_steps = int(getattr(trainer, "global_env_steps", 0))
        demo_pretrain_summary_logged = False
        demo_pretrain_eval_summary_logged = False

        def _current_stage_label(metrics: Mapping[str, float]) -> Optional[str]:
            if "curriculum_stage" not in metrics:
                return None
            stage_index = int(metrics["curriculum_stage"])
            if curriculum_stages and 0 <= stage_index < len(curriculum_stages):
                return str(curriculum_stages[stage_index].get("label", stage_index))
            return str(stage_index)

        def _load_checkpoint_payload(checkpoint_path: Path) -> Dict[str, Any]:
            try:
                return torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except TypeError:
                return torch.load(checkpoint_path, map_location="cpu")

        if should_save_checkpoints or save_final_checkpoint or save_best_checkpoint or save_demo_pretrain_checkpoint:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

        if resume_from_checkpoint:
            checkpoint_path = Path(str(resume_from_checkpoint)).expanduser()
            if not checkpoint_path.is_absolute():
                checkpoint_path = PROJECT_ROOT / checkpoint_path
            if not checkpoint_path.exists():
                raise FileNotFoundError("Checkpoint path does not exist: {0}".format(checkpoint_path))
            checkpoint_payload = _load_checkpoint_payload(checkpoint_path)
            resumed_checkpoint_mode = trainer.load_checkpoint(checkpoint_payload)
            resumed_update = int(trainer.completed_updates)
            resumed_global_env_steps = int(trainer.global_env_steps)
            best_eval_return = float(checkpoint_payload.get("best_eval_return_so_far", float("-inf")))
            if checkpoint_payload.get("demo_pretrain_eval_summary") is not None:
                demo_pretrain_eval_summary = dict(checkpoint_payload["demo_pretrain_eval_summary"])
            if bool(checkpoint_payload.get("is_demo_pretrain_checkpoint", False)):
                demo_pretrain_checkpoint_path = str(checkpoint_path)
                if save_best_checkpoint:
                    best_checkpoint_path = checkpoint_dir / "best_eval.pt"
                    if not best_checkpoint_path.exists():
                        torch.save(checkpoint_payload, best_checkpoint_path)
                        print(
                            "Resume    : seeded best checkpoint from demo-pretrain checkpoint -> {0}".format(
                                best_checkpoint_path
                            )
                        )
            print(
                "Resume    : checkpoint={0}, resume_update={1}, checkpoint_mode={2}".format(
                    checkpoint_path,
                    resumed_update,
                    resumed_checkpoint_mode,
                )
            )

        if tensorboard_enabled:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "TensorBoard logging is enabled, but torch.utils.tensorboard is unavailable. "
                    "Install the 'tensorboard' package or disable spec['tensorboard']['enabled']."
                ) from exc

            tensorboard_root = output_dir / str(tensorboard.get("subdir", "tensorboard"))
            run_timestamp = datetime.now().strftime("%m%d_%H%M%S")
            tensorboard_dir = tensorboard_root / run_timestamp
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(
                log_dir=str(tensorboard_dir),
                flush_secs=int(tensorboard.get("flush_secs", 30)),
            )
            print("TensorBoard: log_dir={0}".format(tensorboard_dir))
            if bool(tensorboard.get("write_static_scalars", True)):
                _log_tensorboard_static_metadata(writer, spec, graph, env_config)
            _log_tensorboard_custom_layout(writer)
            if bool(tensorboard.get("write_config_text", True)):
                writer.add_text(
                    "config/spec_json",
                    "```json\n{0}\n```".format(json.dumps(spec, ensure_ascii=False, indent=2)),
                    0,
                )
            writer.flush()

        def _save_checkpoint(
            filename: str,
            update: int,
            metrics: Mapping[str, float],
            *,
            checkpoint_mode_override: str | None = None,
            best_eval_return_override: float | None = None,
            extra_payload: Mapping[str, Any] | None = None,
            log_prefix: str = "Checkpoint saved",
        ) -> Path:
            checkpoint_path = checkpoint_dir / filename
            payload = trainer.build_checkpoint(
                update=update,
                metrics=metrics,
                checkpoint_mode=checkpoint_mode_override or checkpoint_mode,
            )
            payload["best_eval_return_so_far"] = float(
                best_eval_return if best_eval_return_override is None else best_eval_return_override
            )
            if extra_payload is not None:
                payload.update(dict(extra_payload))
            torch.save(payload, checkpoint_path)
            print("{0}: {1}".format(log_prefix, checkpoint_path))
            return checkpoint_path

        def _build_demo_pretrain_eval_summary(
            *,
            num_episodes: int,
            raw_metrics: Mapping[str, float],
            checkpoint_path: str | None = None,
        ) -> dict[str, Any]:
            summary: dict[str, Any] = {"num_episodes": int(num_episodes)}
            for key, value in raw_metrics.items():
                summary[str(key)] = float(value)
            summary["checkpoint_eval_return_mean"] = float(raw_metrics.get("return_mean", 0.0))
            summary["checkpoint_path"] = checkpoint_path
            return summary

        def _demo_pretrain_eval_metrics(raw_metrics: Mapping[str, float]) -> dict[str, float]:
            metrics: dict[str, float] = {
                "update": float(int(trainer.completed_updates)),
                "global_env_steps": float(int(trainer.global_env_steps)),
            }
            for key, value in raw_metrics.items():
                metrics["eval_{0}".format(key)] = float(value)
            return metrics

        def _on_update(metrics: dict[str, float]) -> None:
            nonlocal best_eval_return
            update = int(metrics["update"])
            recent_metrics.append(dict(metrics))
            if should_save_checkpoints and checkpoint_interval > 0 and update % checkpoint_interval == 0:
                _save_checkpoint("update_{0:06d}.pt".format(update), update=update, metrics=metrics)
                _save_checkpoint("latest.pt", update=update, metrics=metrics)
            if save_best_checkpoint and "eval_return_mean" in metrics:
                eval_return = float(metrics["eval_return_mean"])
                if eval_return > best_eval_return:
                    best_eval_return = eval_return
                    _save_checkpoint("best_eval.pt", update=update, metrics=metrics)
            if writer is not None:
                _log_tensorboard_update_metrics(
                    writer,
                    metrics,
                    curriculum_stages=curriculum_stages,
                    stage_log_state=tensorboard_stage_log_state,
                )
            global_env_steps = int(metrics.get("global_env_steps", update * effective_steps_per_update))
            env_steps = max(global_env_steps - warmup_env_steps, 0)
            should_log_progress = console_progress_logs and (
                update == resumed_update + 1
                or update == int(trainer_config.total_updates)
                or update % max(console_progress_interval, 1) == 0
            )
            should_log_recent_stats = console_training_logs and (
                update == resumed_update + 1
                or update == int(trainer_config.total_updates)
                or update % max(console_log_interval, 1) == 0
            )
            if should_log_progress:
                resumed_training_env_steps = max(resumed_global_env_steps - warmup_env_steps, 0)
                session_env_steps = max(env_steps - resumed_training_env_steps, 0)
                session_total_env_steps = max(total_env_steps - resumed_training_env_steps, 0)
                for line in _format_console_progress_lines(
                    update=update,
                    total_updates=int(trainer_config.total_updates),
                    env_steps=env_steps,
                    total_env_steps=total_env_steps,
                    start_time=training_start_time,
                    eta_env_steps=session_env_steps,
                    eta_total_env_steps=session_total_env_steps,
                ):
                    print(line)
            if should_log_recent_stats:
                for line in _format_console_recent_stats_lines(
                    recent_metrics=list(recent_metrics),
                    latest_metrics=metrics,
                    update=update,
                    total_updates=int(trainer_config.total_updates),
                    env_steps=env_steps,
                    total_env_steps=total_env_steps,
                    stage_label=_current_stage_label(metrics),
                ):
                    print(line)

        should_run_demo_pretrain = (
            int(trainer.completed_updates) == 0
            and not bool(getattr(trainer, "demo_pretrain_completed", False))
            and bool(effective_trainer_config.demo_pretrain_enabled)
        )
        if should_run_demo_pretrain:
            trainer._run_demo_pretrain()
            demo_pretrain_summary = (
                dict(trainer.demo_pretrain_summary)
                if getattr(trainer, "demo_pretrain_summary", None) is not None
                else None
            )
            if demo_pretrain_summary is not None:
                print(
                    "Demo Summary: replay={0:.0f}, actor_bc_updates={1:.0f}, critic_pretrain_updates={2:.0f}, critic_target={3}, target_mean={4:.6f}, target_std={5:.6f}, actor_bc_loss_last={6:.6f}, critic_loss_last={7:.6f}, seconds={{collection:{8:.3f}, actor_bc:{9:.3f}, critic:{10:.3f}}}".format(
                        float(demo_pretrain_summary.get("demo_replay_size_after_collection", 0.0)),
                        float(demo_pretrain_summary.get("actor_bc_updates", 0.0)),
                        float(demo_pretrain_summary.get("critic_pretrain_updates", 0.0)),
                        str(demo_pretrain_summary.get("critic_target_mode", "n_step")),
                        float(demo_pretrain_summary.get("demo_return_target_mean", 0.0)),
                        float(demo_pretrain_summary.get("demo_return_target_std", 0.0)),
                        float(demo_pretrain_summary.get("actor_bc_loss_last", 0.0)),
                        float(demo_pretrain_summary.get("critic_loss_last", 0.0)),
                        float(demo_pretrain_summary.get("seconds_collection", 0.0)),
                        float(demo_pretrain_summary.get("seconds_actor_bc", 0.0)),
                        float(demo_pretrain_summary.get("seconds_critic", 0.0)),
                    )
                )
                if writer is not None:
                    _log_tensorboard_demo_pretrain_summary(writer, demo_pretrain_summary)
                demo_pretrain_summary_logged = True

            demo_pretrain_eval_episodes = max(1, min(int(effective_trainer_config.eval_episodes), 4))
            raw_demo_pretrain_eval = trainer.evaluate(num_episodes=demo_pretrain_eval_episodes)
            pretrain_eval_return = float(raw_demo_pretrain_eval.get("return_mean", 0.0))
            best_eval_return = pretrain_eval_return
            pretrain_eval_metrics = _demo_pretrain_eval_metrics(raw_demo_pretrain_eval)
            print(
                "Demo Pretrain Eval | episodes={0} | return_mean={1:.6f} | cooperation_mean={2:.6f} | gini_mean={3:.6f} | collapse_rate={4:.6f}".format(
                    demo_pretrain_eval_episodes,
                    pretrain_eval_return,
                    float(raw_demo_pretrain_eval.get("cooperation_mean", 0.0)),
                    float(raw_demo_pretrain_eval.get("gini_mean", 0.0)),
                    float(raw_demo_pretrain_eval.get("collapse_rate", 0.0)),
                )
            )

            if save_best_checkpoint:
                _save_checkpoint(
                    "best_eval.pt",
                    update=int(trainer.completed_updates),
                    metrics=pretrain_eval_metrics,
                )

            if save_demo_pretrain_checkpoint:
                checkpoint_path = _save_checkpoint(
                    demo_pretrain_checkpoint_name,
                    update=int(trainer.completed_updates),
                    metrics=pretrain_eval_metrics,
                    checkpoint_mode_override="full_resume",
                    best_eval_return_override=pretrain_eval_return,
                    extra_payload={
                        "is_demo_pretrain_checkpoint": True,
                    },
                    log_prefix="Demo Pretrain Checkpoint saved",
                )
                demo_pretrain_checkpoint_path = str(checkpoint_path)

            demo_pretrain_eval_summary = _build_demo_pretrain_eval_summary(
                num_episodes=demo_pretrain_eval_episodes,
                raw_metrics=raw_demo_pretrain_eval,
                checkpoint_path=demo_pretrain_checkpoint_path,
            )
            if demo_pretrain_checkpoint_path is not None:
                enriched_checkpoint_path = Path(demo_pretrain_checkpoint_path)
                checkpoint_payload = _load_checkpoint_payload(enriched_checkpoint_path)
                checkpoint_payload["demo_pretrain_eval_summary"] = dict(demo_pretrain_eval_summary)
                torch.save(checkpoint_payload, enriched_checkpoint_path)
                if save_best_checkpoint:
                    best_checkpoint_path = checkpoint_dir / "best_eval.pt"
                    if best_checkpoint_path.exists():
                        best_payload = _load_checkpoint_payload(best_checkpoint_path)
                        best_payload["demo_pretrain_eval_summary"] = dict(demo_pretrain_eval_summary)
                        torch.save(best_payload, best_checkpoint_path)
            if writer is not None:
                _log_tensorboard_demo_pretrain_eval_summary(writer, demo_pretrain_eval_summary)
                writer.flush()
                demo_pretrain_eval_summary_logged = True

            if stop_after_demo_pretrain:
                return {
                    "experiment_name": spec["experiment_name"],
                    "run_mode": spec["run_mode"],
                    "network_type": spec["network"]["type"],
                    "trainer_config": asdict(trainer_config),
                    "demo_pretrain_summary": demo_pretrain_summary,
                    "demo_pretrain_eval_summary": demo_pretrain_eval_summary,
                    "demo_pretrain_checkpoint_path": demo_pretrain_checkpoint_path,
                    "stopped_after_demo_pretrain": True,
                    "history": [],
                    "post_training_eval_model_source": (
                        "demo_pretrain_checkpoint" if demo_pretrain_checkpoint_path is not None else "demo_pretrain_eval"
                    ),
                    "post_training_eval_checkpoint": demo_pretrain_checkpoint_path,
                    "post_training_evaluation": [],
                    "final_metrics": dict(pretrain_eval_metrics),
                }

        history = trainer.train(
            num_updates=effective_trainer_config.total_updates,
            on_update=_on_update,
        )
        demo_pretrain_summary = (
            dict(trainer.demo_pretrain_summary)
            if getattr(trainer, "demo_pretrain_summary", None) is not None
            else None
        )
        if demo_pretrain_summary is not None:
            if not demo_pretrain_summary_logged:
                print(
                    "Demo Summary: replay={0:.0f}, actor_bc_updates={1:.0f}, critic_pretrain_updates={2:.0f}, critic_target={3}, target_mean={4:.6f}, target_std={5:.6f}, actor_bc_loss_last={6:.6f}, critic_loss_last={7:.6f}, seconds={{collection:{8:.3f}, actor_bc:{9:.3f}, critic:{10:.3f}}}".format(
                        float(demo_pretrain_summary.get("demo_replay_size_after_collection", 0.0)),
                        float(demo_pretrain_summary.get("actor_bc_updates", 0.0)),
                        float(demo_pretrain_summary.get("critic_pretrain_updates", 0.0)),
                        str(demo_pretrain_summary.get("critic_target_mode", "n_step")),
                        float(demo_pretrain_summary.get("demo_return_target_mean", 0.0)),
                        float(demo_pretrain_summary.get("demo_return_target_std", 0.0)),
                        float(demo_pretrain_summary.get("actor_bc_loss_last", 0.0)),
                        float(demo_pretrain_summary.get("critic_loss_last", 0.0)),
                        float(demo_pretrain_summary.get("seconds_collection", 0.0)),
                        float(demo_pretrain_summary.get("seconds_actor_bc", 0.0)),
                        float(demo_pretrain_summary.get("seconds_critic", 0.0)),
                    )
                )
            if writer is not None and not demo_pretrain_summary_logged:
                _log_tensorboard_demo_pretrain_summary(writer, demo_pretrain_summary)
        if writer is not None and demo_pretrain_eval_summary is not None and not demo_pretrain_eval_summary_logged:
            _log_tensorboard_demo_pretrain_eval_summary(writer, demo_pretrain_eval_summary)
        if history and save_final_checkpoint:
            final_metrics = history[-1]
            _save_checkpoint(
                "final.pt",
                update=int(final_metrics["update"]),
                metrics=final_metrics,
            )

        post_training_eval_model_source = "final_policy"
        post_training_eval_checkpoint = None
        if bool(evaluation.get("use_best_checkpoint_for_post_training_eval", False)):
            if not save_best_checkpoint:
                raise ValueError(
                    "evaluation.use_best_checkpoint_for_post_training_eval=True requires "
                    "training.save_best_checkpoint=True."
                )
            best_checkpoint_path = checkpoint_dir / "best_eval.pt"
            if not best_checkpoint_path.exists():
                raise FileNotFoundError(
                    "Best checkpoint not found for post-training evaluation: {0}".format(best_checkpoint_path)
                )
            best_checkpoint_payload = _load_checkpoint_payload(best_checkpoint_path)
            loaded_checkpoint_mode = trainer.load_checkpoint(best_checkpoint_payload)
            post_training_eval_model_source = "best_eval_checkpoint"
            post_training_eval_checkpoint = str(best_checkpoint_path)
            print(
                "Post-Eval : loaded best checkpoint={0}, checkpoint_mode={1}".format(
                    best_checkpoint_path,
                    loaded_checkpoint_mode,
                )
            )

        evaluation_summaries = run_trained_policy_evaluation(
            spec=spec,
            graph=graph,
            env_config=env_config,
            policy=policy,
            output_dir=output_dir,
        )
        if writer is not None:
            final_env_steps = int(history[-1].get("global_env_steps", 0)) if history else 0
            _log_tensorboard_post_training_evaluation(
                writer,
                evaluation_summaries,
                final_env_steps=final_env_steps,
            )
            writer.flush()

        return {
            "experiment_name": spec["experiment_name"],
            "run_mode": spec["run_mode"],
            "network_type": spec["network"]["type"],
            "trainer_config": asdict(trainer_config),
            "demo_pretrain_summary": demo_pretrain_summary,
            "demo_pretrain_eval_summary": demo_pretrain_eval_summary,
            "demo_pretrain_checkpoint_path": demo_pretrain_checkpoint_path,
            "stopped_after_demo_pretrain": False,
            "history": history,
            "post_training_eval_model_source": post_training_eval_model_source,
            "post_training_eval_checkpoint": post_training_eval_checkpoint,
            "post_training_evaluation": evaluation_summaries,
            "final_metrics": history[-1] if history else {},
        }
    finally:
        if writer is not None:
            writer.close()
        if trainer is not None:
            trainer.close()


def save_results_json(
    spec: Mapping[str, Any],
    graph: Dict[int, List[int]],
    env_config: SPGGConfig,
    results: Mapping[str, Any],
    output_dir: Path,
) -> None:
    if not spec["output"]["save_results_json"]:
        return

    payload = {
        "experiment": spec,
        "graph_summary": graph_summary(graph),
        "env_config": asdict(env_config),
        "results": results,
    }
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if not _runtime_quiet_console(spec):
        print("Results saved to: {0}".format(output_path))


def print_final_summary(results: Mapping[str, Any]) -> None:
    summary = results.get("summary")
    if summary is None:
        summary = results.get("final_metrics", {})
    print("Final summary:")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def run_one_experiment(spec: Mapping[str, Any]) -> Dict[str, Any]:
    np.random.seed(spec["seed"])

    graph = build_graph(spec)
    env_config = build_env_config(spec, graph)
    output_dir = build_output_dir(spec)
    quiet_console = _runtime_quiet_console(spec)
    with experiment_console_log_context(spec, output_dir) as console_log_path:
        if console_log_path is not None and not quiet_console:
            print("Console Log: {0}".format(console_log_path))
        if not quiet_console:
            print_header(spec, graph, env_config)

        if spec["run_mode"] in {"uniform", "proportional", "constant_mix", "pool_power_mix"}:
            results = run_rule_based_mode(spec, graph, env_config, output_dir)
        elif spec["run_mode"] == "gnn_train":
            results = run_gnn_training_mode(spec, graph, env_config, output_dir)
        else:
            raise ValueError("Unsupported run_mode: {0}".format(spec["run_mode"]))

        if not quiet_console:
            print_final_summary(results)
        save_results_json(spec, graph, env_config, results, output_dir)
        return results


def _run_scan_experiment_worker(spec: Mapping[str, Any]) -> Dict[str, Any]:
    results = run_one_experiment(spec)
    return _scan_record_from_results(spec, results)


def _scan_record_from_results(spec: Mapping[str, Any], results: Mapping[str, Any]) -> Dict[str, Any]:
    summary = results["summary"]
    scan_tags = spec["scan_tags"]
    return {
        "experiment_name": spec["experiment_name"],
        "run_mode": scan_tags["run_mode"],
        "run_mode_param_name": scan_tags.get("run_mode_param_name"),
        "run_mode_param_value": scan_tags.get("run_mode_param_value"),
        "run_mode_param_label": scan_tags.get("run_mode_param_label"),
        "warmup_constant_mix_omega": scan_tags.get("warmup_constant_mix_omega"),
        "warmup_pool_power_k": scan_tags.get("warmup_pool_power_k"),
        "num_nodes": int(scan_tags["num_nodes"]),
        "network_type": spec["network"]["type"],
        "network_label": scan_tags["network_label"],
        "resource_consumption_mode": scan_tags["resource_consumption_mode"],
        "resource_consumption_fixed_mode": scan_tags["resource_consumption_fixed_mode"],
        "consumption_label": scan_tags["consumption_label"],
        "strategy_update_rule": scan_tags["strategy_update_rule"],
        "r": float(scan_tags["r"]),
        "final_actual_cooperation_mean": float(summary["final_actual_cooperation_mean"]),
        "final_mean_resource_mean": float(summary["final_mean_resource_mean"]),
        "final_mean_pool_grown_mean": float(summary["final_mean_pool_grown_mean"]),
        "final_mean_consumption_mean": float(summary["final_mean_consumption_mean"]),
        "final_mean_payoff_mean": float(summary["final_mean_payoff_mean"]),
        "final_gini_mean": float(summary["final_gini_mean"]),
        "return_mean": float(summary["return_mean"]),
        "return_std": float(summary["return_std"]),
    }


def _save_scan_summary_tables(
    output_root: Path,
    scan_records: Sequence[Mapping[str, Any]],
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)

    json_path = output_root / "scan_summary.json"
    json_path.write_text(json.dumps(list(scan_records), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Scan summary saved to: {0}".format(json_path))

    csv_path = output_root / "scan_summary.csv"
    fieldnames = [
        "experiment_name",
        "run_mode",
        "run_mode_param_name",
        "run_mode_param_value",
        "run_mode_param_label",
        "warmup_constant_mix_omega",
        "warmup_pool_power_k",
        "num_nodes",
        "network_type",
        "network_label",
        "resource_consumption_mode",
        "resource_consumption_fixed_mode",
        "consumption_label",
        "strategy_update_rule",
        "r",
        "final_actual_cooperation_mean",
        "final_mean_resource_mean",
        "final_mean_pool_grown_mean",
        "final_mean_consumption_mean",
        "final_mean_payoff_mean",
        "final_gini_mean",
        "return_mean",
        "return_std",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in scan_records:
            writer.writerow({field: record.get(field) for field in fieldnames})
    print("Scan CSV saved to: {0}".format(csv_path))


def _save_scan_steady_state_plots(
    output_root: Path,
    scan_records: Sequence[Mapping[str, Any]],
) -> None:
    from Project1.visualization import save_scan_metric_grid

    metrics = [
        "final_actual_cooperation_mean",
        "final_mean_resource_mean",
        "final_mean_pool_grown_mean",
        "final_mean_consumption_mean",
        "final_mean_payoff_mean",
        "final_gini_mean",
    ]

    grouped_records: Dict[tuple[str, str, str, str], List[Mapping[str, Any]]] = {}
    for record in scan_records:
        group_key = (
            str(record["run_mode"]),
            str(record.get("run_mode_param_label") or ""),
            str(record["strategy_update_rule"]),
            str(record["consumption_label"]),
        )
        grouped_records.setdefault(group_key, []).append(record)

    steady_state_dir = output_root / "steady_state_vs_r"
    steady_state_dir.mkdir(parents=True, exist_ok=True)

    for (run_mode, run_mode_param_label, strategy_update_rule, consumption_label), records in grouped_records.items():
        filename_parts = [run_mode]
        title_parts = ["Steady-state vs r", "run_mode={0}".format(run_mode)]
        if run_mode_param_label:
            filename_parts.append(run_mode_param_label)
            first_record = records[0]
            title_parts.append(
                "{0}={1}".format(
                    first_record.get("run_mode_param_name"),
                    first_record.get("run_mode_param_value"),
                )
            )
        filename_parts.extend([strategy_update_rule, consumption_label, "steady_state_vs_r"])
        output_path = steady_state_dir / "{0}.png".format("__".join(filename_parts))
        title_parts.extend(
            [
                "strategy={0}".format(strategy_update_rule),
                "consumption={0}".format(consumption_label),
            ]
        )
        title = " | ".join(title_parts)
        save_scan_metric_grid(
            records=records,
            output_path=output_path,
            metrics=metrics,
            title=title,
            dpi=int(BASE_EXPERIMENT["visualization"]["save_dpi"]),
        )


def run_scan_experiments() -> None:
    specs = build_scan_experiment_specs()
    output_root = Path(SCAN_EXPERIMENT["output_root_dir"]).expanduser()
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    output_root.mkdir(parents=True, exist_ok=True)
    scan_start_time = time.time()

    manifest = {
        "scan_config": deepcopy(SCAN_EXPERIMENT),
        "base_experiment_name": BASE_EXPERIMENT["experiment_name"],
        "base_run_mode": BASE_EXPERIMENT["run_mode"],
        "num_experiments": len(specs),
    }
    manifest_path = output_root / "scan_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Scan manifest saved to: {0}".format(manifest_path))
    print(_console_info("total experiments: {0}".format(len(specs)), phase="scan"))

    scan_records: List[Dict[str, Any]] = []
    if SCAN_EXPERIMENT["parallel"]:
        requested_workers = SCAN_EXPERIMENT["max_workers"]
        cpu_count = os.cpu_count() or 1
        max_workers = cpu_count if requested_workers in (None, 0) else int(requested_workers)
        max_workers = max(1, min(max_workers, len(specs)))
        print(_console_info("running in parallel with {0} worker processes".format(max_workers), phase="scan"))

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_meta = {
                executor.submit(_run_scan_experiment_worker, spec): (index, spec["experiment_name"])
                for index, spec in enumerate(specs, start=1)
            }
            for completed_count, future in enumerate(as_completed(future_to_meta), start=1):
                index, experiment_name = future_to_meta[future]
                scan_records.append(future.result())
                for line in _format_scan_progress_lines(
                    completed_count=completed_count,
                    total_count=len(specs),
                    start_time=scan_start_time,
                    latest_experiment_name="#{0} {1}".format(index, experiment_name),
                ):
                    print(line)
    else:
        for index, spec in enumerate(specs, start=1):
            results = run_one_experiment(spec)
            scan_records.append(_scan_record_from_results(spec, results))
            for line in _format_scan_progress_lines(
                completed_count=index,
                total_count=len(specs),
                start_time=scan_start_time,
                latest_experiment_name="#{0} {1}".format(index, spec["experiment_name"]),
            ):
                print(line)

    scan_records.sort(
        key=lambda item: (
            str(item["run_mode"]),
            str(item.get("run_mode_param_label") or ""),
            str(item["strategy_update_rule"]),
            str(item["consumption_label"]),
            str(item["network_label"]),
            float(item["r"]),
        )
    )
    _save_scan_summary_tables(output_root, scan_records)
    _save_scan_steady_state_plots(output_root, scan_records)


def main() -> None:
    _configure_stdio()
    if SCAN_EXPERIMENT["enabled"]:
        run_scan_experiments()
        return

    specs = build_experiment_specs()
    for index, spec in enumerate(specs, start=1):
        if len(specs) > 1:
            print("\n" + "#" * 80)
            print("Batch experiment {0}/{1}".format(index, len(specs)))
            print("#" * 80)
        run_one_experiment(spec)


if __name__ == "__main__":
    main()
