# GNN / M1 / M2 最小公开包

[English documentation](README_EN.md)

本目录提供训练好的 GNN、已保存的 M1/M2 分配机制，以及 SPGG 环境，
可单独复制后运行完整仿真，不包含 GNN 训练流程。

GNN 权重来源为以下实验的最佳评估 checkpoint：

```text
outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/checkpoints/best_eval.pt
```

`models/gnn.pt` 提取其中用于推理的 Actor 权重和配置，不需要原始实验目录即可加载。

## 文件

| 文件 | 内容 |
| --- | --- |
| `models/gnn.pt` | 从实验使用的 `best_eval.pt` 导出的 Actor 权重和网络配置 |
| `gnn.py` | GNN 网络结构及确定性推理；不含 Critic、优化器或训练逻辑 |
| `models/m1.json`、`models/m2.json` | M1/M2 的资源分箱边界和分配权重 |
| `mechanisms.py` | M1/M2 分配规则，仅依赖 NumPy |
| `env.py` | 完整 SPGG 环境：网络生成、资源和策略更新、观测及 Gini 计算 |
| `environment.json` | 环境参数、初始条件及每种拓扑的 10 个参考图种子 |
| `simulate.py` | 连续仿真、重复运行及统计结果导出 |
| `example_observation.npz` | 50 节点 Regular 网络的一个初始状态，图种子 3055、仿真种子 42 |
| `example.py` | 加载输入并计算 GNN、M1、M2 分配矩阵的示例 |

## 运行

建议 Python 3.10 或更新版本。在本目录运行：

```bash
python -m pip install -r requirements.txt
python simulate.py
```

默认在 CPU 上，让 GNN、M1、M2 分别在同一张 Regular 图上运行 200 步。
每一步都会重新计算分配并更新环境，三个机制使用相同的初始状态和仿真种子。
默认初始合作概率为 0.5，策略按 Fermi 规则演化；环境参数可在 `environment.json` 修改。

```bash
# M2：固定 ER 第 0 张图，运行 200 步，独立重复 50 次
python simulate.py --policy m2 --topology er --graph-index 0 --steps 200 --repeats 50

# GNN：在 BA 图上运行；也可指定自己的建图种子
python simulate.py --policy gnn --topology ba --graph-seed 5698
```

`--topology` 支持 `regular`、`ba`、`er`、`ws`，`--graph-index` 为 0..9。
重复运行时建图种子不变，只改变仿真种子；`--seed` 设置仿真种子的基础值。
默认结果写到本目录的 `results/` 子目录，可用 `--output-dir` 指定其他位置。

| 输出文件 | 内容 |
| --- | --- |
| `trajectories.csv` | 每步 Gini、平均资源和实际合作率；step=0 为初始状态 |
| `episodes.csv` | 每次仿真最后一步的指标及图种子、仿真种子 |
| `summary.csv` | 同一固定图上，多次仿真最终 Gini 的均值和标准差（ddof=0） |
| `run_config.json` | 实际使用的环境配置和种子规则 |

`--steps 200` 统计的是完成 200 次环境更新后的状态。M1/M2 仿真仅依赖 NumPy。

原来的 `example.py` 仍可用于单次分配检查：

```bash
python example.py
python example.py --policy m2
python example.py --policy gnn --device cpu --output allocations.npz
python example.py --observation my_observation.npz
```

也可以直接调用：

```python
from gnn import load_gnn
from mechanisms import load_m1, load_m2
from simulate import load_config, build_environment

env, graph_seed = build_environment(load_config(), topology="regular", steps=200)
policy = load_m2()  # 也可以换成 load_gnn() 或 load_m1()
observation = env.reset(seed=42)
done = False
while not done:
    allocation = policy.allocate(observation)
    observation, reward, done, info = env.step(allocation)
print("Final Gini:", info["gini"])
```

输入是观测字段组成的字典。GNN 使用 `local_mask`、`pool_raw_norm`、
`resource_norm`、`degree_norm`、`strategy_norm`、`x_actual`、`gini` 和 `pool_grown`；
M1/M2 使用 `local_mask`、`resources` 和 `unit_investment`，M2 还使用 `degrees` 选择分组。
`local_mask` 是包含自身的 N×N 布尔矩阵，`gini` 是标量，其余上述字段是长度为 N 的向量。
`env.reset()` 和 `env.step()` 会自动生成上述字段及模型所需的归一化数值。
输出为 N×N NumPy 分配比例矩阵，行表示资源池／发送者，列表示接收者。

## M2 回放说明

M2 先按发送者的度数选择参数组，再按其局部平均资源选择该组中的资源分箱权重：

| 度数 | 参数组 |
| --- | --- |
| 2 | Q1:2 |
| 3 | Q2:3 |
| 4–6 | Q3:4-6 |
| 7–19 | Q4:7-19 |

该选择方式与 `0429_rule_allocation_over_time_ba_degree_conditioned_discrete/compute.py`
的 `_resolve_degree_profile` 一致；该脚本同样读取 0421 目录中保存的权重。
度数不在已拟合的 2–19 范围内时，沿用原策略的 Q4 回退规则；这是跨拓扑运行的边界处理。

早期公开包误把缺失的缓存映射当成没有可用分组，因而始终使用 Q4；现已修正。
早期回退版本的 M2 统计结果不适用于当前分组机制，需要重新运行。

运行环境验证：NumPy 2.4.4、PyTorch 2.11.0（CPU）。
