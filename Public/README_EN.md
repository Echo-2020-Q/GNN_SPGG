# Pretrained GNN, M1 and M2 with the SPGG Environment

This package contains a pretrained GNN allocation policy, saved M1/M2 mechanisms,
and the SPGG environment needed to run complete simulations. This directory can
be copied and used independently of the original project. GNN training code is
not included.

The GNN weights come from this experiment's best-evaluation checkpoint:

```text
outputs/Pool_dynamic/0409_spgg_GNN_50Nodes_200length_Fermi_FixedTopology_StagedTeacher/checkpoints/best_eval.pt
```

`models/gnn.pt` contains the actor weights and inference configuration extracted
from that checkpoint. Loading it does not require the original experiment directory.

## Files

| File | Purpose |
| --- | --- |
| `models/gnn.pt` | Pretrained actor weights and architecture configuration, exported from the experiment's `best_eval.pt` |
| `gnn.py` | GNN architecture and deterministic inference; no critic or optimizer |
| `models/m1.json`, `models/m2.json` | Saved resource-bin boundaries and allocation mixture weights |
| `mechanisms.py` | NumPy implementations of M1 and M2 |
| `env.py` | SPGG environment, graph generation, resource and strategy updates, observations, and Gini calculation |
| `environment.json` | Environment parameters, initial conditions, and ten reference graph seeds per topology |
| `simulate.py` | Complete simulation runs, repeated trials, and CSV output |
| `example.py` | A single allocation calculation on a supplied observation |
| `example_observation.npz` | One initial observation from a 50-node Regular graph; graph seed 3055 and reset seed 42 |
| `requirements.txt` | Python dependencies |

## Quick Start

Use Python 3.10 or newer. Open a terminal in this directory and run:

```bash
python -m pip install -r requirements.txt
python simulate.py
```

By default, GNN, M1 and M2 each run for **200 environment steps** on the same
50-node Regular graph, using the CPU. Each policy starts with the same initial
state and simulation seed. At every step, the policy computes a new allocation,
and the environment updates resources and strategies before the next decision.

The default initial cooperation probability is 0.5, and strategies evolve using
the Fermi rule. These settings can be changed in `environment.json`.

## Simulation Examples

Run M2 on one fixed ER graph for 200 steps, repeated 50 times:

```bash
python simulate.py --policy m2 --topology er --graph-index 0 --steps 200 --repeats 50
```

Run the pretrained GNN on a BA graph with a specified graph seed:

```bash
python simulate.py --policy gnn --topology ba --graph-seed 5698
```

Select a policy with `--policy all`, `gnn`, `m1`, or `m2`. The topology aliases are
`regular`, `ba`, `er`, and `ws`. Use `--graph-index 0` through `9` to select a
reference graph, or `--graph-seed` to override its graph seed.

Repeated runs keep the graph fixed and use different simulation seeds. `--seed`
sets the base used to derive those seeds. M1/M2 simulations require only NumPy;
GNN inference also requires PyTorch. CPU is the default device.

## Outputs

Results are written under `results/` in this directory. Use `--output-dir PATH`
to choose another location.

| Output | Contents |
| --- | --- |
| `trajectories.csv` | Gini, mean resource, and actual cooperation rate at every step |
| `episodes.csv` | Final metrics and graph/simulation seeds for each run |
| `summary.csv` | Mean and population standard deviation of final Gini across repeats on the fixed graph |
| `run_config.json` | Effective environment configuration and seed conventions |

`step=0` is the initial state. With `--steps 200`, final Gini is measured after
**200 environment updates**. It is not averaged over a time window. The standard
deviation uses `ddof=0`; a single run therefore has a reported standard deviation
of zero. Reusing an output directory overwrites its result files.

## Single-Step Example and Python API

To calculate allocations for just one observation:

```bash
python example.py --output allocations.npz
python example.py --policy m2 --observation my_observation.npz
```

`example_observation.npz` is input data, not an environment or a mechanism.
`example.py` does not advance the simulation; use `simulate.py` for full episodes.

The environment and policies can also be called directly:

```python
from mechanisms import load_m2
from simulate import load_config, build_environment

env, graph_seed = build_environment(load_config(), topology="regular", steps=200)
policy = load_m2()
observation = env.reset(seed=42)
done = False

while not done:
    allocation = policy.allocate(observation)
    observation, reward, done, info = env.step(allocation)

print("Final Gini:", info["gini"])
```

Replace `load_m2()` with `load_m1()` from `mechanisms`, or `load_gnn()` from `gnn`,
to use another policy. All three expose `allocate(observation)`, returning an
N-by-N NumPy matrix of allocation fractions: rows are sending pools and columns
are recipients. Each row sums to one, and allocation outside the local group is
zero. `env.reset()` and `env.step()` supply the observation fields and normalized
inputs required by the policies, including the sender degrees used by M2.

## M2 Replay Behavior

M2 first selects a parameter group by sender degree, then selects the group's
resource-bin weights using the local mean resource:

| Sender degree | Parameter group |
| --- | --- |
| 2 | Q1:2 |
| 3 | Q2:3 |
| 4–6 | Q3:4-6 |
| 7–19 | Q4:7-19 |

This follows `_resolve_degree_profile` in
`0429_rule_allocation_over_time_ba_degree_conditioned_discrete/compute.py`, which
uses the weights saved in the 0421 source directory. For degrees outside the
fitted range 2–19, this package retains the original policy's Q4 fallback to
support other network topologies; the 0429 reconstruction script evaluates only
degrees covered by its saved intervals.

An earlier package version incorrectly selected Q4 for every sender because it
did not restore degree selection from the saved intervals. That behavior has
been corrected. M2 statistics from that earlier version must be recomputed.

## Validation

The package was tested from a separate directory on CPU with NumPy 2.4.4 and
PyTorch 2.11.0:

- GNN, M1 and M2 each completed a 200-step simulation.
- Corrected M2 completed 50 independent 200-step runs on the fixed ER graph with
  seed 7221 and base simulation seed 42. The final Gini mean was **0.3487874500**,
  with population standard deviation **0.0899103519**.
- Graph construction, environment configuration, observations, updates and
  rewards were checked against the original environment on all 40 reference graphs.
- All 46 saved resource-bin weight rows and their degree intervals were checked
  against the 0429 reconstruction results. Regression tests cover all four
  groups and the fallback for degrees outside the fitted range.
- GNN and M1 inference previously matched their original implementations on
  48 observations across four topologies; their inference code is unchanged.

These checks cover the tested CPU environment; small numerical differences may
occur with other dependency versions or devices.
