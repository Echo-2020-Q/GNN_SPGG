"""Run GNN, M1 and M2 on the bundled observation, or on another NPZ file."""

import argparse
from pathlib import Path

import numpy as np

from mechanisms import load_m1, load_m2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=("all", "gnn", "m1", "m2"), default="all")
    parser.add_argument("--observation", type=Path, default=Path(__file__).parent / "example_observation.npz")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional NPZ output containing allocation matrices.")
    args = parser.parse_args()
    with np.load(args.observation, allow_pickle=False) as data:
        observation = {name: data[name] for name in data.files}
    selected = ("gnn", "m1", "m2") if args.policy == "all" else (args.policy,)
    output = {}
    for name in selected:
        if name == "gnn":
            import torch
            from gnn import load_gnn

            torch.set_num_threads(1)
            policy = load_gnn(device=args.device)
        else:
            policy = load_m1() if name == "m1" else load_m2()
        allocation = policy.allocate(observation)
        mask = np.asarray(observation["local_mask"], dtype=bool)
        if not np.all(np.isfinite(allocation)) or np.any(allocation < 0):
            raise RuntimeError(f"Invalid allocation from {name}.")
        if np.any(allocation[~mask] != 0) or not np.allclose(allocation.sum(axis=1), 1.0, atol=1e-6):
            raise RuntimeError(f"Allocation constraints failed for {name}.")
        output[name] = allocation
        print(f"{name.upper()}: allocation shape={allocation.shape}, row sums=1, valid")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output, **output)
        print(f"Saved allocations to {args.output}")


if __name__ == "__main__":
    main()
