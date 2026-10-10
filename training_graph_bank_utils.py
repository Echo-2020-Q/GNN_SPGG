from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from analyze_gnn_policy_decisions import build_spec_for_topology, canonicalize_topology_name


GRAPH_SOURCE_MODE_CHOICES = ("training_fixed_graph_bank", "spec_seed", "random_per_episode")


def normalize_graph_source_mode(value: str) -> str:
    normalized = str(value).strip().lower()
    aliases = {
        "training_fixed_graph_bank": "training_fixed_graph_bank",
        "training_graph_bank": "training_fixed_graph_bank",
        "fixed_graph_bank": "training_fixed_graph_bank",
        "graph_bank": "training_fixed_graph_bank",
        "spec_seed": "spec_seed",
        "seed": "spec_seed",
        "results_seed": "spec_seed",
        "random_per_episode": "random_per_episode",
        "random_episode": "random_per_episode",
        "fresh_random": "random_per_episode",
        "fresh_per_episode": "random_per_episode",
        "episode_random": "random_per_episode",
    }
    resolved = aliases.get(normalized)
    if resolved is None:
        raise ValueError(
            f"Unsupported graph source mode: {value}. Expected one of {', '.join(GRAPH_SOURCE_MODE_CHOICES)}."
        )
    return resolved


def _domain_randomization_type_seed(seed_base: int, topology_name: str) -> int:
    return int(seed_base) + sum((index + 1) * ord(char) for index, char in enumerate(str(topology_name)))


def _build_topology_spec_from_training_fixed_graph_bank(
    base_spec: Mapping[str, Any],
    topology_name: str,
    *,
    reference_mean_degree: float,
    graph_bank_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    topology = canonicalize_topology_name(topology_name)
    randomization = dict(base_spec.get("domain_randomization", {}))
    if not bool(randomization.get("enabled", False)):
        raise ValueError("results.json does not have enabled domain_randomization; cannot replay the training graph bank.")
    if not bool(randomization.get("fixed_graph_bank_enabled", False)):
        raise ValueError("results.json does not have fixed_graph_bank_enabled=true; cannot replay the training graph bank.")

    fixed_graph_bank_seed = int(randomization.get("fixed_graph_bank_seed", base_spec["seed"]))
    fixed_graph_bank_size = int(randomization.get("fixed_graph_bank_size_per_type", 0))
    if fixed_graph_bank_size <= 0:
        raise ValueError("fixed_graph_bank_enabled=true but fixed_graph_bank_size_per_type <= 0.")
    if int(graph_bank_index) < 0 or int(graph_bank_index) >= fixed_graph_bank_size:
        raise ValueError(
            f"graph_bank_index={graph_bank_index} is out of range for fixed graph bank size {fixed_graph_bank_size}."
        )

    spec = build_spec_for_topology(
        base_spec,
        topology,
        reference_mean_degree=reference_mean_degree,
    )
    network = dict(spec["network"])
    spec["network"] = network

    num_nodes_choices = tuple(
        int(item) for item in randomization.get("num_nodes_choices", (int(network["num_nodes"]),))
    )
    regular_degree_choices = tuple(
        int(item) for item in randomization.get("regular_degree_choices", (int(network.get("regular_degree", 4)),))
    )
    er_mean_degree_choices = tuple(
        float(item) for item in randomization.get("er_mean_degree_choices", (float(network.get("er_target_mean_degree", 4.0)),))
    )
    ws_degree_choices = tuple(
        int(item) for item in randomization.get("ws_degree_choices", (int(network.get("ws_degree", 4)),))
    )
    ws_rewiring_choices = tuple(
        float(item) for item in randomization.get("ws_rewiring_choices", (float(network.get("ws_rewiring_prob", 0.1)),))
    )
    ba_attachment_choices = tuple(
        int(item) for item in randomization.get("ba_attachment_choices", (int(network.get("ba_attachments_per_new_node", 2)),))
    )

    type_rng = np.random.default_rng(_domain_randomization_type_seed(fixed_graph_bank_seed, topology))
    sampled_num_nodes = int(network["num_nodes"])
    sampled_seed = int(spec["seed"])
    sampled_degree = int(network.get("regular_degree", 0))
    sampled_er_mean_degree = float(network.get("er_target_mean_degree", 0.0))
    sampled_ws_degree = int(network.get("ws_degree", 0))
    sampled_ws_rewiring = float(network.get("ws_rewiring_prob", 0.1))
    sampled_ba_attachment = int(network.get("ba_attachments_per_new_node", 1))

    for _ in range(int(graph_bank_index) + 1):
        sampled_num_nodes = int(type_rng.choice(num_nodes_choices))
        sampled_seed = int(type_rng.integers(0, 2**31 - 1))
        if topology == "regular":
            sampled_degree = int(type_rng.choice(regular_degree_choices))
        elif topology == "erdos_renyi":
            sampled_er_mean_degree = float(type_rng.choice(er_mean_degree_choices))
        elif topology == "small_world":
            sampled_ws_degree = int(type_rng.choice(ws_degree_choices))
            sampled_ws_rewiring = float(type_rng.choice(ws_rewiring_choices))
        elif topology == "scale_free":
            sampled_ba_attachment = int(type_rng.choice(ba_attachment_choices))
        else:
            raise ValueError(f"Unsupported topology for training fixed graph bank replay: {topology}")

    spec["seed"] = int(sampled_seed)
    network["num_nodes"] = int(sampled_num_nodes)
    if topology == "regular":
        network["regular_degree"] = int(sampled_degree)
    elif topology == "erdos_renyi":
        network["er_target_mean_degree"] = float(sampled_er_mean_degree)
        network["er_edge_prob"] = float(sampled_er_mean_degree) / float(max(sampled_num_nodes - 1, 1))
    elif topology == "small_world":
        network["ws_degree"] = int(sampled_ws_degree)
        network["ws_rewiring_prob"] = float(sampled_ws_rewiring)
    elif topology == "scale_free":
        network["ba_attachments_per_new_node"] = int(sampled_ba_attachment)

    metadata = {
        "topology": topology,
        "graph_source_mode": "training_fixed_graph_bank",
        "graph_bank_index": int(graph_bank_index),
        "fixed_graph_bank_seed": int(fixed_graph_bank_seed),
        "fixed_graph_bank_size_per_type": int(fixed_graph_bank_size),
        "fixed_graph_bank_sampling": str(randomization.get("fixed_graph_bank_sampling", "uniform")),
        "graph_seed": int(sampled_seed),
    }
    return spec, metadata


def build_topology_spec_for_analysis(
    base_spec: Mapping[str, Any],
    topology_name: str,
    *,
    reference_mean_degree: float,
    graph_source_mode: str,
    graph_bank_index: int,
    spec_seed: int | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved_mode = normalize_graph_source_mode(graph_source_mode)
    topology = canonicalize_topology_name(topology_name)

    if resolved_mode == "training_fixed_graph_bank":
        return _build_topology_spec_from_training_fixed_graph_bank(
            base_spec,
            topology,
            reference_mean_degree=reference_mean_degree,
            graph_bank_index=int(graph_bank_index),
        )

    spec = build_spec_for_topology(
        base_spec,
        topology,
        reference_mean_degree=reference_mean_degree,
    )
    if spec_seed is not None:
        spec["seed"] = int(spec_seed)
    metadata = {
        "topology": topology,
        "graph_source_mode": str(resolved_mode),
        "graph_bank_index": None,
        "fixed_graph_bank_seed": None,
        "fixed_graph_bank_size_per_type": None,
        "fixed_graph_bank_sampling": None,
        "graph_seed": int(spec["seed"]),
    }
    return spec, metadata
