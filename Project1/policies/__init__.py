from .rule_based import ProportionalContributionPolicy, UniformAllocationPolicy

try:
    from .gnn_rl import (
        AllocationHead,
        EgoLocalGraphNet,
        GNNAllocationPolicy,
        GNNPolicyConfig,
        GlobalCritic,
        PolicyOutput,
        ScoreReadout,
        TwoLayerGraphNetBackbone,
        extract_ego_subgraph,
    )
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
    AllocationHead = None
    EgoLocalGraphNet = None
    GNNAllocationPolicy = None
    GNNPolicyConfig = None
    GlobalCritic = None
    PolicyOutput = None
    ScoreReadout = None
    TwoLayerGraphNetBackbone = None
    extract_ego_subgraph = None

__all__ = [
    "AllocationHead",
    "EgoLocalGraphNet",
    "GNNAllocationPolicy",
    "GNNPolicyConfig",
    "GlobalCritic",
    "PolicyOutput",
    "ScoreReadout",
    "TwoLayerGraphNetBackbone",
    "ProportionalContributionPolicy",
    "UniformAllocationPolicy",
    "extract_ego_subgraph",
]
