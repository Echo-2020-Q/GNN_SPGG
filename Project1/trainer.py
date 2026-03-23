from __future__ import annotations

from Project1.td3 import (
    GraphTD3Config,
    GraphTD3Trainer,
)


TrainerConfig = GraphTD3Config
CentralizedTD3Trainer = GraphTD3Trainer


class CentralizedActorCriticTrainer(GraphTD3Trainer):
    """Backward-compatible alias for the new Graph-TD3 trainer."""


__all__ = [
    "CentralizedActorCriticTrainer",
    "CentralizedTD3Trainer",
    "GraphTD3Config",
    "GraphTD3Trainer",
    "TrainerConfig",
]
