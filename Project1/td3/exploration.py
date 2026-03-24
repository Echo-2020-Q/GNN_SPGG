from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from Project1.policies.gnn_rl import PolicyOutput

from .data import TensorActionRecord


def masked_row_softmax(logits: Tensor, mask: Tensor) -> Tensor:
    mask = mask.to(dtype=torch.bool)
    masked_logits = torch.full_like(logits, torch.finfo(logits.dtype).min)
    masked_logits = torch.where(mask, logits, masked_logits)
    row_max = masked_logits.max(dim=1, keepdim=True).values
    stable_logits = masked_logits - row_max
    exp_logits = torch.exp(stable_logits) * mask.to(dtype=logits.dtype)
    normalizer = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return exp_logits / normalizer


class LogitSpaceExplorer:
    def current_noise_std(self, base_std: float, step: int, decay: float, multiplier: float = 1.0) -> float:
        return float(base_std * (decay ** max(step, 0)) * multiplier)

    def apply_to_policy_output(
        self,
        policy_output: PolicyOutput,
        ego_mask: Tensor,
        pool_values: Tensor,
        noise_std: float,
        noise_clip: float,
    ) -> TensorActionRecord:
        if policy_output.logits is None:
            raise ValueError("policy_output.logits is required for logit-space exploration.")
        return self.apply_to_logits(
            logits=policy_output.logits,
            ego_mask=ego_mask,
            pool_values=pool_values,
            noise_std=noise_std,
            noise_clip=noise_clip,
        )

    def apply_to_logits(
        self,
        logits: Tensor,
        ego_mask: Tensor,
        pool_values: Tensor,
        noise_std: float,
        noise_clip: float,
    ) -> TensorActionRecord:
        mask = ego_mask.to(dtype=torch.bool, device=logits.device)
        noise = torch.randn_like(logits) * float(noise_std)
        if noise_clip > 0.0:
            noise = noise.clamp(-float(noise_clip), float(noise_clip))
        noise = noise * mask.to(dtype=logits.dtype)

        masked_fill_value = torch.finfo(logits.dtype).min
        noisy_logits = torch.where(mask, logits + noise, torch.full_like(logits, masked_fill_value))
        allocation = masked_row_softmax(noisy_logits, mask)
        transfers = allocation * pool_values.view(-1, 1)
        incoming = transfers.sum(dim=0)
        return TensorActionRecord(
            logits=noisy_logits,
            allocation=allocation,
            transfers=transfers,
            incoming=incoming,
            ego_mask=mask,
            pool_values=pool_values,
        )

    def logits_from_allocation(
        self,
        allocation: Tensor | np.ndarray,
        ego_mask: Tensor | np.ndarray,
        device: torch.device | str = "cpu",
    ) -> Tensor:
        allocation_tensor = torch.as_tensor(allocation, dtype=torch.float32, device=device)
        mask = torch.as_tensor(ego_mask, dtype=torch.bool, device=device)
        masked_fill_value = torch.finfo(allocation_tensor.dtype).min
        logits = torch.full_like(allocation_tensor, masked_fill_value)

        positive_mask = mask & (allocation_tensor > 0.0)
        logits = torch.where(
            positive_mask,
            torch.log(allocation_tensor.clamp_min(1e-12)),
            logits,
        )

        rows_without_positive = positive_mask.sum(dim=1) == 0
        if torch.any(rows_without_positive):
            fallback_logits = torch.where(
                mask[rows_without_positive],
                torch.zeros_like(allocation_tensor[rows_without_positive]),
                torch.full_like(allocation_tensor[rows_without_positive], masked_fill_value),
            )
            logits[rows_without_positive] = fallback_logits
        return logits

    def action_from_allocation(
        self,
        allocation: Tensor | np.ndarray,
        ego_mask: Tensor | np.ndarray,
        pool_values: Tensor | np.ndarray,
        noise_std: float = 0.0,
        noise_clip: float = 0.0,
        device: torch.device | str = "cpu",
    ) -> TensorActionRecord:
        logits = self.logits_from_allocation(
            allocation=allocation,
            ego_mask=ego_mask,
            device=device,
        )
        return self.apply_to_logits(
            logits=logits,
            ego_mask=torch.as_tensor(ego_mask, dtype=torch.bool, device=device),
            pool_values=torch.as_tensor(pool_values, dtype=torch.float32, device=device),
            noise_std=noise_std,
            noise_clip=noise_clip,
        )

    def sample_random_logits_action(
        self,
        ego_mask: np.ndarray,
        pool_values: np.ndarray,
        rng: np.random.Generator,
        device: torch.device | str = "cpu",
    ) -> TensorActionRecord:
        mask = torch.as_tensor(ego_mask, dtype=torch.bool, device=device)
        base_logits = torch.as_tensor(rng.standard_normal(size=mask.shape), dtype=torch.float32, device=device)
        pool_tensor = torch.as_tensor(pool_values, dtype=torch.float32, device=device)
        return self.apply_to_logits(
            logits=base_logits,
            ego_mask=mask,
            pool_values=pool_tensor,
            noise_std=0.0,
            noise_clip=0.0,
        )
