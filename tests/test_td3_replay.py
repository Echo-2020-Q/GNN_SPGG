from __future__ import annotations

import importlib.util
import unittest


NUMPY_AVAILABLE = importlib.util.find_spec("numpy") is not None
TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if NUMPY_AVAILABLE and TORCH_AVAILABLE:
    import numpy as np
    import torch

    from Project1.td3.data import TensorActionRecord, TensorTransition
    from Project1.td3.replay import ReplayBuffer


@unittest.skipUnless(NUMPY_AVAILABLE and TORCH_AVAILABLE, "numpy and torch are required for replay tests")
class TensorReplayTests(unittest.TestCase):
    def _make_observation(self, offset: float = 0.0) -> dict[str, np.ndarray]:
        local_mask = np.array(
            [
                [True, True, False],
                [True, True, True],
                [False, True, True],
            ],
            dtype=bool,
        )
        return {
            "x_actual": np.array([1.0, 0.0, 1.0], dtype=np.float64) + offset * 0.0,
            "resource_norm": np.array([0.1, 0.2, 0.3], dtype=np.float64) + offset,
            "pool_raw_norm": np.array([0.4, 0.5, 0.6], dtype=np.float64) + offset,
            "degree_norm": np.array([0.0, 0.1, -0.1], dtype=np.float64),
            "strategy_norm": np.array([0.7, 0.8, 0.9], dtype=np.float64),
            "gini": np.asarray(0.2 + offset, dtype=np.float64),
            "pool_grown": np.array([1.0, 2.0, 3.0], dtype=np.float64) + offset,
            "local_mask": local_mask,
        }

    def _make_action(self, offset: float = 0.0) -> TensorActionRecord:
        allocation = torch.tensor(
            [
                [0.5, 0.5, 0.0],
                [0.2, 0.3, 0.5],
                [0.0, 0.4, 0.6],
            ],
            dtype=torch.float32,
        )
        logits = torch.log(allocation.clamp_min(1e-6)) + float(offset)
        pool_values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32) + float(offset)
        transfers = allocation * pool_values.unsqueeze(-1)
        incoming = transfers.sum(dim=0)
        ego_mask = torch.tensor(
            [
                [True, True, False],
                [True, True, True],
                [False, True, True],
            ],
            dtype=torch.bool,
        )
        return TensorActionRecord(
            logits=logits,
            allocation=allocation,
            transfers=transfers,
            incoming=incoming,
            ego_mask=ego_mask,
            pool_values=pool_values,
        )

    def test_tensor_replay_sample_returns_batched_tensors(self) -> None:
        buffer = ReplayBuffer(capacity=8, seed=7)
        for index in range(4):
            buffer.add(
                TensorTransition.from_step(
                    obs=self._make_observation(offset=0.1 * index),
                    action=self._make_action(offset=0.1 * index),
                    reward=float(index),
                    next_obs=self._make_observation(offset=0.1 * (index + 1)),
                    done=bool(index % 2),
                )
            )

        batch = buffer.sample(batch_size=3)

        self.assertEqual(batch.reward.shape, (3,))
        self.assertEqual(batch.done.shape, (3,))
        self.assertEqual(batch.action.allocation.shape, (3, 3, 3))
        self.assertEqual(batch.obs["resource_norm"].shape, (3, 3))
        self.assertEqual(batch.obs["local_mask"].shape, (3, 3, 3))
        self.assertEqual(batch.obs["local_mask"].dtype, torch.bool)
        self.assertEqual(batch.obs["resource_norm"].dtype, torch.float32)
        self.assertEqual(batch.reward.dtype, torch.float32)
        self.assertTrue(torch.all((batch.done == 0.0) | (batch.done == 1.0)))

    def test_state_dict_round_trip_preserves_sampling(self) -> None:
        buffer = ReplayBuffer(capacity=6, seed=123)
        for index in range(5):
            buffer.add(
                TensorTransition.from_step(
                    obs=self._make_observation(offset=0.05 * index),
                    action=self._make_action(offset=0.05 * index),
                    reward=float(index) + 0.5,
                    next_obs=self._make_observation(offset=0.05 * (index + 1)),
                    done=False,
                )
            )

        state_dict = buffer.state_dict()
        restored = ReplayBuffer(capacity=1, seed=0)
        restored.load_state_dict(state_dict)

        original_batch = buffer.sample(batch_size=4)
        restored_batch = restored.sample(batch_size=4)

        self.assertTrue(torch.equal(original_batch.reward, restored_batch.reward))
        self.assertTrue(torch.equal(original_batch.done, restored_batch.done))
        self.assertTrue(torch.equal(original_batch.obs["resource_norm"], restored_batch.obs["resource_norm"]))
        self.assertTrue(torch.equal(original_batch.action.allocation, restored_batch.action.allocation))

    def test_topology_stratified_sampling_enforces_collapse_cap_per_topology(self) -> None:
        buffer = ReplayBuffer(
            capacity=48,
            seed=5,
            replay_strategy="topology_stratified_mixed",
            topology_names=("regular", "scale_free"),
            recent_fraction=1.0,
            long_term_fraction=0.0,
            demo_fraction=0.0,
        )

        for topology_name in ("regular", "scale_free"):
            for index in range(12):
                collapse_flag = index < 4
                buffer.add(
                    TensorTransition.from_step(
                        obs=self._make_observation(offset=0.01 * index),
                        action=self._make_action(offset=0.01 * index),
                        reward=float(index),
                        next_obs=self._make_observation(offset=0.01 * (index + 1)),
                        done=False,
                        collapse_flag=collapse_flag,
                        topology_name=topology_name,
                    )
                )

        batch = buffer.sample(batch_size=12, max_collapse_ratio=0.20)
        sample_stats = buffer.get_last_sample_stats()
        topology_ids = batch.topology_id.detach().cpu().numpy().astype(np.int64)
        collapse_flags = batch.collapse_flag.detach().cpu().numpy().astype(np.bool_)

        self.assertGreaterEqual(len(np.unique(topology_ids)), 2)
        for topology_id in np.unique(topology_ids):
            topology_mask = topology_ids == int(topology_id)
            topology_collapse_ratio = float(np.mean(collapse_flags[topology_mask]))
            self.assertLessEqual(topology_collapse_ratio, 0.20 + 1e-8)
        self.assertIn("replay_source_frac_recent", sample_stats)
        self.assertIn("replay_topology_frac_regular", sample_stats)
        self.assertIn("replay_topology_frac_scale_free", sample_stats)

    def test_pool_power_demo_buffer_keeps_pool_power_mix_samples(self) -> None:
        buffer = ReplayBuffer(
            capacity=24,
            seed=17,
            replay_strategy="topology_stratified_mixed",
            topology_names=("regular",),
            recent_fraction=0.0,
            long_term_fraction=0.0,
            demo_fraction=1.0,
        )

        for index in range(6):
            buffer.add(
                TensorTransition.from_step(
                    obs=self._make_observation(offset=0.02 * index),
                    action=self._make_action(offset=0.02 * index),
                    reward=float(index),
                    next_obs=self._make_observation(offset=0.02 * (index + 1)),
                    done=False,
                    is_demo=True,
                    topology_name="regular",
                    pool_power_demo_flag=(index % 2 == 0),
                )
            )

        batch = buffer.sample(batch_size=4)
        sample_stats = buffer.get_last_sample_stats()
        self.assertTrue(torch.all(batch.pool_power_demo_flag))
        self.assertAlmostEqual(sample_stats.get("replay_source_frac_demo", 0.0), 1.0)

    def test_sample_demo_fifo_returns_only_demo_transitions(self) -> None:
        buffer = ReplayBuffer(capacity=16, seed=23)
        for index in range(8):
            buffer.add(
                TensorTransition.from_step(
                    obs=self._make_observation(offset=0.03 * index),
                    action=self._make_action(offset=0.03 * index),
                    reward=float(index),
                    next_obs=self._make_observation(offset=0.03 * (index + 1)),
                    done=False,
                    is_demo=(index % 2 == 0),
                    topology_name="regular",
                    pool_power_demo_flag=(index % 2 == 0),
                )
            )

        batch = buffer.sample_demo(batch_size=4)
        self.assertTrue(torch.all(batch.is_demo))
        self.assertTrue(torch.all(batch.pool_power_demo_flag))

    def test_sample_demo_topology_stratified_covers_multiple_topologies(self) -> None:
        buffer = ReplayBuffer(
            capacity=40,
            seed=31,
            replay_strategy="topology_stratified_mixed",
            topology_names=("regular", "scale_free"),
            recent_fraction=0.0,
            long_term_fraction=0.0,
            demo_fraction=1.0,
        )
        for topology_name in ("regular", "scale_free"):
            for index in range(6):
                buffer.add(
                    TensorTransition.from_step(
                        obs=self._make_observation(offset=0.01 * index),
                        action=self._make_action(offset=0.01 * index),
                        reward=float(index),
                        next_obs=self._make_observation(offset=0.01 * (index + 1)),
                        done=False,
                        is_demo=True,
                        collapse_flag=(index == 0),
                        topology_name=topology_name,
                        pool_power_demo_flag=True,
                    )
                )

        batch = buffer.sample_demo(batch_size=8, max_collapse_ratio=0.20)
        sample_stats = buffer.get_last_sample_stats()
        topology_ids = batch.topology_id.detach().cpu().numpy().astype(np.int64)
        collapse_flags = batch.collapse_flag.detach().cpu().numpy().astype(np.bool_)

        self.assertTrue(torch.all(batch.is_demo))
        self.assertGreaterEqual(len(np.unique(topology_ids)), 2)
        for topology_id in np.unique(topology_ids):
            topology_mask = topology_ids == int(topology_id)
            self.assertLessEqual(float(np.mean(collapse_flags[topology_mask])), 0.20 + 1e-8)
        self.assertAlmostEqual(sample_stats.get("replay_source_frac_demo", 0.0), 1.0)

    def test_state_dict_round_trip_preserves_new_replay_metadata(self) -> None:
        buffer = ReplayBuffer(capacity=8, seed=123)
        for index in range(5):
            buffer.add(
                TensorTransition.from_step(
                    obs=self._make_observation(offset=0.05 * index),
                    action=self._make_action(offset=0.05 * index),
                    reward=float(index) + 0.5,
                    next_obs=self._make_observation(offset=0.05 * (index + 1)),
                    done=False,
                    topology_name="regular" if index % 2 == 0 else "scale_free",
                    pool_power_demo_flag=(index % 2 == 0),
                )
            )

        state_dict = buffer.state_dict()
        restored = ReplayBuffer(capacity=1, seed=0)
        restored.load_state_dict(state_dict)

        original_batch = buffer.sample(batch_size=4)
        restored_batch = restored.sample(batch_size=4)

        self.assertTrue(torch.equal(original_batch.topology_id, restored_batch.topology_id))
        self.assertTrue(torch.equal(original_batch.pool_power_demo_flag, restored_batch.pool_power_demo_flag))
