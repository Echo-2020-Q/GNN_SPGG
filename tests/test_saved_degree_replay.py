import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from Public.mechanisms import load_m2
from run_saved_resource_conditioned_spline_policy import _load_saved_conditioned_policy


class SavedDegreeReplayTests(unittest.TestCase):
    def load_without_cached_lookup(self):
        bins = [
            {"bin_label": "Q1:2", "degree_left": 2, "degree_right": 2},
            {"bin_label": "Q2:3", "degree_left": 3, "degree_right": 3},
            {"bin_label": "Q3:4-6", "degree_left": 4, "degree_right": 6},
            {"bin_label": "Q4:7-19", "degree_left": 7, "degree_right": 19},
        ]
        payload = {"adaptive_unitinv_equal_profile": {
            "conditioning_mode": "sender_degree_bin",
            "profile_family": "three_way",
            "condition_spec": {"bins": bins},
            "bin_profiles": {group["bin_label"]: {
                "mode": "binned_3way", "bin_edges_resource": [0, 100],
                "bin_weight_rows": [[index / 3, 1 - index / 3, 0]],
            } for index, group in enumerate(bins)},
        }}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            policy, _, _, _, _ = _load_saved_conditioned_policy(
                path, replay_ego_resource_x_mode="absolute",
                replay_ego_resource_normalizer=None, saved_ego_resource_normalizer=None,
            )
        return policy

    def test_saved_intervals_restore_all_four_groups(self):
        policy = self.load_without_cached_lookup()
        observation = {"degrees": np.array([2, 3, 4, 5, 6, 7, 10, 19])}
        actual = [policy.weights_for_sender(observation, index, 20.0) for index in range(8)]
        self.assertEqual([row[3] for row in actual], [
            "Q1:2", "Q2:3", "Q3:4-6", "Q3:4-6", "Q3:4-6", "Q4:7-19", "Q4:7-19", "Q4:7-19",
        ])
        np.testing.assert_allclose([row[0] for row in actual], [0, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1, 1, 1])

    def test_out_of_range_degrees_keep_original_fallback(self):
        policy = self.load_without_cached_lookup()
        observation = {"degrees": np.array([0, 1, 20])}
        for sender in range(3):
            self.assertEqual(policy._resolve_sender_bin_label(observation, sender), "Q4:7-19")

    def test_public_mechanism_uses_degree_specific_profiles(self):
        policy = load_m2()
        observation = {"degrees": np.array([2, 3, 6, 7])}
        labels = ["Q1:2", "Q2:3", "Q3:4-6", "Q4:7-19"]
        for sender, label in enumerate(labels):
            self.assertIs(policy._profile_for_sender(observation, sender), policy.profiles[label])


if __name__ == "__main__":
    unittest.main()
