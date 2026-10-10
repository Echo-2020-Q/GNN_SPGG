import unittest
from pathlib import Path

import analyze_four_policy_param_sweep_grid as sweep


class PoolPowerKSweepTests(unittest.TestCase):
    def test_sweep_value_overrides_fixed_value(self) -> None:
        actual = sweep._resolve_pool_power_k_for_sweep_point(
            x_var="pool_power_k",
            x_value=22.0,
            fixed_pool_power_k=19.0,
        )
        self.assertAlmostEqual(actual, 22.0)

    def test_fixed_value_is_used_for_other_sweep_variables(self) -> None:
        actual = sweep._resolve_pool_power_k_for_sweep_point(
            x_var="r",
            x_value=3.0,
            fixed_pool_power_k=19.0,
        )
        self.assertAlmostEqual(actual, 19.0)

    def test_invalid_values_are_rejected(self) -> None:
        for invalid_value in (-1.0, float("nan"), float("inf")):
            with self.subTest(invalid_value=invalid_value):
                with self.assertRaisesRegex(ValueError, "pool_power_k"):
                    sweep._resolve_pool_power_k_for_sweep_point(
                        x_var="pool_power_k",
                        x_value=invalid_value,
                        fixed_pool_power_k=19.0,
                    )

    def test_adaptive_policy_is_built_with_current_sweep_value(self) -> None:
        power_k = sweep._resolve_pool_power_k_for_sweep_point(
            x_var="pool_power_k",
            x_value=17.0,
            fixed_pool_power_k=19.0,
        )
        policies, metadata = sweep._build_policy_objects_for_sweep_point(
            m1_summary_json=Path("unused-m1.json"),
            m2_summary_json=Path("unused-m2.json"),
            selected_policies=["adaptive"],
            x_var="pool_power_k",
            x_value=17.0,
            requested_m2_mode="auto",
            requested_m2_current_normalizer=None,
            requested_m2_saved_normalizer=None,
            saved_reference_pc=50.0,
            requested_pc_normalizer_knots=None,
            pool_power_k=power_k,
        )

        self.assertAlmostEqual(policies["adaptive"].power_k, 17.0)
        self.assertAlmostEqual(metadata["pool_power_k"], 17.0)


if __name__ == "__main__":
    unittest.main()
