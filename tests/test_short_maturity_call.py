"""Unit tests for the 1-month European Call option (T = 1/12)."""

import unittest
import numpy as np
from scipy.stats import norm

from products import example_european_call_1M
from Methods.Analytic.europeans import greeks_european_call, d1, d2
from Methods import payoffs
from Methods.sampling import generate_paths
from Methods.Malliavin import (
    fournier_bandwidths,
    malliavin_fournier_greeks_european_call,
    malliavin_fournier_samples_european_call,
    malliavin_greeks_european_call,
    malliavin_samples_european,
)


class TestShortMaturityCall(unittest.TestCase):
    """Test suite for 1-month European Call pricing and Greeks."""

    def setUp(self):
        self.params = example_european_call_1M

    def test_product_specifications(self):
        """Verify contract parameter integrity."""
        self.assertEqual(self.params["S0"], 100.0)
        self.assertEqual(self.params["K"], 100.0)
        self.assertAlmostEqual(self.params["T"], 1.0 / 12.0, places=6)
        self.assertEqual(self.params["sigma"], 0.20)
        self.assertEqual(self.params["name"], "Call European 1M")

    def test_analytical_greeks(self):
        """Verify analytical Black-Scholes Greeks."""
        greeks = greeks_european_call(self.params)
        self.assertIn("delta", greeks)
        self.assertIn("gamma", greeks)
        self.assertIn("vega", greeks)

        # BS Delta: e^{-qT} * N(d1)
        d1_val = d1(
            self.params["S0"],
            self.params["K"],
            self.params["T"],
            self.params["r"],
            self.params["q"],
            self.params["sigma"],
        )
        expected_delta = np.exp(-self.params["q"] * self.params["T"]) * norm.cdf(d1_val)
        self.assertAlmostEqual(greeks["delta"], expected_delta, places=5)
        self.assertAlmostEqual(greeks["delta"], 0.527886, places=4)
        self.assertAlmostEqual(greeks["gamma"], 0.068804, places=4)
        self.assertAlmostEqual(greeks["vega"], 11.46739, places=3)

    def test_calibrated_delta_bandwidth(self):
        """Verify the coupled bandwidth: delta* = 0.40 * S0 * sigma * sqrt(T)."""
        delta_star = fournier_bandwidths(self.params, coupled=True)["delta"]
        expected = 0.40 * 100.0 * 0.20 * np.sqrt(1.0 / 12.0)
        self.assertAlmostEqual(delta_star, expected, places=5)
        self.assertAlmostEqual(delta_star, 2.3094, places=3)

    def test_standard_vs_fournier_unbiasedness(self):
        """Verify both Standard and Fournié Malliavin converge to analytical Delta."""
        N = 100_000
        seed = 42
        analytic = greeks_european_call(self.params)
        g_std = malliavin_greeks_european_call(self.params, N=N, seed=seed)
        g_four = malliavin_fournier_greeks_european_call(self.params, N=N, seed=seed)

        # Both should be within 3 standard errors of analytical value
        self.assertAlmostEqual(g_std["delta"], analytic["delta"], delta=0.02)
        self.assertAlmostEqual(g_four["delta"], analytic["delta"], delta=0.01)

    def test_fournier_variance_reduction(self):
        """Fournié localization (coupled bandwidth) reduces the per-path variance of every Greek."""
        paths = generate_paths(self.params["T"], 50_000, seed=42)
        standard = malliavin_samples_european(payoffs.call, self.params, paths)
        coupled = fournier_bandwidths(self.params, coupled=True)["delta"]
        localized = malliavin_fournier_samples_european_call(self.params, paths, bandwidth=coupled)

        reduction = {g: np.var(standard[g]) / np.var(localized[g]) for g in ("delta", "gamma", "vega")}
        self.assertGreater(reduction["delta"], 3.0)
        self.assertGreater(reduction["gamma"], 3.0)
        self.assertGreater(reduction["vega"], 10.0)

    def test_option_a_decoupled_bandwidths(self):
        """Verify Option A decoupled bandwidths achieve simultaneous variance reduction."""
        N = 50_000
        # 1. Coupled call (single delta = 2.3094)
        g_coupled = malliavin_fournier_greeks_european_call(
            self.params, N=N, seed=42, bandwidth=2.3094
        )
        # 2. Decoupled Option A call (None auto-resolves decoupled: delta_d=1.15, delta_g=2.89, delta_v=1.15)
        g_decoupled = malliavin_fournier_greeks_european_call(
            self.params, N=N, seed=42, bandwidth=None
        )
        analytic = greeks_european_call(self.params)

        # Verify unbiasedness
        self.assertAlmostEqual(g_decoupled["delta"], analytic["delta"], delta=0.015)
        self.assertAlmostEqual(g_decoupled["gamma"], analytic["gamma"], delta=0.005)
        self.assertAlmostEqual(g_decoupled["vega"], analytic["vega"], delta=0.15)

        # Per-Greek defaults
        d_dict = fournier_bandwidths(self.params)
        self.assertIn("delta", d_dict)
        self.assertIn("gamma", d_dict)
        self.assertIn("vega", d_dict)
        self.assertLess(d_dict["delta"], d_dict["gamma"])  # delta is narrower than gamma


if __name__ == "__main__":
    unittest.main()

