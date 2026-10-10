"""Unit tests for discrete arithmetic Asian option pricing and Greek computations."""

import unittest
import numpy as np
from products import (
    example_asian_call_ITM,
    example_asian_call_ATM,
    example_asian_call_OTM,
    example_asian_digital_ITM,
    example_asian_digital_ATM,
    example_asian_digital_OTM,
    example_asian_barrier_up_and_out_ITM,
    example_asian_barrier_up_and_out_ATM,
    example_asian_barrier_up_and_out_OTM,
)
from pricers.asians_analytic import (
    compute_discrete_asian_moments,
    price_asian_call_moment_matching,
    price_asian_digital_call_moment_matching,
    price_asian_barrier_up_and_out_call_moment_matching,
)
from pricers.asians_MC import (
    generate_asian_paths,
    MC_asian_call_option,
    MC_asian_digital_call_option,
    MC_asian_barrier_up_and_out_call_option,
)
from Methods.Analytic.asians import (
    greeks_asian_call,
    greeks_asian_digital_call,
    greeks_asian_barrier_up_and_out_call,
)
from Methods.MonteCarlo.FDMC_asian import (
    FDMC_greeks_asian_call,
    FDMC_greeks_asian_digital_call,
    FDMC_greeks_asian_barrier_up_and_out_call,
)


class TestAsianGreeks(unittest.TestCase):
    """Test suite for discrete arithmetic Asian option pricing and Greeks."""

    def test_discrete_moments(self):
        """Verify that moments are positive and variance matches theoretical bounds."""
        S0, T, r, q, sigma, freq = 100.0, 1.0, 0.05, 0.02, 0.20, 12
        M1, M2, muA, vA = compute_discrete_asian_moments(S0, T, r, q, sigma, freq)

        self.assertGreater(M1, 0.0)
        self.assertGreater(M2, M1**2)  # Variance must be strictly positive
        self.assertGreater(vA, 0.0)
        # Effective vol vA is approximately sigma * sqrt(T/3) ~ 0.115
        self.assertTrue(0.08 < vA < 0.16)

    def test_moment_matching_pricing_ordering(self):
        """Verify standard financial ordering properties for MM pricing."""
        p_itm = price_asian_call_moment_matching(100.0, 80.0, 1.0, 0.05, 0.02, 0.20, 12)
        p_atm = price_asian_call_moment_matching(100.0, 100.0, 1.0, 0.05, 0.02, 0.20, 12)
        p_otm = price_asian_call_moment_matching(100.0, 125.0, 1.0, 0.05, 0.02, 0.20, 12)

        # Monotonicity in strike
        self.assertGreater(p_itm, p_atm)
        self.assertGreater(p_atm, p_otm)
        self.assertGreater(p_otm, 0.0)

        # Digital option prices bounded by discount factor
        d_atm = price_asian_digital_call_moment_matching(100.0, 100.0, 1.0, 0.05, 0.02, 0.20, 12)
        self.assertTrue(0.0 < d_atm < np.exp(-0.05 * 1.0))

        # Barrier call price strictly less than vanilla call price
        b_atm = price_asian_barrier_up_and_out_call_moment_matching(100.0, 100.0, 1.0, 0.05, 0.02, 0.20, 135.0, 12)
        self.assertTrue(0.0 < b_atm < p_atm)

    def test_monte_carlo_pricing_consistency(self):
        """Verify Monte Carlo price converges to close proximity of MM price ATM."""
        paths = generate_asian_paths(1.0, 12, N=200_000, seed=42)
        mc_call = MC_asian_call_option(100.0, 100.0, 1.0, 0.05, 0.02, 0.20, 12, paths=paths)
        mm_call = price_asian_call_moment_matching(100.0, 100.0, 1.0, 0.05, 0.02, 0.20, 12)

        # Relative difference ATM should be under 1%
        rel_diff = abs(mc_call - mm_call) / mm_call
        self.assertLess(rel_diff, 0.01)

    def test_moment_matching_greeks(self):
        """Verify Greeks computed via Finite Differences on MM pricing have expected signs."""
        greeks_call = greeks_asian_call(example_asian_call_ATM)
        self.assertTrue(0.0 < greeks_call["delta"] < 1.0)
        self.assertGreater(greeks_call["gamma"], 0.0)
        self.assertGreater(greeks_call["vega"], 0.0)

        greeks_dig = greeks_asian_digital_call(example_asian_digital_ATM)
        self.assertGreater(greeks_dig["delta"], 0.0)

        greeks_bar = greeks_asian_barrier_up_and_out_call(example_asian_barrier_up_and_out_ATM)
        self.assertTrue(0.0 < greeks_bar["delta"] < greeks_call["delta"])

    def test_fdmc_vs_mm_delta_atm(self):
        """Verify that FDMC Delta with Common Numbers matches MM Delta ATM within reasonable MC error."""
        paths = generate_asian_paths(1.0, 12, N=200_000, seed=42)
        fdmc_g = FDMC_greeks_asian_call(example_asian_call_ATM, eps=0.01, N=200_000, paths=paths)
        mm_g = greeks_asian_call(example_asian_call_ATM, eps=1e-5)

        # Delta should be very close (~0.55)
        self.assertLess(abs(fdmc_g["delta"] - mm_g["delta"]), 0.02)


if __name__ == "__main__":
    unittest.main()
