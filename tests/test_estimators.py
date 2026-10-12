"""Tests for per-path statistics, seeding and the general Fournié Gamma."""

import unittest
import numpy as np

from products import (
    example_european_call_ATM,
    example_european_digital_ATM,
    example_european_call_terminal_knockout,
)
from Benchmarks.europeans import (
    benchmark_greeks_european_call,
    benchmark_greeks_european_digital,
    benchmark_greeks_european_barrier_up_and_out,
)
from Methods import payoffs
from Methods.sampling import generate_paths, estimate_from_samples, simulate_terminal_spot
from Methods.MonteCarlo.FDMC_european import FDMC_geeks_european_call
from Methods.Malliavin import (
    malliavin_greeks_european_call,
    malliavin_samples_european,
    malliavin_fournier_greeks_european_call,
    malliavin_fournier_greeks_european_digital_call,
    malliavin_fournier_greeks_european_barrier_up_and_out_call,
)

GREEKS = ("delta", "gamma", "vega")


class TestPerPathStatistics(unittest.TestCase):

    def test_stats_match_numpy(self):
        params = example_european_call_ATM
        W_T = generate_paths(params["T"], 50_000, seed=7)
        res = malliavin_greeks_european_call(params, paths=W_T, return_stats=True)
        samples = malliavin_samples_european(payoffs.call, params, W_T)
        for g in GREEKS:
            self.assertAlmostEqual(res[g], float(np.mean(samples[g])), places=12)
            self.assertAlmostEqual(res[f"{g}_var"] / np.var(samples[g], ddof=1), 1.0, places=10)
            self.assertAlmostEqual(res[f"{g}_se"], np.sqrt(res[f"{g}_var"] / len(W_T)), places=12)

    def test_chunked_equals_single_block(self):
        params = example_european_call_ATM
        N, seed = 25_000, 3
        sample_fn = lambda W: malliavin_samples_european(payoffs.call, params, W)
        whole = estimate_from_samples(sample_fn, params["T"], N, seed, return_stats=True)
        chunked = estimate_from_samples(sample_fn, params["T"], N, seed, return_stats=True, chunk_size=7_000)
        for k, v in whole.items():
            self.assertAlmostEqual(chunked[k] / v, 1.0, places=10)

    def test_fdmc_samples_match_price_differences(self):
        """The mean of the per-path quotients equals the difference quotient of MC prices."""
        params, eps = example_european_call_ATM, 0.01
        S0, T, r, q, sigma = params["S0"], params["T"], params["r"], params["q"], params["sigma"]
        W_T = generate_paths(T, 20_000, seed=11)

        def price(s0, sig):
            return np.exp(-r * T) * np.mean(payoffs.call(simulate_terminal_spot(s0, T, r, q, sig, W_T), params))

        res = FDMC_geeks_european_call(params, eps=eps, paths=W_T)
        self.assertAlmostEqual(res["delta"], (price(S0 + eps, sigma) - price(S0 - eps, sigma)) / (2 * eps), places=10)
        self.assertAlmostEqual(
            res["gamma"], (price(S0 + eps, sigma) - 2 * price(S0, sigma) + price(S0 - eps, sigma)) / eps**2, places=6
        )
        self.assertAlmostEqual(res["vega"], (price(S0, sigma + eps) - price(S0, sigma - eps)) / (2 * eps), places=8)

    def test_seeds_are_independent_and_reproducible(self):
        params = example_european_call_ATM
        a = malliavin_greeks_european_call(params, N=10_000, seed=1)
        b = malliavin_greeks_european_call(params, N=10_000, seed=1)
        c = malliavin_greeks_european_call(params, N=10_000, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a["delta"], c["delta"])


class TestFournierGamma(unittest.TestCase):
    N = 2_000_000

    def _assert_unbiased(self, res, bench, greeks=GREEKS, z=4.5):
        for g in greeks:
            self.assertLess(abs(res[g] - bench[g]), z * res[f"{g}_se"], msg=f"{g}: {res[g]} vs {bench[g]}")

    def test_call_general_gamma_unbiased(self):
        params = example_european_call_ATM
        res = malliavin_fournier_greeks_european_call(params, N=self.N, seed=5, return_stats=True)
        self._assert_unbiased(res, benchmark_greeks_european_call(params))

    def test_call_shortcut_gamma_unbiased_with_lower_variance(self):
        params = example_european_call_ATM
        kw = dict(N=self.N, seed=5, return_stats=True)
        general = malliavin_fournier_greeks_european_call(params, **kw)
        shortcut = malliavin_fournier_greeks_european_call(params, gamma_formula="shortcut", **kw)
        self._assert_unbiased(shortcut, benchmark_greeks_european_call(params), greeks=("gamma",))
        self.assertLess(shortcut["gamma_var"], general["gamma_var"])

    def test_digital_unbiased(self):
        params = example_european_digital_ATM
        res = malliavin_fournier_greeks_european_digital_call(params, N=self.N, seed=6, return_stats=True)
        self._assert_unbiased(res, benchmark_greeks_european_digital(params))

    def test_terminal_knockout_unbiased_for_every_cutoff(self):
        params = example_european_call_terminal_knockout
        bench = benchmark_greeks_european_barrier_up_and_out(params)
        for cutoff in ("gaussian", "quintic", "cauchy"):
            res = malliavin_fournier_greeks_european_barrier_up_and_out_call(
                params, N=self.N, seed=8, cutoff_type=cutoff, return_stats=True
            )
            self._assert_unbiased(res, bench)

    def test_terminal_knockout_sum_cutoff_bias(self):
        """psi_K + psi_H: biased with the gaussian (psi(K) != 1), identical to the union for compact cutoffs."""
        params = example_european_call_terminal_knockout
        bench = benchmark_greeks_european_barrier_up_and_out(params)
        kw = dict(N=self.N, seed=9, return_stats=True)
        biased = malliavin_fournier_greeks_european_barrier_up_and_out_call(
            params, cutoff_type="gaussian", cutoff_combination="sum", **kw
        )
        self.assertGreater(abs(biased["delta"] - bench["delta"]), 50 * biased["delta_se"])
        self.assertGreater(abs(biased["gamma"] - bench["gamma"]), 50 * biased["gamma_se"])

        summed = malliavin_fournier_greeks_european_barrier_up_and_out_call(
            params, cutoff_type="quintic", cutoff_combination="sum", **kw
        )
        union = malliavin_fournier_greeks_european_barrier_up_and_out_call(params, cutoff_type="quintic", **kw)
        for g in GREEKS:
            self.assertAlmostEqual(summed[g], union[g], places=12)


if __name__ == "__main__":
    unittest.main()
