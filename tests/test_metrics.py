"""Unit and regression tests for Utilities metrics, convergence, and comparisons."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

import Utilities as utils


class TestMetricsAndComparisons(unittest.TestCase):
    """Test suite for metrics and comparison refactored utilities."""

    def setUp(self):
        """Set up synthetic benchmark and simulation data."""
        self.benchmarks_df = pd.DataFrame([
            {"Product": "Call European ATM", "delta": 0.58685, "gamma": 0.01895, "vega": 37.90116, "Method": "Analytical"},
            {"Product": "Digital European ATM", "delta": 0.01895, "gamma": -0.0002369, "vega": -0.47376, "Method": "Analytical"},
            {"Product": "Zero Benchmark Option", "delta": 0.0, "gamma": 0.0, "vega": 0.0, "Method": "Analytical"},
        ])

        # Synthetic FDMC results with multiple sample sizes and eps
        fdmc_rows = []
        for prod in ["Call European ATM", "Digital European ATM"]:
            b_row = self.benchmarks_df[self.benchmarks_df["Product"] == prod].iloc[0]
            for n in [10_000, 100_000, 1_000_000]:
                for eps in [0.1, 0.01, 0.001]:
                    noise = 0.01 / np.sqrt(n)
                    bias = 0.001 * (eps ** 2) if "Call" in prod else 0.01 * eps
                    fdmc_rows.append({
                        "Product": prod,
                        "N": n,
                        "eps": eps,
                        "delta": b_row["delta"] + bias + noise,
                        "gamma": b_row["gamma"] + bias * 5 + noise * 2,
                        "vega": b_row["vega"] + bias * 10 + noise * 5,
                        "Method": "FDMC",
                    })
        self.fdmc_df = pd.DataFrame(fdmc_rows)

        # Synthetic Malliavin results (no eps)
        mal_rows = []
        for prod in ["Call European ATM", "Digital European ATM"]:
            b_row = self.benchmarks_df[self.benchmarks_df["Product"] == prod].iloc[0]
            for n in [10_000, 100_000, 1_000_000]:
                noise = 0.005 / np.sqrt(n)
                mal_rows.append({
                    "Product": prod,
                    "N": n,
                    "delta": b_row["delta"] + noise,
                    "gamma": b_row["gamma"] + noise * 1.5,
                    "vega": b_row["vega"] + noise * 3,
                    "Method": "Malliavin",
                })
        self.malliavin_df = pd.DataFrame(mal_rows)

        # Synthetic Fournié results (no eps)
        four_rows = []
        for prod in ["Call European ATM", "Digital European ATM"]:
            b_row = self.benchmarks_df[self.benchmarks_df["Product"] == prod].iloc[0]
            for n in [10_000, 100_000, 1_000_000]:
                noise = 0.001 / np.sqrt(n)
                four_rows.append({
                    "Product": prod,
                    "N": n,
                    "delta": b_row["delta"] + noise,
                    "gamma": b_row["gamma"] + noise * 0.5,
                    "vega": b_row["vega"] + noise,
                    "Method": "Fournié Malliavin",
                })
        self.fournier_df = pd.DataFrame(four_rows)

        # Timing DataFrames
        self.fdmc_timing = pd.DataFrame([
            {"Product": "Call European ATM", "N": 10_000, "time_ms_mean": 15.0, "time_ms_std": 1.0, "ms_per_10k_paths": 15.0},
            {"Product": "Call European ATM", "N": 100_000, "time_ms_mean": 150.0, "time_ms_std": 5.0, "ms_per_10k_paths": 15.0},
            {"Product": "Call European ATM", "N": 1_000_000, "time_ms_mean": 1500.0, "time_ms_std": 20.0, "ms_per_10k_paths": 15.0},
            {"Product": "Digital European ATM", "N": 10_000, "time_ms_mean": 15.0, "time_ms_std": 1.0, "ms_per_10k_paths": 15.0},
            {"Product": "Digital European ATM", "N": 100_000, "time_ms_mean": 150.0, "time_ms_std": 5.0, "ms_per_10k_paths": 15.0},
            {"Product": "Digital European ATM", "N": 1_000_000, "time_ms_mean": 1500.0, "time_ms_std": 20.0, "ms_per_10k_paths": 15.0},
        ])

        self.mal_timing = pd.DataFrame([
            {"Product": "Call European ATM", "N": 10_000, "time_ms_mean": 8.0, "time_ms_std": 0.5, "ms_per_10k_paths": 8.0},
            {"Product": "Call European ATM", "N": 100_000, "time_ms_mean": 80.0, "time_ms_std": 2.0, "ms_per_10k_paths": 8.0},
            {"Product": "Call European ATM", "N": 1_000_000, "time_ms_mean": 800.0, "time_ms_std": 10.0, "ms_per_10k_paths": 8.0},
            {"Product": "Digital European ATM", "N": 10_000, "time_ms_mean": 8.0, "time_ms_std": 0.5, "ms_per_10k_paths": 8.0},
            {"Product": "Digital European ATM", "N": 100_000, "time_ms_mean": 80.0, "time_ms_std": 2.0, "ms_per_10k_paths": 8.0},
            {"Product": "Digital European ATM", "N": 1_000_000, "time_ms_mean": 800.0, "time_ms_std": 10.0, "ms_per_10k_paths": 8.0},
        ])

        self.four_timing = pd.DataFrame([
            {"Product": "Call European ATM", "N": 10_000, "time_ms_mean": 10.0, "time_ms_std": 0.5, "ms_per_10k_paths": 10.0},
            {"Product": "Call European ATM", "N": 100_000, "time_ms_mean": 100.0, "time_ms_std": 2.0, "ms_per_10k_paths": 10.0},
            {"Product": "Call European ATM", "N": 1_000_000, "time_ms_mean": 1000.0, "time_ms_std": 10.0, "ms_per_10k_paths": 10.0},
            {"Product": "Digital European ATM", "N": 10_000, "time_ms_mean": 10.0, "time_ms_std": 0.5, "ms_per_10k_paths": 10.0},
            {"Product": "Digital European ATM", "N": 100_000, "time_ms_mean": 100.0, "time_ms_std": 2.0, "ms_per_10k_paths": 10.0},
            {"Product": "Digital European ATM", "N": 1_000_000, "time_ms_mean": 1000.0, "time_ms_std": 10.0, "ms_per_10k_paths": 10.0},
        ])

    def test_compute_error_metrics(self):
        """Test compute_error_metrics calculations, signed bias, and NaN handling."""
        metrics = utils.compute_error_metrics(self.fdmc_df, self.benchmarks_df)

        # Check required columns
        for g in ["delta", "gamma", "vega"]:
            self.assertIn(f"{g}_bench", metrics.columns)
            self.assertIn(f"{g}_abs_error", metrics.columns)
            self.assertIn(f"{g}_rel_error", metrics.columns)
            self.assertIn(f"{g}_bias", metrics.columns)
            self.assertIn(f"{g}_rel_bias", metrics.columns)
            self.assertIn(f"{g}_log10_abs_error", metrics.columns)
            self.assertIn(f"{g}_log10_rel_error", metrics.columns)

        # Check math correctness
        first_row = metrics.iloc[0]
        self.assertAlmostEqual(first_row["delta_abs_error"], abs(first_row["delta"] - first_row["delta_bench"]))
        self.assertAlmostEqual(first_row["delta_bias"], first_row["delta"] - first_row["delta_bench"])

        # Check zero benchmark NaN preservation (zero benchmark should have NaN relative error and log error)
        zero_df = pd.DataFrame([{"Product": "Zero Benchmark Option", "delta": 0.05, "gamma": 0.01, "vega": 0.2}])
        zero_metrics = utils.compute_error_metrics(zero_df, self.benchmarks_df)
        self.assertTrue(np.isnan(zero_metrics["delta_rel_error"].iloc[0]))
        self.assertTrue(np.isnan(zero_metrics["delta_log10_rel_error"].iloc[0]))

    def test_is_catastrophic_eps_and_series(self):
        """Test catastrophic regime detection on scalars and series."""
        # 1. Total collapse to zero
        self.assertTrue(utils.is_catastrophic_eps("Call European ATM", "delta", 0.01, 1e-8, 0.58))
        self.assertFalse(utils.is_catastrophic_eps("Call European ATM", "delta", 0.01, 0.58, 0.58))

        # 2. General Gamma < 0.01
        self.assertTrue(utils.is_catastrophic_eps("Call European ATM", "gamma", 0.001, 0.018, 0.018))
        self.assertFalse(utils.is_catastrophic_eps("Call European ATM", "gamma", 0.01, 0.018, 0.018))

        # 3. Discontinuous Gamma < 0.1
        self.assertTrue(utils.is_catastrophic_eps("Digital European ATM", "gamma", 0.05, -0.0002, -0.0002))
        self.assertFalse(utils.is_catastrophic_eps("Digital European ATM", "gamma", 0.1, -0.0002, -0.0002))

        # 4. Discontinuous Delta < 0.01
        self.assertTrue(utils.is_catastrophic_eps("Call with terminal knock-out", "delta", 0.005, 0.0015, 0.0015))
        self.assertFalse(utils.is_catastrophic_eps("Call with terminal knock-out", "delta", 0.01, 0.0015, 0.0015))

        # Vectorized series test
        prods = pd.Series(["Call European ATM", "Digital European ATM", "Call European ATM"])
        greeks = pd.Series(["gamma", "gamma", "gamma"])
        eps_s = pd.Series([0.001, 0.05, 0.05])
        ests = pd.Series([0.018, -0.0002, 0.018])
        benches = pd.Series([0.018, -0.0002, 0.018])

        flags = utils.is_catastrophic_eps_series(prods, greeks, eps_s, ests, benches)
        self.assertTrue(flags.iloc[0])  # call gamma < 0.01
        self.assertTrue(flags.iloc[1])  # digital gamma < 0.1
        self.assertFalse(flags.iloc[2]) # call gamma >= 0.01

    def test_extract_fixed_or_optimal_eps(self):
        """Test extracting fixed and optimal epsilon subsets."""
        metrics = utils.compute_error_metrics(self.fdmc_df, self.benchmarks_df)

        # Fixed eps
        fixed_df = utils.extract_fixed_or_optimal_eps(metrics, fdmc_eps=0.01)
        self.assertFalse(fixed_df.empty)
        self.assertIn("optimal_eps", fixed_df.columns)
        self.assertIn("min_abs_error", fixed_df.columns)
        self.assertIn("min_rel_error_pct", fixed_df.columns)
        self.assertIn("fdmc_value", fixed_df.columns)
        self.assertIn("benchmark_value", fixed_df.columns)
        self.assertTrue((fixed_df["optimal_eps"] == 0.01).all())

        # Optimal eps without catastrophic filter (pure mathematical minimum)
        opt_df = utils.extract_fixed_or_optimal_eps(metrics, fdmc_eps="optimal", exclude_catastrophic=False)
        self.assertFalse(opt_df.empty)
        # Optimal error must be <= fixed error for all slices
        for prod in metrics["Product"].unique():
            for g in ["delta", "gamma", "vega"]:
                sub_opt = opt_df[(opt_df["Product"] == prod) & (opt_df["Greek"] == g) & (opt_df["N"] == 1_000_000)]
                sub_fix = fixed_df[(fixed_df["Product"] == prod) & (fixed_df["Greek"] == g) & (fixed_df["N"] == 1_000_000)]
                if not sub_opt.empty and not sub_fix.empty:
                    self.assertLessEqual(sub_opt["min_abs_error"].iloc[0], sub_fix["min_abs_error"].iloc[0] + 1e-12)

        # Missing eps column (e.g. Malliavin) should not raise KeyError
        mal_metrics = utils.compute_error_metrics(self.malliavin_df, self.benchmarks_df)
        mal_extracted = utils.extract_fixed_or_optimal_eps(mal_metrics)
        self.assertFalse(mal_extracted.empty)
        self.assertTrue(np.isnan(mal_extracted["optimal_eps"].iloc[0]))

    def test_generate_error_pivot(self):
        """Test reshaping metrics into an N x eps pivot table."""
        metrics = utils.compute_error_metrics(self.fdmc_df, self.benchmarks_df)
        pivot = utils.generate_error_pivot(metrics, "Call European ATM", "gamma", metric="abs_error")

        self.assertEqual(list(pivot.index), [10_000, 100_000, 1_000_000])
        self.assertEqual(list(pivot.columns), [0.1, 0.01, 0.001])  # Descending order

    def test_compute_comparative_summary_table(self):
        """Test side-by-side executive comparative table generation."""
        f_met = utils.compute_error_metrics(self.fdmc_df, self.benchmarks_df)
        m_met = utils.compute_error_metrics(self.malliavin_df, self.benchmarks_df)
        four_met = utils.compute_error_metrics(self.fournier_df, self.benchmarks_df)

        # 1. 2-way comparison (FDMC vs Std Malliavin)
        comp_2way = utils.compute_comparative_summary_table(f_met, m_met, n_choice=1_000_000)
        self.assertIn("Accuracy Ratio (FDMC/Mal)", comp_2way.columns)
        self.assertIn("Ratio (FDMC/Std Mal)", comp_2way.columns)
        self.assertIn("Std Malliavin", comp_2way.columns)

        # 2. 3-way comparison (FDMC vs Std Malliavin vs Fournié)
        comp_3way = utils.compute_comparative_summary_table(
            f_met, m_met, n_choice=1_000_000, fournier_metrics_df=four_met
        )
        self.assertIn("Fournié Malliavin", comp_3way.columns)
        self.assertIn("Ratio (FDMC/Fournié)", comp_3way.columns)
        self.assertIn("Fournié Gain vs Std", comp_3way.columns)
        self.assertIn("Ratio (FDMC/Std Mal)", comp_3way.columns)
        self.assertIn("Accuracy Ratio (FDMC/Mal)", comp_3way.columns)

    def test_find_min_acceptable_N_comparison(self):
        """Test finding minimum sample size and speedups across methods."""
        f_met = utils.compute_error_metrics(self.fdmc_df, self.benchmarks_df)
        m_met = utils.compute_error_metrics(self.malliavin_df, self.benchmarks_df)
        four_met = utils.compute_error_metrics(self.fournier_df, self.benchmarks_df)

        # Without Fournié
        res_2way = utils.find_min_acceptable_N_comparison(
            f_met, m_met,
            fdmc_timing_df=self.fdmc_timing,
            malliavin_timing_df=self.mal_timing,
        )
        self.assertIn("FDMC Min N*", res_2way.columns)
        self.assertIn("Malliavin Min N*", res_2way.columns)
        self.assertIn("Speedup (FDMC/Mal Time)", res_2way.columns)

        # With Fournié
        res_3way = utils.find_min_acceptable_N_comparison(
            f_met, m_met,
            fdmc_timing_df=self.fdmc_timing,
            malliavin_timing_df=self.mal_timing,
            fournier_metrics_df=four_met,
            fournier_timing_df=self.four_timing,
        )
        self.assertIn("Fournié Min N*", res_3way.columns)
        self.assertIn("Fournié Time (ms)", res_3way.columns)
        self.assertIn("Speedup (FDMC/Fournié Time)", res_3way.columns)
        self.assertIn("Fournié Speedup vs Mal", res_3way.columns)

    def test_empirical_convergence_rate(self):
        """Test asymptotic empirical convergence rate alpha fitting."""
        # Create pure O(N^-0.5) data
        n_vals = [10_000, 40_000, 160_000, 640_000, 2_560_000]
        err_vals = [1.0 / np.sqrt(n) for n in n_vals]
        df = pd.DataFrame({
            "Product": "Call European ATM",
            "N": n_vals,
            "gamma_abs_error": err_vals,
        })

        fit = utils.compute_empirical_convergence_rate(df, "Call European ATM", greek="gamma")
        self.assertAlmostEqual(fit["alpha"], 0.5, places=3)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=3)

    def test_load_simulation_results_isolated(self):
        """Test that load_simulation_results can load FDMC and benchmarks without crashing."""
        b_df, f_df = utils.load_simulation_results(
            benchmarks_path="./results/benchmarks.csv",
            fdmc_path="./results/FDMC_results.csv",
        )
        self.assertFalse(b_df.empty)
        self.assertFalse(f_df.empty)
        self.assertIn("delta", b_df.columns)
        self.assertIn("delta", f_df.columns)

    def test_real_dataset_pipeline(self):
        """Regression test verifying real results CSVs run through the full analysis pipeline."""
        b_df, f_df, m_df, four_df = utils.load_all_simulation_results()

        f_met = utils.compute_error_metrics(f_df, b_df)
        m_met = utils.compute_error_metrics(m_df, b_df)
        four_met = utils.compute_error_metrics(four_df, b_df)

        # Check summary table
        summary = utils.compute_comparative_summary_table(
            f_met, m_met, n_choice=100_000_000, fournier_metrics_df=four_met
        )
        self.assertEqual(len(summary), 9)  # 3 products * 3 greeks

    def test_summarize_cutoff_and_delta(self):
        """Test cutoff function and delta sensitivity summarizers."""
        bench = {"delta": 0.5868, "gamma": 0.0189, "vega": 37.90}
        cutoff_dict = {
            "gaussian": {"delta": 0.5870, "gamma": 0.0190, "vega": 37.91},
            "cubic": {"delta": 0.5865, "gamma": 0.0188, "vega": 37.89},
        }
        df_cutoff = utils.summarize_cutoff_functions(cutoff_dict, bench)
        self.assertEqual(len(df_cutoff), 2)
        self.assertIn("Delta Abs Error", df_cutoff.columns)
        self.assertIn("Gamma Bias", df_cutoff.columns)

        delta_dict = {
            2.0: {"delta": 0.5870, "gamma": 0.0190, "vega": 37.91},
            5.0: {"delta": 0.5866, "gamma": 0.0189, "vega": 37.90},
        }
        df_delta = utils.summarize_delta_strike_sensitivities(delta_dict, bench)
        self.assertEqual(len(df_delta), 2)
        self.assertIn("Delta Strike (δ)", df_delta.columns)
        self.assertIn("Vega Abs Error", df_delta.columns)

    def test_find_optimal_delta(self):
        """Test finding optimal delta from smoothing grid."""
        grid_data = pd.DataFrame([
            {"cutoff_type": "gaussian", "delta_strike": 2.0, "gamma": 0.0191, "gamma_abs_error": 0.0002, "gamma_var": 0.00001, "gamma_std": 0.003},
            {"cutoff_type": "gaussian", "delta_strike": 5.0, "gamma": 0.0190, "gamma_abs_error": 0.0001, "gamma_var": 0.000005, "gamma_std": 0.0022},
            {"cutoff_type": "cubic", "delta_strike": 2.0, "gamma": 0.0192, "gamma_abs_error": 0.0003, "gamma_var": 0.00002, "gamma_std": 0.004},
            {"cutoff_type": "cubic", "delta_strike": 5.0, "gamma": 0.0189, "gamma_abs_error": 0.00005, "gamma_var": 0.000008, "gamma_std": 0.0028},
        ])

        opt_var = utils.find_optimal_delta(grid_data, greek="gamma", criterion="var")
        self.assertEqual(len(opt_var), 2)
        # For gaussian, min var is at delta=5.0
        gauss_opt = opt_var[opt_var["Cutoff Function"] == "gaussian"].iloc[0]
        self.assertEqual(gauss_opt["Optimal δ* (var)"], 5.0)

        # Test abs_error criterion
        opt_err = utils.find_optimal_delta(grid_data, greek="gamma", criterion="abs_error")
        cubic_opt = opt_err[opt_err["Cutoff Function"] == "cubic"].iloc[0]
        self.assertEqual(cubic_opt["Optimal δ* (abs_error)"], 5.0)

    def test_construct_comparative_pareto_dataset(self):
        """Test comparative pareto dataset construction including Fournié."""
        f_met = utils.compute_error_metrics(self.fdmc_df, self.benchmarks_df)
        m_met = utils.compute_error_metrics(self.malliavin_df, self.benchmarks_df)
        four_met = utils.compute_error_metrics(self.fournier_df, self.benchmarks_df)

        # 2-way Pareto
        p_2way = utils.construct_comparative_pareto_dataset(
            f_met, m_met, self.fdmc_timing, self.mal_timing
        )
        self.assertIn("Method", p_2way.columns)
        self.assertEqual(set(p_2way["Method"].unique()), {"FDMC", "Malliavin"})
        self.assertIn("sim_value", p_2way.columns)
        self.assertIn("fdmc_value", p_2way.columns)

        # 3-way Pareto
        p_3way = utils.construct_comparative_pareto_dataset(
            f_met, m_met, self.fdmc_timing, self.mal_timing,
            fournier_metrics_df=four_met, fournier_timing_df=self.four_timing,
        )
        self.assertEqual(set(p_3way["Method"].unique()), {"FDMC", "Malliavin", "Fournié Malliavin"})

    def test_compute_fournier_delta(self):
        """Test Fournié delta gap calculation for vanilla, digital, and barrier options."""
        from Methods.Malliavin import compute_fournier_delta
        from products import example_european_call_ATM, example_european_digital_ATM, example_european_call_terminal_knockout

        d_call = compute_fournier_delta(example_european_call_ATM, is_barrier=False)
        d_dig = compute_fournier_delta(example_european_digital_ATM, is_barrier=False)
        d_bar = compute_fournier_delta(example_european_call_terminal_knockout, is_barrier=True)

        self.assertAlmostEqual(d_call, 8.0, places=6)
        self.assertAlmostEqual(d_dig, 8.0, places=6)
        self.assertAlmostEqual(d_bar, 2.625, places=6)

        # Custom barrier with wide corridor
        wide_bar = {"S0": 100.0, "sigma": 0.2, "T": 1.0, "K": 100.0, "H": 130.0}
        d_wide = compute_fournier_delta(wide_bar, is_barrier=True)
        # 0.40 * 100 * 0.2 * 1 = 8.0 < (130 - 100) / 2 = 15.0
        self.assertAlmostEqual(d_wide, 8.0, places=6)

    def test_localized_digital_gamma_and_cutoff_second_deriv(self):
        """Test cutoff analytical second derivatives and distinct localized digital gamma."""
        from Methods.Malliavin.malliavin_european_fournier import (
            get_cutoff,
            malliavin_fournier_greeks_european_digital_call,
        )
        from Methods.Malliavin.malliavin_european_standard import (
            malliavin_greeks_european_digital_call,
        )
        from products import example_european_digital_ATM

        # 1. Verify second derivatives against numerical second differences
        K, delta, eps = 100.0, 8.0, 1e-5
        x = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        for ctype in ["gaussian", "quintic", "cauchy"]:
            psi, p1, p2 = get_cutoff(x, K, delta=delta, cutoff_type=ctype, return_second_deriv=True)
            psi_p, _ = get_cutoff(x + eps, K, delta=delta, cutoff_type=ctype)
            psi_m, _ = get_cutoff(x - eps, K, delta=delta, cutoff_type=ctype)
            num_p2 = (psi_p - 2.0 * psi + psi_m) / (eps ** 2)
            np.testing.assert_allclose(p2, num_p2, rtol=1e-3, atol=1e-4)

        # 2. Verify Fournié localized digital gamma differs from unlocalized standard Malliavin
        std_res = malliavin_greeks_european_digital_call(example_european_digital_ATM, N=50_000, seed=123)
        four_res = malliavin_fournier_greeks_european_digital_call(
            example_european_digital_ATM, N=50_000, seed=123, cutoff_type="gaussian", delta_strike=8.0
        )
        self.assertNotEqual(std_res["gamma"], four_res["gamma"])

    def test_budget_comparison_with_2500ms(self):
        """Test that evaluate_fixed_cpu_budget_comparison includes 2500ms and 5 budget tiers."""
        f_met = utils.compute_error_metrics(self.fdmc_df, self.benchmarks_df)
        m_met = utils.compute_error_metrics(self.malliavin_df, self.benchmarks_df)
        four_met = utils.compute_error_metrics(self.fournier_df, self.benchmarks_df)

        comp = utils.evaluate_fixed_cpu_budget_comparison(
            f_met, m_met, self.fdmc_timing, self.mal_timing,
            fournier_metrics_df=four_met, fournier_timing_df=self.four_timing,
        )
        budgets = sorted(comp["Budget (ms)"].unique())
        self.assertEqual(budgets, [250.0, 500.0, 1000.0, 2500.0, 5000.0])
        self.assertIn("Fournié Abs Error", comp.columns)
        self.assertIn("Malliavin Abs Error", comp.columns)

    def test_plot_runtime_vs_simulations(self):
        """Test that plot_runtime_vs_simulations creates a valid matplotlib Figure."""
        joint = utils.benchmark_joint_fdmc_vs_malliavin(
            n_sims=[10_000, 100_000], num_repeats=1, include_fournier=True
        )
        fig = utils.plot_runtime_vs_simulations(joint)
        self.assertIsNotNone(fig)
        self.assertEqual(len(fig.axes), len(joint["Product"].unique()))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_budget_simulations_vs_time(self):
        """Test that plot_budget_simulations_vs_time creates a valid matplotlib Figure for bars and lines."""
        f_met = utils.compute_error_metrics(self.fdmc_df, self.benchmarks_df)
        m_met = utils.compute_error_metrics(self.malliavin_df, self.benchmarks_df)
        four_met = utils.compute_error_metrics(self.fournier_df, self.benchmarks_df)

        comp = utils.evaluate_fixed_cpu_budget_comparison(
            f_met, m_met, self.fdmc_timing, self.mal_timing,
            budgets_ms=[250.0, 500.0],
            fournier_metrics_df=four_met, fournier_timing_df=self.four_timing,
        )
        import matplotlib.pyplot as plt
        fig_bars = utils.plot_budget_simulations_vs_time(comp, chart_type="bars")
        self.assertIsNotNone(fig_bars)
        plt.close(fig_bars)

        fig_lines = utils.plot_budget_simulations_vs_time(comp, chart_type="lines")
        self.assertIsNotNone(fig_lines)
        plt.close(fig_lines)


if __name__ == "__main__":
    unittest.main()


