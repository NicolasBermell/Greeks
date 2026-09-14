"""Profiling and efficiency (Pareto frontier & CPU budget) utilities for Greeks computations."""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Methods.MonteCarlo.FDMC_european import (
    _delta_european_barrier_up_and_out_call,
    _delta_european_call,
    _delta_european_digital_call,
    _gamma_european_barrier_up_and_out_call,
    _gamma_european_call,
    _gamma_european_digital_call,
    _vega_european_barrier_up_and_out_call,
    _vega_european_call,
    _vega_european_digital_call,
    FDMC_geeks_european_barrier_up_and_out_call,
    FDMC_geeks_european_call,
    FDMC_geeks_european_digital_call,
    generate_paths,
)
from Methods.Malliavin import (
    _delta_malliavin_european_barrier_up_and_out_call,
    _delta_malliavin_european_call,
    _delta_malliavin_european_digital_call,
    _gamma_malliavin_european_barrier_up_and_out_call,
    _gamma_malliavin_european_call,
    _gamma_malliavin_european_digital_call,
    _vega_malliavin_european_barrier_up_and_out_call,
    _vega_malliavin_european_call,
    _vega_malliavin_european_digital_call,
    malliavin_greeks_european_barrier_up_and_out_call,
    malliavin_greeks_european_call,
    malliavin_greeks_european_digital_call,
    malliavin_fournier_greeks_european_barrier_up_and_out_call,
    malliavin_fournier_greeks_european_call,
    malliavin_fournier_greeks_european_digital_call,
)
from products import (
    example_european_barrier_up_and_out_95,
    example_european_call_ATM,
    example_european_digital_ATM,
)
from config import FDMC_FIXED_EPS

GREEK_FN_MAP = {
    "Call European ATM": {
        "delta": _delta_european_call,
        "gamma": _gamma_european_call,
        "vega": _vega_european_call,
        "params": example_european_call_ATM,
    },
    "Digital European ATM": {
        "delta": _delta_european_digital_call,
        "gamma": _gamma_european_digital_call,
        "vega": _vega_european_digital_call,
        "params": example_european_digital_ATM,
    },
    "Barrier European Up and Out 95": {
        "delta": _delta_european_barrier_up_and_out_call,
        "gamma": _gamma_european_barrier_up_and_out_call,
        "vega": _vega_european_barrier_up_and_out_call,
        "params": example_european_barrier_up_and_out_95,
    },
}

PRODUCT_FN_MAP = {
    "Call European ATM": (FDMC_geeks_european_call, example_european_call_ATM),
    "Digital European ATM": (FDMC_geeks_european_digital_call, example_european_digital_ATM),
    "Barrier European Up and Out 95": (
        FDMC_geeks_european_barrier_up_and_out_call,
        example_european_barrier_up_and_out_95,
    ),
}

MALLIAVIN_PRODUCT_FN_MAP = {
    "Call European ATM": (malliavin_greeks_european_call, example_european_call_ATM),
    "Digital European ATM": (malliavin_greeks_european_digital_call, example_european_digital_ATM),
    "Barrier European Up and Out 95": (
        malliavin_greeks_european_barrier_up_and_out_call,
        example_european_barrier_up_and_out_95,
    ),
}

FOURNIER_PRODUCT_FN_MAP = {
    "Call European ATM": (malliavin_fournier_greeks_european_call, example_european_call_ATM),
    "Digital European ATM": (malliavin_fournier_greeks_european_digital_call, example_european_digital_ATM),
    "Barrier European Up and Out 95": (
        malliavin_fournier_greeks_european_barrier_up_and_out_call,
        example_european_barrier_up_and_out_95,
    ),
}


def benchmark_per_greek_execution_time(
    n_sims: list[int] = [10_000, 100_000, 1_000_000],
    num_repeats: int = 3,
    seed: int = 1984,
    pregenerate_paths: bool = True,
) -> pd.DataFrame:
    """Measures wall-clock execution time (ms) for each Greek individually across sample sizes N.

    Args:
        n_sims: List of simulation sample sizes N.
        num_repeats: Number of timing iterations to average.
        seed: Random seed for reproducibility.
        pregenerate_paths: If True, generates Brownian motion paths W_T once and passes them
            to Greek functions, measuring marginal sensitivity runtime without 7x redundant path generation.

    Returns:
        pd.DataFrame: Product, Greek, N, mean time (ms), time per 1k paths (ms).
    """
    records = []

    for prod_name, config in GREEK_FN_MAP.items():
        params = config["params"]
        for greek in ["delta", "gamma", "vega"]:
            fn = config[greek]
            for n in n_sims:
                times = []
                paths = generate_paths(params["T"], n, seed=seed) if pregenerate_paths else None
                for _ in range(num_repeats):
                    t0 = time.perf_counter()
                    if pregenerate_paths:
                        _ = fn(params, eps=1e-2, N=n, seed=seed, paths=paths)
                    else:
                        _ = fn(params, eps=1e-2, N=n, seed=seed)
                    t1 = time.perf_counter()
                    times.append((t1 - t0) * 1000.0)  # ms

                mean_t = float(np.mean(times))
                std_t = float(np.std(times))
                records.append({
                    "Product": prod_name,
                    "Greek": greek,
                    "N": n,
                    "time_ms_mean": mean_t,
                    "time_ms_std": std_t,
                    "ms_per_10k_paths": mean_t / (n / 10_000.0),
                })

    return pd.DataFrame(records)


def benchmark_fdmc_execution_time(
    n_sims: list[int] = [10_000, 100_000, 250_000, 1_000_000, 2_000_000],
    num_repeats: int = 3,
) -> pd.DataFrame:
    """Measures total wall-clock execution time (ms) across all Greeks combined."""
    records = []
    for prod_name, (fn, params) in PRODUCT_FN_MAP.items():
        for n in n_sims:
            times = []
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                _ = fn(params, eps=1e-2, N=n)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            mean_t = float(np.mean(times))
            records.append({
                "Product": prod_name,
                "N": n,
                "time_ms_mean": mean_t,
                "time_ms_std": float(np.std(times)),
                "ms_per_10k_paths": mean_t / (n / 10_000.0),
            })
    return pd.DataFrame(records)


def construct_pareto_dataset(
    metrics_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    fdmc_eps: float | dict[str, float] | str = FDMC_FIXED_EPS,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Merges error metrics with timing data to build the Pareto efficiency dataset.

    Args:
        metrics_df: DataFrame with error metrics.
        timing_df: Timing DataFrame (can be total or per-Greek).
        fdmc_eps: Fixed epsilon (e.g. 0.01) or 'optimal'.
        criterion: 'abs_error' or 'rel_error'.
        exclude_catastrophic: If True, excludes catastrophic step sizes.

    Returns:
        pd.DataFrame: Pareto dataset.
    """
    from Utilities.metrics import extract_fixed_or_optimal_eps

    opt_df = extract_fixed_or_optimal_eps(
        metrics_df, fdmc_eps=fdmc_eps, criterion=criterion, exclude_catastrophic=exclude_catastrophic
    )

    merge_cols = ["Product", "N"]
    if "Greek" in timing_df.columns:
        merge_cols.append("Greek")

    pareto_df = pd.merge(opt_df, timing_df, on=merge_cols, how="inner")
    err_col = "min_abs_error" if criterion == "abs_error" else "min_rel_error_pct"

    pareto_df["ln_time"] = np.log(pareto_df["time_ms_mean"].clip(lower=1e-3))
    pareto_df["ln_error"] = np.log(pareto_df[err_col].clip(lower=1e-8))
    return pareto_df


def plot_pareto_efficiency(
    pareto_df: pd.DataFrame,
    product: str,
    criterion: str = "abs_error",
    figsize: tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plots the Pareto efficiency curve: Time (ms) vs. Absolute (or Relative) Error.

    Args:
        pareto_df: Pareto dataset.
        product: Contract name.
        criterion: 'abs_error' or 'rel_error'.
        figsize: Figure dimensions.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    sub = pareto_df[pareto_df["Product"] == product]

    fig, ax = plt.subplots(figsize=figsize)
    greeks = sub["Greek"].unique()
    colors = {"delta": "#1f77b4", "gamma": "#e76f51", "vega": "#2a9d8f"}
    markers = {"delta": "o", "gamma": "^", "vega": "s"}
    err_col = "min_abs_error" if criterion == "abs_error" else "min_rel_error_pct"
    ylabel = "Absolute Error" if criterion == "abs_error" else "Relative Error (%)"

    for greek in greeks:
        greek_data = sub[sub["Greek"] == greek].sort_values("time_ms_mean")
        ax.plot(
            greek_data["time_ms_mean"],
            greek_data[err_col],
            marker=markers.get(greek, "o"),
            color=colors.get(greek, "gray"),
            label=f"{greek.capitalize()} (FDMC)",
            linewidth=2,
            markersize=8,
        )

        for _, row in greek_data.iterrows():
            ax.annotate(
                f"N={int(row['N']):_}",
                (row["time_ms_mean"], row[err_col]),
                textcoords="offset points",
                xytext=(6, 5),
                fontsize=8.5,
                alpha=0.85,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Joint CPU Execution Time (ms) [All Greeks]")
    ax.set_ylabel(f"Minimum {ylabel}")
    ax.set_title(f"{product}: Pareto Efficiency Frontier (Joint Time vs. {ylabel})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)

    plt.tight_layout()
    return fig


def evaluate_fixed_cpu_budget(
    metrics_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    budgets_ms: list[float] = [250.0, 500.0, 1000.0, 2500.0, 5000.0],
    fdmc_eps: float | dict[str, float] | str = FDMC_FIXED_EPS,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Evaluates how many simulations N and what accuracy is achievable under joint CPU budgets.

    Computes all Greeks jointly using shared Brownian motion paths and reused perturbed prices.

    Args:
        metrics_df: Error metrics DataFrame.
        timing_df: Timing DataFrame (joint or per-Greek).
        budgets_ms: List of target joint budgets in ms (default: [250, 500, 1000, 2500, 5000]).
        fdmc_eps: Fixed epsilon (e.g. 0.01) or 'optimal'.
        criterion: Error metric to evaluate ('abs_error' or 'rel_error_pct').
        exclude_catastrophic: If True, filters out catastrophic epsilon values.

    Returns:
        pd.DataFrame: Table comparing budget, achievable joint N, achieved error, and status.
    """
    from Utilities.metrics import extract_fixed_or_optimal_eps

    opt_df = extract_fixed_or_optimal_eps(
        metrics_df, fdmc_eps=fdmc_eps, criterion=criterion, exclude_catastrophic=exclude_catastrophic
    )
    err_col = "min_abs_error" if criterion == "abs_error" else "min_rel_error_pct"

    results = []
    products = opt_df["Product"].unique()
    greeks = ["delta", "gamma", "vega"]

    # Compute rate ms / simulation path across all Greeks
    ms_per_sim = timing_df.copy()
    if "ms_per_10k_paths" in ms_per_sim.columns:
        ms_per_sim["rate"] = ms_per_sim["ms_per_10k_paths"] / 10_000.0
    else:
        ms_per_sim["rate"] = ms_per_sim["time_ms_mean"] / ms_per_sim["N"]

    has_greek_col = "Greek" in ms_per_sim.columns
    if has_greek_col:
        timing_rate = ms_per_sim.groupby(["Product", "Greek"])["rate"].mean().to_dict()
    else:
        timing_rate = ms_per_sim.groupby("Product")["rate"].mean().to_dict()

    for budget in budgets_ms:
        for prod in products:
            for greek in greeks:
                if has_greek_col:
                    ms_per_path = timing_rate.get((prod, greek), 0.0001)
                else:
                    ms_per_path = timing_rate.get(prod, 0.0001)

                # Maximum paths achievable within the joint budget
                n_budget = int(budget / ms_per_path)

                sub = opt_df[(opt_df["Product"] == prod) & (opt_df["Greek"] == greek)].sort_values("N")
                all_n = sub["N"].values

                log_n = np.log10(all_n)
                log_err = np.log10(sub[err_col].clip(lower=1e-8).values)

                # Interpolate error at target N (bounded by max evaluated N or log-extrapolated)
                target_log_n = np.log10(max(n_budget, all_n[0]))
                if target_log_n > log_n[-1]:
                    interp_log_err = log_err[-1] - 0.5 * (target_log_n - log_n[-1])
                else:
                    interp_log_err = np.interp(target_log_n, log_n, log_err)
                achieved_err = 10.0 ** interp_log_err

                best_eps = sub.iloc[-1]["optimal_eps"]

                results.append({
                    "Product": prod,
                    "Greek": greek,
                    "Budget (ms)": budget,
                    "Joint Achievable N": f"{n_budget:_}",
                    "Achieved Abs Error": achieved_err,
                    "Optimal ε*": best_eps,
                    "FDMC ε": best_eps,
                })

    return pd.DataFrame(results)


def plot_budget_accuracy_bars(
    budget_df: pd.DataFrame,
    product: str,
    figsize: tuple[int, int] = (12, 5.5),
) -> plt.Figure:
    """Plots a grouped bar chart of achieved absolute error across joint CPU budgets (250ms - 5000ms).

    Args:
        budget_df: DataFrame from evaluate_fixed_cpu_budget.
        product: Contract name.
        figsize: Figure dimensions.

    Returns:
        plt.Figure: Matplotlib figure.
    """
    sub = budget_df[budget_df["Product"] == product].copy()

    fig, ax = plt.subplots(figsize=figsize)
    budgets = sorted(sub["Budget (ms)"].unique())
    greeks = ["delta", "gamma", "vega"]
    x = np.arange(len(greeks))
    n_budgets = len(budgets)
    width = 0.8 / n_budgets

    colors = ["#3a86ff", "#06d6a0", "#ffbe0b", "#e63946"]

    for idx, b in enumerate(budgets):
        b_data = sub[sub["Budget (ms)"] == b]
        errs = [b_data[b_data["Greek"] == g]["Achieved Abs Error"].values[0] for g in greeks]
        n_vals = [b_data[b_data["Greek"] == g]["Joint Achievable N"].values[0] for g in greeks]

        offset = (idx - (n_budgets - 1) / 2) * width
        rects = ax.bar(x + offset, errs, width, label=f"{int(b)} ms Budget", color=colors[idx % len(colors)])

        for r, err, n_str in zip(rects, errs, n_vals):
            val_str = f"{err:.2e}" if err < 0.001 else f"{err:.4f}"
            ax.text(
                r.get_x() + r.get_width() / 2.0,
                err * 1.08,
                f"N={n_str}\n{val_str}",
                ha="center",
                va="bottom",
                fontsize=7.5,
                fontweight="bold",
            )

    ax.set_yscale("log")
    ax.set_ylabel("Achieved Absolute Error")
    ax.set_title(f"{product}: Absolute Error Achievable Under Joint CPU Budgets (250 ms – 5000 ms)")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\Delta$", r"$\Gamma$", r"$\mathcal{V}$"])
    ax.legend(title="Joint CPU Budget", frameon=True)
    ax.grid(axis="y", which="both", alpha=0.3)

    plt.tight_layout()
    return fig


def benchmark_malliavin_execution_time(
    n_sims: list[int] = [10_000, 100_000, 250_000, 1_000_000, 2_000_000],
    num_repeats: int = 3,
) -> pd.DataFrame:
    """Measures total wall-clock execution time (ms) for Malliavin across all Greeks combined."""
    records = []
    for prod_name, (fn, params) in MALLIAVIN_PRODUCT_FN_MAP.items():
        for n in n_sims:
            times = []
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                _ = fn(params, N=n)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            mean_t = float(np.mean(times))
            records.append({
                "Product": prod_name,
                "N": n,
                "time_ms_mean": mean_t,
                "time_ms_std": float(np.std(times)),
                "ms_per_10k_paths": mean_t / (n / 10_000.0),
                "Method": "Malliavin",
            })
    return pd.DataFrame(records)


def benchmark_fournier_execution_time(
    n_sims: list[int] = [10_000, 100_000, 250_000, 1_000_000, 2_000_000],
    num_repeats: int = 3,
    cutoff_type: str = "gaussian",
    delta_strike: float = 10.0,
) -> pd.DataFrame:
    """Measures total wall-clock execution time (ms) for Fournié Localized Malliavin across all Greeks combined."""
    records = []
    for prod_name, (fn, params) in FOURNIER_PRODUCT_FN_MAP.items():
        for n in n_sims:
            times = []
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                _ = fn(params, N=n, cutoff_type=cutoff_type, delta_strike=delta_strike)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            mean_t = float(np.mean(times))
            records.append({
                "Product": prod_name,
                "N": n,
                "time_ms_mean": mean_t,
                "time_ms_std": float(np.std(times)),
                "ms_per_10k_paths": mean_t / (n / 10_000.0),
                "Method": "Fournier Malliavin",
            })
    return pd.DataFrame(records)


def benchmark_joint_fdmc_vs_malliavin(
    n_sims: list[int] = [10_000, 100_000, 250_000, 1_000_000, 2_000_000],
    num_repeats: int = 3,
    include_fournier: bool = True,
) -> pd.DataFrame:
    """Compares joint wall-clock execution times (ms) between FDMC, Malliavin, and Fournié Malliavin."""
    fdmc_timing = benchmark_fdmc_execution_time(n_sims=n_sims, num_repeats=num_repeats)
    mal_timing = benchmark_malliavin_execution_time(n_sims=n_sims, num_repeats=num_repeats)

    merged = pd.merge(
        fdmc_timing, mal_timing,
        on=["Product", "N"],
        suffixes=("_fdmc", "_mal")
    )
    merged["speedup_factor"] = merged["time_ms_mean_fdmc"] / merged["time_ms_mean_mal"]

    if include_fournier:
        four_timing = benchmark_fournier_execution_time(n_sims=n_sims, num_repeats=num_repeats)
        merged = pd.merge(
            merged, four_timing,
            on=["Product", "N"]
        ).rename(columns={
            "time_ms_mean": "time_ms_mean_fournier",
            "ms_per_10k_paths": "ms_per_10k_paths_fournier",
        })
        merged["speedup_factor_fournier"] = merged["time_ms_mean_fdmc"] / merged["time_ms_mean_fournier"]

        return merged[[
            "Product", "N",
            "time_ms_mean_fdmc", "ms_per_10k_paths_fdmc",
            "time_ms_mean_mal", "ms_per_10k_paths_mal",
            "time_ms_mean_fournier", "ms_per_10k_paths_fournier",
            "speedup_factor", "speedup_factor_fournier"
        ]]

    return merged[[
        "Product", "N",
        "time_ms_mean_fdmc", "ms_per_10k_paths_fdmc",
        "time_ms_mean_mal", "ms_per_10k_paths_mal",
        "speedup_factor"
    ]]


def evaluate_fixed_cpu_budget_comparison(
    fdmc_metrics_df: pd.DataFrame,
    malliavin_metrics_df: pd.DataFrame,
    fdmc_timing_df: pd.DataFrame,
    malliavin_timing_df: pd.DataFrame,
    budgets_ms: list[float] = [250.0, 500.0, 1000.0, 2500.0, 5000.0],
    fdmc_eps: float | dict[str, float] | str = FDMC_FIXED_EPS,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
    fournier_metrics_df: pd.DataFrame | None = None,
    fournier_timing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluates achievable simulations and error under fixed CPU budgets for FDMC vs Malliavin (and Fournié)."""
    b_fdmc = evaluate_fixed_cpu_budget(
        fdmc_metrics_df, fdmc_timing_df, budgets_ms=budgets_ms,
        fdmc_eps=fdmc_eps, criterion=criterion, exclude_catastrophic=exclude_catastrophic
    )

    # Rate ms per path for Malliavin
    t_mal = malliavin_timing_df.copy()
    if "ms_per_10k_paths" in t_mal.columns:
        t_mal["rate"] = t_mal["ms_per_10k_paths"] / 10_000.0
    else:
        t_mal["rate"] = t_mal["time_ms_mean"] / t_mal["N"]
    mal_rates = t_mal.groupby("Product")["rate"].mean().to_dict()

    # Rate ms per path for Fournié if provided
    four_rates = {}
    if fournier_timing_df is not None:
        t_four = fournier_timing_df.copy()
        if "ms_per_10k_paths" in t_four.columns:
            t_four["rate"] = t_four["ms_per_10k_paths"] / 10_000.0
        else:
            t_four["rate"] = t_four["time_ms_mean"] / t_four["N"]
        four_rates = t_four.groupby("Product")["rate"].mean().to_dict()

    records = []
    greeks = ["delta", "gamma", "vega"]
    products = malliavin_metrics_df["Product"].unique()

    for b in budgets_ms:
        for prod in products:
            # Malliavin achievable paths and log coordinates
            ms_per_sim_mal = mal_rates.get(prod, 0.001)
            raw_n_mal = int(b / ms_per_sim_mal)

            sub_mal = malliavin_metrics_df[malliavin_metrics_df["Product"] == prod].sort_values("N")
            all_n_mal = sub_mal["N"].values
            log_n_mal = np.log10(all_n_mal)
            target_log_n_mal = np.log10(max(raw_n_mal, all_n_mal[0]))

            # Fournié achievable paths and log coordinates if available
            raw_n_four = np.nan
            sub_four = None
            if fournier_metrics_df is not None and four_rates:
                ms_per_sim_four = four_rates.get(prod, 0.001)
                raw_n_four = int(b / ms_per_sim_four)
                sub_four = fournier_metrics_df[fournier_metrics_df["Product"] == prod].sort_values("N")
                all_n_four = sub_four["N"].values
                log_n_four = np.log10(all_n_four)
                target_log_n_four = np.log10(max(raw_n_four, all_n_four[0]))

            for greek in greeks:
                col_err = f"{greek}_{criterion}"

                # Continuous log-linear interpolation for Malliavin error
                log_err_mal = np.log10(sub_mal[col_err].clip(lower=1e-12).values)
                if target_log_n_mal > log_n_mal[-1]:
                    # Theoretical Monte Carlo O(1/sqrt(N)) decay extrapolation
                    interp_log_mal = log_err_mal[-1] - 0.5 * (target_log_n_mal - log_n_mal[-1])
                else:
                    interp_log_mal = np.interp(target_log_n_mal, log_n_mal, log_err_mal)
                mal_err = float(10.0 ** interp_log_mal)

                # Find matching FDMC row
                fdmc_match = b_fdmc[
                    (b_fdmc["Budget (ms)"] == b) &
                    (b_fdmc["Product"] == prod) &
                    (b_fdmc["Greek"] == greek)
                ]
                if not fdmc_match.empty:
                    f_row = fdmc_match.iloc[0]
                    fdmc_n = f_row["Joint Achievable N"]
                    fdmc_err = f_row["Achieved Abs Error"]
                    fdmc_eps_val = f_row.get("FDMC ε", f_row.get("Optimal ε*", np.nan))
                else:
                    fdmc_n = np.nan
                    fdmc_err = np.nan
                    fdmc_eps_val = np.nan

                gain = (fdmc_err / mal_err) if (mal_err > 0 and not np.isnan(fdmc_err)) else np.nan

                rec = {
                    "Budget (ms)": b,
                    "Product": prod,
                    "Greek": greek,
                    "FDMC Achievable N": fdmc_n,
                    "FDMC ε": fdmc_eps_val,
                    "FDMC Opt ε*": fdmc_eps_val,
                    "FDMC Abs Error": fdmc_err,
                    "Malliavin Achievable N": f"{raw_n_mal:_}",
                    "Malliavin Abs Error": mal_err,
                    "Accuracy Gain Ratio": gain,
                }

                if sub_four is not None and not sub_four.empty:
                    log_err_four = np.log10(sub_four[col_err].clip(lower=1e-12).values)
                    if target_log_n_four > log_n_four[-1]:
                        interp_log_four = log_err_four[-1] - 0.5 * (target_log_n_four - log_n_four[-1])
                    else:
                        interp_log_four = np.interp(target_log_n_four, log_n_four, log_err_four)
                    four_err = float(10.0 ** interp_log_four)

                    gain_four_fdmc = (fdmc_err / four_err) if (four_err > 0 and not np.isnan(fdmc_err)) else np.nan
                    gain_four_mal = (mal_err / four_err) if (four_err > 0 and not np.isnan(mal_err)) else np.nan
                    rec["Fournier Achievable N"] = f"{raw_n_four:_}"
                    rec["Fournier Abs Error"] = four_err
                    rec["Fournier Gain vs FDMC"] = gain_four_fdmc
                    rec["Fournier Gain vs Std"] = gain_four_mal

                records.append(rec)

    return pd.DataFrame(records)


def construct_comparative_pareto_dataset(
    fdmc_metrics_df: pd.DataFrame,
    malliavin_metrics_df: pd.DataFrame,
    fdmc_timing_df: pd.DataFrame,
    malliavin_timing_df: pd.DataFrame,
    fdmc_eps: float | dict[str, float] | str = FDMC_FIXED_EPS,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
    fournier_metrics_df: pd.DataFrame | None = None,
    fournier_timing_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merges error metrics and timing for FDMC, Standard Malliavin (and optional Fournié) into a Pareto dataset."""
    p_fdmc = construct_pareto_dataset(
        fdmc_metrics_df, fdmc_timing_df, fdmc_eps=fdmc_eps, criterion=criterion, exclude_catastrophic=exclude_catastrophic
    )
    p_fdmc["Method"] = "FDMC"

    mal_records = []
    greeks = ["delta", "gamma", "vega"]
    products = malliavin_metrics_df["Product"].unique()
    n_values = sorted(malliavin_metrics_df["N"].unique())

    t_mal = malliavin_timing_df.set_index(["Product", "N"])

    for prod in products:
        for n in n_values:
            row_mal = malliavin_metrics_df[
                (malliavin_metrics_df["Product"] == prod) &
                (malliavin_metrics_df["N"] == n)
            ]
            if row_mal.empty or (prod, n) not in t_mal.index:
                continue
            r_mal = row_mal.iloc[0]
            t_row = t_mal.loc[(prod, n)]
            t_mean = t_row["time_ms_mean"]

            for greek in greeks:
                val = r_mal[greek]
                bench = r_mal[f"{greek}_bench"]
                abs_err = r_mal[f"{greek}_abs_error"]
                rel_err = r_mal[f"{greek}_rel_error"]

                mal_records.append({
                    "Product": prod,
                    "N": n,
                    "Greek": greek,
                    "optimal_eps": np.nan,
                    "eps": np.nan,
                    "min_abs_error": abs_err,
                    "abs_error": abs_err,
                    "min_rel_error_pct": rel_err,
                    "rel_error_pct": rel_err,
                    "fdmc_value": val,
                    "sim_value": val,
                    "value": val,
                    "benchmark_value": bench,
                    "time_ms_mean": t_mean,
                    "time_ms_std": t_row.get("time_ms_std", 0.0),
                    "ms_per_10k_paths": t_row.get("ms_per_10k_paths", 0.0),
                    "ln_time": np.log(max(t_mean, 1e-3)),
                    "ln_error": np.log(max(abs_err if criterion == "abs_error" else rel_err, 1e-8)),
                    "Method": "Malliavin",
                })

    dfs = [p_fdmc, pd.DataFrame(mal_records)]

    # Optional Fournié integration
    if fournier_metrics_df is not None and fournier_timing_df is not None:
        four_records = []
        t_four = fournier_timing_df.set_index(["Product", "N"])
        four_n_values = sorted(fournier_metrics_df["N"].unique())

        for prod in products:
            for n in four_n_values:
                row_four = fournier_metrics_df[
                    (fournier_metrics_df["Product"] == prod) &
                    (fournier_metrics_df["N"] == n)
                ]
                if row_four.empty or (prod, n) not in t_four.index:
                    continue
                r_four = row_four.iloc[0]
                t_row_four = t_four.loc[(prod, n)]
                t_mean_four = t_row_four["time_ms_mean"]

                for greek in greeks:
                    val = r_four[greek]
                    bench = r_four[f"{greek}_bench"]
                    abs_err = r_four[f"{greek}_abs_error"]
                    rel_err = r_four[f"{greek}_rel_error"]

                    four_records.append({
                        "Product": prod,
                        "N": n,
                        "Greek": greek,
                        "optimal_eps": np.nan,
                        "eps": np.nan,
                        "min_abs_error": abs_err,
                        "abs_error": abs_err,
                        "min_rel_error_pct": rel_err,
                        "rel_error_pct": rel_err,
                        "fdmc_value": val,
                        "sim_value": val,
                        "value": val,
                        "benchmark_value": bench,
                        "time_ms_mean": t_mean_four,
                        "time_ms_std": t_row_four.get("time_ms_std", 0.0),
                        "ms_per_10k_paths": t_row_four.get("ms_per_10k_paths", 0.0),
                        "ln_time": np.log(max(t_mean_four, 1e-3)),
                        "ln_error": np.log(max(abs_err if criterion == "abs_error" else rel_err, 1e-8)),
                        "Method": "Fournié Malliavin",
                    })
        if four_records:
            dfs.append(pd.DataFrame(four_records))

    return pd.concat(dfs, ignore_index=True)


def plot_comparative_pareto_efficiency(
    pareto_df: pd.DataFrame,
    product: str,
    criterion: str = "abs_error",
    fdmc_label: str = "FDMC (ε=0.01)",
    figsize: tuple[int, int] = (12, 6),
) -> plt.Figure:
    """Plots comparative Pareto efficiency curves for FDMC vs Malliavin on the same axes."""
    sub = pareto_df[pareto_df["Product"] == product]

    fig, ax = plt.subplots(figsize=figsize)
    greeks = ["delta", "gamma", "vega"]
    colors = {"delta": "#1f77b4", "gamma": "#e76f51", "vega": "#2a9d8f"}
    err_col = "min_abs_error" if criterion == "abs_error" else "min_rel_error_pct"
    ylabel = "Absolute Error" if criterion == "abs_error" else "Relative Error (%)"

    for greek in greeks:
        greek_fdmc = sub[(sub["Greek"] == greek) & (sub["Method"] == "FDMC")].sort_values("time_ms_mean")
        greek_mal = sub[(sub["Greek"] == greek) & (sub["Method"] == "Malliavin")].sort_values("time_ms_mean")

        color = colors.get(greek, "gray")

        if not greek_fdmc.empty:
            ax.plot(
                greek_fdmc["time_ms_mean"],
                greek_fdmc[err_col],
                marker="o",
                linestyle="--",
                color=color,
                label=f"{greek.capitalize()} ({fdmc_label})",
                linewidth=1.8,
                alpha=0.7,
                markersize=6,
            )

        if not greek_mal.empty:
            ax.plot(
                greek_mal["time_ms_mean"],
                greek_mal[err_col],
                marker="s",
                linestyle="-",
                color=color,
                label=f"{greek.capitalize()} (Malliavin)",
                linewidth=2.5,
                markersize=8,
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Joint CPU Execution Time (ms) [Delta, Gamma, Vega]")
    ax.set_ylabel(f"Achieved {ylabel}")
    ax.set_title(f"{product}: Pareto Efficiency Frontier (FDMC vs. Malliavin)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True, fontsize=9, ncol=2)

    plt.tight_layout()
    return fig


def plot_comparative_budget_accuracy_bars(
    comp_budget_df: pd.DataFrame,
    product: str,
    fdmc_label: str = "FDMC (ε=0.01)",
    figsize: tuple[int, int] = (14, 6),
) -> plt.Figure:
    """Plots comparative bar charts showing FDMC vs Malliavin (and Fournié) absolute error under each CPU budget."""
    sub = comp_budget_df[comp_budget_df["Product"] == product]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    greeks = ["delta", "gamma", "vega"]
    greek_names = [r"$\Delta$ (Delta)", r"$\Gamma$ (Gamma)", r"$\mathcal{V}$ (Vega)"]

    budgets = sorted(sub["Budget (ms)"].unique())
    x = np.arange(len(budgets))
    has_four = "Fournier Abs Error" in sub.columns and not sub["Fournier Abs Error"].isna().all()
    width = 0.26 if has_four else 0.35

    for ax, greek, gname in zip(axes, greeks, greek_names):
        g_data = sub[sub["Greek"] == greek].sort_values("Budget (ms)")

        fdmc_errs = g_data["FDMC Abs Error"].tolist()
        mal_errs = g_data["Malliavin Abs Error"].tolist()

        if has_four:
            four_errs = g_data["Fournier Abs Error"].tolist()
            ax.bar(x - width, fdmc_errs, width, label=fdmc_label, color="#e07a5f")
            ax.bar(x, mal_errs, width, label="Std Malliavin", color="#2a9d8f")
            ax.bar(x + width, four_errs, width, label="Fournié Malliavin", color="#3d5a80")
        else:
            ax.bar(x - width / 2, fdmc_errs, width, label=fdmc_label, color="#e07a5f")
            ax.bar(x + width / 2, mal_errs, width, label="Malliavin", color="#2a9d8f")

        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(b)} ms" for b in budgets])
        ax.set_xlabel("Joint Latency Budget")
        ax.set_ylabel("Absolute Error")
        ax.set_title(f"{gname}")
        ax.grid(axis="y", which="both", alpha=0.3)
        if greek == "delta":
            ax.legend(frameon=True, fontsize=9)

    title_suffix = " (FDMC vs. Std Malliavin vs. Fournié Malliavin)" if has_four else " (FDMC vs. Malliavin)"
    fig.suptitle(f"{product}: Absolute Error Under Institutional Latency SLAs{title_suffix}", fontsize=13)
    plt.tight_layout()
    return fig

