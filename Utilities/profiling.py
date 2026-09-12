"""Profiling and efficiency (Pareto frontier & CPU budget) utilities for Greeks computations."""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Methods.MonteCarlo.FDMC import (
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
)
from products import (
    example_european_barrier_up_and_out_95,
    example_european_call_ATM,
    example_european_digital_ATM,
)

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


def benchmark_per_greek_execution_time(
    n_sims: list[int] = [10_000, 100_000, 1_000_000],
    num_repeats: int = 3,
    seed: int = 1984,
) -> pd.DataFrame:
    """Measures wall-clock execution time (ms) for each Greek individually across sample sizes N.

    Args:
        n_sims: List of simulation sample sizes N.
        num_repeats: Number of timing iterations to average.
        seed: Random seed for reproducibility.

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
                for _ in range(num_repeats):
                    t0 = time.perf_counter()
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
    n_sims: list[int] = [10_000, 100_000, 1_000_000],
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

            records.append({
                "Product": prod_name,
                "N": n,
                "time_ms_mean": float(np.mean(times)),
                "time_ms_std": float(np.std(times)),
            })
    return pd.DataFrame(records)


def construct_pareto_dataset(
    metrics_df: pd.DataFrame,
    timing_df: pd.DataFrame,
    criterion: str = "abs_error",
) -> pd.DataFrame:
    """Merges error metrics with timing data to build the Pareto efficiency dataset.

    Args:
        metrics_df: DataFrame with error metrics.
        timing_df: Timing DataFrame (can be total or per-Greek).
        criterion: 'abs_error' or 'rel_error'.

    Returns:
        pd.DataFrame: Pareto dataset.
    """
    from Utilities.metrics import extract_optimal_eps

    opt_df = extract_optimal_eps(metrics_df, criterion=criterion)

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
    ax.set_xlabel("CPU Execution Time (ms)")
    ax.set_ylabel(f"Minimum {ylabel}")
    ax.set_title(f"{product}: Pareto Efficiency Frontier (Time vs. {ylabel})")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=True)

    plt.tight_layout()
    return fig


def evaluate_fixed_cpu_budget(
    metrics_df: pd.DataFrame,
    per_greek_timing_df: pd.DataFrame,
    budgets_ms: list[float] = [100.0, 200.0],
    criterion: str = "abs_error",
) -> pd.DataFrame:
    """Evaluates how many simulations N and what accuracy is achievable under fixed CPU time budgets (per Greek).

    Args:
        metrics_df: Error metrics DataFrame.
        per_greek_timing_df: Per-Greek timing DataFrame from benchmark_per_greek_execution_time.
        budgets_ms: List of target budgets in ms (e.g. [100, 200]).
        criterion: 'abs_error' or 'rel_error'.

    Returns:
        pd.DataFrame: Table comparing budget, achievable N, achieved error, and status.
    """
    from Utilities.metrics import extract_optimal_eps

    opt_df = extract_optimal_eps(metrics_df, criterion=criterion)
    err_col = "min_abs_error" if criterion == "abs_error" else "min_rel_error_pct"

    results = []
    products = opt_df["Product"].unique()
    greeks = ["delta", "gamma", "vega"]

    # Calculate average time per simulation path (ms / path) for each (product, greek)
    timing_rate = per_greek_timing_df.groupby(["Product", "Greek"]).apply(
        lambda df: (df["time_ms_mean"] / df["N"]).mean()
    ).to_dict()

    for budget in budgets_ms:
        for prod in products:
            for greek in greeks:
                ms_per_path = timing_rate.get((prod, greek), 0.0001)
                # Max paths achievable within budget
                n_budget = int(budget / ms_per_path)

                # Round to nearest round number for practical execution
                # Find the closest evaluated N in our metrics dataset to interpolate/lookup
                sub = opt_df[(opt_df["Product"] == prod) & (opt_df["Greek"] == greek)].sort_values("N")
                all_n = sub["N"].values

                # Use log-linear interpolation of error vs N
                log_n = np.log10(all_n)
                log_err = np.log10(sub[err_col].clip(lower=1e-8).values)

                target_log_n = np.log10(max(n_budget, all_n[0]))
                interp_log_err = np.interp(target_log_n, log_n, log_err)
                achieved_err = 10.0 ** interp_log_err

                # Best eps at largest N
                best_eps = sub.iloc[-1]["optimal_eps"]

                results.append({
                    "Product": prod,
                    "Greek": greek,
                    "Budget (ms)": budget,
                    "Achievable N": f"{n_budget:_}",
                    "Achieved Abs Error": achieved_err,
                    "Optimal ε*": best_eps,
                })

    return pd.DataFrame(results)


def plot_budget_accuracy_bars(
    budget_df: pd.DataFrame,
    product: str,
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plots a grouped bar chart of achieved absolute error across 100 ms vs 200 ms CPU budgets.

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
    width = 0.35

    colors = ["#3a86ff", "#06d6a0"]

    for idx, b in enumerate(budgets):
        b_data = sub[sub["Budget (ms)"] == b]
        errs = [b_data[b_data["Greek"] == g]["Achieved Abs Error"].values[0] for g in greeks]
        n_vals = [b_data[b_data["Greek"] == g]["Achievable N"].values[0] for g in greeks]

        rects = ax.bar(x + (idx - 0.5) * width, errs, width, label=f"{int(b)} ms Budget", color=colors[idx])

        for r, err, n_str in zip(rects, errs, n_vals):
            ax.text(
                r.get_x() + r.get_width() / 2.0,
                err * 1.08,
                f"N={n_str}\n{err:.2e}" if err < 0.01 else f"N={n_str}\n{err:.4f}",
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
            )

    ax.set_yscale("log")
    ax.set_ylabel("Achieved Absolute Error")
    ax.set_title(f"{product}: Absolute Error Achievable Under 100 ms vs. 200 ms Budget (per Greek)")
    ax.set_xticks(x)
    ax.set_xticklabels([r"$\Delta$", r"$\Gamma$", r"$\mathcal{V}$"])
    ax.legend(frameon=True)
    ax.grid(axis="y", which="both", alpha=0.3)

    plt.tight_layout()
    return fig
