"""Wall-clock timing utilities, plus the single-run FDMC budget and Pareto tools used by analysis_FDMC.ipynb."""

import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Methods.MonteCarlo.FDMC_european import (
    FDMC_greeks_european_call_terminal_knockout,
    FDMC_greeks_european_call,
    FDMC_greeks_european_digital_call,
)
from Methods.Malliavin import (
    malliavin_greeks_european_call_terminal_knockout,
    malliavin_greeks_european_call,
    malliavin_greeks_european_digital_call,
    malliavin_fournier_greeks_european_call_terminal_knockout,
    malliavin_fournier_greeks_european_call,
    malliavin_fournier_greeks_european_digital_call,
)
from products import (
    example_european_call_terminal_knockout,
    example_european_call_ATM,
    example_european_digital_ATM,
)
from config import FDMC_FIXED_EPS


PRODUCT_FN_MAP = {
    "Call European ATM": (FDMC_greeks_european_call, example_european_call_ATM),
    "Digital European ATM": (FDMC_greeks_european_digital_call, example_european_digital_ATM),
    "Call with terminal knock-out": (
        FDMC_greeks_european_call_terminal_knockout,
        example_european_call_terminal_knockout,
    ),
}

MALLIAVIN_PRODUCT_FN_MAP = {
    "Call European ATM": (malliavin_greeks_european_call, example_european_call_ATM),
    "Digital European ATM": (malliavin_greeks_european_digital_call, example_european_digital_ATM),
    "Call with terminal knock-out": (
        malliavin_greeks_european_call_terminal_knockout,
        example_european_call_terminal_knockout,
    ),
}

FOURNIER_PRODUCT_FN_MAP = {
    "Call European ATM": (malliavin_fournier_greeks_european_call, example_european_call_ATM),
    "Digital European ATM": (malliavin_fournier_greeks_european_digital_call, example_european_digital_ATM),
    "Call with terminal knock-out": (
        malliavin_fournier_greeks_european_call_terminal_knockout,
        example_european_call_terminal_knockout,
    ),
}


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
    bandwidth: float | dict[str, float] | None = None,
) -> pd.DataFrame:
    """Measures total wall-clock execution time (ms) for Fournié Localized Malliavin across all Greeks combined."""
    records = []
    for prod_name, (fn, params) in FOURNIER_PRODUCT_FN_MAP.items():
        for n in n_sims:
            times = []
            for _ in range(num_repeats):
                t0 = time.perf_counter()
                _ = fn(params, N=n, cutoff_type=cutoff_type, bandwidth=bandwidth)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000.0)

            mean_t = float(np.mean(times))
            records.append({
                "Product": prod_name,
                "N": n,
                "time_ms_mean": mean_t,
                "time_ms_std": float(np.std(times)),
                "ms_per_10k_paths": mean_t / (n / 10_000.0),
                "Method": "Fournié Malliavin",
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


def plot_runtime_vs_simulations(
    joint_timing_df: pd.DataFrame,
    figsize: tuple[int, int] = (16, 5),
) -> plt.Figure:
    """Plots joint wall-clock execution time (ms) on the y-axis vs. number of simulations on the x-axis.

    This visualizes computational scaling and throughput: for any target time on the y-axis,
    it shows the number of simulations completed in that time across FDMC, Std Malliavin,
    and Fournié Localized Malliavin.

    Args:
        joint_timing_df: DataFrame from benchmark_joint_fdmc_vs_malliavin.
        figsize: Figure dimensions (width, height).

    Returns:
        plt.Figure: Matplotlib figure object.
    """
    products = joint_timing_df["Product"].unique()
    fig, axes = plt.subplots(1, len(products), figsize=figsize, sharey=True)
    if len(products) == 1:
        axes = [axes]

    has_fournier = "time_ms_mean_fournier" in joint_timing_df.columns

    for ax, prod in zip(axes, products):
        sub = joint_timing_df[joint_timing_df["Product"] == prod].sort_values("N")
        n_vals = sub["N"].values

        # FDMC
        ax.plot(
            n_vals,
            sub["time_ms_mean_fdmc"],
            marker="o",
            linestyle="--",
            color="#e07a5f",
            linewidth=2.2,
            markersize=7,
            label="FDMC (ε=0.01)",
        )

        # Std Malliavin
        ax.plot(
            n_vals,
            sub["time_ms_mean_mal"],
            marker="s",
            linestyle="-",
            color="#2a9d8f",
            linewidth=2.5,
            markersize=7,
            label="Std Malliavin",
        )

        # Fournié Malliavin
        if has_fournier:
            ax.plot(
                n_vals,
                sub["time_ms_mean_fournier"],
                marker="D",
                linestyle="-.",
                color="#3d5a80",
                linewidth=2.2,
                markersize=7,
                label="Fournié Malliavin",
            )

        # Annotate mean speedup
        mean_speedup = sub["speedup_factor"].mean()
        speedup_text = f"Std Malliavin Speedup: {mean_speedup:.1f}×"
        if has_fournier and "speedup_factor_fournier" in sub.columns:
            mean_four_speedup = sub["speedup_factor_fournier"].mean()
            speedup_text += f"\nFournié Speedup: {mean_four_speedup:.1f}×"

        ax.text(
            0.05,
            0.92,
            speedup_text,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight="bold",
            va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f8f9fa", edgecolor="#ced4da", alpha=0.9),
        )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of Simulations in that Time ($N$)", fontsize=10.5)
        ax.set_title(f"{prod}", fontsize=12, fontweight="bold")
        ax.grid(True, which="both", alpha=0.3)

        if ax == axes[0]:
            ax.set_ylabel("Joint Wall-Clock Time (ms)", fontsize=11)
            ax.legend(frameon=True, fontsize=9.5, loc="lower right")

    fig.suptitle(
        "Computational Scalability: Joint Execution Time (ms) vs. Number of Simulations in that Time",
        fontsize=13.5,
        y=1.02,
    )
    plt.tight_layout()
    return fig
