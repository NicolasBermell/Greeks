"""Metrics calculation and data preparation utilities for Greeks analysis."""

import numpy as np
import pandas as pd


def load_simulation_results(
    benchmarks_path: str = "./results/benchmarks.csv",
    fdmc_path: str = "./results/FDMC_results.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads benchmarks and FDMC simulation results from CSV files.

    Args:
        benchmarks_path: Path to the benchmarks CSV file.
        fdmc_path: Path to the FDMC results CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (benchmarks_df, fdmc_df)
    """
    benchmarks_df = pd.read_csv(benchmarks_path)
    fdmc_df = pd.read_csv(fdmc_path)

    # Standardize column types
    for df in [benchmarks_df, fdmc_df]:
        for col in ["delta", "gamma", "vega"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    if "N" in fdmc_df.columns:
        fdmc_df["N"] = fdmc_df["N"].astype(int)
    if "eps" in fdmc_df.columns:
        fdmc_df["eps"] = fdmc_df["eps"].astype(float)

    return benchmarks_df, fdmc_df


def compute_error_metrics(
    fdmc_df: pd.DataFrame,
    benchmarks_df: pd.DataFrame,
) -> pd.DataFrame:
    """Computes absolute and relative error metrics between FDMC and benchmarks.

    Args:
        fdmc_df: DataFrame of FDMC simulation results.
        benchmarks_df: DataFrame of benchmark analytical results.

    Returns:
        pd.DataFrame: Merged DataFrame enriched with:
            - <greek>_benchmark: Benchmark value
            - <greek>_abs_error: |FDMC - Benchmark|
            - <greek>_rel_error: (|FDMC - Benchmark| / |Benchmark|) * 100 (%)
            - <greek>_log10_rel_error: log10(relative error), clipped at 1e-12
    """
    bench_renamed = benchmarks_df.copy()
    bench_cols = {
        "delta": "delta_bench",
        "gamma": "gamma_bench",
        "vega": "vega_bench",
    }
    bench_renamed = bench_renamed.rename(columns=bench_cols)
    if "Method" in bench_renamed.columns:
        bench_renamed = bench_renamed.drop(columns=["Method"])

    merged = pd.merge(fdmc_df, bench_renamed, on="Product", how="left")

    for greek in ["delta", "gamma", "vega"]:
        fdmc_val = merged[greek]
        bench_val = merged[f"{greek}_bench"]

        abs_err = (fdmc_val - bench_val).abs()
        merged[f"{greek}_abs_error"] = abs_err

        # Relative error in %
        denom = bench_val.abs().replace(0.0, np.nan)
        rel_err = (abs_err / denom) * 100.0
        merged[f"{greek}_rel_error"] = rel_err

        # Log10 relative error for Pareto / sensitivity plots
        # Clip below 1e-6% to avoid log(0) issues
        clipped_rel_err = rel_err.fillna(1e6).clip(lower=1e-6)
        merged[f"{greek}_log10_rel_error"] = np.log10(clipped_rel_err)

    return merged


def is_catastrophic_eps(product: str, greek: str, eps: float, fdmc_val: float, bench_val: float) -> bool:
    """Detects whether an epsilon step size is in the catastrophic failure regime.

    Catastrophic regimes include:
    1. Estimator Collapse to Zero: |FDMC| < 1e-7 while analytical benchmark is non-zero
       (e.g., fine eps for discontinuous digital/barrier where zero simulated paths cross the strike).
    2. Gamma Roundoff Explosion / Cancellation: Central finite differences dividing by eps^2
       suffer catastrophic floating-point cancellation or variance explosion for eps < 0.01
       (and eps < 0.1 for discontinuous options).
    3. Discontinuous Boundary Collapse: Delta for digital/barrier options collapses when eps < 0.01.
    """
    if abs(bench_val) > 1e-5 and abs(fdmc_val) < 1e-7:
        return True
    if greek == "gamma" and eps < 0.01:
        return True
    if "Digital" in product or "Barrier" in product:
        if greek == "gamma" and eps < 0.1:
            return True
        if greek == "delta" and eps < 0.01:
            return True
    return False


def extract_optimal_eps(
    metrics_df: pd.DataFrame,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Finds the optimal step size eps* that minimizes the chosen error metric (default: abs_error).

    Args:
        metrics_df: DataFrame enriched with error metrics.
        criterion: 'abs_error' or 'rel_error'.
        exclude_catastrophic: If True, filters out catastrophic epsilon values (e.g. collapsed
            zero estimators or catastrophic roundoff zones for Gamma).

    Returns:
        pd.DataFrame: Summary table of optimal eps* and corresponding errors.
    """
    records = []
    products = metrics_df["Product"].unique()
    greeks = ["delta", "gamma", "vega"]
    n_values = sorted(metrics_df["N"].unique())

    for prod in products:
        prod_data = metrics_df[metrics_df["Product"] == prod]
        for n in n_values:
            n_data = prod_data[prod_data["N"] == n]
            if n_data.empty:
                continue

            for greek in greeks:
                col = f"{greek}_{criterion}"
                valid = n_data.dropna(subset=[col])
                if valid.empty:
                    continue

                if exclude_catastrophic:
                    non_cat = valid[~valid.apply(
                        lambda r: is_catastrophic_eps(prod, greek, r["eps"], r[greek], r[f"{greek}_bench"]),
                        axis=1,
                    )]
                    if not non_cat.empty:
                        valid = non_cat

                best_idx = valid[col].idxmin()
                best_row = valid.loc[best_idx]

                records.append({
                    "Product": prod,
                    "N": n,
                    "Greek": greek,
                    "optimal_eps": best_row["eps"],
                    "min_abs_error": best_row[f"{greek}_abs_error"],
                    "min_rel_error_pct": best_row[f"{greek}_rel_error"],
                    "fdmc_value": best_row[greek],
                    "benchmark_value": best_row[f"{greek}_bench"],
                })

    return pd.DataFrame(records)


def generate_error_pivot(
    metrics_df: pd.DataFrame,
    product: str,
    greek: str,
    metric: str = "abs_error",
) -> pd.DataFrame:
    """Generates an (N x eps) pivot table for a given product, greek, and metric.

    Args:
        metrics_df: DataFrame enriched with error metrics.
        product: Name of the product (e.g. 'Call European ATM').
        greek: Greek sensitivity ('delta', 'gamma', 'vega').
        metric: 'abs_error', 'rel_error', or 'value'.

    Returns:
        pd.DataFrame: Pivot table with N as index and eps as columns.
    """
    prod_data = metrics_df[metrics_df["Product"] == product]
    if metric == "value":
        col = greek
    else:
        col = f"{greek}_{metric}"

    pivot = prod_data.pivot(index="N", columns="eps", values=col)
    pivot = pivot.sort_index(ascending=True)
    # Sort columns in descending order of eps (from 0.1 down to 1e-8)
    sorted_cols = sorted(pivot.columns, reverse=True)
    return pivot[sorted_cols]


def compute_summary_table(
    metrics_df: pd.DataFrame,
    n_choice: int | None = None,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Generates an executive summary table comparing Benchmark vs best FDMC for each Product and Greek.

    Args:
        metrics_df: Enriched metrics DataFrame.
        n_choice: Specific N to evaluate, or None to pick the largest N.
        criterion: 'abs_error' or 'rel_error'.
        exclude_catastrophic: If True, excludes catastrophic step sizes.

    Returns:
        pd.DataFrame: Formatted summary comparison.
    """
    if n_choice is None:
        n_choice = int(metrics_df["N"].max())

    opt_df = extract_optimal_eps(metrics_df, criterion=criterion, exclude_catastrophic=exclude_catastrophic)
    opt_subset = opt_df[opt_df["N"] == n_choice]

    table = opt_subset[[
        "Product",
        "Greek",
        "benchmark_value",
        "fdmc_value",
        "optimal_eps",
        "min_abs_error",
        "min_rel_error_pct",
    ]].copy()

    table.columns = [
        "Product",
        "Greek",
        "Benchmark",
        f"FDMC (N={n_choice:_})",
        "Optimal ε*",
        "Abs Error",
        "Rel Error (%)",
    ]

    return table


def find_min_acceptable_N(
    metrics_df: pd.DataFrame,
    tolerances: dict[str, float] | None = None,
    criterion: str = "abs_error",
    timing_df: pd.DataFrame | None = None,
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Finds the minimum number of simulations N needed to achieve an acceptable error tolerance.

    Args:
        metrics_df: Enriched metrics DataFrame.
        tolerances: Dictionary of target tolerances per Greek (e.g. {'delta': 0.001, 'gamma': 0.0005, 'vega': 0.05}).
        criterion: 'abs_error' or 'rel_error'.
        timing_df: Optional timing DataFrame (joint or per-Greek) to report wall-clock time.
        exclude_catastrophic: If True, excludes catastrophic step sizes.

    Returns:
        pd.DataFrame: Table of minimum required N, optimal eps, achieved error, and joint wall-clock execution time.
    """
    if tolerances is None:
        if criterion == "abs_error":
            tolerances = {
                "delta": 0.001,    # within 10 bps of delta
                "gamma": 0.0005,   # within 0.0005 gamma
                "vega": 0.05,      # within 0.05 vega points
            }
        else:
            tolerances = {
                "delta": 0.5,      # 0.5% relative error
                "gamma": 1.0,      # 1.0% relative error
                "vega": 0.5,       # 0.5% relative error
            }

    opt_df = extract_optimal_eps(metrics_df, criterion=criterion, exclude_catastrophic=exclude_catastrophic)
    err_col = f"min_{criterion}" if criterion == "abs_error" else "min_rel_error_pct"

    # Compute rate ms / path if timing_df is provided
    ms_rates = {}
    has_greek_col = False
    if timing_df is not None:
        ms_per_sim = timing_df.copy()
        if "ms_per_10k_paths" in ms_per_sim.columns:
            ms_per_sim["rate"] = ms_per_sim["ms_per_10k_paths"] / 10_000.0
        else:
            ms_per_sim["rate"] = ms_per_sim["time_ms_mean"] / ms_per_sim["N"]

        has_greek_col = "Greek" in ms_per_sim.columns
        if has_greek_col:
            ms_rates = ms_per_sim.groupby(["Product", "Greek"])["rate"].mean().to_dict()
        else:
            ms_rates = ms_per_sim.groupby("Product")["rate"].mean().to_dict()

    results = []
    products = opt_df["Product"].unique()
    greeks = ["delta", "gamma", "vega"]

    for prod in products:
        for greek in greeks:
            if has_greek_col:
                ms_per_path = ms_rates.get((prod, greek), None)
            else:
                ms_per_path = ms_rates.get(prod, None)

            target_tol = tolerances.get(greek, 0.01)
            sub = opt_df[(opt_df["Product"] == prod) & (opt_df["Greek"] == greek)].sort_values("N")

            passing = sub[sub[err_col] <= target_tol]
            if not passing.empty:
                first_pass = passing.iloc[0]
                min_n = int(first_pass["N"])
                t_est = (ms_per_path * min_n) if ms_per_path is not None else np.nan

                results.append({
                    "Product": prod,
                    "Greek": greek,
                    "Target Tolerance": target_tol,
                    "Min Required N": min_n,
                    "Optimal ε*": first_pass["optimal_eps"],
                    "Achieved Abs Error": first_pass["min_abs_error"],
                    "Achieved Rel Error (%)": first_pass["min_rel_error_pct"],
                    "Joint Time (ms)": t_est,
                    "Status": "Passed",
                })
            else:
                best_ever = sub.loc[sub[err_col].idxmin()]
                min_n = int(sub["N"].max())
                t_est = (ms_per_path * min_n) if ms_per_path is not None else np.nan

                results.append({
                    "Product": prod,
                    "Greek": greek,
                    "Target Tolerance": target_tol,
                    "Min Required N": min_n,
                    "Optimal ε*": best_ever["optimal_eps"],
                    "Achieved Abs Error": best_ever["min_abs_error"],
                    "Achieved Rel Error (%)": best_ever["min_rel_error_pct"],
                    "Joint Time (ms)": t_est,
                    "Status": f"Exceeds Tol (best={best_ever[err_col]:.4f})",
                })

    res_df = pd.DataFrame(results)
    if timing_df is None:
        res_df = res_df.drop(columns=["Joint Time (ms)"])
    return res_df

