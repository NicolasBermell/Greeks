"""Metrics, convergence, error analysis, and comparative benchmark utilities for Greek sensitivities."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from config import FDMC_FIXED_EPS


# ==============================================================================
# 1. File Loading & Simulation Results I/O
# ==============================================================================

def _coerce_results_types(df: pd.DataFrame) -> pd.DataFrame:
    """Standardizes column types and safely parses numeric sensitivities and parameters."""
    out = df.copy()
    greeks_and_params = [
        "delta", "gamma", "vega", "theta", "rho",
        "S0", "K", "T", "r", "q", "sigma", "H", "eps", "bandwidth",
        "delta_var", "delta_se", "gamma_var", "gamma_se", "vega_var", "vega_se",
    ]
    for col in greeks_and_params:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    if "N" in out.columns:
        out["N"] = pd.to_numeric(out["N"], errors="coerce").astype("int64")

    return out


def load_simulation_results(
    benchmarks_path: str | Path = "./results/benchmarks.csv",
    fdmc_path: str | Path = "./results/FDMC_results.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads benchmarks and FDMC simulation results from CSV files.

    Independent of Malliavin result files so it succeeds even when only FDMC
    has been computed.

    Args:
        benchmarks_path: Path to the benchmarks CSV file.
        fdmc_path: Path to the FDMC results CSV file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (benchmarks_df, fdmc_df)
    """
    benchmarks_path = Path(benchmarks_path)
    fdmc_path = Path(fdmc_path)

    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Benchmarks CSV file not found at: {benchmarks_path}")
    if not fdmc_path.exists():
        raise FileNotFoundError(f"FDMC results CSV file not found at: {fdmc_path}")

    benchmarks_df = _coerce_results_types(pd.read_csv(benchmarks_path))
    fdmc_df = _coerce_results_types(pd.read_csv(fdmc_path))

    return benchmarks_df, fdmc_df


def load_all_simulation_results(
    benchmarks_path: str | Path = "./results/benchmarks.csv",
    fdmc_path: str | Path = "./results/FDMC_results.csv",
    malliavin_path: str | Path = "./results/malliavin_results.csv",
    malliavin_fournier_path: str | Path = "./results/malliavin_fournier_results.csv",
    optional_fournier: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Loads benchmarks, FDMC, Standard Malliavin, and Fournié Malliavin simulation results from CSV files.

    Args:
        benchmarks_path: Path to the benchmarks CSV file.
        fdmc_path: Path to the FDMC results CSV file.
        malliavin_path: Path to the Standard Malliavin results CSV file.
        malliavin_fournier_path: Path to the Fournié Localized Malliavin results CSV file.
        optional_fournier: If True and the Fournié file is missing, returns None for it instead of raising.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]: 
            (benchmarks_df, fdmc_df, malliavin_df, malliavin_fournier_df)
    """
    benchmarks_df, fdmc_df = load_simulation_results(benchmarks_path, fdmc_path)

    m_path = Path(malliavin_path)
    if not m_path.exists():
        raise FileNotFoundError(f"Malliavin results CSV file not found at: {m_path}")
    malliavin_df = _coerce_results_types(pd.read_csv(m_path))

    four_path = Path(malliavin_fournier_path)
    if not four_path.exists():
        if optional_fournier:
            malliavin_fournier_df = None
        else:
            raise FileNotFoundError(f"Fournié Malliavin results CSV file not found at: {four_path}")
    else:
        malliavin_fournier_df = _coerce_results_types(pd.read_csv(four_path))

    return benchmarks_df, fdmc_df, malliavin_df, malliavin_fournier_df


# ==============================================================================
# 2. Metric Computation & Error Enrichment
# ==============================================================================

def compute_error_metrics(
    fdmc_df: pd.DataFrame,
    benchmarks_df: pd.DataFrame,
    sim_df: pd.DataFrame | None = None,
    greeks: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Computes absolute error, relative error, signed bias, and log errors vs. analytical benchmarks.

    Enriches simulation datasets with precision financial metrics while preserving
    strict backward compatibility with existing schemas.

    Args:
        fdmc_df: Primary simulation DataFrame (can be FDMC, Malliavin, Fournié, etc.).
            Named `fdmc_df` for backward compatibility with existing call sites.
        benchmarks_df: DataFrame of benchmark analytical or reference results.
        sim_df: Optional alias for `fdmc_df` to provide more general semantics.
        greeks: Optional list of Greek columns to evaluate. If None, dynamically detects
            common Greek columns present in both datasets.

    Returns:
        pd.DataFrame: Merged DataFrame enriched with:
            - <greek>_bench: Benchmark reference value
            - <greek>_abs_error: |Estimate - Benchmark|
            - <greek>_rel_error: (|Estimate - Benchmark| / |Benchmark|) * 100 (%)
            - <greek>_bias: Estimate - Benchmark (signed error)
            - <greek>_rel_bias: ((Estimate - Benchmark) / |Benchmark|) * 100 (%)
            - <greek>_log10_abs_error: log10(|Estimate - Benchmark|), clipped at 1e-12
            - <greek>_log10_rel_error: log10(rel_error), clipped at 1e-6 (preserves NaNs)
    """
    input_df = sim_df if sim_df is not None else fdmc_df

    # Prepare benchmark data
    bench = benchmarks_df.copy()
    if "Method" in bench.columns:
        bench = bench.drop(columns=["Method"])

    # Determine candidate Greeks
    candidate_greeks = ["delta", "gamma", "vega", "theta", "rho"]
    if greeks is None:
        target_greeks = [g for g in candidate_greeks if g in input_df.columns and g in bench.columns]
    else:
        target_greeks = [g for g in greeks if g in input_df.columns and g in bench.columns]

    # Rename benchmark columns to avoid collisions
    rename_dict = {g: f"{g}_bench" for g in target_greeks}
    bench_sub = bench[["Product"] + target_greeks].rename(columns=rename_dict)

    # Left merge on Product
    merged = pd.merge(input_df, bench_sub, on="Product", how="left")

    # Compute enriched metrics for each Greek
    for g in target_greeks:
        val = merged[g]
        bench_val = merged[f"{g}_bench"]

        # 1. Signed bias and absolute error
        signed_err = val - bench_val
        abs_err = signed_err.abs()
        merged[f"{g}_bias"] = signed_err
        merged[f"{g}_abs_error"] = abs_err

        # 2. Relative error in %
        denom = bench_val.abs().replace(0.0, np.nan)
        rel_err = (abs_err / denom) * 100.0
        rel_bias = (signed_err / denom) * 100.0
        merged[f"{g}_rel_error"] = rel_err
        merged[f"{g}_rel_bias"] = rel_bias

        # 3. Log10 metrics
        # Clip absolute error
        valid_abs = abs_err.where(abs_err.notna(), np.nan)
        clipped_abs = valid_abs.clip(lower=1e-12)
        merged[f"{g}_log10_abs_error"] = np.log10(clipped_abs)

        # Clip relative error (preserve NaN without fabricating fake errors)
        valid_rel = rel_err.where(rel_err.notna(), np.nan)
        clipped_rel = valid_rel.clip(lower=1e-6)
        merged[f"{g}_log10_rel_error"] = np.log10(clipped_rel)

    return merged


# ==============================================================================
# 3. Catastrophic Step Size Detection
# ==============================================================================

def is_catastrophic_eps(
    product: str,
    greek: str,
    eps: float,
    fdmc_val: float,
    bench_val: float,
    tol: float = 1e-7,
) -> bool:
    """Detects whether a finite-difference step size eps is in a known catastrophic numerical regime.

    Catastrophic regimes include:
    1. Estimator Collapse to Zero: |FDMC| < 1e-7 while analytical benchmark is non-zero
       (e.g., fine eps for a discontinuous digital or knock-out where zero simulated paths cross the strike).
    2. Gamma Roundoff Explosion / Floating-Point Cancellation: Central finite differences
       dividing by eps^2 suffer catastrophic cancellation or variance explosion for eps < 0.01
       (and eps < 0.1 for discontinuous options).
    3. Discontinuous Boundary Collapse: Delta for digital/knock-out options collapses when eps < 0.01.

    Args:
        product: Option product name.
        greek: Sensitivity name ('delta', 'gamma', 'vega').
        eps: Step size epsilon.
        fdmc_val: Computed Monte Carlo finite-difference Greek value.
        bench_val: Analytical benchmark Greek value.
        tol: Floating-point precision margin to prevent binary representation edge cases.

    Returns:
        bool: True if the step size is in the catastrophic failure regime.
    """
    g = str(greek).lower()
    p = str(product).lower()

    # 1. Total estimator collapse
    if abs(bench_val) > 1e-5 and abs(fdmc_val) < 1e-7:
        return True

    # 2. General Gamma roundoff explosion for eps < 0.01
    if g == "gamma" and eps < (0.01 - tol):
        return True

    # 3. Discontinuous contracts (Digital & terminal knock-out)
    if "digital" in p or "barrier" in p or "knock-out" in p:
        if g == "gamma" and eps < (0.1 - tol):
            return True
        if g == "delta" and eps < (0.01 - tol):
            return True

    return False


def is_catastrophic_eps_series(
    products: pd.Series,
    greeks: pd.Series | str,
    eps_series: pd.Series,
    est_series: pd.Series,
    bench_series: pd.Series,
    tol: float = 1e-7,
) -> pd.Series:
    """Vectorized catastrophic failure regime detector for Pandas Series.

    Args:
        products: Series of product names.
        greeks: Series of Greek names or single string Greek.
        eps_series: Series of epsilon step sizes.
        est_series: Series of computed estimates.
        bench_series: Series of benchmark values.
        tol: Floating-point precision margin.

    Returns:
        pd.Series of bool: True for rows in catastrophic regime.
    """
    p_lower = products.astype(str).str.lower()
    is_discontinuous = p_lower.str.contains("digital|barrier|knock-out", na=False)

    if isinstance(greeks, str):
        g_lower = greeks.lower()
        is_gamma = (g_lower == "gamma")
        is_delta = (g_lower == "delta")
    else:
        g_series_lower = greeks.astype(str).str.lower()
        is_gamma = (g_series_lower == "gamma")
        is_delta = (g_series_lower == "delta")

    # Condition 1: Estimator collapse to zero
    cond1 = (bench_series.abs() > 1e-5) & (est_series.abs() < 1e-7)

    # Condition 2: General gamma cancellation
    cond2 = is_gamma & (eps_series < (0.01 - tol))

    # Condition 3: Discontinuous Gamma (< 0.1)
    cond3 = is_discontinuous & is_gamma & (eps_series < (0.1 - tol))

    # Condition 4: Discontinuous Delta (< 0.01)
    cond4 = is_discontinuous & is_delta & (eps_series < (0.01 - tol))

    return cond1 | cond2 | cond3 | cond4


# ==============================================================================
# 4. Step Size Selection: Fixed vs. Ex-Post Optimal
# ==============================================================================

def extract_fixed_or_optimal_eps(
    metrics_df: pd.DataFrame,
    fdmc_eps: float | Mapping[str, float] | Mapping[tuple[str, str], float] | str = FDMC_FIXED_EPS,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Extracts simulation results for a designated step size policy (fixed or ex-post optimal eps*).

    Provides seamless backward compatibility with legacy downstream extractors, returning
    all historical columns (`optimal_eps`, `min_abs_error`, `min_rel_error_pct`, `fdmc_value`,
    `benchmark_value`) along with standardized modern aliases (`eps`, `abs_error`, `rel_error`, `sim_value`).

    Args:
        metrics_df: DataFrame enriched with error metrics.
        fdmc_eps: Step size evaluation policy:
            - float (e.g. 0.01): Fix epsilon globally across all products and Greeks.
            - dict: Per-product or per-Greek mapping (e.g. {'Call European ATM': 0.01} or {'gamma': 0.01}).
            - 'optimal': Ex-post optimal eps* minimizing the selected error metric.
        criterion: 'abs_error', 'rel_error', 'bias', or 'log10_abs_error'.
        exclude_catastrophic: If True (and fdmc_eps == 'optimal'), filters catastrophic step sizes.

    Returns:
        pd.DataFrame: Formatted selection table maintaining identical schema across all policies.
    """
    is_optimal = isinstance(fdmc_eps, str) and fdmc_eps.lower() == "optimal"
    if is_optimal:
        return extract_optimal_eps(metrics_df, criterion=criterion, exclude_catastrophic=exclude_catastrophic)

    records = []
    products = metrics_df["Product"].unique()
    n_values = sorted(metrics_df["N"].unique())
    candidate_greeks = ["delta", "gamma", "vega", "theta", "rho"]
    available_greeks = [g for g in candidate_greeks if g in metrics_df.columns]

    has_eps_col = "eps" in metrics_df.columns

    for prod in products:
        prod_data = metrics_df[metrics_df["Product"] == prod]
        for n in n_values:
            n_data = prod_data[prod_data["N"] == n]
            if n_data.empty:
                continue

            for greek in available_greeks:
                err_col = f"{greek}_{criterion}"
                if err_col not in n_data.columns:
                    err_col = f"{greek}_abs_error"
                if err_col not in n_data.columns:
                    continue

                valid = n_data.dropna(subset=[greek, f"{greek}_bench"])
                if valid.empty:
                    continue

                if not has_eps_col:
                    chosen_row = valid.iloc[0]
                    chosen_eps = np.nan
                else:
                    # Resolve target epsilon
                    if isinstance(fdmc_eps, Mapping):
                        if (prod, greek) in fdmc_eps:
                            target_eps = fdmc_eps[(prod, greek)]
                        elif prod in fdmc_eps:
                            target_eps = fdmc_eps[prod]
                        elif greek in fdmc_eps:
                            target_eps = fdmc_eps[greek]
                        else:
                            target_eps = FDMC_FIXED_EPS
                    else:
                        target_eps = float(fdmc_eps)

                    # Match target eps with floating tolerance
                    eps_matches = valid[np.isclose(valid["eps"], target_eps, atol=1e-6)]
                    if not eps_matches.empty:
                        chosen_row = eps_matches.iloc[0]
                    else:
                        closest_idx = (valid["eps"] - target_eps).abs().idxmin()
                        chosen_row = valid.loc[closest_idx]
                    chosen_eps = chosen_row["eps"]

                abs_err = chosen_row.get(f"{greek}_abs_error", np.nan)
                rel_err = chosen_row.get(f"{greek}_rel_error", np.nan)
                val = chosen_row[greek]
                bench = chosen_row[f"{greek}_bench"]

                records.append({
                    "Product": prod,
                    "N": n,
                    "Greek": greek,
                    "optimal_eps": chosen_eps,
                    "eps": chosen_eps,
                    "min_abs_error": abs_err,
                    "abs_error": abs_err,
                    "min_rel_error_pct": rel_err,
                    "rel_error_pct": rel_err,
                    "fdmc_value": val,
                    "sim_value": val,
                    "benchmark_value": bench,
                })

    cols = [
        "Product", "N", "Greek", "optimal_eps", "eps",
        "min_abs_error", "abs_error", "min_rel_error_pct", "rel_error_pct",
        "fdmc_value", "sim_value", "benchmark_value",
    ]
    return pd.DataFrame(records, columns=cols if not records else None)


def extract_optimal_eps(
    metrics_df: pd.DataFrame,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Finds the optimal step size eps* minimizing the chosen error metric (default: abs_error).

    Args:
        metrics_df: DataFrame enriched with error metrics.
        criterion: Error metric to minimize ('abs_error' or 'rel_error').
        exclude_catastrophic: If True, filters out catastrophic epsilon step sizes.

    Returns:
        pd.DataFrame: Summary table of optimal eps* and corresponding errors.
    """
    records = []
    products = metrics_df["Product"].unique()
    n_values = sorted(metrics_df["N"].unique())
    candidate_greeks = ["delta", "gamma", "vega", "theta", "rho"]
    available_greeks = [g for g in candidate_greeks if g in metrics_df.columns]
    has_eps_col = "eps" in metrics_df.columns

    for prod in products:
        prod_data = metrics_df[metrics_df["Product"] == prod]
        for n in n_values:
            n_data = prod_data[prod_data["N"] == n]
            if n_data.empty:
                continue

            for greek in available_greeks:
                err_col = f"{greek}_{criterion}"
                if err_col not in n_data.columns:
                    err_col = f"{greek}_abs_error"
                if err_col not in n_data.columns:
                    continue

                valid = n_data.dropna(subset=[err_col])
                if valid.empty:
                    continue

                if not has_eps_col:
                    best_row = valid.iloc[0]
                    chosen_eps = np.nan
                else:
                    if exclude_catastrophic:
                        mask_cat = is_catastrophic_eps_series(
                            valid["Product"], greek, valid["eps"], valid[greek], valid[f"{greek}_bench"]
                        )
                        non_cat = valid[~mask_cat]
                        if not non_cat.empty:
                            valid = non_cat

                    best_idx = valid[err_col].idxmin()
                    best_row = valid.loc[best_idx]
                    chosen_eps = best_row["eps"]

                abs_err = best_row.get(f"{greek}_abs_error", np.nan)
                rel_err = best_row.get(f"{greek}_rel_error", np.nan)
                val = best_row[greek]
                bench = best_row[f"{greek}_bench"]

                records.append({
                    "Product": prod,
                    "N": n,
                    "Greek": greek,
                    "optimal_eps": chosen_eps,
                    "eps": chosen_eps,
                    "min_abs_error": abs_err,
                    "abs_error": abs_err,
                    "min_rel_error_pct": rel_err,
                    "rel_error_pct": rel_err,
                    "fdmc_value": val,
                    "sim_value": val,
                    "benchmark_value": bench,
                })

    return pd.DataFrame(records)


# ==============================================================================
# 5. Pivoting & Tabular Analysis
# ==============================================================================

def generate_error_pivot(
    metrics_df: pd.DataFrame,
    product: str,
    greek: str,
    metric: str = "abs_error",
) -> pd.DataFrame:
    """Generates an (N x eps) pivot table for a given product, greek, and metric.

    Uses `pivot_table` with mean aggregation to safely prevent duplicate index exceptions.

    Args:
        metrics_df: DataFrame enriched with error metrics.
        product: Name of the product (e.g. 'Call European ATM').
        greek: Sensitivity ('delta', 'gamma', 'vega').
        metric: 'abs_error', 'rel_error', 'bias', or 'value'.

    Returns:
        pd.DataFrame: Pivot table with N as index and eps as descending columns.
    """
    prod_data = metrics_df[metrics_df["Product"] == product]
    if prod_data.empty:
        return pd.DataFrame()

    if "eps" not in prod_data.columns:
        raise KeyError(f"DataFrame does not contain 'eps' column required for step size pivot.")

    if metric == "value":
        col = greek
    else:
        col = f"{greek}_{metric}"

    if col not in prod_data.columns:
        raise KeyError(f"Metric column '{col}' not found in DataFrame.")

    pivot = prod_data.pivot_table(index="N", columns="eps", values=col, aggfunc="mean")
    pivot = pivot.sort_index(ascending=True)

    sorted_cols = sorted(pivot.columns, reverse=True)
    return pivot[sorted_cols]


# ==============================================================================
# 6. Executive & Comparative Summary Tables
# ==============================================================================

def compute_summary_table(
    metrics_df: pd.DataFrame,
    n_choice: int | None = None,
    fdmc_eps: float | Mapping[str, float] | Mapping[tuple[str, str], float] | str = FDMC_FIXED_EPS,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Generates an executive summary table comparing Benchmark vs FDMC for each Product and Greek.

    Args:
        metrics_df: Enriched metrics DataFrame.
        n_choice: Specific N to evaluate, or None to pick the largest N.
        fdmc_eps: Fixed epsilon (e.g. 0.01) or 'optimal'.
        criterion: 'abs_error' or 'rel_error'.
        exclude_catastrophic: If True, excludes catastrophic step sizes.

    Returns:
        pd.DataFrame: Formatted summary comparison.
    """
    if n_choice is None:
        n_choice = int(metrics_df["N"].max())

    opt_df = extract_fixed_or_optimal_eps(
        metrics_df, fdmc_eps=fdmc_eps, criterion=criterion, exclude_catastrophic=exclude_catastrophic
    )
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

    is_opt = isinstance(fdmc_eps, str) and fdmc_eps.lower() == "optimal"
    eps_label = "Optimal ε*" if is_opt else (f"Fixed ε ({fdmc_eps})" if isinstance(fdmc_eps, (int, float)) else "Fixed ε")
    fdmc_header = (
        f"FDMC (opt ε*, N={n_choice:_})"
        if is_opt
        else (f"FDMC (ε={fdmc_eps}, N={n_choice:_})" if isinstance(fdmc_eps, (int, float)) else f"FDMC (N={n_choice:_})")
    )

    table.columns = [
        "Product",
        "Greek",
        "Benchmark",
        fdmc_header,
        eps_label,
        "Abs Error",
        "Rel Error (%)",
    ]

    return table


def compute_comparative_summary_table(
    fdmc_metrics_df: pd.DataFrame,
    malliavin_metrics_df: pd.DataFrame,
    n_choice: int | None = None,
    fdmc_eps: float | Mapping[str, float] | Mapping[tuple[str, str], float] | str = FDMC_FIXED_EPS,
    criterion: str = "abs_error",
    exclude_catastrophic: bool = True,
    fournier_metrics_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Generates an executive comparative table: Benchmark vs FDMC vs Standard Malliavin (vs Fournié Malliavin).

    Maintains dual compatibility across schema versions by providing both
    `Ratio (FDMC/Std Mal)` and `Accuracy Ratio (FDMC/Mal)` regardless of whether Fournié is included.

    Args:
        fdmc_metrics_df: Enriched FDMC metrics DataFrame.
        malliavin_metrics_df: Enriched Standard Malliavin metrics DataFrame.
        n_choice: Sample size N to evaluate (defaults to max available N).
        fdmc_eps: Fixed epsilon (e.g. 0.01) or 'optimal'.
        criterion: Error metric to evaluate ('abs_error' or 'rel_error').
        exclude_catastrophic: If True, excludes catastrophic FDMC step sizes.
        fournier_metrics_df: Optional enriched Fournié Localized Malliavin metrics DataFrame.

    Returns:
        pd.DataFrame: Formatted side-by-side comparative table.
    """
    if n_choice is None:
        n_choice = int(malliavin_metrics_df["N"].max())

    opt_fdmc = extract_fixed_or_optimal_eps(
        fdmc_metrics_df, fdmc_eps=fdmc_eps, criterion=criterion, exclude_catastrophic=exclude_catastrophic
    )
    sub_fdmc = opt_fdmc[opt_fdmc["N"] == n_choice].set_index(["Product", "Greek"])
    sub_mal = malliavin_metrics_df[malliavin_metrics_df["N"] == n_choice].copy()
    sub_four = (
        fournier_metrics_df[fournier_metrics_df["N"] == n_choice].copy()
        if fournier_metrics_df is not None
        else None
    )

    is_opt = isinstance(fdmc_eps, str) and fdmc_eps.lower() == "optimal"
    fdmc_col_name = (
        "FDMC (opt ε*)"
        if is_opt
        else (f"FDMC (ε={fdmc_eps})" if isinstance(fdmc_eps, (int, float)) else "FDMC (Fixed ε)")
    )
    eps_col_name = "Optimal ε*" if is_opt else "FDMC ε"

    records = []
    greeks = ["delta", "gamma", "vega"]
    products = malliavin_metrics_df["Product"].unique()

    use_rel = (criterion == "rel_error")

    for prod in products:
        for greek in greeks:
            row_mal = sub_mal[sub_mal["Product"] == prod]
            if row_mal.empty:
                continue
            r_mal = row_mal.iloc[0]

            bench_val = r_mal.get(f"{greek}_bench", np.nan)
            mal_val = r_mal.get(greek, np.nan)
            mal_abs_err = r_mal.get(f"{greek}_abs_error", np.nan)
            mal_rel_err = r_mal.get(f"{greek}_rel_error", np.nan)

            mal_err_comp = mal_rel_err if use_rel else mal_abs_err

            fdmc_val = np.nan
            fdmc_eps_val = np.nan
            fdmc_abs_err = np.nan
            fdmc_rel_err = np.nan

            if (prod, greek) in sub_fdmc.index:
                r_fdmc = sub_fdmc.loc[(prod, greek)]
                fdmc_val = r_fdmc["fdmc_value"]
                fdmc_eps_val = r_fdmc["optimal_eps"]
                fdmc_abs_err = r_fdmc["min_abs_error"]
                fdmc_rel_err = r_fdmc["min_rel_error_pct"]

            fdmc_err_comp = fdmc_rel_err if use_rel else fdmc_abs_err

            # Ratio of FDMC error to Malliavin error
            ratio = (
                (fdmc_err_comp / mal_err_comp)
                if (mal_err_comp > 0 and not np.isnan(fdmc_err_comp))
                else np.nan
            )

            rec = {
                "Product": prod,
                "Greek": greek,
                "Benchmark": bench_val,
                fdmc_col_name: fdmc_val,
                eps_col_name: fdmc_eps_val,
                "FDMC Abs Error": fdmc_abs_err,
                "Std Malliavin": mal_val,
                "Std Mal Abs Error": mal_abs_err,
                "FDMC Rel Err (%)": fdmc_rel_err,
                "Std Mal Rel Err (%)": mal_rel_err,
                "Accuracy Ratio (FDMC/Mal)": ratio,
                "Ratio (FDMC/Std Mal)": ratio,
            }

            if sub_four is not None:
                row_four = sub_four[sub_four["Product"] == prod]
                if not row_four.empty:
                    r_four = row_four.iloc[0]
                    four_val = r_four.get(greek, np.nan)
                    four_abs_err = r_four.get(f"{greek}_abs_error", np.nan)
                    four_rel_err = r_four.get(f"{greek}_rel_error", np.nan)

                    four_err_comp = four_rel_err if use_rel else four_abs_err

                    ratio_four = (
                        (fdmc_err_comp / four_err_comp)
                        if (four_err_comp > 0 and not np.isnan(fdmc_err_comp))
                        else np.nan
                    )
                    gain_four_vs_std = (
                        (mal_err_comp / four_err_comp)
                        if (four_err_comp > 0 and not np.isnan(mal_err_comp))
                        else np.nan
                    )

                    rec["Fournié Malliavin"] = four_val
                    rec["Fournié Abs Error"] = four_abs_err
                    rec["Fournié Rel Err (%)"] = four_rel_err
                    rec["Ratio (FDMC/Fournié)"] = ratio_four
                    rec["Fournié Gain vs Std"] = gain_four_vs_std

            records.append(rec)

    return pd.DataFrame(records)


# ==============================================================================
# 7. Sample Size & Convergence Solvers (Min Required N*)
# ==============================================================================

def find_min_acceptable_N(
    metrics_df: pd.DataFrame,
    tolerances: Mapping[str, float] | None = None,
    fdmc_eps: float | Mapping[str, float] | Mapping[tuple[str, str], float] | str = FDMC_FIXED_EPS,
    criterion: str = "abs_error",
    timing_df: pd.DataFrame | None = None,
    exclude_catastrophic: bool = True,
) -> pd.DataFrame:
    """Finds the minimum number of simulations N needed to achieve target error tolerances.

    Args:
        metrics_df: Enriched metrics DataFrame.
        tolerances: Dictionary of target tolerances per Greek (e.g. {'delta': 0.005, 'gamma': 0.0002, 'vega': 0.05}).
        fdmc_eps: Fixed epsilon (e.g. 0.01) or 'optimal'.
        criterion: 'abs_error' or 'rel_error'.
        timing_df: Optional timing DataFrame (joint or per-Greek) to compute wall-clock execution time.
        exclude_catastrophic: If True, excludes catastrophic step sizes.

    Returns:
        pd.DataFrame: Table of minimum required N, chosen eps, achieved error, and joint wall-clock execution time.
    """
    if tolerances is None:
        if criterion == "abs_error":
            tolerances = {
                "delta": 0.005,    # within 0.005 (50 bps)
                "gamma": 0.0002,   # within 0.0002 (~1% error)
                "vega": 0.05,      # within 0.05
            }
        else:
            tolerances = {
                "delta": 0.5,      # 0.5% relative error
                "gamma": 1.0,      # 1.0% relative error
                "vega": 0.5,       # 0.5% relative error
            }

    opt_df = extract_fixed_or_optimal_eps(
        metrics_df, fdmc_eps=fdmc_eps, criterion=criterion, exclude_catastrophic=exclude_catastrophic
    )
    err_col = "min_abs_error" if criterion == "abs_error" else "min_rel_error_pct"

    # Compute rate ms / path if timing_df is provided
    ms_rates = {}
    has_greek_col = False
    if timing_df is not None:
        ms_per_sim = timing_df.copy()
        if "ms_per_10k_paths" in ms_per_sim.columns:
            ms_per_sim["rate"] = ms_per_sim["ms_per_10k_paths"] / 10_000.0
        elif "time_ms_mean" in ms_per_sim.columns and "N" in ms_per_sim.columns:
            ms_per_sim["rate"] = ms_per_sim["time_ms_mean"] / ms_per_sim["N"]
        else:
            ms_per_sim["rate"] = 0.0

        has_greek_col = "Greek" in ms_per_sim.columns
        if has_greek_col:
            ms_rates = ms_per_sim.groupby(["Product", "Greek"])["rate"].mean().to_dict()
        else:
            ms_rates = ms_per_sim.groupby("Product")["rate"].mean().to_dict()

    results = []
    products = opt_df["Product"].unique()
    candidate_greeks = ["delta", "gamma", "vega"]
    available_greeks = [g for g in candidate_greeks if g in opt_df["Greek"].unique()]

    for prod in products:
        for greek in available_greeks:
            if has_greek_col:
                ms_per_path = ms_rates.get((prod, greek), None)
            else:
                ms_per_path = ms_rates.get(prod, None)

            target_tol = tolerances.get(greek, 0.01)
            sub = opt_df[(opt_df["Product"] == prod) & (opt_df["Greek"] == greek)].sort_values("N")

            if sub.empty:
                continue

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
                    "FDMC ε": first_pass["optimal_eps"],
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
                    "FDMC ε": best_ever["optimal_eps"],
                    "Achieved Abs Error": best_ever["min_abs_error"],
                    "Achieved Rel Error (%)": best_ever["min_rel_error_pct"],
                    "Joint Time (ms)": t_est,
                    "Status": f"Exceeds Tol (best={best_ever[err_col]:.4f})",
                })

    res_df = pd.DataFrame(results)
    if timing_df is None and "Joint Time (ms)" in res_df.columns:
        res_df = res_df.drop(columns=["Joint Time (ms)"])
    return res_df


# ==============================================================================
# 8. Cutoff & Delta Strike Sensitivity Summarizers
# ==============================================================================


# ==============================================================================
# 9. Fournié Smoothing Grid & Variance Calibration
# ==============================================================================

def compute_smoothing_and_gap_grid(
    params: Mapping[str, float],
    benchmarks: Mapping[str, float],
    contract_type: str = "call",
    cutoff_types: Sequence[str] | None = None,
    deltas: Sequence[float] | None = None,
    N: int = 100_000,
    seed: int = 42,
    num_reps_for_variance: int = 20,
    variance_sample_size: int = 10_000,
) -> pd.DataFrame:
    """Computes Greeks, errors, and empirical estimator variance across a 2D grid of (Cutoff Function x Gap Delta).

    Args:
        params: Option parameter dictionary (S0, K, T, r, q, sigma).
        benchmarks: Dictionary of analytical benchmark values {delta: ..., gamma: ..., vega: ...}.
        contract_type: 'call', 'digital', or 'knock-out' ('barrier' is accepted as an alias).
        cutoff_types: List of cutoff smoothing functions.
        deltas: List of smoothing half-widths delta from the strike.
        N: Number of simulations for point estimate.
        seed: Random seed for reproducibility.
        num_reps_for_variance: Number of Monte Carlo replications to compute sample variance.
        variance_sample_size: Sample size per replication for variance calculation.

    Returns:
        pd.DataFrame: Comprehensive sensitivity and variance dataset across all (function, delta) pairs.
    """
    from Methods.sampling import generate_paths
    from Methods.Malliavin.malliavin_european_fournier import (
        malliavin_fournier_greeks_european_call_terminal_knockout,
        malliavin_fournier_greeks_european_call,
        malliavin_fournier_greeks_european_digital_call,
    )

    if cutoff_types is None:
        cutoff_types = ["gaussian", "cubic", "cosine", "quintic", "cauchy"]
    if deltas is None:
        deltas = [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0, 40.0, 50.0]

    fn_map = {
        "call": malliavin_fournier_greeks_european_call,
        "digital": malliavin_fournier_greeks_european_digital_call,
        "barrier": malliavin_fournier_greeks_european_call_terminal_knockout,
        "knock-out": malliavin_fournier_greeks_european_call_terminal_knockout,
    }
    calc_fn = fn_map.get(contract_type.strip().lower(), malliavin_fournier_greeks_european_call)

    # Pre-generate paths for point estimate
    paths = generate_paths(params["T"], N, seed=seed)

    # Pre-generate paths for variance replications
    rep_paths_list = [
        generate_paths(params["T"], variance_sample_size, seed=seed + 1000 + i)
        for i in range(num_reps_for_variance)
    ]

    records = []

    for ctype in cutoff_types:
        for d in deltas:
            # 1. Point estimate
            res = calc_fn(params, N=N, paths=paths, cutoff_type=ctype, bandwidth=d)

            # 2. Empirical variance across independent replications
            rep_deltas, rep_gammas, rep_vegas = [], [], []
            for r_paths in rep_paths_list:
                rep_res = calc_fn(
                    params, N=variance_sample_size, paths=r_paths, cutoff_type=ctype, bandwidth=d
                )
                rep_deltas.append(rep_res["delta"])
                rep_gammas.append(rep_res["gamma"])
                rep_vegas.append(rep_res["vega"])

            v_delta = float(np.var(rep_deltas, ddof=1))
            v_gamma = float(np.var(rep_gammas, ddof=1))
            v_vega = float(np.var(rep_vegas, ddof=1))

            # Errors vs benchmark
            d_bench = benchmarks.get("delta", np.nan)
            g_bench = benchmarks.get("gamma", np.nan)
            v_bench = benchmarks.get("vega", np.nan)

            d_err = abs(res["delta"] - d_bench) if not np.isnan(d_bench) else np.nan
            g_err = abs(res["gamma"] - g_bench) if not np.isnan(g_bench) else np.nan
            v_err = abs(res["vega"] - v_bench) if not np.isnan(v_bench) else np.nan

            records.append({
                "cutoff_type": ctype,
                "bandwidth": d,
                "delta": res["delta"],
                "gamma": res["gamma"],
                "vega": res["vega"],
                "delta_abs_error": d_err,
                "gamma_abs_error": g_err,
                "vega_abs_error": v_err,
                "delta_rel_error": (d_err / abs(d_bench) * 100.0) if (d_bench != 0 and not np.isnan(d_bench)) else np.nan,
                "gamma_rel_error": (g_err / abs(g_bench) * 100.0) if (g_bench != 0 and not np.isnan(g_bench)) else np.nan,
                "vega_rel_error": (v_err / abs(v_bench) * 100.0) if (v_bench != 0 and not np.isnan(v_bench)) else np.nan,
                "delta_var": v_delta,
                "gamma_var": v_gamma,
                "vega_var": v_vega,
                "delta_std": np.sqrt(v_delta),
                "gamma_std": np.sqrt(v_gamma),
                "vega_std": np.sqrt(v_vega),
            })

    return pd.DataFrame(records)


def find_optimal_delta(
    grid_df: pd.DataFrame,
    greek: str = "gamma",
    criterion: str = "var",
) -> pd.DataFrame:
    """Finds the optimal smoothing half-width delta* for each cutoff function.

    Args:
        grid_df: DataFrame returned by `compute_smoothing_and_gap_grid`.
        greek: Sensitivity ('delta', 'gamma', 'vega').
        criterion: 'var' (sample variance), 'std' (standard deviation), or 'abs_error' (absolute error).

    Returns:
        pd.DataFrame: Table of optimal delta* and minimum metric per cutoff function.
    """
    g = greek.lower()
    c = criterion.lower()

    if c in ["var", "variance"]:
        col = f"{g}_var"
    elif c in ["std", "stdev"]:
        col = f"{g}_std"
    elif c in ["abs_error", "abs"]:
        col = f"{g}_abs_error"
    elif c in ["rel_error", "rel"]:
        col = f"{g}_rel_error"
    else:
        col = f"{g}_{criterion}"

    if col not in grid_df.columns:
        raise KeyError(f"Optimization metric '{col}' not found in smoothing grid dataset.")

    rows = []
    for ctype in grid_df["cutoff_type"].unique():
        sub = grid_df[grid_df["cutoff_type"] == ctype]
        best_idx = sub[col].idxmin()
        best_row = sub.loc[best_idx]
        rows.append({
            "Cutoff Function": ctype,
            "Greek": g.capitalize(),
            f"Optimal δ* ({criterion})": best_row["bandwidth"],
            f"Min {criterion.capitalize()}": best_row[col],
            "Value": best_row[g],
            "Abs Error": best_row[f"{g}_abs_error"],
            "Variance": best_row[f"{g}_var"],
        })
    return pd.DataFrame(rows)


# ==============================================================================
# 10. Quantitative Diagnostics: Empirical Convergence Rate
# ==============================================================================


def generate_summary_csv(
    output_path: str | Path = "./results/summary.csv",
    benchmarks_path: str | Path = "./results/benchmarks.csv",
    fdmc_euro_path: str | Path = "./results/FDMC_results.csv",
    malliavin_euro_path: str | Path = "./results/malliavin_results.csv",
    malliavin_fournier_path: str | Path = "./results/malliavin_fournier_results.csv",
    fdmc_fixed_eps: float = 0.01,
) -> pd.DataFrame:
    """Generates an up-to-date summary CSV comparing benchmarks, FDMC (at fixed eps), and Malliavin.

    For each method and product, extracts the Greek sensitivities computed with the
    maximum number of simulations (the most precise estimate), using fdmc_fixed_eps
    (default 0.01) for all FDMC estimators.

    Args:
        output_path: File path to save the summary CSV.
        benchmarks_path: Path to benchmarks.csv.
        fdmc_euro_path: Path to European FDMC_results.csv.
        malliavin_euro_path: Path to European malliavin_results.csv.
        malliavin_fournier_path: Path to European malliavin_fournier_results.csv.
        fdmc_fixed_eps: Perturbation step size for FDMC (default 0.01).

    Returns:
        pd.DataFrame: Summary table with columns Product, each Greek with its per-path variance
            and standard error, Method, N, eps.
    """
    benchmarks_path = Path(benchmarks_path)
    fdmc_euro_path = Path(fdmc_euro_path)
    malliavin_euro_path = Path(malliavin_euro_path)
    malliavin_fournier_path = Path(malliavin_fournier_path)

    sections = []

    # 1. Malliavin Results at Max N (Standard and Fournié)
    mal_dfs = []
    if malliavin_euro_path.exists():
        mal_dfs.append(pd.read_csv(malliavin_euro_path))
    if malliavin_fournier_path.exists():
        mal_dfs.append(pd.read_csv(malliavin_fournier_path))

    if mal_dfs:
        combined_mal = pd.concat(mal_dfs, ignore_index=True)
        mal_rows = []
        for (prod, method) in combined_mal[["Product", "Method"]].drop_duplicates().values:
            prod_df = combined_mal[(combined_mal["Product"] == prod) & (combined_mal["Method"] == method)]
            max_n = prod_df["N"].max()
            row = prod_df[prod_df["N"] == max_n].iloc[0].copy()
            mal_rows.append(row)
        df_mal = pd.DataFrame(mal_rows)
        if "eps" not in df_mal.columns:
            df_mal["eps"] = None
        sections.append(df_mal)

    # 2. FDMC Results at Max N with eps == fdmc_fixed_eps
    fdmc_dfs = []
    if fdmc_euro_path.exists():
        fdmc_dfs.append(pd.read_csv(fdmc_euro_path))

    if fdmc_dfs:
        combined_fdmc = pd.concat(fdmc_dfs, ignore_index=True)
        fdmc_rows = []
        for prod in combined_fdmc["Product"].unique():
            prod_df = combined_fdmc[combined_fdmc["Product"] == prod]
            # Filter by fixed eps
            eps_filtered = prod_df[np.isclose(prod_df["eps"], fdmc_fixed_eps)]
            if not eps_filtered.empty:
                max_n = eps_filtered["N"].max()
                row = eps_filtered[eps_filtered["N"] == max_n].iloc[0].copy()
            else:
                max_n = prod_df["N"].max()
                row = prod_df[prod_df["N"] == max_n].iloc[0].copy()
            fdmc_rows.append(row)
        df_fdmc = pd.DataFrame(fdmc_rows)
        sections.append(df_fdmc)

    # 3. Reference Benchmarks
    if benchmarks_path.exists():
        bench_df = pd.read_csv(benchmarks_path)
        bench_df["N"] = None
        bench_df["eps"] = None
        sections.append(bench_df)

    if not sections:
        raise ValueError("No simulation or benchmark datasets were found to generate summary.csv.")

    summary_df = pd.concat(sections, ignore_index=True)
    col_order = [
        "Product",
        "delta", "delta_var", "delta_se",
        "gamma", "gamma_var", "gamma_se",
        "vega", "vega_var", "vega_se",
        "Method", "N", "eps",
    ]
    summary_df = summary_df[[c for c in col_order if c in summary_df.columns]]

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_p, index=False)
    print(f"Summary table successfully saved to {out_p} ({len(summary_df)} entries).")
    return summary_df


# ==============================================================================
# 10. Independent Replications: Bias, Standard Error, RMSE and Efficiency
# ==============================================================================

def estimator_label(method: str, eps: float | None) -> str:
    """'FDMC (ε=0.01)' for FDMC rows, the method name otherwise."""
    if eps is None or pd.isna(eps):
        return method
    return f"{method} (ε={eps:g})"


def summarize_replications(
    raw_df: pd.DataFrame,
    benchmarks_df: pd.DataFrame,
    greeks: Sequence[str] = ("delta", "gamma", "vega"),
) -> pd.DataFrame:
    """Summarizes R independent runs per (Method, eps, Product, N) against analytical benchmarks.

    Output is long format, one row per Greek, with:
        R              number of replications
        mean           average estimate over the R runs
        bias           mean - benchmark
        bias_se        sd / sqrt(R), uncertainty of the bias estimate
        sd             standard deviation of the estimate across runs (empirical SE at this N)
        mean_se        average within-run SE, sqrt(var / N); should match sd
        rmse           sqrt(mean((estimate - benchmark)^2))
        var_per_path   average within-run per-path variance
        abs_err_q10/q50/q90   quantiles of |estimate - benchmark| across runs (for bands)
        coverage_95    share of runs with |estimate - benchmark| <= 1.96 * se
        time_per_path_s  median over runs of wall time / N (all three Greeks together)
    """
    bench = benchmarks_df.set_index("Product")
    df = raw_df.copy()
    if "eps" not in df.columns:
        df["eps"] = np.nan

    records = []
    for (method, eps, product, n), grp in df.groupby(["Method", "eps", "Product", "N"], dropna=False):
        time_per_path = float(np.median(grp["time_s"] / n)) if "time_s" in grp.columns else np.nan
        for greek in greeks:
            true = float(bench.loc[product, greek])
            est = grp[greek].to_numpy(dtype=float)
            err = est - true
            se = grp[f"{greek}_se"].to_numpy(dtype=float)
            r = len(est)
            sd = float(np.std(est, ddof=1)) if r > 1 else np.nan
            records.append({
                "Method": method,
                "eps": eps,
                "Estimator": estimator_label(method, eps),
                "Product": product,
                "Greek": greek,
                "N": int(n),
                "R": r,
                "benchmark": true,
                "mean": float(np.mean(est)),
                "bias": float(np.mean(err)),
                "bias_se": sd / np.sqrt(r) if r > 1 else np.nan,
                "sd": sd,
                "mean_se": float(np.mean(se)),
                "rmse": float(np.sqrt(np.mean(err**2))),
                "var_per_path": float(np.mean(grp[f"{greek}_var"])),
                "abs_err_q10": float(np.quantile(np.abs(err), 0.10)),
                "abs_err_q50": float(np.quantile(np.abs(err), 0.50)),
                "abs_err_q90": float(np.quantile(np.abs(err), 0.90)),
                "coverage_95": float(np.mean(np.abs(err) <= 1.96 * se)),
                "time_per_path_s": time_per_path,
            })
    return pd.DataFrame(records)


def required_paths(var_per_path: float, tol: float, z: float = 1.96) -> float:
    """N* = z^2 * Var / tol^2: paths needed so the half-width of the z-interval is tol."""
    return z**2 * var_per_path / tol**2


def compute_efficiency_table(
    summary_df: pd.DataFrame,
    fdmc_eps: float = FDMC_FIXED_EPS,
    rel_tol: float = 0.01,
    z: float = 1.96,
) -> pd.DataFrame:
    """Per-path variance, time per path and efficiency (variance x time) per estimator.

    Uses the largest N of each estimator. Efficiency = var_per_path * time_per_path_s is the
    work needed per unit of inverse squared error; lower is better. Ratios are relative to
    FDMC at ``fdmc_eps`` on the same product and Greek:
        n_ratio_vs_FDMC          = Var / Var_FDMC   (paths needed for equal SE)
        efficiency_ratio_vs_FDMC = (Var * t) / (Var_FDMC * t_FDMC)   (CPU time for equal SE)
    n_star / time_star give the paths and seconds to reach a 95% half-width of
    rel_tol * |benchmark|. They ignore bias, so ``bias_exceeds_tol`` flags FDMC rows where the
    truncation bias alone is larger than the tolerance.
    """
    idx = summary_df.groupby(["Estimator", "Product", "Greek"])["N"].idxmax()
    top = summary_df.loc[idx].copy()

    top["time_per_path_ns"] = top["time_per_path_s"] * 1e9
    top["efficiency"] = top["var_per_path"] * top["time_per_path_s"]

    ref = top[(top["Method"] == "FDMC") & np.isclose(top["eps"].fillna(-1.0), fdmc_eps)]
    ref = ref.set_index(["Product", "Greek"])[["var_per_path", "efficiency"]]
    keys = list(zip(top["Product"], top["Greek"]))
    top["n_ratio_vs_FDMC"] = top["var_per_path"].to_numpy() / ref.loc[keys, "var_per_path"].to_numpy()
    top["efficiency_ratio_vs_FDMC"] = top["efficiency"].to_numpy() / ref.loc[keys, "efficiency"].to_numpy()

    tol = rel_tol * top["benchmark"].abs()
    top["tol"] = tol
    top["n_star"] = required_paths(top["var_per_path"], tol, z)
    top["time_star_s"] = top["n_star"] * top["time_per_path_s"]
    top["bias_exceeds_tol"] = (top["bias"].abs() > tol) & (top["bias"].abs() > 3 * top["bias_se"])

    cols = [
        "Product", "Greek", "Estimator", "Method", "eps", "N", "benchmark", "bias", "bias_se",
        "var_per_path", "time_per_path_ns", "efficiency", "n_ratio_vs_FDMC", "efficiency_ratio_vs_FDMC",
        "tol", "n_star", "time_star_s", "bias_exceeds_tol",
    ]
    return top[cols].sort_values(["Product", "Greek", "efficiency"]).reset_index(drop=True)


def expected_error_under_budget(
    efficiency_df: pd.DataFrame,
    budgets_ms: Sequence[float] = (250.0, 1000.0, 5000.0),
) -> pd.DataFrame:
    """Expected RMSE when each estimator gets a fixed CPU time budget.

    N = budget / time_per_path, and RMSE = sqrt(bias^2 + Var / N). The bias term is kept only
    when significant (|bias| > 3 bias_se), so unbiased methods are not given a fake noise floor.
    Long format: one row per (Product, Greek, Estimator, budget_ms).
    """
    rows = []
    for _, row in efficiency_df.iterrows():
        bias = row["bias"] if abs(row["bias"]) > 3 * row["bias_se"] else 0.0
        for budget in budgets_ms:
            n = budget * 1e-3 / (row["time_per_path_ns"] * 1e-9)
            rows.append({
                "Product": row["Product"],
                "Greek": row["Greek"],
                "Estimator": row["Estimator"],
                "budget_ms": budget,
                "N": int(n),
                "expected_rmse": float(np.sqrt(bias**2 + row["var_per_path"] / n)),
            })
    return pd.DataFrame(rows)
