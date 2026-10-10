"""Execution script to run high-precision Asian benchmarks, Moment Matching Greeks, and multi-N FDMC simulations."""

from pathlib import Path
from pandas import DataFrame, concat
from config import (
    ASIAN_BENCHMARK_N_SIMS,
    ASIAN_BENCHMARK_EPS,
    ASIAN_FDMC_N_SIMS_ARRAY,
    ASIAN_FDMC_EPS_ARRAY,
    ASIAN_SEED,
    ASIAN_BENCHMARKS_FILE,
    ASIAN_FDMC_RESULTS_FILE,
    ASIAN_MOMENT_MATCHING_RESULTS_FILE,
)
from products import asian_products
from Benchmarks.asian import benchmark_all_asian
from Methods.Analytic.asians import compute_all_asian_moment_matching_greeks
from Methods.MonteCarlo.FDMC_asian import (
    compute_all_asian_FDMC_greeks,
    compute_all_asian_conditional_FDMC_greeks,
)
from pricers.asians_MC import generate_asian_paths


def run_asian_moment_matching_greeks() -> DataFrame:
    """Computes Greeks using Finite Differences on Moment Matching pricing formulas across all Asian products."""
    print("Computing Asian Greeks via Moment Matching + Finite Differences...")
    results = {}
    for name, params in asian_products.items():
        greeks = compute_all_asian_moment_matching_greeks(params, eps=1e-5)
        results[name] = greeks

    df = DataFrame(results).T
    df.index.name = "Product"
    df["Method"] = "Moment Matching (FD)"
    df["eps"] = 1e-5
    return df


def run_asian_multi_N_FDMC(
    n_sims_list: list[int] | None = None,
    eps_list: list[float] | None = None,
    seed: int = ASIAN_SEED,
) -> DataFrame:
    """Runs Standard FDMC simulations across varying sample sizes N and perturbation sizes eps."""
    if n_sims_list is None:
        n_sims_list = ASIAN_FDMC_N_SIMS_ARRAY
    if eps_list is None:
        eps_list = [0.01]  # Standard production epsilon

    sample_params = next(iter(asian_products.values()))
    T = float(sample_params["T"])
    freq = int(sample_params.get("frequency", 12))

    all_dfs = []
    for N in n_sims_list:
        print(f"Generating {N:,} paths for multi-N Standard FDMC run...")
        paths = generate_asian_paths(T, freq, N, seed)
        for eps in eps_list:
            print(f"  Running Standard FDMC N={N:,}, eps={eps}...")
            current_results = {}
            for name, params in asian_products.items():
                greeks = compute_all_asian_FDMC_greeks(params, eps=eps, N=N, seed=seed, paths=paths)
                current_results[name] = greeks

            df = DataFrame(current_results).T
            df.index.name = "Product"
            df["Method"] = "FDMC"
            df["N"] = N
            df["eps"] = eps
            all_dfs.append(df)

    results_df = concat(all_dfs)
    return results_df



def main():
    Path("results").mkdir(parents=True, exist_ok=True)

    # 1. Moment Matching Greeks
    mm_df = run_asian_moment_matching_greeks()
    mm_df.to_csv(ASIAN_MOMENT_MATCHING_RESULTS_FILE)
    print(f"\n--- Moment Matching Greeks (saved to {ASIAN_MOMENT_MATCHING_RESULTS_FILE}) ---")
    print(mm_df)

    # 2. High-Precision Conditional MC Benchmark (N = 5,000,000 with Common Numbers)
    if not Path(ASIAN_BENCHMARKS_FILE).exists():
        benchmark_df = benchmark_all_asian(asian_products, N=ASIAN_BENCHMARK_N_SIMS, seed=ASIAN_SEED)
        benchmark_df.to_csv(ASIAN_BENCHMARKS_FILE)
        print(f"\n--- Conditional MC Benchmark Greeks (saved to {ASIAN_BENCHMARKS_FILE}) ---")
        print(benchmark_df)
    else:
        print(f"\n--- Benchmark already exists at {ASIAN_BENCHMARKS_FILE} ---")

    # 3. Multi-N Convergence Grid
    fdmc_df = run_asian_multi_N_FDMC()
    fdmc_df.to_csv(ASIAN_FDMC_RESULTS_FILE)
    print(f"\n--- Multi-N FDMC Results (saved to {ASIAN_FDMC_RESULTS_FILE}) ---")
    print(f"Total rows: {len(fdmc_df)}")


if __name__ == "__main__":
    main()
