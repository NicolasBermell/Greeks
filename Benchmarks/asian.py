"""High-Precision Asian Options Benchmark via Conditional Monte Carlo (CMC) with Common Numbers.

Uses Conditional Monte Carlo (conditioning on the filtration F_{t_{m-1}}) combined with
large-scale simulation (5,000,000 paths) and Common Random Numbers to establish an exact,
infinitely smooth, low-variance ground truth benchmark for discrete arithmetic Asian Greeks.
"""

import sys
from pathlib import Path
from typing import Any
import numpy as np
from pandas import DataFrame

# Ensure project root is in sys.path
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import (
    ASIAN_BENCHMARK_N_SIMS,
    ASIAN_BENCHMARK_EPS,
    ASIAN_SEED,
    ASIAN_BENCHMARKS_FILE,
)
from products import asian_products
from Methods.MonteCarlo.FDMC_asian import (
    FDMC_conditional_greeks_asian_call,
    FDMC_conditional_greeks_asian_digital_call,
    FDMC_conditional_greeks_asian_barrier_up_and_out_call,
)
from pricers.asians_MC import generate_asian_paths


def benchmark_greeks_asian_call(
    params: dict[str, Any],
    N: int = ASIAN_BENCHMARK_N_SIMS,
    seed: int = ASIAN_SEED,
    eps: float = ASIAN_BENCHMARK_EPS,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes benchmark Greeks for a Vanilla Asian Call via Conditional Monte Carlo."""
    return FDMC_conditional_greeks_asian_call(params, eps=eps, N=N, seed=seed, paths=paths)


def benchmark_greeks_asian_digital(
    params: dict[str, Any],
    N: int = ASIAN_BENCHMARK_N_SIMS,
    seed: int = ASIAN_SEED,
    eps: float = ASIAN_BENCHMARK_EPS,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes benchmark Greeks for a Digital Asian Call via Conditional Monte Carlo (smooth expectation)."""
    return FDMC_conditional_greeks_asian_digital_call(params, eps=eps, N=N, seed=seed, paths=paths)


def benchmark_greeks_asian_barrier_up_and_out(
    params: dict[str, Any],
    N: int = ASIAN_BENCHMARK_N_SIMS,
    seed: int = ASIAN_SEED,
    eps: float = ASIAN_BENCHMARK_EPS,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes benchmark Greeks for a Barrier Up-and-Out Asian Call via Conditional Monte Carlo."""
    return FDMC_conditional_greeks_asian_barrier_up_and_out_call(params, eps=eps, N=N, seed=seed, paths=paths)


def benchmark_all_asian(
    products: dict[str, dict[str, Any]] | None = None,
    N: int = ASIAN_BENCHMARK_N_SIMS,
    seed: int = ASIAN_SEED,
    eps: float = ASIAN_BENCHMARK_EPS,
) -> DataFrame:
    """Computes benchmark Greeks across all provided Asian products using Conditional Monte Carlo."""
    if products is None:
        products = asian_products

    # Pre-generate shared paths with Common Random Numbers (CN)
    sample_params = next(iter(products.values()))
    T = float(sample_params["T"])
    freq = int(sample_params.get("frequency", 12))

    print(f"Pre-generating {N:,} Brownian motion paths with Common Numbers (seed={seed})...")
    paths = generate_asian_paths(T, freq, N, seed)

    results = {}
    for name, params in products.items():
        print(f"Computing Conditional MC benchmark Greeks for {name} (N={N:,}, eps={eps})...")
        prod_name_lower = name.lower()
        if "barrier" in prod_name_lower:
            greeks = benchmark_greeks_asian_barrier_up_and_out(params, N=N, seed=seed, eps=eps, paths=paths)
        elif "digital" in prod_name_lower:
            greeks = benchmark_greeks_asian_digital(params, N=N, seed=seed, eps=eps, paths=paths)
        else:
            greeks = benchmark_greeks_asian_call(params, N=N, seed=seed, eps=eps, paths=paths)
        results[name] = greeks

    df = DataFrame(results).T
    df.index.name = "Product"
    df["Method"] = "Conditional MC Benchmark (CN)"
    df["N"] = N
    df["eps"] = eps
    return df


if __name__ == "__main__":
    df = benchmark_all_asian()
    Path(ASIAN_BENCHMARKS_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ASIAN_BENCHMARKS_FILE)
    print("\n--- Benchmark Asian Greeks (Conditional MC) ---")
    print(df)
    print(f"\nSaved benchmarks to {ASIAN_BENCHMARKS_FILE}")
