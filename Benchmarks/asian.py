"""Asian benchmark Greeks via Conditional Monte Carlo (CMC) with shared random numbers.

Conditioning on F_{t_{m-1}} makes the payoff smooth in S0 and sigma, so central differences on
5,000,000 shared paths give a low-variance reference for the discrete arithmetic Asian Greeks.
Each Greek is stored with its per-path variance and standard error.
"""

from typing import Any

from pandas import DataFrame

from config import ASIAN_BENCHMARK_EPS, ASIAN_BENCHMARK_N_SIMS, ASIAN_SEED
from products import asian_products
from Methods.MonteCarlo.FDMC_asian import compute_all_asian_conditional_FDMC_greeks
from Methods.sampling import GREEK_STAT_COLUMNS
from pricers.asians_MC import generate_asian_paths


def benchmark_all_asian(
    products: dict[str, dict[str, Any]] | None = None,
    N: int = ASIAN_BENCHMARK_N_SIMS,
    seed: int = ASIAN_SEED,
    eps: float = ASIAN_BENCHMARK_EPS,
) -> DataFrame:
    """Benchmark Greeks for every Asian product, all on the same paths."""
    if products is None:
        products = asian_products

    sample_params = next(iter(products.values()))
    T = float(sample_params["T"])
    freq = int(sample_params.get("frequency", 12))

    print(f"Pre-generating {N:,} Brownian paths (seed={seed})...")
    paths = generate_asian_paths(T, freq, N, seed)

    results = {}
    for name, params in products.items():
        print(f"Computing Conditional MC benchmark Greeks for {name} (N={N:,}, eps={eps})...")
        results[name] = compute_all_asian_conditional_FDMC_greeks(
            params, eps=eps, N=N, seed=seed, paths=paths, return_stats=True
        )

    df = DataFrame(results).T[GREEK_STAT_COLUMNS]
    df.index.name = "Product"
    df["Method"] = "Conditional MC Benchmark (CN)"
    df["N"] = N
    df["eps"] = eps
    return df
