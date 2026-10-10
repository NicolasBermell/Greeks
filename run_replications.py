"""Independent replications of every European estimator, with per-path variance and timing.

For each N in REPLICATION_N_ARRAY and each of R seeds, every (product, estimator) pair is run
once with np.random.default_rng(seed). Each run records the three Greeks, their per-path
variances and SEs, and the wall time to compute all three Greeks together (path generation
included):
    FDMC:              5 payoff evaluations per path (S0 ± eps, S0, sigma ± eps)
    Malliavin:         1 payoff evaluation per path
    Malliavin Fournié: 1 payoff evaluation plus the cutoff evaluations (one per distinct bandwidth)
Two extra runs are kept for comparison: the old call-only Gamma shortcut, and the knock-out
with psi = psi_K + psi_H (biased with the gaussian cutoff).
Estimators are interleaved within each seed so that machine-load drift affects them equally.

Outputs:
    results/replications_raw.csv      one row per run
    results/replications_summary.csv  bias, SE, RMSE, per-path variance, time per path
    results/efficiency.csv            variance x time per path, N and CPU ratios vs FDMC
    results/figures/error_vs_N_*.png  RMSE vs N with 10-90% bands and 1/sqrt(N) extrapolation
"""

import re
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import (
    EFFICIENCY_FILE,
    FIGURES_DIR,
    MALLIAVIN_FOURNIER_CUTOFF_TYPE,
    REPLICATION_BASE_SEED,
    REPLICATION_FDMC_EPS_ARRAY,
    REPLICATION_N_ARRAY,
    REPLICATION_R,
    REPLICATION_RAW_FILE,
    REPLICATION_SUMMARY_FILE,
)
from products import example_european_call_ATM, example_european_call_terminal_knockout
from run_all_greeks import EUROPEAN_PRODUCTS
from run_benchmarks import benchmark_all
from Utilities.metrics import compute_efficiency_table, summarize_replications
from Utilities.plotting import plot_replication_error_vs_N


def build_estimators() -> list[tuple[str, float | None, dict, callable]]:
    """(Method, eps, product params, fn(seed, N) -> stats dict) for every run configuration."""
    estimators = []
    for params, fdmc_fn, mall_fn, four_fn, bandwidths in EUROPEAN_PRODUCTS:
        for eps in REPLICATION_FDMC_EPS_ARRAY:
            estimators.append(("FDMC", eps, params,
                               lambda seed, N, p=params, f=fdmc_fn, e=eps: f(p, eps=e, N=N, seed=seed, return_stats=True)))
        estimators.append(("Malliavin", None, params,
                           lambda seed, N, p=params, f=mall_fn: f(p, N=N, seed=seed, return_stats=True)))
        estimators.append(("Malliavin Fournié", None, params,
                           lambda seed, N, p=params, f=four_fn, b=bandwidths: f(
                               p, N=N, seed=seed, cutoff_type=MALLIAVIN_FOURNIER_CUTOFF_TYPE,
                               delta_strike=b, return_stats=True)))
        if params is example_european_call_terminal_knockout:
            # psi = psi_K + psi_H: biased with the gaussian cutoff (psi(K) = 1 + psi_H(K)). Kept to show that.
            estimators.append(("Malliavin Fournié (ψ_K+ψ_H)", None, params,
                               lambda seed, N, p=params, f=four_fn, b=bandwidths: f(
                                   p, N=N, seed=seed, cutoff_type=MALLIAVIN_FOURNIER_CUTOFF_TYPE,
                                   delta_strike=b, return_stats=True, cutoff_combination="sum")))
        if params is example_european_call_ATM:
            # Old call-only Gamma: (K/S0) x localized digital Delta. Kept for the remark comparing variances.
            estimators.append(("Malliavin Fournié (Γ shortcut)", None, params,
                               lambda seed, N, p=params, f=four_fn, b=bandwidths: f(
                                   p, N=N, seed=seed, cutoff_type=MALLIAVIN_FOURNIER_CUTOFF_TYPE,
                                   delta_strike=b, return_stats=True, gamma_formula="shortcut")))
    return estimators


def run_replications() -> pd.DataFrame:
    estimators = build_estimators()
    rows = []
    for N in REPLICATION_N_ARRAY:
        print(f"Replications N={N:_}: {REPLICATION_R} seeds x {len(estimators)} estimators")
        t_start = time.perf_counter()
        for r in range(REPLICATION_R):
            seed = REPLICATION_BASE_SEED + r
            for method, eps, params, fn in estimators:
                t0 = time.perf_counter()
                res = fn(seed, N)
                elapsed = time.perf_counter() - t0
                rows.append({"Product": params["name"], "Method": method, "eps": eps, "N": N,
                             "seed": seed, **res, "time_s": elapsed})
        print(f"  done in {time.perf_counter() - t_start:.1f} s")
    return pd.DataFrame(rows)


def main():
    Path(REPLICATION_RAW_FILE).parent.mkdir(parents=True, exist_ok=True)
    raw = run_replications()
    raw.to_csv(REPLICATION_RAW_FILE, index=False)

    benchmarks = benchmark_all().reset_index()
    summary = summarize_replications(raw, benchmarks)
    summary.to_csv(REPLICATION_SUMMARY_FILE, index=False)

    efficiency = compute_efficiency_table(summary)
    efficiency.to_csv(EFFICIENCY_FILE, index=False)

    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    for product in summary["Product"].unique():
        fig = plot_replication_error_vs_N(summary, product)
        slug = re.sub(r"[^a-z0-9]+", "_", product.lower()).strip("_")
        fig.savefig(Path(FIGURES_DIR) / f"error_vs_N_{slug}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    print(f"Saved {REPLICATION_RAW_FILE}, {REPLICATION_SUMMARY_FILE}, {EFFICIENCY_FILE} and figures in {FIGURES_DIR}")


if __name__ == "__main__":
    main()
