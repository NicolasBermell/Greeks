"""Finite Difference Monte Carlo (FDMC) Greeks for discrete arithmetic average Asian options.

Central differences with shared random numbers: the Brownian paths are generated once and reused
for every bumped evaluation (S0 +- eps, S0, sigma +- eps). Two variants:
- standard: the payoff of the simulated average;
- conditional: the payoff's conditional expectation given F_{t_{m-1}} (smooth in S0 and sigma),
  used for the benchmark.
With ``return_stats=True`` each Greek also gets its per-path variance and standard error.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from Methods.sampling import estimate_from_samples
from pricers.asians_MC import (
    asian_barrier_up_and_out_call_payoffs,
    asian_call_payoffs,
    asian_digital_call_payoffs,
    conditional_asian_barrier_up_and_out_call_payoffs,
    conditional_asian_call_payoffs,
    conditional_asian_digital_call_payoffs,
    generate_asian_paths,
)


def _asian_fdmc_greeks(
    payoff_fn: Callable[..., np.ndarray],
    params: dict[str, Any],
    eps: float,
    N: int,
    seed: int,
    paths: np.ndarray | None,
    return_stats: bool,
    with_barrier: bool = False,
) -> dict[str, float]:
    """Per-path central differences of ``payoff_fn`` (5 evaluations per path on shared paths)."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)
    discount = np.exp(-r * T)
    extra = (float(params["H"]),) if with_barrier else ()

    def discounted_payoff(s0: float, sig: float) -> np.ndarray:
        return discount * payoff_fn(s0, K, T, r, q, sig, *extra, freq, seed, N, paths)

    def samples(_paths: np.ndarray) -> dict[str, np.ndarray]:
        V_plus, V_minus, V_0 = discounted_payoff(S0 + eps, sigma), discounted_payoff(S0 - eps, sigma), discounted_payoff(S0, sigma)
        V_sig_plus, V_sig_minus = discounted_payoff(S0, sigma + eps), discounted_payoff(S0, sigma - eps)
        return {
            "delta": (V_plus - V_minus) / (2.0 * eps),
            "gamma": (V_plus - 2.0 * V_0 + V_minus) / eps**2,
            "vega": (V_sig_plus - V_sig_minus) / (2.0 * eps),
        }

    return estimate_from_samples(samples, T, N, seed, paths=paths, return_stats=return_stats)


# ==============================================================================
# 1. Standard FDMC
# ==============================================================================

def FDMC_greeks_asian_call(params, eps=0.01, N=10000, seed=42, paths=None, return_stats=False) -> dict[str, float]:
    return _asian_fdmc_greeks(asian_call_payoffs, params, eps, N, seed, paths, return_stats)


def FDMC_greeks_asian_digital_call(params, eps=0.01, N=10000, seed=42, paths=None, return_stats=False) -> dict[str, float]:
    return _asian_fdmc_greeks(asian_digital_call_payoffs, params, eps, N, seed, paths, return_stats)


def FDMC_greeks_asian_barrier_up_and_out_call(params, eps=0.01, N=10000, seed=42, paths=None, return_stats=False) -> dict[str, float]:
    return _asian_fdmc_greeks(asian_barrier_up_and_out_call_payoffs, params, eps, N, seed, paths, return_stats, with_barrier=True)


def compute_all_asian_FDMC_greeks(params, eps=0.01, N=10000, seed=42, paths=None, return_stats=False) -> dict[str, float]:
    """Dispatches on the product: barrier if params has H, digital if the name says so, else call."""
    name = str(params.get("name", "")).lower()
    if "H" in params:
        return FDMC_greeks_asian_barrier_up_and_out_call(params, eps, N, seed, paths, return_stats)
    if "digital" in name:
        return FDMC_greeks_asian_digital_call(params, eps, N, seed, paths, return_stats)
    return FDMC_greeks_asian_call(params, eps, N, seed, paths, return_stats)


# ==============================================================================
# 2. Conditional Monte Carlo FDMC (smooth, low variance; used as the benchmark)
# ==============================================================================

def FDMC_conditional_greeks_asian_call(params, eps=0.01, N=10000, seed=42, paths=None, return_stats=False) -> dict[str, float]:
    return _asian_fdmc_greeks(conditional_asian_call_payoffs, params, eps, N, seed, paths, return_stats)


def FDMC_conditional_greeks_asian_digital_call(params, eps=0.01, N=10000, seed=42, paths=None, return_stats=False) -> dict[str, float]:
    return _asian_fdmc_greeks(conditional_asian_digital_call_payoffs, params, eps, N, seed, paths, return_stats)


def FDMC_conditional_greeks_asian_barrier_up_and_out_call(params, eps=0.01, N=10000, seed=42, paths=None, return_stats=False) -> dict[str, float]:
    return _asian_fdmc_greeks(
        conditional_asian_barrier_up_and_out_call_payoffs, params, eps, N, seed, paths, return_stats, with_barrier=True
    )


def compute_all_asian_conditional_FDMC_greeks(params, eps=0.01, N=10000, seed=42, paths=None, return_stats=False) -> dict[str, float]:
    """Conditional-MC version of ``compute_all_asian_FDMC_greeks``."""
    name = str(params.get("name", "")).lower()
    if "H" in params:
        return FDMC_conditional_greeks_asian_barrier_up_and_out_call(params, eps, N, seed, paths, return_stats)
    if "digital" in name:
        return FDMC_conditional_greeks_asian_digital_call(params, eps, N, seed, paths, return_stats)
    return FDMC_conditional_greeks_asian_call(params, eps, N, seed, paths, return_stats)
