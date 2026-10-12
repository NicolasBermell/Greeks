"""Finite Difference Monte Carlo (FDMC) Greeks for European options.

Central differences with shared random numbers: every bumped price uses the same W_T.
Delta, Gamma and Vega need 5 payoff evaluations per path (S0 +- eps, S0, sigma +- eps).
"""

from __future__ import annotations

import numpy as np

from Methods import payoffs
from Methods.sampling import estimate_from_samples, generate_paths, simulate_terminal_spot


def FDMC_samples_european(payoff_fn, params, eps: float, W_T: np.ndarray) -> dict[str, np.ndarray]:
    """Per-path central difference quotients with shared random numbers.

    Uses 5 payoff evaluations per path: S0 + eps, S0, S0 - eps, sigma + eps, sigma - eps.
    The mean of each array is the usual FDMC estimate; its variance is the per-path variance.
    """
    S0, T, r, q, sigma = params["S0"], params["T"], params["r"], params["q"], params["sigma"]
    discount = np.exp(-r * T)

    def discounted_payoff(s0: float, sig: float) -> np.ndarray:
        return discount * payoff_fn(simulate_terminal_spot(s0, T, r, q, sig, W_T), params)

    V_plus = discounted_payoff(S0 + eps, sigma)
    V_minus = discounted_payoff(S0 - eps, sigma)
    V_0 = discounted_payoff(S0, sigma)
    V_sig_plus = discounted_payoff(S0, sigma + eps)
    V_sig_minus = discounted_payoff(S0, sigma - eps)

    return {
        "delta": (V_plus - V_minus) / (2 * eps),
        "gamma": (V_plus - 2 * V_0 + V_minus) / (eps**2),
        "vega": (V_sig_plus - V_sig_minus) / (2 * eps),
    }


def _FDMC_greeks(payoff_fn, params, eps, N, seed, paths, return_stats) -> dict[str, float]:
    return estimate_from_samples(
        lambda W_T: FDMC_samples_european(payoff_fn, params, eps, W_T),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )


# ==============================================================================
# 1. Vanilla European Call Greeks
# ==============================================================================


def FDMC_geeks_european_call(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Delta, Gamma and Vega by central differences (5 evaluations per path, shared random numbers).

    With ``return_stats`` also returns '{greek}_var' (variance of the per-path difference
    quotient) and '{greek}_se'.
    """
    return _FDMC_greeks(payoffs.call, params, eps, N, seed, paths, return_stats)

# ==============================================================================
# 2. Digital European Call Greeks
# ==============================================================================


def FDMC_geeks_european_digital_call(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Delta, Gamma and Vega by central differences (5 evaluations per path, shared random numbers).

    With ``return_stats`` also returns '{greek}_var' (variance of the per-path difference
    quotient) and '{greek}_se'.
    """
    return _FDMC_greeks(payoffs.digital_call, params, eps, N, seed, paths, return_stats)

# ==============================================================================
# 3. Call with Terminal Knock-Out Greeks
# ==============================================================================


def FDMC_geeks_european_barrier_up_and_out_call(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Delta, Gamma and Vega by central differences (5 evaluations per path, shared random numbers).

    With ``return_stats`` also returns '{greek}_var' (variance of the per-path difference
    quotient) and '{greek}_se'.
    """
    return _FDMC_greeks(payoffs.call_terminal_knockout, params, eps, N, seed, paths, return_stats)

