"""Malliavin Calculus Monte Carlo Greeks for European Options.

This module implements exact Malliavin weights for computing first- and second-order
Greeks (Delta, Gamma, Vega) for European-style options:
1. Vanilla European Call: Payoff = max(S_T - K, 0)
2. Digital European Call: Payoff = 1_{S_T > K}
3. Call with terminal knock-out: Payoff = max(S_T - K, 0) * 1_{S_T <= H}

Mathematical Formulation (Black-Scholes Model):
------------------------------------------------
Under the risk-neutral measure Q:
    S_T = S_0 * exp((r - q - 0.5 * sigma^2) * T + sigma * W_T)
where W_T ~ N(0, T).

Using Malliavin integration by parts (duality between the Malliavin derivative D
and the Skorokhod integral delta), sensitivity derivatives of E[Phi(S_T)] are
transferred entirely onto the underlying Gaussian space as Malliavin weights:

    Delta = e^{-rT} * E[ Phi(S_T) * W_Delta ]
    Gamma = e^{-rT} * E[ Phi(S_T) * W_Gamma ]
    Vega  = e^{-rT} * E[ Phi(S_T) * W_Vega ]

where the exact Malliavin weights are:
    W_Delta = W_T / (S_0 * sigma * T)
    W_Gamma = (W_T^2 - sigma * T * W_T - T) / (S_0^2 * sigma^2 * T^2)
    W_Vega  = (W_T^2 - sigma * T * W_T - T) / (sigma * T) = S_0^2 * sigma * T * W_Gamma

Key Advantages over Finite Difference Monte Carlo (FDMC):
- Zero Truncation Bias: Bias is identically zero (O(eps^2) = 0).
- No Catastrophic Cancellation: Eliminates the 1/eps and 1/eps^2 variance explosion
  and floating-point cancellation for discontinuous payoffs (digital and terminal knock-out).
- Computational Efficiency: Delta, Gamma, and Vega are evaluated simultaneously in a
  single simulation pass (1 path generation, 1 S_T evaluation) without perturbing spots.
"""

from __future__ import annotations

import numpy as np

from Methods import payoffs
from Methods.sampling import estimate_from_samples, generate_paths, simulate_terminal_spot


def compute_malliavin_weights(
    S0: float,
    T: float,
    sigma: float,
    W_T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes exact Malliavin weights for Delta, Gamma, and Vega.

    Args:
        S0 (float): Initial underlying asset price.
        T (float): Time to maturity.
        sigma (float): Annualized volatility.
        W_T (np.ndarray): Brownian motion values W_T ~ N(0, T).

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: (pi_delta, pi_gamma, pi_vega)
    """
    vol_t = sigma * T
    s0_vol_t = S0 * vol_t

    pi_delta = W_T / s0_vol_t
    # Shared numerator for Gamma and Vega weights: W_T^2 - sigma*T*W_T - T
    numerator = W_T**2 - vol_t * W_T - T
    pi_gamma = numerator / (s0_vol_t**2)
    pi_vega = numerator / vol_t

    return pi_delta, pi_gamma, pi_vega


def malliavin_samples_european(payoff_fn, params: dict[str, float], W_T: np.ndarray) -> dict[str, np.ndarray]:
    """Per-path discounted samples e^{-rT} * Phi(S_T) * W_theta for Delta, Gamma and Vega."""
    S0, T, r, q, sigma = params["S0"], params["T"], params["r"], params["q"], params["sigma"]
    S_T = simulate_terminal_spot(S0, T, r, q, sigma, W_T)
    discounted_payoff = np.exp(-r * T) * payoff_fn(S_T, params)
    pi_delta, pi_gamma, pi_vega = compute_malliavin_weights(S0, T, sigma, W_T)
    return {
        "delta": discounted_payoff * pi_delta,
        "gamma": discounted_payoff * pi_gamma,
        "vega": discounted_payoff * pi_vega,
    }


def _malliavin_greeks(payoff_fn, params, N, seed, paths, return_stats) -> dict[str, float]:
    return estimate_from_samples(
        lambda W_T: malliavin_samples_european(payoff_fn, params, W_T),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )


# ==============================================================================
# 1. Vanilla European Call Greeks
# ==============================================================================


def malliavin_greeks_european_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Computes all Greeks (Delta, Gamma, Vega) for a Vanilla European Call jointly.

    All Greeks are computed simultaneously from a single simulation pass.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma.
        N: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        paths: Optional pre-generated Brownian motion increments W_T.
        return_stats: If True, also return '{greek}_var' (per-path variance) and '{greek}_se'.

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    return _malliavin_greeks(payoffs.call, params, N, seed, paths, return_stats)


# ==============================================================================
# 2. Digital European Call Greeks
# ==============================================================================


def malliavin_greeks_european_digital_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Computes all Greeks (Delta, Gamma, Vega) for a Digital Call jointly.

    Payoff is 1_{S_T > K}. Unlike FDMC which collapses or suffers massive variance,
    Malliavin integrates the step function against smooth weights.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma.
        N: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        paths: Optional pre-generated Brownian motion increments W_T.
        return_stats: If True, also return '{greek}_var' (per-path variance) and '{greek}_se'.

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    return _malliavin_greeks(payoffs.digital_call, params, N, seed, paths, return_stats)


# ==============================================================================
# 3. Call with Terminal Knock-Out Greeks
# ==============================================================================


def malliavin_greeks_european_call_terminal_knockout(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Computes all Greeks (Delta, Gamma, Vega) for a call with terminal knock-out jointly.

    Payoff is max(S_T - K, 0) * 1_{S_T <= H}. The jump discontinuity at H causes
    catastrophic finite difference issues in FDMC, but is handled smoothly by Malliavin weights.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma, and H.
        N: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        paths: Optional pre-generated Brownian motion increments W_T.
        return_stats: If True, also return '{greek}_var' (per-path variance) and '{greek}_se'.

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    return _malliavin_greeks(payoffs.call_terminal_knockout, params, N, seed, paths, return_stats)
