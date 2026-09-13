"""Malliavin Calculus Monte Carlo Greeks for European Options.

This module implements exact Malliavin weights for computing first- and second-order
Greeks (Delta, Gamma, Vega) for European-style options:
1. Vanilla European Call: Payoff = max(S_T - K, 0)
2. Digital European Call: Payoff = 1_{S_T > K}
3. Terminal Barrier Up-and-Out Call: Payoff = max(S_T - K, 0) * 1_{S_T <= H}

Mathematical Formulation (Black-Scholes Model):
------------------------------------------------
Under the risk-neutral measure Q:
    S_T = S_0 * exp((r - q - 0.5 * sigma^2) * T + sigma * W_T)
where W_T ~ N(0, T).

Using Malliavin integration by parts (duality between the Malliavin derivative D
and the Skorokhod integral delta), sensitivity derivatives of E[Phi(S_T)] are
transferred entirely onto the underlying Gaussian space as Malliavin weights:

    Delta = e^{-rT} * E[ Phi(S_T) * pi^Delta ]
    Gamma = e^{-rT} * E[ Phi(S_T) * pi^Gamma ]
    Vega  = e^{-rT} * E[ Phi(S_T) * pi^Vega ]

where the exact Malliavin weights are:
    pi^Delta = W_T / (S_0 * sigma * T)
    pi^Gamma = (W_T^2 - sigma * T * W_T - T) / (S_0^2 * sigma^2 * T^2)
    pi^Vega  = (W_T^2 - sigma * T * W_T - T) / (sigma * T) = S_0^2 * sigma * T * pi^Gamma

Key Advantages over Finite Difference Monte Carlo (FDMC):
- Zero Truncation Bias: Bias is identically zero (O(eps^2) = 0).
- No Catastrophic Cancellation: Eliminates the 1/eps and 1/eps^2 variance explosion
  and floating-point cancellation for discontinuous payoffs (Digital and Barrier options).
- Computational Efficiency: Delta, Gamma, and Vega are evaluated simultaneously in a
  single simulation pass (1 path generation, 1 S_T evaluation) without perturbing spots.
"""

from __future__ import annotations

import numpy as np


def generate_paths(T: float, N: int, seed: int = 42) -> np.ndarray:
    """Generates standard Brownian motion increments W_T ~ N(0, T) for N simulations.

    Generating W_T once ensures Common Random Numbers (CRN) are preserved across
    all sensitivities (Delta, Gamma, Vega) without redundant random number generation.

    Args:
        T (float): Time to maturity in years (T > 0).
        N (int): Number of Monte Carlo simulations.
        seed (int): Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (N,) containing Brownian increments W_T.
    """
    np.random.seed(seed)
    return np.random.normal(0.0, np.sqrt(T), N)


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


def simulate_terminal_spot(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    W_T: np.ndarray,
) -> np.ndarray:
    """Simulates terminal asset prices S_T under the Black-Scholes model.

    Args:
        S0 (float): Initial underlying asset price.
        T (float): Time to maturity.
        r (float): Risk-free interest rate.
        q (float): Dividend yield.
        sigma (float): Annualized volatility.
        W_T (np.ndarray): Brownian motion values W_T ~ N(0, T).

    Returns:
        np.ndarray: Simulated terminal prices S_T.
    """
    drift = (r - q - 0.5 * sigma**2) * T
    return S0 * np.exp(drift + sigma * W_T)


# ==============================================================================
# 1. Vanilla European Call Greeks
# ==============================================================================

def _delta_malliavin_european_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Delta of a European call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    pi_delta = paths / (S0 * sigma * T)
    return float(np.exp(-r * T) * np.mean(payoff * pi_delta))


def _gamma_malliavin_european_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Gamma of a European call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    numerator = paths**2 - sigma * T * paths - T
    pi_gamma = numerator / ((S0 * sigma * T)**2)
    return float(np.exp(-r * T) * np.mean(payoff * pi_gamma))


def _vega_malliavin_european_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Vega of a European call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    numerator = paths**2 - sigma * T * paths - T
    pi_vega = numerator / (sigma * T)
    return float(np.exp(-r * T) * np.mean(payoff * pi_vega))


def malliavin_greeks_european_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes all Greeks (Delta, Gamma, Vega) for a Vanilla European Call jointly.

    All Greeks are computed simultaneously from a single simulation pass.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma.
        N: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        paths: Optional pre-generated Brownian motion increments W_T.

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    discount = np.exp(-r * T)

    pi_delta, pi_gamma, pi_vega = compute_malliavin_weights(S0, T, sigma, paths)

    return {
        "delta": float(discount * np.mean(payoff * pi_delta)),
        "gamma": float(discount * np.mean(payoff * pi_gamma)),
        "vega": float(discount * np.mean(payoff * pi_vega)),
    }


# ==============================================================================
# 2. Digital European Call Greeks
# ==============================================================================

def _delta_malliavin_european_digital_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Delta of a European digital call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = (S_T > K).astype(float)
    pi_delta = paths / (S0 * sigma * T)
    return float(np.exp(-r * T) * np.mean(payoff * pi_delta))


def _gamma_malliavin_european_digital_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Gamma of a European digital call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = (S_T > K).astype(float)
    numerator = paths**2 - sigma * T * paths - T
    pi_gamma = numerator / ((S0 * sigma * T)**2)
    return float(np.exp(-r * T) * np.mean(payoff * pi_gamma))


def _vega_malliavin_european_digital_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Vega of a European digital call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = (S_T > K).astype(float)
    numerator = paths**2 - sigma * T * paths - T
    pi_vega = numerator / (sigma * T)
    return float(np.exp(-r * T) * np.mean(payoff * pi_vega))


def malliavin_greeks_european_digital_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes all Greeks (Delta, Gamma, Vega) for a Digital Call jointly.

    Payoff is 1_{S_T > K}. Unlike FDMC which collapses or suffers massive variance,
    Malliavin integrates the step function against smooth weights.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma.
        N: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        paths: Optional pre-generated Brownian motion increments W_T.

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = (S_T > K).astype(float)
    discount = np.exp(-r * T)

    pi_delta, pi_gamma, pi_vega = compute_malliavin_weights(S0, T, sigma, paths)

    return {
        "delta": float(discount * np.mean(payoff * pi_delta)),
        "gamma": float(discount * np.mean(payoff * pi_gamma)),
        "vega": float(discount * np.mean(payoff * pi_vega)),
    }


# ==============================================================================
# 3. Barrier Option (Up-and-Out Call) Greeks
# ==============================================================================

def _delta_malliavin_european_barrier_up_and_out_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Delta of a European barrier up-and-out call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = params.get("H", 105.25)
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    payoff[S_T > H] = 0.0
    pi_delta = paths / (S0 * sigma * T)
    return float(np.exp(-r * T) * np.mean(payoff * pi_delta))


def _gamma_malliavin_european_barrier_up_and_out_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Gamma of a European barrier up-and-out call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = params.get("H", 105.25)
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    payoff[S_T > H] = 0.0
    numerator = paths**2 - sigma * T * paths - T
    pi_gamma = numerator / ((S0 * sigma * T)**2)
    return float(np.exp(-r * T) * np.mean(payoff * pi_gamma))


def _vega_malliavin_european_barrier_up_and_out_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    """Computes Vega of a European barrier up-and-out call using Malliavin Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = params.get("H", 105.25)
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    payoff[S_T > H] = 0.0
    numerator = paths**2 - sigma * T * paths - T
    pi_vega = numerator / (sigma * T)
    return float(np.exp(-r * T) * np.mean(payoff * pi_vega))


def malliavin_greeks_european_barrier_up_and_out_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes all Greeks (Delta, Gamma, Vega) for a Terminal Barrier Up-and-Out Call jointly.

    Payoff is max(S_T - K, 0) * 1_{S_T <= H}. The jump discontinuity at H causes
    catastrophic finite difference issues in FDMC, but is handled smoothly by Malliavin weights.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma, and H.
        N: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        paths: Optional pre-generated Brownian motion increments W_T.

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = params.get("H", 105.25)
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    payoff[S_T > H] = 0.0
    discount = np.exp(-r * T)

    pi_delta, pi_gamma, pi_vega = compute_malliavin_weights(S0, T, sigma, paths)

    return {
        "delta": float(discount * np.mean(payoff * pi_delta)),
        "gamma": float(discount * np.mean(payoff * pi_gamma)),
        "vega": float(discount * np.mean(payoff * pi_vega)),
    }


# ==============================================================================
# 4. Multi-Product Greeks Suite
# ==============================================================================

def malliavin_all_greeks_european(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    """Calculates all Greeks for Call, Digital Call, and Barrier Up-and-Out options.

    Generates paths W_T once at the top level so that all 3 products and their
    respective Greeks share the exact same underlying simulation paths (Common Random Numbers).

    Args:
        params: Dictionary containing option parameters (S0, K, T, r, q, sigma, optional H).
        N: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        paths: Optional pre-generated Brownian motion increments W_T.

    Returns:
        dict[str, dict[str, float]]: Dictionary mapping product key to its Greeks dict.
    """
    if paths is None:
        paths = generate_paths(params["T"], N, seed)

    return {
        "call": malliavin_greeks_european_call(params, N=N, seed=seed, paths=paths),
        "digital": malliavin_greeks_european_digital_call(params, N=N, seed=seed, paths=paths),
        "barrier": malliavin_greeks_european_barrier_up_and_out_call(params, N=N, seed=seed, paths=paths),
    }


if __name__ == "__main__":
    test_params = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.02,
        "sigma": 0.20,
        "H": 105.25,
    }
    print("Malliavin All Greeks test (N=100,000):")
    results = malliavin_all_greeks_european(test_params, N=100000)
    for prod, greeks in results.items():
        print(f"{prod}: {greeks}")
