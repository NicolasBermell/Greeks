"""Fournié et al. (1999) Localized Malliavin Calculus Monte Carlo Greeks for European Options.

Reference:
    Fournié, E., Lasry, J. M., Lebuchoux, J., Lions, P. L., & Touzi, N. (1999).
    Applications of Malliavin calculus to Monte Carlo methods in finance.
    Finance and Stochastics, 3(4), 391-412. (Theorem 2.3 & Section 4).

Mathematical Formulation:
-------------------------
Standard Malliavin weights transfer derivatives from the payoff onto the Brownian motion:
    pi^Delta = W_T / (S_0 * sigma * T)
    pi^Gamma = (W_T^2 - sigma * T * W_T - T) / (S_0^2 * sigma^2 * T^2)
    pi^Vega  = (W_T^2 - sigma * T * W_T - T) / (sigma * T)

For smooth payoffs (e.g. Vanilla Call), these polynomial weights (specifically W_T^2)
introduce substantial sample variance compared to the pathwise derivative (IPA).

Fournié et al. (Theorem 2.3) introduced a localizing function psi(S_T) centered around
the singularity (strike K) with half-width delta:
    - Inside [K - delta, K + delta]: psi(S_T) ~ 1. Uses Malliavin integration by parts (IBP).
    - Outside [K - delta, K + delta]: psi(S_T) ~ 0. Uses the low-variance Pathwise Derivative (IPA).
    - Transition boundary: Corrected by the boundary product-rule derivative -psi'(S_T).

Exact Localized Formulas:
-------------------------
1. Localized Delta:
   Delta_loc = e^{-rT} * E[ Phi(S_T) * (psi * pi^Delta - psi' * S_T / S0)
                          + Phi'(S_T) * (1 - psi) * S_T / S0 ]

2. Localized Vega:
   Vega_loc  = e^{-rT} * E[ Phi(S_T) * (psi * pi^Vega - psi' * S_T * (W_T - sigma*T))
                          + Phi'(S_T) * (1 - psi) * S_T * (W_T - sigma*T) ]

3. Localized Gamma:
   Gamma_loc = e^{-rT} * E[ Phi'(S_T) * (psi * pi^Delta - psi' * S_T / S0)
                          + Phi''(S_T) * (1 - psi) * (S_T / S0)^2 ]
   (For Call options, Phi''(S_T) = delta(S_T - K), so the outer term vanishes).

Supported Cutoff Smoothing Functions:
-------------------------------------
- 'gaussian': psi(S_T) = exp(-0.5 * ((S_T - K) / delta)^2)
- 'cubic': Compact Hermite C^1 smoothstep: 1 - 3u^2 + 2u^3 for u = |S_T - K| / delta <= 1
- 'cosine': Compact C^1 trigonometric cutoff: 0.5 * (1 + cos(pi * u))
- 'quintic': Compact C^2 smoothstep: 1 - 10u^3 + 15u^4 - 6u^5
- 'cauchy': Rational Lorentzian cutoff: 1 / (1 + u^2)
"""

from __future__ import annotations

import numpy as np


def generate_paths(T: float, N: int, seed: int = 42) -> np.ndarray:
    """Generates standard Brownian motion increments W_T ~ N(0, T) for N simulations."""
    np.random.seed(seed)
    return np.random.normal(0.0, np.sqrt(T), N)


def simulate_terminal_spot(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    W_T: np.ndarray,
) -> np.ndarray:
    """Simulates terminal asset prices S_T under the Black-Scholes model."""
    drift = (r - q - 0.5 * sigma**2) * T
    return S0 * np.exp(drift + sigma * W_T)


def compute_malliavin_weights(
    S0: float,
    T: float,
    sigma: float,
    W_T: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes exact standard Malliavin weights for Delta, Gamma, and Vega."""
    vol_t = sigma * T
    s0_vol_t = S0 * vol_t

    pi_delta = W_T / s0_vol_t
    numerator = W_T**2 - vol_t * W_T - T
    pi_gamma = numerator / (s0_vol_t**2)
    pi_vega = numerator / vol_t

    return pi_delta, pi_gamma, pi_vega


def get_cutoff(
    S_T: np.ndarray,
    K: float,
    delta: float = 10.0,
    cutoff_type: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the localizing cutoff function psi(S_T) and its first derivative psi'(S_T).

    Args:
        S_T: Simulated terminal prices.
        K: Strike price (singularity location).
        delta: Half-width / characteristic scale of the smoothing window around K (delta > 0).
        cutoff_type: Type of smooth function:
            - 'gaussian': C^inf exponential Gaussian bump
            - 'cubic': C^1 compact Hermite smoothstep spline
            - 'cosine': C^1 compact trigonometric cutoff
            - 'quintic': C^2 compact quintic smoothstep spline
            - 'cauchy': Rational Lorentzian cutoff

    Returns:
        tuple[np.ndarray, np.ndarray]: (psi, psi_prime)
    """
    if delta <= 0.0:
        raise ValueError(f"Smoothing parameter delta must be positive, got {delta}")

    diff = S_T - K
    u = np.abs(diff) / delta
    sign = np.sign(diff)

    if cutoff_type == "gaussian":
        psi = np.exp(-0.5 * (diff / delta)**2)
        psi_prime = - (diff / (delta**2)) * psi

    elif cutoff_type == "cubic":
        mask = u <= 1.0
        psi = np.where(mask, 1.0 - 3.0 * u**2 + 2.0 * u**3, 0.0)
        psi_prime = np.where(mask, -sign * (6.0 * u - 6.0 * u**2) / delta, 0.0)

    elif cutoff_type == "cosine":
        mask = u <= 1.0
        psi = np.where(mask, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)
        psi_prime = np.where(mask, -sign * (0.5 * np.pi / delta) * np.sin(np.pi * u), 0.0)

    elif cutoff_type == "quintic":
        mask = u <= 1.0
        psi = np.where(mask, 1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5, 0.0)
        psi_prime = np.where(mask, -sign * (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / delta, 0.0)

    elif cutoff_type == "cauchy":
        denom = 1.0 + u**2
        psi = 1.0 / denom
        psi_prime = - 2.0 * diff / (delta**2 * denom**2)

    else:
        raise ValueError(f"Unknown cutoff_type: '{cutoff_type}'. Supported: 'gaussian', 'cubic', 'cosine', 'quintic', 'cauchy'")

    return psi, psi_prime


# ==============================================================================
# 1. Vanilla European Call: Fournié Localized Greeks
# ==============================================================================

def malliavin_fournier_greeks_european_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    delta_strike: float = 8.0,
) -> dict[str, float]:
    """Computes Delta, Gamma, and Vega of a European Call using Fournié et al. localization.

    Eliminates the high variance of standard Malliavin weights on smooth payoffs by
    applying Malliavin IBP only in the localized band [K - delta, K + delta], and the
    pathwise derivative (IPA) outside.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma.
        N: Number of Monte Carlo simulations.
        seed: Random seed for reproducibility.
        paths: Optional pre-generated Brownian motion increments W_T.
        cutoff_type: Smoothing function ('gaussian', 'cubic', 'cosine', 'quintic', 'cauchy').
        delta_strike: Half-width of smoothing window around the strike K (calibrated optimal: 0.40 * S0 * sigma * sqrt(T)).

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    payoff = np.maximum(S_T - K, 0.0)
    payoff_prime = (S_T > K).astype(float)
    discount = np.exp(-r * T)

    pi_delta, _, pi_vega = compute_malliavin_weights(S0, T, sigma, paths)
    psi, psi_prime = get_cutoff(S_T, K, delta=delta_strike, cutoff_type=cutoff_type)

    # 1. Localized Delta:
    # Inner: Phi * (psi * pi_delta - psi' * S_T / S0)
    # Outer: Phi' * (1 - psi) * S_T / S0
    delta_inner = payoff * (psi * pi_delta - psi_prime * (S_T / S0))
    delta_outer = payoff_prime * (1.0 - psi) * (S_T / S0)
    delta_samples = delta_inner + delta_outer

    # 2. Localized Vega:
    # Pathwise vega weight: S_T * (W_T - sigma * T)
    pw_vega = S_T * (paths - sigma * T)
    vega_inner = payoff * (psi * pi_vega - psi_prime * pw_vega)
    vega_outer = payoff_prime * (1.0 - psi) * pw_vega
    vega_samples = vega_inner + vega_outer

    # 3. Localized Gamma:
    # Scale by (K / S0) because Gamma_call = (K / S0) * Delta_digital
    gamma_samples = (K / S0) * payoff_prime * (psi * pi_delta - psi_prime * (S_T / S0))

    return {
        "delta": float(discount * np.mean(delta_samples)),
        "gamma": float(discount * np.mean(gamma_samples)),
        "vega": float(discount * np.mean(vega_samples)),
    }


# ==============================================================================
# 2. Digital European Call: Fournié Localized Greeks
# ==============================================================================

def malliavin_fournier_greeks_european_digital_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    delta_strike: float = 8.0,
) -> dict[str, float]:
    """Computes Delta, Gamma, and Vega of a Digital Call using Fournié et al. localization.

    Payoff is 1_{S_T > K}. Differentiable derivative is 0 everywhere except K, so
    the outer pathwise term is zero, and localization smooths the Malliavin integration.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma.
        N: Number of simulations.
        seed: Random seed.
        paths: Optional pre-generated paths.
        cutoff_type: Smoothing function.
        delta_strike: Smoothing width around K.

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
    psi, psi_prime = get_cutoff(S_T, K, delta=delta_strike, cutoff_type=cutoff_type)

    # Localized sensitivities for digital call
    pw_vega = S_T * (paths - sigma * T)
    delta_samples = payoff * (psi * pi_delta - psi_prime * (S_T / S0))
    vega_samples = payoff * (psi * pi_vega - psi_prime * pw_vega)
    # Exact Skorokhod duality weight for the indicator step jump
    gamma_samples = payoff * pi_gamma

    return {
        "delta": float(discount * np.mean(delta_samples)),
        "gamma": float(discount * np.mean(gamma_samples)),
        "vega": float(discount * np.mean(vega_samples)),
    }


# ==============================================================================
# 3. Terminal Barrier Up-and-Out Call: Fournié Localized Greeks
# ==============================================================================

def malliavin_fournier_greeks_european_barrier_up_and_out_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    delta_strike: float = 2.0,
) -> dict[str, float]:
    """Computes Greeks of a Terminal Barrier Up-and-Out Call using Fournié localization.

    Payoff is max(S_T - K, 0) * 1_{S_T <= H}.
    Decomposes into two continuous calls and an indicator step jump:
        Payoff = (S_T - K)^+ - (S_T - H)^+ - (H - K) * 1_{S_T > H}
    This resolves the boundary jump at H without destructive window overlap or missing mass.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma, and H.
        N: Number of simulations.
        seed: Random seed.
        paths: Optional pre-generated paths.
        cutoff_type: Smoothing function.
        delta_strike: Smoothing width around boundaries (constrained by corridor (H-K)/2).

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = params.get("H", 105.25)
    if paths is None:
        paths = generate_paths(T, N, seed)

    S_T = simulate_terminal_spot(S0, T, r, q, sigma, paths)
    discount = np.exp(-r * T)

    # Corridor constraint: enforce delta <= (H - K) / 2.5 to prevent destructive window overlap
    corridor = H - K
    effective_delta = min(delta_strike, corridor / 2.5) if corridor > 0 else delta_strike

    pi_delta, pi_gamma, pi_vega = compute_malliavin_weights(S0, T, sigma, paths)
    pw_vega = S_T * (paths - sigma * T)

    # 1. Lower Strike K: Fournié localized call
    psi_k, psi_k_prime = get_cutoff(S_T, K, delta=effective_delta, cutoff_type=cutoff_type)
    payoff_k = np.maximum(S_T - K, 0.0)
    prime_k = (S_T > K).astype(float)
    d_k = payoff_k * (psi_k * pi_delta - psi_k_prime * (S_T / S0)) + prime_k * (1.0 - psi_k) * (S_T / S0)
    v_k = payoff_k * (psi_k * pi_vega - psi_k_prime * pw_vega) + prime_k * (1.0 - psi_k) * pw_vega
    g_k = (K / S0) * prime_k * (psi_k * pi_delta - psi_k_prime * (S_T / S0))

    # 2. Upper Barrier H: Fournié localized call
    psi_h, psi_h_prime = get_cutoff(S_T, H, delta=effective_delta, cutoff_type=cutoff_type)
    payoff_h = np.maximum(S_T - H, 0.0)
    prime_h = (S_T > H).astype(float)
    d_h = payoff_h * (psi_h * pi_delta - psi_h_prime * (S_T / S0)) + prime_h * (1.0 - psi_h) * (S_T / S0)
    v_h = payoff_h * (psi_h * pi_vega - psi_h_prime * pw_vega) + prime_h * (1.0 - psi_h) * pw_vega
    g_h = (H / S0) * prime_h * (psi_h * pi_delta - psi_h_prime * (S_T / S0))

    # 3. Boundary Step Jump at H: Exact Skorokhod duality for indicator -(H - K) * 1_{S_T > H}
    dig_h = (S_T > H).astype(float)
    d_dig = dig_h * pi_delta
    v_dig = dig_h * pi_vega
    g_dig = dig_h * pi_gamma

    delta_samples = d_k - d_h - (H - K) * d_dig
    gamma_samples = g_k - g_h - (H - K) * g_dig
    vega_samples = v_k - v_h - (H - K) * v_dig

    return {
        "delta": float(discount * np.mean(delta_samples)),
        "gamma": float(discount * np.mean(gamma_samples)),
        "vega": float(discount * np.mean(vega_samples)),
    }


# ==============================================================================
# 4. Multi-Product Suite & Variance Diagnostic Tools
# ==============================================================================

def malliavin_fournier_all_greeks_european(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    delta_strike: float = 8.0,
    barrier_delta: float = 2.0,
) -> dict[str, dict[str, float]]:
    """Calculates all Greeks for Call, Digital Call, and Barrier Up-and-Out options jointly."""
    if paths is None:
        paths = generate_paths(params["T"], N, seed)

    return {
        "call": malliavin_fournier_greeks_european_call(
            params, N=N, seed=seed, paths=paths, cutoff_type=cutoff_type, delta_strike=delta_strike
        ),
        "digital": malliavin_fournier_greeks_european_digital_call(
            params, N=N, seed=seed, paths=paths, cutoff_type=cutoff_type, delta_strike=delta_strike
        ),
        "barrier": malliavin_fournier_greeks_european_barrier_up_and_out_call(
            params, N=N, seed=seed, paths=paths, cutoff_type=cutoff_type, delta_strike=barrier_delta
        ),
    }


def compare_cutoff_functions(
    params: dict[str, float],
    N: int = 100000,
    delta_strike: float = 10.0,
    seed: int = 42,
    cutoff_types: list[str] = ["gaussian", "cubic", "cosine", "quintic", "cauchy"],
) -> dict[str, dict[str, float]]:
    """Compares Greek values across different cutoff smoothing functions for the same paths."""
    paths = generate_paths(params["T"], N, seed)
    results = {}
    for ctype in cutoff_types:
        results[ctype] = malliavin_fournier_greeks_european_call(
            params, N=N, seed=seed, paths=paths, cutoff_type=ctype, delta_strike=delta_strike
        )
    return results


def compare_delta_strike_sensitivities(
    params: dict[str, float],
    N: int = 100000,
    deltas: list[float] = [1.0, 5.0, 10.0, 20.0, 30.0],
    cutoff_type: str = "gaussian",
    seed: int = 42,
) -> dict[float, dict[str, float]]:
    """Compares Greek values across different smoothing half-widths delta around strike."""
    paths = generate_paths(params["T"], N, seed)
    results = {}
    for d in deltas:
        results[d] = malliavin_fournier_greeks_european_call(
            params, N=N, seed=seed, paths=paths, cutoff_type=cutoff_type, delta_strike=d
        )
    return results


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
    print("Fournié Localized Malliavin Greeks Test (N=100,000, delta=10.0, gaussian):")
    res = malliavin_fournier_all_greeks_european(test_params, N=100000)
    for prod, greeks in res.items():
        print(f"  {prod}: {greeks}")

    print("\nComparing Cutoff Functions for Call European ATM (N=100,000):")
    cutoff_res = compare_cutoff_functions(test_params, N=100000)
    for ctype, greeks in cutoff_res.items():
        print(f"  Cutoff '{ctype}': {greeks}")

    print("\nComparing Smoothing Half-Widths delta for Call European ATM (N=100,000):")
    delta_res = compare_delta_strike_sensitivities(test_params, N=100000)
    for d, greeks in delta_res.items():
        print(f"  delta={d:4.1f}: {greeks}")
