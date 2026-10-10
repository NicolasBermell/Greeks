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
Split Phi = Phi * psi + Phi * (1 - psi). The first part uses the Malliavin weight,
the second the pathwise derivative (S_T is linear in S0, and dS_T/dsigma = S_T (W_T - sigma T)):

1. Localized Delta:
   Delta_loc = e^{-rT} * E[ Phi psi W_Delta + (Phi' (1 - psi) - Phi psi') * S_T / S0 ]

2. Localized Vega:
   Vega_loc  = e^{-rT} * E[ Phi psi W_Vega + (Phi' (1 - psi) - Phi psi') * S_T (W_T - sigma T) ]

3. Localized Gamma (general formula, any payoff):
   Gamma_loc = e^{-rT} * E[ Phi psi W_Gamma
                            + (Phi'' (1 - psi) - 2 Phi' psi' - Phi psi'') * (S_T / S0)^2 ]

Phi' and Phi'' are the classical (almost-everywhere) derivatives. The Dirac masses of the
distributional derivatives sit at the singularities, where psi = 1 and psi' = 0, so they drop out.

Products:
   - Call:       Phi' = 1{S_T > K},           Phi'' = 0, psi centered at K.
   - Digital:    Phi' = 0,                    Phi'' = 0, psi centered at K.
   - Call with terminal knock-out, Phi = (S_T - K)^+ 1{S_T <= H}:
                 Phi' = 1{K < S_T <= H},      Phi'' = 0, one cutoff covering both K and H:
                 psi = 1 - (1 - psi_K)(1 - psi_H), which equals psi_K + psi_H when the two
                 bumps do not overlap (compact cutoffs with delta <= (H - K) / 2).
                 The plain sum psi_K + psi_H is available as cutoff_combination="sum" to show
                 its bias with non-compact cutoffs (psi(K) = 1 + psi_H(K) != 1 for the gaussian).

The old call-only Gamma shortcut, Gamma_call = (K / S0) * (localized digital Delta), is kept
as ``gamma_formula="shortcut"`` for comparison. It relies on the call's homogeneity in (S0, K).

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

from Methods.Malliavin.malliavin_european_standard import (
    compute_malliavin_weights,
    generate_paths,
    simulate_terminal_spot,
)
from Methods.sampling import GREEKS, estimate_from_samples


def get_cutoff(
    S_T: np.ndarray,
    K: float,
    delta: float = 10.0,
    cutoff_type: str = "gaussian",
    return_second_deriv: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates the localizing cutoff function psi(S_T) and its first/second derivatives.

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
        return_second_deriv: If True, returns (psi, psi_prime, psi_double_prime).

    Returns:
        tuple: (psi, psi_prime) if return_second_deriv=False else (psi, psi_prime, psi_double_prime)
    """
    if delta <= 0.0:
        raise ValueError(f"Smoothing parameter delta must be positive, got {delta}")

    diff = S_T - K
    u = np.abs(diff) / delta
    sign = np.sign(diff)

    if cutoff_type == "gaussian":
        psi = np.exp(-0.5 * (diff / delta)**2)
        psi_prime = - (diff / (delta**2)) * psi
        psi_double_prime = (-1.0 / delta**2 + (diff**2) / (delta**4)) * psi

    elif cutoff_type == "cubic":
        mask = u <= 1.0
        psi = np.where(mask, 1.0 - 3.0 * u**2 + 2.0 * u**3, 0.0)
        psi_prime = np.where(mask, -sign * (6.0 * u - 6.0 * u**2) / delta, 0.0)
        psi_double_prime = np.where(mask, (-6.0 + 12.0 * u) / delta**2, 0.0)

    elif cutoff_type == "cosine":
        mask = u <= 1.0
        psi = np.where(mask, 0.5 * (1.0 + np.cos(np.pi * u)), 0.0)
        psi_prime = np.where(mask, -sign * (0.5 * np.pi / delta) * np.sin(np.pi * u), 0.0)
        psi_double_prime = np.where(mask, -0.5 * (np.pi / delta)**2 * np.cos(np.pi * u), 0.0)

    elif cutoff_type == "quintic":
        mask = u <= 1.0
        psi = np.where(mask, 1.0 - 10.0 * u**3 + 15.0 * u**4 - 6.0 * u**5, 0.0)
        psi_prime = np.where(mask, -sign * (30.0 * u**2 - 60.0 * u**3 + 30.0 * u**4) / delta, 0.0)
        psi_double_prime = np.where(mask, (-60.0 * u + 180.0 * u**2 - 120.0 * u**3) / delta**2, 0.0)

    elif cutoff_type == "cauchy":
        u2 = u**2
        denom = 1.0 + u2
        psi = 1.0 / denom
        psi_prime = - 2.0 * diff / (delta**2 * denom**2)
        psi_double_prime = (6.0 * u2 - 2.0) / (delta**2 * denom**3)

    else:
        raise ValueError(f"Unknown cutoff_type: '{cutoff_type}'. Supported: 'gaussian', 'cubic', 'cosine', 'quintic', 'cauchy'")

    if return_second_deriv:
        return psi, psi_prime, psi_double_prime
    return psi, psi_prime


def compute_fournier_delta(
    params: dict[str, float],
    is_barrier: bool = False,
    decoupled: bool = False,
    greek: str | None = None,
) -> float | dict[str, float]:
    """Computes the calibrated optimal Fournié delta gap bandwidth.

    Rule of thumb (Unified base):
        delta^* = 0.40 * S_0 * sigma * sqrt(T)
    Option A (Decoupled Per-Greek Bandwidths):
        - Delta: delta^*_delta = 0.20 * S_0 * sigma * sqrt(T)
        - Gamma: delta^*_gamma = 0.50 * S_0 * sigma * sqrt(T)
        - Vega:  delta^*_vega  = 0.20 * S_0 * sigma * sqrt(T)
    For barrier options with corridor width H - K:
        Enforces corridor non-overlap constraint delta <= (H - K) / 2.0.

    Args:
        params: Dictionary containing S0, sigma, T, and optionally H, K.
        is_barrier: Whether the contract is a barrier option.
        decoupled: If True, returns a dictionary with decoupled per-Greek bandwidths.
        greek: Optional specific Greek ('delta', 'gamma', 'vega'). If specified, returns float.

    Returns:
        float | dict[str, float]: Calibrated smoothing half-width delta(s).
    """
    S0 = float(params["S0"])
    sigma = float(params["sigma"])
    T = float(params["T"])
    diffusion_scale = S0 * sigma * np.sqrt(T)

    if greek is not None:
        g = greek.lower()
        multiplier = 0.50 if g == "gamma" else 0.20
        delta_val = multiplier * diffusion_scale
        if is_barrier:
            H = float(params.get("H", 0.0))
            K = float(params.get("K", 0.0))
            if H > K:
                delta_val = min(delta_val, (H - K) / 2.0)
        return float(delta_val)

    if decoupled:
        deltas = {
            "delta": 0.20 * diffusion_scale,
            "gamma": 0.50 * diffusion_scale,
            "vega": 0.20 * diffusion_scale,
        }
        if is_barrier:
            H = float(params.get("H", 0.0))
            K = float(params.get("K", 0.0))
            if H > K:
                corridor_cap = (H - K) / 2.0
                deltas = {k: min(v, corridor_cap) for k, v in deltas.items()}
        return deltas

    base_delta = 0.40 * diffusion_scale
    if is_barrier:
        H = float(params.get("H", 0.0))
        K = float(params.get("K", 0.0))
        if H > K:
            corridor_cap = (H - K) / 2.0
            return float(min(base_delta, corridor_cap))
    return float(base_delta)


def _resolve_deltas(
    delta_strike: float | dict[str, float] | None,
    params: dict[str, float],
    is_barrier: bool = False,
) -> tuple[float, float, float]:
    """Resolves decoupled (delta_d, delta_g, delta_v) from user input or default calibration."""
    if delta_strike is None:
        # Option A decoupled by default
        d_dict = compute_fournier_delta(params, is_barrier=is_barrier, decoupled=True)
        return float(d_dict["delta"]), float(d_dict["gamma"]), float(d_dict["vega"])
    elif isinstance(delta_strike, dict):
        d_dict = compute_fournier_delta(params, is_barrier=is_barrier, decoupled=True)
        return (
            float(delta_strike.get("delta", d_dict["delta"])),
            float(delta_strike.get("gamma", d_dict["gamma"])),
            float(delta_strike.get("vega", d_dict["vega"])),
        )
    else:
        d = float(delta_strike)
        return d, d, d



# ==============================================================================
# 1. Generic localized estimator (general Fournié formula)
# ==============================================================================

def _combined_cutoff(
    S_T: np.ndarray,
    centers: tuple[float, ...],
    delta: float,
    cutoff_type: str,
    combination: str = "union",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One cutoff covering several singularities, with its first two derivatives.

    'union': psi = 1 - prod_c (1 - psi_c). Equals 1 with zero slope at every center, which is
             what the localized formulas need, so the estimator is unbiased for any cutoff type.
    'sum':   psi = sum_c psi_c. Same as 'union' when the bumps do not overlap (compact cutoffs
             with delta <= half the gap). With non-compact cutoffs (gaussian, cauchy) psi != 1 at
             the centers, the Dirac terms no longer vanish and the estimator is biased.
    With one center both reduce to the cutoff around it.
    """
    if combination == "sum":
        parts = [get_cutoff(S_T, c, delta=delta, cutoff_type=cutoff_type, return_second_deriv=True) for c in centers]
        return tuple(sum(p[i] for p in parts) for i in range(3))
    if combination != "union":
        raise ValueError(f"Unknown cutoff combination: '{combination}'. Supported: 'union', 'sum'")

    u, u1, u2 = None, None, None  # u = prod (1 - psi_c)
    for c in centers:
        psi_c, psi1_c, psi2_c = get_cutoff(S_T, c, delta=delta, cutoff_type=cutoff_type, return_second_deriv=True)
        v, v1, v2 = 1.0 - psi_c, -psi1_c, -psi2_c
        if u is None:
            u, u1, u2 = v, v1, v2
        else:
            u, u1, u2 = u * v, u1 * v + u * v1, u2 * v + 2.0 * u1 * v1 + u * v2
    return 1.0 - u, -u1, -u2


def _fournier_samples(
    params: dict[str, float],
    W_T: np.ndarray,
    payoff_fn,
    centers: tuple[float, ...],
    deltas: tuple[float, float, float],
    cutoff_type: str,
    gamma_formula: str = "general",
    cutoff_combination: str = "union",
) -> dict[str, np.ndarray]:
    """Per-path discounted samples of the localized Delta, Gamma and Vega.

    ``payoff_fn(S_T)`` returns (Phi, Phi'), the payoff and its classical derivative
    (Phi'' = 0 a.e. for every product here). ``deltas`` are the (Delta, Gamma, Vega)
    bandwidths; cutoffs are evaluated once per distinct bandwidth.
    """
    S0, T, r, q, sigma = params["S0"], params["T"], params["r"], params["q"], params["sigma"]
    S_T = simulate_terminal_spot(S0, T, r, q, sigma, W_T)
    discount = np.exp(-r * T)
    phi, phi1 = payoff_fn(S_T)
    w_delta, w_gamma, w_vega = compute_malliavin_weights(S0, T, sigma, W_T)
    x = S_T / S0                      # dS_T / dS0
    pw_vega = S_T * (W_T - sigma * T)  # dS_T / dsigma

    cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    psi = {}
    for greek, d in zip(GREEKS, deltas):
        if d not in cache:
            cache[d] = _combined_cutoff(S_T, centers, d, cutoff_type, cutoff_combination)
        psi[greek] = cache[d]

    psi_d, psi1_d, _ = psi["delta"]
    delta = phi * psi_d * w_delta + (phi1 * (1.0 - psi_d) - phi * psi1_d) * x

    psi_v, psi1_v, _ = psi["vega"]
    vega = phi * psi_v * w_vega + (phi1 * (1.0 - psi_v) - phi * psi1_v) * pw_vega

    psi_g, psi1_g, psi2_g = psi["gamma"]
    if gamma_formula == "general":
        gamma = phi * psi_g * w_gamma - (2.0 * phi1 * psi1_g + phi * psi2_g) * x**2
    elif gamma_formula == "shortcut":
        # Call only: Gamma = (K / S0) * localized digital Delta (homogeneity of the call in (S0, K)).
        gamma = (params["K"] / S0) * phi1 * (psi_g * w_delta - psi1_g * x)
    else:
        raise ValueError(f"Unknown gamma_formula: '{gamma_formula}'. Supported: 'general', 'shortcut'")

    return {"delta": discount * delta, "gamma": discount * gamma, "vega": discount * vega}


def _knockout_deltas(
    delta_strike: float | dict[str, float] | None,
    params: dict[str, float],
) -> tuple[float, float, float]:
    """Bandwidths for the knock-out, capped at (H - K) / 2 so the two bumps do not overlap."""
    d_delta, d_gamma, d_vega = _resolve_deltas(delta_strike, params, is_barrier=True)
    corridor = params["H"] - params["K"]
    if corridor > 0:
        cap = corridor / 2.0
        d_delta, d_gamma, d_vega = min(d_delta, cap), min(d_gamma, cap), min(d_vega, cap)
    return d_delta, d_gamma, d_vega


# ==============================================================================
# 2. Per-path samples for each product
# ==============================================================================

def malliavin_fournier_samples_european_call(
    params: dict[str, float],
    W_T: np.ndarray,
    cutoff_type: str = "gaussian",
    delta_strike: float | dict[str, float] | None = None,
    gamma_formula: str = "general",
) -> dict[str, np.ndarray]:
    K = params["K"]
    return _fournier_samples(
        params, W_T,
        lambda S_T: (np.maximum(S_T - K, 0.0), (S_T > K).astype(float)),
        (K,), _resolve_deltas(delta_strike, params), cutoff_type, gamma_formula,
    )


def malliavin_fournier_samples_european_digital_call(
    params: dict[str, float],
    W_T: np.ndarray,
    cutoff_type: str = "gaussian",
    delta_strike: float | dict[str, float] | None = None,
) -> dict[str, np.ndarray]:
    K = params["K"]
    return _fournier_samples(
        params, W_T,
        lambda S_T: ((S_T > K).astype(float), np.zeros_like(S_T)),
        (K,), _resolve_deltas(delta_strike, params), cutoff_type,
    )


def malliavin_fournier_samples_european_call_terminal_knockout(
    params: dict[str, float],
    W_T: np.ndarray,
    cutoff_type: str = "gaussian",
    delta_strike: float | dict[str, float] | None = None,
    cutoff_combination: str = "union",
) -> dict[str, np.ndarray]:
    K, H = params["K"], params["H"]

    def payoff(S_T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        alive = (S_T > K) & (S_T <= H)
        return np.where(alive, S_T - K, 0.0), alive.astype(float)

    return _fournier_samples(
        params, W_T, payoff, (K, H), _knockout_deltas(delta_strike, params), cutoff_type,
        cutoff_combination=cutoff_combination,
    )


# ==============================================================================
# 3. Public estimators (mean, and optionally per-path variance and SE)
# ==============================================================================

def malliavin_fournier_greeks_european_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    delta_strike: float | dict[str, float] | None = None,
    return_stats: bool = False,
    gamma_formula: str = "general",
) -> dict[str, float]:
    """Delta, Gamma and Vega of a European call with Fournié et al. localization around K.

    Args:
        params: Dictionary containing S0, K, T, r, q, sigma.
        N: Number of Monte Carlo simulations.
        seed: Seed for ``np.random.default_rng``.
        paths: Optional pre-generated Brownian motion values W_T.
        cutoff_type: Smoothing function ('gaussian', 'cubic', 'cosine', 'quintic', 'cauchy').
        delta_strike: Half-width(s) around strike K:
            - float: Shared bandwidth across all Greeks.
            - dict: Decoupled per-Greek bandwidths {'delta': ..., 'gamma': ..., 'vega': ...}.
            - None: Defaults to decoupled calibration (0.20, 0.50, 0.20) * S0 * sigma * sqrt(T).
        return_stats: If True, also return '{greek}_var' (per-path variance) and '{greek}_se'.
        gamma_formula: 'general' (default) or 'shortcut' (Gamma = (K/S0) * localized digital Delta).

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    return estimate_from_samples(
        lambda W_T: malliavin_fournier_samples_european_call(params, W_T, cutoff_type, delta_strike, gamma_formula),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )


def malliavin_fournier_greeks_european_digital_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    delta_strike: float | dict[str, float] | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Delta, Gamma and Vega of a digital call 1{S_T > K} with Fournié localization around K.

    Same arguments as ``malliavin_fournier_greeks_european_call`` (without ``gamma_formula``).
    """
    return estimate_from_samples(
        lambda W_T: malliavin_fournier_samples_european_digital_call(params, W_T, cutoff_type, delta_strike),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )


def malliavin_fournier_greeks_european_barrier_up_and_out_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    delta_strike: float | dict[str, float] | None = None,
    return_stats: bool = False,
    cutoff_combination: str = "union",
) -> dict[str, float]:
    """Delta, Gamma and Vega of a call with terminal knock-out, (S_T - K)^+ 1{S_T <= H}.

    Localizes directly with one cutoff covering K and H (no call - call - digital split).
    Bandwidths are capped at (H - K) / 2. Same arguments as the call version, with ``params``
    also containing H, plus ``cutoff_combination``: 'union' (default, psi = 1 - (1 - psi_K)(1 - psi_H),
    unbiased) or 'sum' (psi = psi_K + psi_H, biased for non-compact cutoffs such as the gaussian).
    """
    return estimate_from_samples(
        lambda W_T: malliavin_fournier_samples_european_call_terminal_knockout(
            params, W_T, cutoff_type, delta_strike, cutoff_combination
        ),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )


# ==============================================================================
# 4. Multi-Product Suite & Variance Diagnostic Tools
# ==============================================================================


def malliavin_fournier_all_greeks_european(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    delta_strike: float | dict[str, float] | None = None,
    barrier_delta: float | dict[str, float] | None = None,
) -> dict[str, dict[str, float]]:
    """Calculates all Greeks for the call, digital call and call with terminal knock-out jointly."""
    if paths is None:
        paths = generate_paths(params["T"], N, seed)

    if delta_strike is None:
        delta_strike = compute_fournier_delta(params, is_barrier=False)
    if barrier_delta is None:
        barrier_delta = compute_fournier_delta(params, is_barrier=True)

    return {
        "call": malliavin_fournier_greeks_european_call(
            params, N=N, seed=seed, paths=paths, cutoff_type=cutoff_type, delta_strike=delta_strike
        ),
        "digital": malliavin_fournier_greeks_european_digital_call(
            params, N=N, seed=seed, paths=paths, cutoff_type=cutoff_type, delta_strike=delta_strike
        ),
        "call_terminal_knockout": malliavin_fournier_greeks_european_barrier_up_and_out_call(
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
