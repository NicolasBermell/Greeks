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

Bandwidths:
-----------
Default per-Greek half-widths delta = c * S0 * sigma * sqrt(T) with c = 0.20 (Delta, Vega) and
0.50 (Gamma) ("Option A"), or a single c = 0.40 for all Greeks ("coupled"). For the knock-out
they are capped at (H - K) / 2. See ``fournier_bandwidths``.

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

from Methods import payoffs
from Methods.Malliavin.malliavin_european_standard import compute_malliavin_weights
from Methods.sampling import GREEKS, estimate_from_samples, simulate_terminal_spot

# Bandwidth = multiplier * S0 * sigma * sqrt(T).
BANDWIDTH_MULTIPLIERS = {"delta": 0.20, "gamma": 0.50, "vega": 0.20}
COUPLED_BANDWIDTH_MULTIPLIER = 0.40

Bandwidth = float | dict[str, float] | None


def get_cutoff(
    S_T: np.ndarray,
    K: float,
    bandwidth: float = 10.0,
    cutoff_type: str = "gaussian",
    return_second_deriv: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluates the localizing cutoff function psi(S_T) and its first/second derivatives.

    Args:
        S_T: Simulated terminal prices.
        K: Strike price (singularity location).
        bandwidth: Half-width delta of the smoothing window around K (delta > 0).
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
    if bandwidth <= 0.0:
        raise ValueError(f"Bandwidth must be positive, got {bandwidth}")
    delta = bandwidth

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


def fournier_bandwidths(params: dict[str, float], coupled: bool = False) -> dict[str, float]:
    """Default Fournié half-width per Greek: multiplier * S0 * sigma * sqrt(T).

    Per-Greek multipliers by default (narrow for Delta and Vega, wide for Gamma), or one
    multiplier for all Greeks if ``coupled``. If ``params`` has a knock-out level H > K, every
    bandwidth is capped at (H - K) / 2 so the bumps around K and H do not overlap.
    """
    scale = params["S0"] * params["sigma"] * np.sqrt(params["T"])
    bandwidths = {
        g: (COUPLED_BANDWIDTH_MULTIPLIER if coupled else BANDWIDTH_MULTIPLIERS[g]) * scale for g in GREEKS
    }
    return _cap_for_knockout(bandwidths, params)


def _cap_for_knockout(bandwidths: dict[str, float], params: dict[str, float]) -> dict[str, float]:
    if "H" in params and params["H"] > params["K"]:
        cap = (params["H"] - params["K"]) / 2.0
        return {g: min(b, cap) for g, b in bandwidths.items()}
    return bandwidths


def _resolve_bandwidths(bandwidth: Bandwidth, params: dict[str, float]) -> tuple[float, float, float]:
    """(Delta, Gamma, Vega) half-widths from a float (all Greeks), a dict (missing Greeks take the
    default) or None (defaults), capped for the knock-out."""
    if bandwidth is None:
        resolved = fournier_bandwidths(params)
    elif isinstance(bandwidth, dict):
        resolved = {**fournier_bandwidths(params), **{g: float(bandwidth[g]) for g in GREEKS if g in bandwidth}}
    else:
        resolved = {g: float(bandwidth) for g in GREEKS}
    resolved = _cap_for_knockout(resolved, params)
    return resolved["delta"], resolved["gamma"], resolved["vega"]


# ==============================================================================
# 1. Generic localized estimator (general Fournié formula)
# ==============================================================================

def _combined_cutoff(
    S_T: np.ndarray,
    centers: tuple[float, ...],
    bandwidth: float,
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
        parts = [get_cutoff(S_T, c, bandwidth=bandwidth, cutoff_type=cutoff_type, return_second_deriv=True) for c in centers]
        return tuple(sum(p[i] for p in parts) for i in range(3))
    if combination != "union":
        raise ValueError(f"Unknown cutoff combination: '{combination}'. Supported: 'union', 'sum'")

    u, u1, u2 = None, None, None  # u = prod (1 - psi_c)
    for c in centers:
        psi_c, psi1_c, psi2_c = get_cutoff(S_T, c, bandwidth=bandwidth, cutoff_type=cutoff_type, return_second_deriv=True)
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
    payoff_derivative_fn,
    centers: tuple[float, ...],
    bandwidths: tuple[float, float, float],
    cutoff_type: str,
    gamma_formula: str = "general",
    cutoff_combination: str = "union",
) -> dict[str, np.ndarray]:
    """Per-path discounted samples of the localized Delta, Gamma and Vega.

    ``payoff_fn`` and ``payoff_derivative_fn`` give Phi and its classical derivative Phi'
    (Phi'' = 0 a.e. for every product here). ``bandwidths`` are the (Delta, Gamma, Vega)
    half-widths; cutoffs are evaluated once per distinct bandwidth.
    """
    S0, T, r, q, sigma = params["S0"], params["T"], params["r"], params["q"], params["sigma"]
    S_T = simulate_terminal_spot(S0, T, r, q, sigma, W_T)
    discount = np.exp(-r * T)
    phi, phi1 = payoff_fn(S_T, params), payoff_derivative_fn(S_T, params)
    w_delta, w_gamma, w_vega = compute_malliavin_weights(S0, T, sigma, W_T)
    x = S_T / S0                      # dS_T / dS0
    pw_vega = S_T * (W_T - sigma * T)  # dS_T / dsigma

    cache: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    psi = {}
    for greek, d in zip(GREEKS, bandwidths):
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


# ==============================================================================
# 2. Per-path samples for each product
# ==============================================================================

def malliavin_fournier_samples_european_call(
    params: dict[str, float],
    W_T: np.ndarray,
    cutoff_type: str = "gaussian",
    bandwidth: Bandwidth = None,
    gamma_formula: str = "general",
) -> dict[str, np.ndarray]:
    return _fournier_samples(
        params, W_T, payoffs.call, payoffs.call_derivative,
        (params["K"],), _resolve_bandwidths(bandwidth, params), cutoff_type, gamma_formula,
    )


def malliavin_fournier_samples_european_digital_call(
    params: dict[str, float],
    W_T: np.ndarray,
    cutoff_type: str = "gaussian",
    bandwidth: Bandwidth = None,
) -> dict[str, np.ndarray]:
    return _fournier_samples(
        params, W_T, payoffs.digital_call, payoffs.digital_call_derivative,
        (params["K"],), _resolve_bandwidths(bandwidth, params), cutoff_type,
    )


def malliavin_fournier_samples_european_call_terminal_knockout(
    params: dict[str, float],
    W_T: np.ndarray,
    cutoff_type: str = "gaussian",
    bandwidth: Bandwidth = None,
    cutoff_combination: str = "union",
) -> dict[str, np.ndarray]:
    return _fournier_samples(
        params, W_T, payoffs.call_terminal_knockout, payoffs.call_terminal_knockout_derivative,
        (params["K"], params["H"]), _resolve_bandwidths(bandwidth, params), cutoff_type,
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
    bandwidth: Bandwidth = None,
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
        bandwidth: Half-width(s) of the cutoff:
            - float: Same bandwidth for all Greeks.
            - dict: Per-Greek bandwidths {'delta': ..., 'gamma': ..., 'vega': ...}; missing ones use the default.
            - None: ``fournier_bandwidths(params)``, i.e. (0.20, 0.50, 0.20) * S0 * sigma * sqrt(T).
        return_stats: If True, also return '{greek}_var' (per-path variance) and '{greek}_se'.
        gamma_formula: 'general' (default) or 'shortcut' (Gamma = (K/S0) * localized digital Delta).

    Returns:
        dict[str, float]: {'delta': ..., 'gamma': ..., 'vega': ...}
    """
    return estimate_from_samples(
        lambda W_T: malliavin_fournier_samples_european_call(params, W_T, cutoff_type, bandwidth, gamma_formula),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )


def malliavin_fournier_greeks_european_digital_call(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    bandwidth: Bandwidth = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Delta, Gamma and Vega of a digital call 1{S_T > K} with Fournié localization around K.

    Same arguments as ``malliavin_fournier_greeks_european_call`` (without ``gamma_formula``).
    """
    return estimate_from_samples(
        lambda W_T: malliavin_fournier_samples_european_digital_call(params, W_T, cutoff_type, bandwidth),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )


def malliavin_fournier_greeks_european_call_terminal_knockout(
    params: dict[str, float],
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    cutoff_type: str = "gaussian",
    bandwidth: Bandwidth = None,
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
            params, W_T, cutoff_type, bandwidth, cutoff_combination
        ),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )
