"""Greeks for discrete arithmetic average Asian options using Finite Differences on Moment Matching pricing formulas.

Methodology:
1. Pricing: Exact moments of discrete arithmetic average -> mapped to lognormal parameters -> Black-Scholes formulation.
2. Greeks: Central Finite Differences applied to the closed-form Moment Matching pricing formula:
   - Delta = (V(S0 + eps) - V(S0 - eps)) / (2 * eps)
   - Gamma = (V(S0 + eps) - 2*V(S0) + V(S0 - eps)) / (eps^2)
   - Vega  = (V(sigma + eps) - V(sigma - eps)) / (2 * eps)
"""

from __future__ import annotations
from typing import Any
from pricers.asians_analytic import (
    price_asian_call_moment_matching,
    price_asian_digital_call_moment_matching,
    price_asian_barrier_up_and_out_call_moment_matching,
    price_asian_option_moment_matching,
)


def greeks_asian_call(params: dict[str, Any], eps: float = 1e-5) -> dict[str, float]:
    r"""Computes Delta, Gamma, and Vega for a Vanilla Asian Call using Finite Differences on the MM pricing formula."""
    S0 = float(params["S0"])
    K = float(params["K"])
    T = float(params["T"])
    r = float(params["r"])
    q = float(params["q"])
    sigma = float(params["sigma"])
    freq = int(params.get("frequency", 12))

    # Delta & Gamma via spot perturbation
    V_plus = price_asian_call_moment_matching(S0 + eps, K, T, r, q, sigma, freq)
    V_minus = price_asian_call_moment_matching(S0 - eps, K, T, r, q, sigma, freq)
    V_0 = price_asian_call_moment_matching(S0, K, T, r, q, sigma, freq)

    delta = (V_plus - V_minus) / (2.0 * eps)
    gamma = (V_plus - 2.0 * V_0 + V_minus) / (eps**2)

    # Vega via volatility perturbation
    V_sig_plus = price_asian_call_moment_matching(S0, K, T, r, q, sigma + eps, freq)
    V_sig_minus = price_asian_call_moment_matching(S0, K, T, r, q, sigma - eps, freq)
    vega = (V_sig_plus - V_sig_minus) / (2.0 * eps)

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


def greeks_asian_digital_call(params: dict[str, Any], eps: float = 1e-5) -> dict[str, float]:
    r"""Computes Delta, Gamma, and Vega for a Digital Asian Call using Finite Differences on the MM pricing formula."""
    S0 = float(params["S0"])
    K = float(params["K"])
    T = float(params["T"])
    r = float(params["r"])
    q = float(params["q"])
    sigma = float(params["sigma"])
    freq = int(params.get("frequency", 12))

    V_plus = price_asian_digital_call_moment_matching(S0 + eps, K, T, r, q, sigma, freq)
    V_minus = price_asian_digital_call_moment_matching(S0 - eps, K, T, r, q, sigma, freq)
    V_0 = price_asian_digital_call_moment_matching(S0, K, T, r, q, sigma, freq)

    delta = (V_plus - V_minus) / (2.0 * eps)
    gamma = (V_plus - 2.0 * V_0 + V_minus) / (eps**2)

    V_sig_plus = price_asian_digital_call_moment_matching(S0, K, T, r, q, sigma + eps, freq)
    V_sig_minus = price_asian_digital_call_moment_matching(S0, K, T, r, q, sigma - eps, freq)
    vega = (V_sig_plus - V_sig_minus) / (2.0 * eps)

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


def greeks_asian_barrier_up_and_out_call(params: dict[str, Any], eps: float = 1e-5) -> dict[str, float]:
    r"""Computes Delta, Gamma, and Vega for a Barrier Up-and-Out Asian Call using Finite Differences on the MM pricing formula."""
    S0 = float(params["S0"])
    K = float(params["K"])
    T = float(params["T"])
    r = float(params["r"])
    q = float(params["q"])
    sigma = float(params["sigma"])
    H = float(params.get("H", 120.0))
    freq = int(params.get("frequency", 12))

    V_plus = price_asian_barrier_up_and_out_call_moment_matching(S0 + eps, K, T, r, q, sigma, H, freq)
    V_minus = price_asian_barrier_up_and_out_call_moment_matching(S0 - eps, K, T, r, q, sigma, H, freq)
    V_0 = price_asian_barrier_up_and_out_call_moment_matching(S0, K, T, r, q, sigma, H, freq)

    delta = (V_plus - V_minus) / (2.0 * eps)
    gamma = (V_plus - 2.0 * V_0 + V_minus) / (eps**2)

    V_sig_plus = price_asian_barrier_up_and_out_call_moment_matching(S0, K, T, r, q, sigma + eps, H, freq)
    V_sig_minus = price_asian_barrier_up_and_out_call_moment_matching(S0, K, T, r, q, sigma - eps, H, freq)
    vega = (V_sig_plus - V_sig_minus) / (2.0 * eps)

    return {"delta": float(delta), "gamma": float(gamma), "vega": float(vega)}


def compute_all_asian_moment_matching_greeks(params: dict[str, Any], eps: float = 1e-5) -> dict[str, float]:
    """Convenience dispatcher to compute Greeks for any Asian contract."""
    name = str(params.get("name", "")).lower()
    if "barrier" in name or "h" in [k.lower() for k in params.keys()]:
        return greeks_asian_barrier_up_and_out_call(params, eps)
    elif "digital" in name:
        return greeks_asian_digital_call(params, eps)
    else:
        return greeks_asian_call(params, eps)
