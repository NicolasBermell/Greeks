"""Finite Difference Monte Carlo (FDMC) Greek calculations for discrete arithmetic average Asian options.

Uses Common Random Numbers (CRN / CN) where Brownian paths are generated once and reused across
perturbed asset prices (S0 +/- eps) and volatilities (sigma +/- eps) to eliminate Monte Carlo noise
and drastically reduce the variance of sensitivity estimators.
"""

from __future__ import annotations
from typing import Any
import numpy as np

from pricers.asians_MC import (
    generate_asian_paths,
    MC_asian_call_option,
    MC_asian_digital_call_option,
    MC_asian_barrier_up_and_out_call_option,
)


# ==============================================================================
# 1. Vanilla Asian Call Greeks
# ==============================================================================


def _gamma_asian_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    V_plus: float | None = None,
    V_minus: float | None = None,
) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    if V_plus is None:
        V_plus = MC_asian_call_option(S0 + eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    if V_minus is None:
        V_minus = MC_asian_call_option(S0 - eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    V_0 = MC_asian_call_option(S0, K, T, r, q, sigma, freq, seed, N, paths=paths)
    return float((V_plus - 2.0 * V_0 + V_minus) / (eps**2))


def _vega_asian_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = MC_asian_call_option(S0, K, T, r, q, sigma + eps, freq, seed, N, paths=paths)
    V_minus = MC_asian_call_option(S0, K, T, r, q, sigma - eps, freq, seed, N, paths=paths)
    return float((V_plus - V_minus) / (2.0 * eps))


def FDMC_greeks_asian_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = MC_asian_call_option(S0 + eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    V_minus = MC_asian_call_option(S0 - eps, K, T, r, q, sigma, freq, seed, N, paths=paths)

    delta = float((V_plus - V_minus) / (2.0 * eps))
    gamma = _gamma_asian_call(params, eps, N, seed, paths=paths, V_plus=V_plus, V_minus=V_minus)
    vega = _vega_asian_call(params, eps, N, seed, paths=paths)

    return {"delta": delta, "gamma": gamma, "vega": vega}


FDMC_geeks_asian_call = FDMC_greeks_asian_call


# ==============================================================================
# 2. Digital Asian Call Greeks
# ==============================================================================


def _gamma_asian_digital_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    V_plus: float | None = None,
    V_minus: float | None = None,
) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    if V_plus is None:
        V_plus = MC_asian_digital_call_option(S0 + eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    if V_minus is None:
        V_minus = MC_asian_digital_call_option(S0 - eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    V_0 = MC_asian_digital_call_option(S0, K, T, r, q, sigma, freq, seed, N, paths=paths)
    return float((V_plus - 2.0 * V_0 + V_minus) / (eps**2))


def _vega_asian_digital_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = MC_asian_digital_call_option(S0, K, T, r, q, sigma + eps, freq, seed, N, paths=paths)
    V_minus = MC_asian_digital_call_option(S0, K, T, r, q, sigma - eps, freq, seed, N, paths=paths)
    return float((V_plus - V_minus) / (2.0 * eps))


def FDMC_greeks_asian_digital_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = MC_asian_digital_call_option(S0 + eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    V_minus = MC_asian_digital_call_option(S0 - eps, K, T, r, q, sigma, freq, seed, N, paths=paths)

    delta = float((V_plus - V_minus) / (2.0 * eps))
    gamma = _gamma_asian_digital_call(params, eps, N, seed, paths=paths, V_plus=V_plus, V_minus=V_minus)
    vega = _vega_asian_digital_call(params, eps, N, seed, paths=paths)

    return {"delta": delta, "gamma": gamma, "vega": vega}


FDMC_geeks_asian_digital_call = FDMC_greeks_asian_digital_call


# ==============================================================================
# 3. Barrier Up-and-Out Asian Call Greeks
# ==============================================================================


def _gamma_asian_barrier_up_and_out_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    V_plus: float | None = None,
    V_minus: float | None = None,
) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = float(params.get("H", 120.0))
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    if V_plus is None:
        V_plus = MC_asian_barrier_up_and_out_call_option(S0 + eps, K, T, r, q, sigma, H, freq, seed, N, paths=paths)
    if V_minus is None:
        V_minus = MC_asian_barrier_up_and_out_call_option(S0 - eps, K, T, r, q, sigma, H, freq, seed, N, paths=paths)
    V_0 = MC_asian_barrier_up_and_out_call_option(S0, K, T, r, q, sigma, H, freq, seed, N, paths=paths)
    return float((V_plus - 2.0 * V_0 + V_minus) / (eps**2))


def _vega_asian_barrier_up_and_out_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = float(params.get("H", 120.0))
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = MC_asian_barrier_up_and_out_call_option(S0, K, T, r, q, sigma + eps, H, freq, seed, N, paths=paths)
    V_minus = MC_asian_barrier_up_and_out_call_option(S0, K, T, r, q, sigma - eps, H, freq, seed, N, paths=paths)
    return float((V_plus - V_minus) / (2.0 * eps))


def FDMC_greeks_asian_barrier_up_and_out_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = float(params.get("H", 120.0))
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = MC_asian_barrier_up_and_out_call_option(S0 + eps, K, T, r, q, sigma, H, freq, seed, N, paths=paths)
    V_minus = MC_asian_barrier_up_and_out_call_option(S0 - eps, K, T, r, q, sigma, H, freq, seed, N, paths=paths)

    delta = float((V_plus - V_minus) / (2.0 * eps))
    gamma = _gamma_asian_barrier_up_and_out_call(params, eps, N, seed, paths=paths, V_plus=V_plus, V_minus=V_minus)
    vega = _vega_asian_barrier_up_and_out_call(params, eps, N, seed, paths=paths)

    return {"delta": delta, "gamma": gamma, "vega": vega}


FDMC_geeks_asian_barrier_up_and_out_call = FDMC_greeks_asian_barrier_up_and_out_call


def compute_all_asian_FDMC_greeks(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Convenience dispatcher to compute standard FDMC Greeks for any Asian contract."""
    name = str(params.get("name", "")).lower()
    if "barrier" in name or "h" in [k.lower() for k in params.keys()]:
        return FDMC_greeks_asian_barrier_up_and_out_call(params, eps, N, seed, paths)
    elif "digital" in name:
        return FDMC_greeks_asian_digital_call(params, eps, N, seed, paths)
    else:
        return FDMC_greeks_asian_call(params, eps, N, seed, paths)


# ==============================================================================
# 4. Conditional Monte Carlo (CMC) Greeks (Low-Variance & Smooth for Discontinuous Payoffs)
# ==============================================================================

from pricers.asians_MC import (
    conditional_MC_asian_call_option,
    conditional_MC_asian_digital_call_option,
    conditional_MC_asian_barrier_up_and_out_call_option,
)


def FDMC_conditional_greeks_asian_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes Delta, Gamma, Vega for a Vanilla Asian Call using Conditional Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = conditional_MC_asian_call_option(S0 + eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    V_minus = conditional_MC_asian_call_option(S0 - eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    V_0 = conditional_MC_asian_call_option(S0, K, T, r, q, sigma, freq, seed, N, paths=paths)

    delta = float((V_plus - V_minus) / (2.0 * eps))
    gamma = float((V_plus - 2.0 * V_0 + V_minus) / (eps**2))

    V_sig_plus = conditional_MC_asian_call_option(S0, K, T, r, q, sigma + eps, freq, seed, N, paths=paths)
    V_sig_minus = conditional_MC_asian_call_option(S0, K, T, r, q, sigma - eps, freq, seed, N, paths=paths)
    vega = float((V_sig_plus - V_sig_minus) / (2.0 * eps))

    return {"delta": delta, "gamma": gamma, "vega": vega}


def FDMC_conditional_greeks_asian_digital_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes Delta, Gamma, Vega for a Digital Asian Call using Conditional Monte Carlo (smooth Gaussian expectation)."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = conditional_MC_asian_digital_call_option(S0 + eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    V_minus = conditional_MC_asian_digital_call_option(S0 - eps, K, T, r, q, sigma, freq, seed, N, paths=paths)
    V_0 = conditional_MC_asian_digital_call_option(S0, K, T, r, q, sigma, freq, seed, N, paths=paths)

    delta = float((V_plus - V_minus) / (2.0 * eps))
    gamma = float((V_plus - 2.0 * V_0 + V_minus) / (eps**2))

    V_sig_plus = conditional_MC_asian_digital_call_option(S0, K, T, r, q, sigma + eps, freq, seed, N, paths=paths)
    V_sig_minus = conditional_MC_asian_digital_call_option(S0, K, T, r, q, sigma - eps, freq, seed, N, paths=paths)
    vega = float((V_sig_plus - V_sig_minus) / (2.0 * eps))

    return {"delta": delta, "gamma": gamma, "vega": vega}


def FDMC_conditional_greeks_asian_barrier_up_and_out_call(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Computes Delta, Gamma, Vega for a Barrier Asian Call using Conditional Monte Carlo."""
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    H = float(params.get("H", 135.0))
    freq = int(params.get("frequency", 12))
    if paths is None:
        paths = generate_asian_paths(T, freq, N, seed)

    V_plus = conditional_MC_asian_barrier_up_and_out_call_option(S0 + eps, K, T, r, q, sigma, H, freq, seed, N, paths=paths)
    V_minus = conditional_MC_asian_barrier_up_and_out_call_option(S0 - eps, K, T, r, q, sigma, H, freq, seed, N, paths=paths)
    V_0 = conditional_MC_asian_barrier_up_and_out_call_option(S0, K, T, r, q, sigma, H, freq, seed, N, paths=paths)

    delta = float((V_plus - V_minus) / (2.0 * eps))
    gamma = float((V_plus - 2.0 * V_0 + V_minus) / (eps**2))

    V_sig_plus = conditional_MC_asian_barrier_up_and_out_call_option(S0, K, T, r, q, sigma + eps, H, freq, seed, N, paths=paths)
    V_sig_minus = conditional_MC_asian_barrier_up_and_out_call_option(S0, K, T, r, q, sigma - eps, H, freq, seed, N, paths=paths)
    vega = float((V_sig_plus - V_sig_minus) / (2.0 * eps))

    return {"delta": delta, "gamma": gamma, "vega": vega}


def compute_all_asian_conditional_FDMC_greeks(
    params: dict[str, Any],
    eps: float = 0.01,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
) -> dict[str, float]:
    """Convenience dispatcher to compute Conditional FDMC Greeks for any Asian contract."""
    name = str(params.get("name", "")).lower()
    if "barrier" in name or "h" in [k.lower() for k in params.keys()]:
        return FDMC_conditional_greeks_asian_barrier_up_and_out_call(params, eps, N, seed, paths)
    elif "digital" in name:
        return FDMC_conditional_greeks_asian_digital_call(params, eps, N, seed, paths)
    else:
        return FDMC_conditional_greeks_asian_call(params, eps, N, seed, paths)
