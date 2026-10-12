"""European payoffs Phi(S_T) and their classical (almost-everywhere) derivatives Phi'(S_T).

Every estimator (FDMC, standard Malliavin, Fournié) takes its payoff from here, so a product
is defined in one place. The derivatives ignore the Dirac masses at the singularities; the
Fournié estimator relies on its cutoff being 1 with zero slope there (see
malliavin_european_fournier.py). Phi'' = 0 almost everywhere for all three products.
"""

from __future__ import annotations

import numpy as np


def call(S_T: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """(S_T - K)^+"""
    return np.maximum(S_T - params["K"], 0.0)


def call_derivative(S_T: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """1{S_T > K}"""
    return (S_T > params["K"]).astype(float)


def digital_call(S_T: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """1{S_T > K}"""
    return (S_T > params["K"]).astype(float)


def digital_call_derivative(S_T: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """0 almost everywhere (the jump at K is a Dirac mass)."""
    return np.zeros_like(S_T)


def call_terminal_knockout(S_T: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """(S_T - K)^+ 1{S_T <= H}, checked at maturity only."""
    return np.maximum(S_T - params["K"], 0.0) * (S_T <= params["H"])


def call_terminal_knockout_derivative(S_T: np.ndarray, params: dict[str, float]) -> np.ndarray:
    """1{K < S_T <= H} (the jump of -(H - K) at H is a Dirac mass)."""
    return ((S_T > params["K"]) & (S_T <= params["H"])).astype(float)
