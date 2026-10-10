"""Analytical pricing module for discrete arithmetic average Asian options.

Approximates the distribution of the discrete arithmetic average A = (1/m) sum_{i=1}^m S_{t_i}
by matching the first two moments (M1 = E[A], M2 = E[A^2]) to a lognormal distribution,
yielding Black-Scholes-like closed-form formulas for:
1. Vanilla Asian Call: E[e^{-rT} (A - K)^+]
2. Digital Asian Call: E[e^{-rT} 1_{A > K}]
3. Barrier Asian Up-and-Out Call: E[e^{-rT} (A - K)^+ 1_{A <= H}]
"""

from __future__ import annotations
from typing import Any
import numpy as np
from scipy.stats import norm


def compute_discrete_asian_moments(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
) -> tuple[float, float, float, float]:
    r"""Computes the exact first and second moments of the discrete arithmetic average.

    For observation dates $t_i = i \Delta t$ ($i=1, \dots, m$ with $\Delta t = T / m$):
        $A = \frac{1}{m} \sum_{i=1}^m S_{t_i}$
        $M_1 = \mathbb{E}[A] = \frac{S_0}{m} \sum_{i=1}^m e^{(r-q)t_i}$
        $\mathbb{E}[S_{t_i} S_{t_j}] = S_0^2 \exp\left( (r - q)(t_i + t_j) + \sigma^2 \min(t_i, t_j) \right)$
        $M_2 = \mathbb{E}[A^2] = \frac{1}{m^2} \sum_{i=1}^m \sum_{j=1}^m \mathbb{E}[S_{t_i} S_{t_j}]$

    Then maps to lognormal parameters:
        $v_A^2 = \ln(M_2 / M_1^2)$
        $\mu_A = \ln(M_1) - \frac{1}{2} v_A^2$

    Args:
        S0: Initial spot price ($S_0 > 0$).
        T: Time to maturity in years ($T > 0$).
        r: Annualized risk-free rate.
        q: Annualized dividend yield.
        sigma: Annualized volatility ($\sigma > 0$).
        frequency: Number of discrete averaging dates $m \ge 1$.

    Returns:
        tuple[float, float, float, float]: (M1, M2, mu_A, v_A)
    """
    m = int(frequency)
    if m < 1:
        raise ValueError(f"Frequency m must be at least 1, got {frequency}")

    dt = T / m
    t = np.linspace(dt, T, m)

    # 1. First moment M1 = E[A]
    E_S = S0 * np.exp((r - q) * t)
    M1 = float(np.mean(E_S))

    # 2. Second moment M2 = E[A^2]
    ti_grid, tj_grid = np.meshgrid(t, t, indexing="ij")
    cov_grid = (r - q) * (ti_grid + tj_grid) + (sigma**2) * np.minimum(ti_grid, tj_grid)
    E_Si_Sj = (S0**2) * np.exp(cov_grid)
    M2 = float(np.mean(E_Si_Sj))

    # 3. Lognormal matching
    vA2 = np.log(M2 / (M1**2))
    vA = float(np.sqrt(max(vA2, 1e-16)))
    muA = float(np.log(M1) - 0.5 * vA2)

    return M1, M2, muA, vA


def price_asian_call_moment_matching(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
) -> float:
    r"""Prices a discrete arithmetic average Asian Call option using lognormal moment matching.

    Using Black's formula with forward $M_1 = \mathbb{E}[A]$ and total volatility $v_A$:
        $d_1 = \frac{\ln(M_1 / K) + \frac{1}{2} v_A^2}{v_A}$
        $d_2 = d_1 - v_A$
        $C = e^{-rT} (M_1 \Phi(d_1) - K \Phi(d_2))$
    """
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    M1, _, _, vA = compute_discrete_asian_moments(S0, T, r, q, sigma, frequency)
    d1 = (np.log(M1 / K) + 0.5 * vA**2) / vA
    d2 = d1 - vA

    disc = np.exp(-r * T)
    return float(disc * (M1 * norm.cdf(d1) - K * norm.cdf(d2)))


def price_asian_digital_call_moment_matching(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
) -> float:
    r"""Prices a discrete arithmetic average Digital Asian Call (Cash-or-Nothing).

    Payoff: $1_{A > K}$
    Price: $e^{-rT} \mathbb{Q}(A > K) \approx e^{-rT} \Phi(d_2)$
    """
    if S0 <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    M1, _, _, vA = compute_discrete_asian_moments(S0, T, r, q, sigma, frequency)
    d2 = (np.log(M1 / K) - 0.5 * vA**2) / vA

    disc = np.exp(-r * T)
    return float(disc * norm.cdf(d2))


def price_asian_barrier_up_and_out_call_moment_matching(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    H: float,
    frequency: int = 12,
) -> float:
    r"""Prices a discrete arithmetic average Barrier Up-and-Out Call option.

    Payoff: $(A - K)^+ \cdot \mathbf{1}_{A \le H}$
    Under lognormal approximation:
        $B = C(K) - C(H) - (H - K) D(H)$
    where $C(\cdot)$ is the vanilla Asian call and $D(\cdot)$ is the digital Asian call.
    """
    if K >= H or S0 <= 0 or T <= 0 or sigma <= 0:
        return 0.0

    call_K = price_asian_call_moment_matching(S0, K, T, r, q, sigma, frequency)
    call_H = price_asian_call_moment_matching(S0, H, T, r, q, sigma, frequency)
    dig_H = price_asian_digital_call_moment_matching(S0, H, T, r, q, sigma, frequency)

    price = call_K - call_H - (H - K) * dig_H
    return float(max(price, 0.0))


def price_asian_option_moment_matching(params: dict[str, Any]) -> float:
    """Convenience dispatcher to price any Asian product defined in params."""
    S0 = float(params["S0"])
    K = float(params["K"])
    T = float(params["T"])
    r = float(params["r"])
    q = float(params["q"])
    sigma = float(params["sigma"])
    frequency = int(params.get("frequency", 12))
    name = str(params.get("name", "")).lower()

    if "barrier" in name or "h" in [k.lower() for k in params.keys()]:
        H = float(params.get("H", 120.0))
        return price_asian_barrier_up_and_out_call_moment_matching(S0, K, T, r, q, sigma, H, frequency)
    elif "digital" in name:
        return price_asian_digital_call_moment_matching(S0, K, T, r, q, sigma, frequency)
    else:
        return price_asian_call_moment_matching(S0, K, T, r, q, sigma, frequency)
