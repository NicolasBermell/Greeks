"""Monte Carlo simulation pricer for discrete arithmetic average Asian options.

Uses Common Random Numbers (CRN / CN) across all Greeks and perturbed parameter evaluations.
Supports:
1. Vanilla Asian Call: E[e^{-rT} (A - K)^+]
2. Digital Asian Call: E[e^{-rT} 1_{A > K}]
3. Barrier Asian Up-and-Out Call: E[e^{-rT} (A - K)^+ 1_{A <= H}]
"""

from __future__ import annotations
import numpy as np


def generate_asian_paths(
    T: float,
    frequency: int = 12,
    N: int = 10000,
    seed: int = 42,
) -> np.ndarray:
    r"""Generates Brownian motion paths $W(t_i)$ for $N$ simulations at discrete observation dates.

    Observation dates $t_i = i \Delta t$ with $\Delta t = T / m$.
    $W(t_i) = \sum_{k=1}^i \sqrt{\Delta t} Z_k$, where $Z_k \sim \mathcal{N}(0, 1)$ i.i.d.

    Generating $W$ once preserves Common Random Numbers (CRN / CN) across all finite difference
    perturbations ($S_0 \pm \epsilon, \sigma \pm \epsilon$) and eliminates Monte Carlo noise between evaluations.

    Args:
        T: Time to maturity in years ($T > 0$).
        frequency: Number of discrete averaging dates $m \ge 1$.
        N: Number of Monte Carlo paths.
        seed: Random seed for reproducibility.

    Returns:
        np.ndarray: Array of shape (N, m) containing Brownian motion values $W(t_i)$.
    """
    m = int(frequency)
    dt = T / m
    rng = np.random.default_rng(seed)
    # Generate increments dW ~ N(0, dt)
    dW = rng.normal(0.0, np.sqrt(dt), size=(N, m))
    W = np.cumsum(dW, axis=1)
    return W


def _simulate_arithmetic_average(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
    seed: int = 42,
    N: int = 10000,
    paths: np.ndarray | None = None,
) -> np.ndarray:
    """Helper to simulate the arithmetic average A = (1/m) sum_{i=1}^m S_{t_i} for N paths."""
    m = int(frequency)
    if paths is None:
        paths = generate_asian_paths(T, m, N, seed)
    else:
        N, m = paths.shape

    dt = T / m
    t = np.linspace(dt, T, m)
    drift = (r - q - 0.5 * sigma**2) * t  # shape (m,)
    S_paths = S0 * np.exp(drift + sigma * paths)  # shape (N, m)
    A = np.mean(S_paths, axis=1)  # shape (N,)
    return A


def MC_asian_call_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
    seed: int = 42,
    N: int = 10000,
    paths: np.ndarray | None = None,
) -> float:
    """Computes the price of a discrete arithmetic average Asian Call option using Monte Carlo."""
    A = _simulate_arithmetic_average(S0, T, r, q, sigma, frequency, seed, N, paths)
    payoffs = np.maximum(A - K, 0.0)
    return float(np.mean(payoffs) * np.exp(-r * T))


def MC_asian_digital_call_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
    seed: int = 42,
    N: int = 10000,
    paths: np.ndarray | None = None,
) -> float:
    """Computes the price of a discrete arithmetic average Digital Asian Call (Cash-or-Nothing)."""
    A = _simulate_arithmetic_average(S0, T, r, q, sigma, frequency, seed, N, paths)
    payoffs = (A > K).astype(float)
    return float(np.mean(payoffs) * np.exp(-r * T))


def MC_asian_barrier_up_and_out_call_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    H: float,
    frequency: int = 12,
    seed: int = 42,
    N: int = 10000,
    paths: np.ndarray | None = None,
) -> float:
    """Computes the price of a discrete arithmetic average Barrier Up-and-Out Call option."""
    if K >= H:
        return 0.0
    A = _simulate_arithmetic_average(S0, T, r, q, sigma, frequency, seed, N, paths)
    payoffs = np.where((A >= K) & (A <= H), A - K, 0.0)
    return float(np.mean(payoffs) * np.exp(-r * T))


# ==============================================================================
# Conditional Monte Carlo (CMC) Pricers via Tower Property Conditioning on F_{t_{m-1}}
# ==============================================================================

def _simulate_conditional_averaging_state(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
    seed: int = 42,
    N: int = 10000,
    paths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    r"""Simulates the partial trajectory up to time $t_{m-1}$ for Conditional Monte Carlo.

    Conditional on $\mathcal{F}_{t_{m-1}}$, the partial sum $\Sigma_{m-1} = \sum_{i=1}^{m-1} S_{t_i}$
    and the previous spot price $S_{t_{m-1}}$ are known constants, while the final asset price
    $S_{t_m}$ is a lognormal random variable.

    Returns:
        tuple[np.ndarray, np.ndarray, float]: (Sigma_prev, S_last, dt)
    """
    m = int(frequency)
    dt = T / m
    if paths is None:
        paths = generate_asian_paths(T, m, N, seed)
    else:
        N = paths.shape[0]

    # Use paths up to m-1
    t_prev = np.linspace(dt, T - dt, m - 1)
    drift_prev = (r - q - 0.5 * sigma**2) * t_prev
    # paths[:, :m-1] has shape (N, m-1)
    S_prev = S0 * np.exp(drift_prev + sigma * paths[:, :m-1])
    Sigma_prev = np.sum(S_prev, axis=1)  # shape (N,)
    S_last = S_prev[:, -1]  # shape (N,)
    return Sigma_prev, S_last, dt


def _cond_call_payoff(
    S_last: np.ndarray,
    k_star: np.ndarray,
    dt: float,
    r: float,
    q: float,
    sigma: float,
    m: int,
) -> np.ndarray:
    """Analytical conditional expectation E[(S_{t_m} - k_star)^+ | F_{t_{m-1}}] / m."""
    from scipy.stats import norm
    N = S_last.shape[0]
    res = np.zeros(N)
    pos = k_star > 0
    neg = ~pos

    if np.any(neg):
        res[neg] = S_last[neg] * np.exp((r - q) * dt) - k_star[neg]

    if np.any(pos):
        vol_sqrt_dt = sigma * np.sqrt(dt)
        d1 = (np.log(S_last[pos] / k_star[pos]) + (r - q + 0.5 * sigma**2) * dt) / vol_sqrt_dt
        d2 = d1 - vol_sqrt_dt
        res[pos] = S_last[pos] * np.exp((r - q) * dt) * norm.cdf(d1) - k_star[pos] * norm.cdf(d2)

    return res / m


def _cond_digital_payoff(
    S_last: np.ndarray,
    k_star: np.ndarray,
    dt: float,
    r: float,
    q: float,
    sigma: float,
) -> np.ndarray:
    """Analytical conditional probability Q(S_{t_m} > k_star | F_{t_{m-1}})."""
    from scipy.stats import norm
    N = S_last.shape[0]
    res = np.zeros(N)
    pos = k_star > 0
    neg = ~pos

    if np.any(neg):
        res[neg] = 1.0

    if np.any(pos):
        vol_sqrt_dt = sigma * np.sqrt(dt)
        d2 = (np.log(S_last[pos] / k_star[pos]) + (r - q - 0.5 * sigma**2) * dt) / vol_sqrt_dt
        res[pos] = norm.cdf(d2)

    return res


def conditional_MC_asian_call_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
    seed: int = 42,
    N: int = 10000,
    paths: np.ndarray | None = None,
) -> float:
    """Computes the price of a Vanilla Asian Call using Conditional Monte Carlo (smooth Black-Scholes expectation)."""
    m = int(frequency)
    Sigma_prev, S_last, dt = _simulate_conditional_averaging_state(S0, T, r, q, sigma, frequency, seed, N, paths)
    K_star = m * K - Sigma_prev
    c_pay = _cond_call_payoff(S_last, K_star, dt, r, q, sigma, m)
    return float(np.mean(c_pay) * np.exp(-r * T))


def conditional_MC_asian_digital_call_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    frequency: int = 12,
    seed: int = 42,
    N: int = 10000,
    paths: np.ndarray | None = None,
) -> float:
    """Computes the price of a Digital Asian Call using Conditional Monte Carlo (smooth Gaussian CDF expectation)."""
    m = int(frequency)
    Sigma_prev, S_last, dt = _simulate_conditional_averaging_state(S0, T, r, q, sigma, frequency, seed, N, paths)
    K_star = m * K - Sigma_prev
    d_pay = _cond_digital_payoff(S_last, K_star, dt, r, q, sigma)
    return float(np.mean(d_pay) * np.exp(-r * T))


def conditional_MC_asian_barrier_up_and_out_call_option(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    H: float,
    frequency: int = 12,
    seed: int = 42,
    N: int = 10000,
    paths: np.ndarray | None = None,
) -> float:
    """Computes the price of a Barrier Up-and-Out Asian Call using Conditional Monte Carlo."""
    if K >= H:
        return 0.0
    m = int(frequency)
    Sigma_prev, S_last, dt = _simulate_conditional_averaging_state(S0, T, r, q, sigma, frequency, seed, N, paths)
    K_star = m * K - Sigma_prev
    H_star = m * H - Sigma_prev

    call_K = _cond_call_payoff(S_last, K_star, dt, r, q, sigma, m)
    call_H = _cond_call_payoff(S_last, H_star, dt, r, q, sigma, m)
    dig_H = _cond_digital_payoff(S_last, H_star, dt, r, q, sigma)

    b_pay = np.maximum(call_K - call_H - (H - K) * dig_H, 0.0)
    return float(np.mean(b_pay) * np.exp(-r * T))

