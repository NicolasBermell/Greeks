"""Shared random-number generation and per-path statistics for the European estimators.

Every run draws its Brownian increments from its own ``np.random.default_rng(seed)``,
so independent replications only need distinct seeds and never touch NumPy's global state.

All estimators in this project are sample means of i.i.d. per-path quantities Y_i, so
each run reports, per Greek:
    estimate = mean(Y_i)
    var      = sample variance of Y_i (ddof=1), i.e. the variance *per path*
    se       = sqrt(var / N)
The number of paths needed for a tolerance ``tol`` at confidence ``z`` is
N* = z^2 * var / tol^2, so the ratio of required paths between two unbiased methods
is just the ratio of their per-path variances.
"""

from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

GREEKS = ("delta", "gamma", "vega")

# Paths per block when N is large; keeps memory bounded (FDMC holds ~10 arrays per block).
DEFAULT_CHUNK_SIZE = 2_000_000


def generate_paths(T: float, N: int, seed: int = 42) -> np.ndarray:
    """Draws N terminal Brownian values W_T ~ N(0, T) from ``default_rng(seed)``."""
    return np.random.default_rng(seed).standard_normal(N) * np.sqrt(T)


def simulate_terminal_spot(
    S0: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    W_T: np.ndarray,
) -> np.ndarray:
    """Black-Scholes terminal price S_T = S0 exp((r - q - sigma^2/2) T + sigma W_T)."""
    return S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * W_T)


def iter_paths(T: float, N: int, seed: int, chunk_size: int = DEFAULT_CHUNK_SIZE) -> Iterator[np.ndarray]:
    """Yields W_T in blocks. Concatenating the blocks gives exactly ``generate_paths(T, N, seed)``."""
    rng = np.random.default_rng(seed)
    sqrt_t = np.sqrt(T)
    remaining = N
    while remaining > 0:
        n = min(chunk_size, remaining)
        yield rng.standard_normal(n) * sqrt_t
        remaining -= n


class RunningStats:
    """Streaming mean and variance (Chan et al. parallel update) for several named series."""

    def __init__(self, names: tuple[str, ...] = GREEKS):
        self.names = names
        self.n = 0
        self.mean = {k: 0.0 for k in names}
        self.m2 = {k: 0.0 for k in names}

    def update(self, samples: dict[str, np.ndarray]) -> None:
        n_b = len(next(iter(samples.values())))
        if n_b == 0:
            return
        n_a = self.n
        n = n_a + n_b
        for k in self.names:
            x = samples[k]
            mean_b = float(np.mean(x))
            m2_b = float(np.sum((x - mean_b) ** 2))
            d = mean_b - self.mean[k]
            self.mean[k] += d * n_b / n
            self.m2[k] += m2_b + d * d * n_a * n_b / n
        self.n = n

    def result(self, return_stats: bool = False) -> dict[str, float]:
        out: dict[str, float] = {}
        for k in self.names:
            out[k] = self.mean[k]
            if return_stats:
                var = self.m2[k] / (self.n - 1) if self.n > 1 else float("nan")
                out[f"{k}_var"] = var
                out[f"{k}_se"] = float(np.sqrt(var / self.n))
        return out


def estimate_from_samples(
    sample_fn: Callable[[np.ndarray], dict[str, np.ndarray]],
    T: float,
    N: int,
    seed: int,
    paths: np.ndarray | None = None,
    return_stats: bool = False,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, float]:
    """Runs ``sample_fn`` (W_T -> per-path samples for each Greek) over all paths.

    If ``paths`` is given it is used as-is; otherwise paths are drawn from
    ``default_rng(seed)`` in blocks of ``chunk_size``.

    Returns {'delta', 'gamma', 'vega'} and, with ``return_stats``, also
    '{greek}_var' (per-path variance) and '{greek}_se' (standard error of the mean).
    """
    stats = RunningStats()
    blocks = [paths] if paths is not None else iter_paths(T, N, seed, chunk_size)
    for W_T in blocks:
        stats.update(sample_fn(W_T))
    return stats.result(return_stats)
