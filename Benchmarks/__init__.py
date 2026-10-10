"""Benchmarks package."""

from Benchmarks.europeans import (
    benchmark_greeks_european_call,
    benchmark_greeks_european_digital,
    benchmark_greeks_european_barrier_up_and_out,
    _compute_all_european_greeks,
)
from Benchmarks.asian import (
    benchmark_greeks_asian_call,
    benchmark_greeks_asian_digital,
    benchmark_greeks_asian_barrier_up_and_out,
    benchmark_all_asian,
)

benchmark_all_european = _compute_all_european_greeks

__all__ = [
    "benchmark_greeks_european_call",
    "benchmark_greeks_european_digital",
    "benchmark_greeks_european_barrier_up_and_out",
    "benchmark_all_european",
    "_compute_all_european_greeks",
    "benchmark_greeks_asian_call",
    "benchmark_greeks_asian_digital",
    "benchmark_greeks_asian_barrier_up_and_out",
    "benchmark_all_asian",
]
