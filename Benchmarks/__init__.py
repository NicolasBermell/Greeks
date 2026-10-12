"""Benchmarks package."""

from Benchmarks.europeans import (
    benchmark_greeks_european_call,
    benchmark_greeks_european_digital,
    benchmark_greeks_european_barrier_up_and_out,
)
from Benchmarks.asian import (
    benchmark_greeks_asian_call,
    benchmark_greeks_asian_digital,
    benchmark_greeks_asian_barrier_up_and_out,
    benchmark_all_asian,
)

__all__ = [
    "benchmark_greeks_european_call",
    "benchmark_greeks_european_digital",
    "benchmark_greeks_european_barrier_up_and_out",
    "benchmark_greeks_asian_call",
    "benchmark_greeks_asian_digital",
    "benchmark_greeks_asian_barrier_up_and_out",
    "benchmark_all_asian",
]
