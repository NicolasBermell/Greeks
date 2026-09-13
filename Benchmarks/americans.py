import sys
from pathlib import Path
import importlib
from pandas import DataFrame
from PDE_1D.PDE_Solver_ql import (
    PDE_ql_geeks_american_call,
    PDE_ql_geeks_american_digital_call,
    PDE_ql_geeks_american_barrier_up_and_out,
)

def benchmark_greeks_american_call(params: dict[str, float]) -> dict[str, float]:
    return PDE_ql_geeks_american_call(params)


def benchmark_greeks_american_digital(params: dict[str, float]) -> dict[str, float]:
    return PDE_ql_geeks_american_digital_call(params)


def benchmark_greeks_american_barrier_up_and_out(params: dict[str, float]) -> dict[str, float]:
    return PDE_ql_geeks_american_barrier_up_and_out(params)


def _compute_all_american_greeks(params: dict[str, float]) -> DataFrame:
    """Computes the Greeks of all American options using numerical PDE / tree benchmarks."""
    results = {
        "call": benchmark_greeks_american_call(params),
        "digital": benchmark_greeks_american_digital(params),
        "up_and_out": benchmark_greeks_american_barrier_up_and_out(params),
    }
    return DataFrame(results)


if __name__ == "__main__":
    params = {
        "S0": 100,
        "K": 100,
        "T": 1,
        "r": 0.05,
        "q": 0.02,
        "H": 105.25,
        "sigma": 0.20,
    }
    results = _compute_all_american_greeks(params)
    print(params)
    print(results)
