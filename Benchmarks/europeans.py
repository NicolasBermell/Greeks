import QuantLib as ql
import numpy as np
from scipy.stats import norm
from pandas import DataFrame

from Methods.Analytic.europeans import greeks_european_call, greeks_european_barrier_up_and_out_call, greeks_european_digital_call

def benchmark_greeks_european_call(params: dict[str, float]) -> dict[str, float]:
    return greeks_european_call(params)

def benchmark_greeks_european_digital(params: dict[str, float]) -> dict[str, float]:
    return greeks_european_digital_call(params)

def benchmark_greeks_european_barrier_up_and_out(params: dict[str, float]) -> dict[str, float]:
    return greeks_european_barrier_up_and_out_call(params)


def _compute_all_european_greeks(params: dict[str, float]) -> DataFrame:
    """Computes the Greeks of all European options using analytical formulas."""
    results = {
        "call": greeks_european_call(params),
        "digital": greeks_european_digital_call(params),
        "up_and_out": greeks_european_barrier_up_and_out_call(params),
    }
    return DataFrame(results)

if __name__ == "__main__":
    params = {
        "S0": 100,
        "K": 100,
        "T": 1,
        "r": 0.05,
        "q": 0.02,
        "H": 110,
        "sigma": 0.20,
    }
    results = _compute_all_european_greeks(params)
    print(params)
    print(results)
    
        