import sys
from pathlib import Path
import QuantLib as ql
import numpy as np
from scipy.stats import norm
from pandas import DataFrame

# Ensure project root is in sys.path for direct script execution
root_dir = str(Path(__file__).resolve().parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from Methods.Analytic.europeans import (
    greeks_european_call,
    greeks_european_barrier_up_and_out_call,
    greeks_european_digital_call,
)

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
        "call_terminal_knockout": greeks_european_barrier_up_and_out_call(params),
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
    results = _compute_all_european_greeks(params)
    print(params)
    print(results)
    
        