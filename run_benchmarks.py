from products import example_european_call_ATM
from Benchmarks.europeans import *
from products import *

def benchmark_all_european():
    benchmark_european_call = benchmark_greeks_european_call(example_european_call_ATM)
    benchmark_european_digital = benchmark_greeks_european_digital(example_european_digital_ATM)
    benchmark_european_barrier_up_and_out = benchmark_greeks_european_barrier_up_and_out(example_european_barrier_up_and_out_95)

    results = DataFrame({
        example_european_call_ATM["name"]: benchmark_european_call,
        example_european_digital_ATM["name"]: benchmark_european_digital,
        example_european_barrier_up_and_out_95["name"]: benchmark_european_barrier_up_and_out,
    }).T

    results.index.name = "Product"
    results["Method"] = "Analytical"
    return results

def benchmark_all():
    return benchmark_all_european()