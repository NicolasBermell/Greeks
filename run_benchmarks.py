from pandas import DataFrame, concat
from Benchmarks.europeans import (
    benchmark_greeks_european_call,
    benchmark_greeks_european_digital,
    benchmark_greeks_european_barrier_up_and_out,
)
from Benchmarks.americans import (
    benchmark_greeks_american_call,
    benchmark_greeks_american_digital,
    benchmark_greeks_american_barrier_up_and_out,
)
from products import (
    example_european_call_ATM,
    example_european_digital_ATM,
    example_european_barrier_up_and_out_95,
    example_american_call_ATM,
    example_american_digital_ATM,
    example_american_barrier_up_and_out_95,
)


def benchmark_all_european() -> DataFrame:
    benchmark_european_call = benchmark_greeks_european_call(example_european_call_ATM)
    benchmark_european_digital = benchmark_greeks_european_digital(example_european_digital_ATM)
    benchmark_european_barrier_up_and_out = benchmark_greeks_european_barrier_up_and_out(
        example_european_barrier_up_and_out_95
    )

    results = DataFrame({
        example_european_call_ATM["name"]: benchmark_european_call,
        example_european_digital_ATM["name"]: benchmark_european_digital,
        example_european_barrier_up_and_out_95["name"]: benchmark_european_barrier_up_and_out,
    }).T

    results.index.name = "Product"
    results["Method"] = "Analytical"
    return results


def benchmark_all_american() -> DataFrame:
    benchmark_american_call = benchmark_greeks_american_call(example_american_call_ATM)
    benchmark_american_digital = benchmark_greeks_american_digital(example_american_digital_ATM)
    benchmark_american_barrier_up_and_out = benchmark_greeks_american_barrier_up_and_out(
        example_american_barrier_up_and_out_95
    )

    results = DataFrame({
        example_american_call_ATM["name"]: benchmark_american_call,
        example_american_digital_ATM["name"]: benchmark_american_digital,
        example_american_barrier_up_and_out_95["name"]: benchmark_american_barrier_up_and_out,
    }).T

    results.index.name = "Product"
    results["Method"] = "PDE / Tree"
    return results


def benchmark_all() -> DataFrame:
    return concat([benchmark_all_european(), benchmark_all_american()])


if __name__ == "__main__":
    print(benchmark_all())