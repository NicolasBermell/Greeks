from pandas import DataFrame
from Benchmarks.europeans import (
    benchmark_greeks_european_call,
    benchmark_greeks_european_digital,
    benchmark_greeks_european_call_terminal_knockout,
)
from products import (
    example_european_call_1M,
    example_european_call_ATM,
    example_european_digital_ATM,
    example_european_call_terminal_knockout,
)


def benchmark_all_european() -> DataFrame:
    benchmark_european_call = benchmark_greeks_european_call(example_european_call_ATM)
    benchmark_european_digital = benchmark_greeks_european_digital(example_european_digital_ATM)
    benchmark_call_terminal_knockout = benchmark_greeks_european_call_terminal_knockout(
        example_european_call_terminal_knockout
    )

    results = DataFrame({
        example_european_call_ATM["name"]: benchmark_european_call,
        example_european_digital_ATM["name"]: benchmark_european_digital,
        example_european_call_terminal_knockout["name"]: benchmark_call_terminal_knockout,
        example_european_call_1M["name"]: benchmark_greeks_european_call(example_european_call_1M),
    }).T

    results.index.name = "Product"
    results["Method"] = "Analytical"
    return results


def benchmark_all() -> DataFrame:
    return benchmark_all_european()


if __name__ == "__main__":
    b_df = benchmark_all()
    b_df.to_csv("results/benchmarks.csv")
    print(b_df)
    print("Saved benchmarks to results/benchmarks.csv")
