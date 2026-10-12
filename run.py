from run_all_greeks import run_all_greeks
from run_benchmarks import benchmark_all
from run_replications import main as run_replications
from run_gap_sweep import main as run_gap_sweep
from Utilities.metrics import generate_summary_csv

# Greeks computation comparison using different methods.

def main():
    print("Starting computation.")

    print("Starting benchmarks...")
    benchmarks = benchmark_all()
    benchmarks.to_csv("./results/benchmarks.csv")

    print("Starting European Greeks computations...")
    run_all_greeks()

    print("Running independent replications, timing and efficiency...")
    run_replications()

    print("Running the Fournié gap sweep...")
    run_gap_sweep()

    print("Generating summary CSV...")
    generate_summary_csv("./results/summary.csv", fdmc_fixed_eps=0.01)

    print("Computation finished.")


if __name__ == "__main__":
    main()