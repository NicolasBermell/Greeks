from run_all_greeks import run_all_greeks
from run_benchmarks import benchmark_all

# Greeks computation comparison using different methods.

def main():
    
    print("Starting computation.")

    print("Starting benchmarks...")
    benchmarks = benchmark_all()
    benchmarks.to_csv("./results/benchmarks.csv")

    print("Starting greeks computations...")
    run_all_greeks()

    print("Computation finished.")

if __name__ == "__main__":
    main()