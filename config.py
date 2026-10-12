# Simulation Parameters for FDMC
FDMC_N_SIMS_ARRAY = [10000, 100000, 250_000, 1000000, 2_000_000, 5_000_000, 10_000_000, 30_000_000, 50_000_000, 100_000_000]
FDMC_EPS_ARRAY = [1e-1, 1e-2, 1e-3]
FDMC_FIXED_EPS = 0.01  # Fixed production step size benchmark across all Greeks & products
FDMC_SEED = 1984
FDMC_RESULTS_FILE = "./results/FDMC_results.csv"

# Simulation Parameters for Malliavin (Standard)
MALLIAVIN_N_SIMS_ARRAY = [10000, 100000, 250_000, 1000000, 2_000_000, 5_000_000, 10_000_000, 30_000_000, 50_000_000, 100_000_000]
MALLIAVIN_SEED = 1984
MALLIAVIN_RESULTS_FILE = "./results/malliavin_results.csv"

# Simulation Parameters for Fournié Localized Malliavin
# Bandwidths come from fournier_bandwidths(params) in malliavin_european_fournier.py.
MALLIAVIN_FOURNIER_RESULTS_FILE = "./results/malliavin_fournier_results.csv"
MALLIAVIN_FOURNIER_CUTOFF_TYPE = "gaussian"

# Independent replications: R seeds per (method, product, N), each run with np.random.default_rng(seed).
# Larger N is extrapolated with SE ∝ 1/sqrt(N) from the per-path variance instead of being simulated.
REPLICATION_N_ARRAY = [10_000, 100_000, 1_000_000]
REPLICATION_R = 100
REPLICATION_BASE_SEED = 20_000  # run r uses seed REPLICATION_BASE_SEED + r
REPLICATION_FDMC_EPS_ARRAY = [1e-1, 1e-2, 1e-3]
REPLICATION_RAW_FILE = "./results/replications_raw.csv"
REPLICATION_SUMMARY_FILE = "./results/replications_summary.csv"
EFFICIENCY_FILE = "./results/efficiency.csv"
FIGURES_DIR = "./results/figures"

# Fournié gap sweep: per-path variance vs the gap l = multiplier * S0 * sigma * sqrt(T), same l for
# all three Greeks, Gaussian cutoff, one set of shared paths per product.
GAP_SWEEP_MULTIPLIERS = [0.02, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.85, 0.95, 1.0, 1.05, 1.1, 1.2, 1.4, 1.7, 2.0, 3.0, 5.0, 10.0]
GAP_SWEEP_N = 2_000_000
GAP_SWEEP_SEED = 31_415
GAP_SWEEP_FILE = "./results/gap_sweep.csv"

# Simulation Parameters for Asian Options
ASIAN_BENCHMARK_N_SIMS = 5_000_000
ASIAN_BENCHMARK_EPS = 0.01
ASIAN_FDMC_N_SIMS_ARRAY = [10_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 5_000_000]
ASIAN_FDMC_EPS_ARRAY = [1e-1, 1e-2, 1e-3]
ASIAN_SEED = 1984
ASIAN_BENCHMARKS_FILE = "./results/asian_benchmarks.csv"
ASIAN_FDMC_RESULTS_FILE = "./results/asian_FDMC_results.csv"
ASIAN_MOMENT_MATCHING_RESULTS_FILE = "./results/asian_moment_matching_results.csv"