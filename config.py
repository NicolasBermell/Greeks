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
MALLIAVIN_FOURNIER_RESULTS_FILE = "./results/malliavin_fournier_results.csv"
MALLIAVIN_FOURNIER_CUTOFF_TYPE = "gaussian"
MALLIAVIN_FOURNIER_DELTA_STRIKE = 8.0  # Calibrated optimal rule: 0.40 * S0 * sigma * sqrt(T)
MALLIAVIN_FOURNIER_BARRIER_DELTA = 2.625  # Corridor constraint: min(0.40 * S0 * sigma * sqrt(T), (H - K) / 2) = 2.625