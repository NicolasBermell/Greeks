import numpy as np

def MC_european_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float, seed: int = 42, N: int = 10000, paths: np.ndarray | None = None) -> float:
    """Computes the price of a European call option using Monte Carlo simulation.
    Args:
        S0 (float): Initial underlying asset price (S0 > 0).
        K (float): Strike price of the option (K > 0).
        T (float): Time to maturity in years (T > 0).
        r (float): Annualized risk-free interest rate in decimal form (e.g., 0.05 for 5%).
        q (float): Annualized continuous dividend yield in decimal form (e.g., 0.02 for 2%).
        sigma (float): Annualized volatility in decimal form (sigma > 0, e.g., 0.20 for 20%).
        seed (int): Seed for reproducibility.
        N (int): Number of Monte Carlo simulations.
        paths (np.ndarray | None): Optional pre-generated Brownian motion values W_T ~ N(0, T).
    Returns:
        float: Price of the European call option.
    """
    if paths is None:
        np.random.seed(seed)
        paths = np.random.normal(0, np.sqrt(T), N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * paths)
    return float(np.mean(np.maximum(S_T - K, 0)) * np.exp(-r * T))

def MC_european_barrier_up_and_out_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float, H: float, seed: int = 42, N: int = 10000, paths: np.ndarray | None = None) -> float:
    if paths is None:
        np.random.seed(seed)
        paths = np.random.normal(0, np.sqrt(T), N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * paths)
    
    payoffs = np.maximum(S_T - K, 0)
    payoffs[S_T > H] = 0

    return float(np.mean(payoffs) * np.exp(-r * T))

def MC_european_digital_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float, seed: int = 42, N: int = 10000, paths: np.ndarray | None = None) -> float:
    if paths is None:
        np.random.seed(seed)
        paths = np.random.normal(0, np.sqrt(T), N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * paths)

    payoffs = np.maximum(S_T - K, 0)
    payoffs[S_T > K] = 1

    return float(np.mean(payoffs) * np.exp(-r * T))