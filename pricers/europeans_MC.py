import numpy as np

def MC_european_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float, seed: int, N: int) -> float:
    """Computes the price of a European call option using Monte Carlo simulation.
    Args:
        params (dict[str, float]): Dictionary containing the option and market parameters:
            - "S0" (float): Initial underlying asset price ($S_0 > 0$).
            - "K" (float): Strike price of the option ($K > 0$).
            - "T" (float): Time to maturity in years ($T > 0$).
            - "r" (float): Annualized risk-free interest rate in decimal form (e.g., 0.05 for 5%).
            - "q" (float): Annualized continuous dividend yield in decimal form (e.g., 0.02 for 2%).
            - "sigma" (float): Annualized volatility in decimal form ($\sigma > 0$, e.g., 0.20 for 20%).
            - "seed" (int): Seed for reproducibility.
            - "N" (int): Number of Monte Carlo simulations.
    Returns:
        dict[str, float]: Dictionary containing the price of the European call option.
    """
    np.random.seed(seed)
    W_T = np.random.normal(0, np.sqrt(T), N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * W_T)
    return np.mean(np.maximum(S_T - K, 0)) * np.exp(-r * T)

def MC_european_barrier_up_and_out_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float, H: float, seed: int, N: int) -> float:
    np.random.seed(seed)
    W_T = np.random.normal(0, np.sqrt(T), N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * W_T)
    
    payoffs = np.maximum(S_T - K, 0)
    payoffs[S_T > H] = 0

    return np.mean(payoffs) * np.exp(-r * T)

def MC_european_digital_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float, seed: int, N: int) -> float:
    np.random.seed(seed)
    W_T = np.random.normal(0, np.sqrt(T), N)
    S_T = S0 * np.exp((r - q - 0.5 * sigma**2) * T + sigma * W_T)

    payoffs = np.maximum(S_T - K, 0)
    payoffs[S_T > K] = 1

    return np.mean(payoffs) * np.exp(-r * T)