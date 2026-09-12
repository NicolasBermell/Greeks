from Benchmarks import europeans
from pricers.europeans_MC import *

def _delta_european_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], seed, N

    V_plus = MC_european_call_option(S0 + eps, K, T, r, q, sigma, seed, N)
    V_minus = MC_european_call_option(S0 - eps, K, T, r, q, sigma, seed, N)
    return (V_plus - V_minus) / (2 * eps)

def _gamma_european_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], seed, N

    V_plus = MC_european_call_option(S0 + eps, K, T, r, q, sigma, seed, N)
    V_minus = MC_european_call_option(S0 - eps, K, T, r, q, sigma, seed, N)
    return (V_plus - 2 * MC_european_call_option(S0, K, T, r, q, sigma, seed, N) + V_minus) / (eps ** 2)

def _vega_european_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], seed, N

    V_plus = MC_european_call_option(S0, K, T, r, q, sigma + eps, seed, N)
    V_minus = MC_european_call_option(S0, K, T, r, q, sigma - eps, seed, N)
    return (V_plus - V_minus) / (2 * eps)

def FDMC_geeks_european_call(params, eps: float = 1e-6, N: int = 10000, seed: int = 42) -> dict[str, float]:
    delta = _delta_european_call(params, eps, N, seed)
    gamma = _gamma_european_call(params, eps, N, seed)
    vega = _vega_european_call(params, eps, N, seed)
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        }

# Digital Call
def _delta_european_digital_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], seed, N

    V_plus = MC_european_digital_call_option(S0 + eps, K, T, r, q, sigma, seed, N)
    V_minus = MC_european_digital_call_option(S0 - eps, K, T, r, q, sigma, seed, N)
    return (V_plus - V_minus) / (2 * eps)

def _gamma_european_digital_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], seed, N

    V_plus = MC_european_digital_call_option(S0 + eps, K, T, r, q, sigma, seed, N)
    V_minus = MC_european_digital_call_option(S0 - eps, K, T, r, q, sigma, seed, N)
    return (V_plus - 2 * MC_european_digital_call_option(S0, K, T, r, q, sigma, seed, N) + V_minus) / (eps ** 2)

def _vega_european_digital_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], seed, N

    V_plus = MC_european_digital_call_option(S0, K, T, r, q, sigma + eps, seed, N)
    V_minus = MC_european_digital_call_option(S0, K, T, r, q, sigma - eps, seed, N)
    return (V_plus - V_minus) / (2 * eps)

def FDMC_geeks_european_digital_call(params, eps: float = 1e-6, N: int = 10000, seed: int = 42) -> dict[str, float]:
    delta = _delta_european_digital_call(params, eps, N, seed)
    gamma = _gamma_european_digital_call(params, eps, N, seed)
    vega = _vega_european_digital_call(params, eps, N, seed)
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }

# Barrier Option (Up-and-Out Call)
def _delta_european_barrier_up_and_out_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, H, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], params["H"], seed, N

    V_plus = MC_european_barrier_up_and_out_call_option(S0 + eps, K, T, r, q, sigma, H, seed, N)
    V_minus = MC_european_barrier_up_and_out_call_option(S0 - eps, K, T, r, q, sigma, H, seed, N)
    return (V_plus - V_minus) / (2 * eps)

def _gamma_european_barrier_up_and_out_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, H, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], params["H"], seed, N

    V_plus = MC_european_barrier_up_and_out_call_option(S0 + eps, K, T, r, q, sigma, H, seed, N)
    V_minus = MC_european_barrier_up_and_out_call_option(S0 - eps, K, T, r, q, sigma, H, seed, N)
    return (V_plus - 2 * MC_european_barrier_up_and_out_call_option(S0, K, T, r, q, sigma, H, seed, N) + V_minus) / (eps ** 2)

def _vega_european_barrier_up_and_out_call(params, eps, N, seed):
    S0, K, T, r, q, sigma, H, seed, N = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], params["H"], seed, N

    V_plus = MC_european_barrier_up_and_out_call_option(S0, K, T, r, q, sigma + eps, H, seed, N)
    V_minus = MC_european_barrier_up_and_out_call_option(S0, K, T, r, q, sigma - eps, H, seed, N)
    return (V_plus - V_minus) / (2 * eps)

def FDMC_geeks_european_barrier_up_and_out_call(params, eps: float = 1e-6, N: int = 10000, seed: int = 42) -> dict[str, float]:
    delta = _delta_european_barrier_up_and_out_call(params, eps, N, seed)
    gamma = _gamma_european_barrier_up_and_out_call(params, eps, N, seed)
    vega = _vega_european_barrier_up_and_out_call(params, eps, N, seed)
    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }

def FDMC_all_greeks(params, eps: float = 1e-6, N: int = 10000, seed: int = 42) -> dict[str, float]:
    return {
        "call": FDMC_geeks_european_call(params, eps, N, seed),
        "digital": FDMC_geeks_european_digital_call(params, eps, N, seed),
        "barrier": FDMC_geeks_european_barrier_up_and_out_call(params, eps, N, seed),
    }


if __name__ == "__main__":
    params = {
        "S0": 100.0,
        "K": 100.0,
        "T": 1.0,
        "r": 0.05,
        "q": 0.02,
        "sigma": 0.20,
        "seed": 42,
        "N": 10000,
        "H": 120.0,
    }
    print(FDMC_all_greeks(params))
