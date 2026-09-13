import numpy as np
from pricers.europeans_MC import (
    MC_european_barrier_up_and_out_call_option,
    MC_european_call_option,
    MC_european_digital_call_option,
)


def generate_paths(T: float, N: int, seed: int = 42) -> np.ndarray:
    """Generates standard Brownian motion increments W_T ~ N(0, T) for N simulations.

    Generating W_T once ensures Common Random Numbers (CRN) are preserved across
    all sensitivities (Delta, Gamma, Vega) without redundant random number generation.
    """
    np.random.seed(seed)
    return np.random.normal(0.0, np.sqrt(T), N)


# ==============================================================================
# 1. Vanilla European Call Greeks
# ==============================================================================

def _delta_european_call(params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    V_plus = MC_european_call_option(S0 + eps, K, T, r, q, sigma, seed, N, paths=paths)
    V_minus = MC_european_call_option(S0 - eps, K, T, r, q, sigma, seed, N, paths=paths)
    return (V_plus - V_minus) / (2 * eps)


def _gamma_european_call(
    params,
    eps: float = 1e-6,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    V_plus: float | None = None,
    V_minus: float | None = None,
) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    if V_plus is None:
        V_plus = MC_european_call_option(S0 + eps, K, T, r, q, sigma, seed, N, paths=paths)
    if V_minus is None:
        V_minus = MC_european_call_option(S0 - eps, K, T, r, q, sigma, seed, N, paths=paths)
    V_0 = MC_european_call_option(S0, K, T, r, q, sigma, seed, N, paths=paths)
    return (V_plus - 2 * V_0 + V_minus) / (eps ** 2)


def _vega_european_call(params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    V_plus = MC_european_call_option(S0, K, T, r, q, sigma + eps, seed, N, paths=paths)
    V_minus = MC_european_call_option(S0, K, T, r, q, sigma - eps, seed, N, paths=paths)
    return (V_plus - V_minus) / (2 * eps)


def FDMC_geeks_european_call(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None
) -> dict[str, float]:
    if paths is None:
        paths = generate_paths(params["T"], N, seed)

    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    V_plus = MC_european_call_option(S0 + eps, K, T, r, q, sigma, seed, N, paths=paths)
    V_minus = MC_european_call_option(S0 - eps, K, T, r, q, sigma, seed, N, paths=paths)

    delta = (V_plus - V_minus) / (2 * eps)
    gamma = _gamma_european_call(params, eps, N, seed, paths=paths, V_plus=V_plus, V_minus=V_minus)
    vega = _vega_european_call(params, eps, N, seed, paths=paths)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }

# ==============================================================================
# 2. Digital European Call Greeks
# ==============================================================================

def _delta_european_digital_call(params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    V_plus = MC_european_digital_call_option(S0 + eps, K, T, r, q, sigma, seed, N, paths=paths)
    V_minus = MC_european_digital_call_option(S0 - eps, K, T, r, q, sigma, seed, N, paths=paths)
    return (V_plus - V_minus) / (2 * eps)


def _gamma_european_digital_call(
    params,
    eps: float = 1e-6,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    V_plus: float | None = None,
    V_minus: float | None = None,
) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    if V_plus is None:
        V_plus = MC_european_digital_call_option(S0 + eps, K, T, r, q, sigma, seed, N, paths=paths)
    if V_minus is None:
        V_minus = MC_european_digital_call_option(S0 - eps, K, T, r, q, sigma, seed, N, paths=paths)
    V_0 = MC_european_digital_call_option(S0, K, T, r, q, sigma, seed, N, paths=paths)
    return (V_plus - 2 * V_0 + V_minus) / (eps ** 2)


def _vega_european_digital_call(params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None) -> float:
    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    if paths is None:
        paths = generate_paths(T, N, seed)

    V_plus = MC_european_digital_call_option(S0, K, T, r, q, sigma + eps, seed, N, paths=paths)
    V_minus = MC_european_digital_call_option(S0, K, T, r, q, sigma - eps, seed, N, paths=paths)
    return (V_plus - V_minus) / (2 * eps)


def FDMC_geeks_european_digital_call(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None
) -> dict[str, float]:
    if paths is None:
        paths = generate_paths(params["T"], N, seed)

    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]
    V_plus = MC_european_digital_call_option(S0 + eps, K, T, r, q, sigma, seed, N, paths=paths)
    V_minus = MC_european_digital_call_option(S0 - eps, K, T, r, q, sigma, seed, N, paths=paths)

    delta = (V_plus - V_minus) / (2 * eps)
    gamma = _gamma_european_digital_call(params, eps, N, seed, paths=paths, V_plus=V_plus, V_minus=V_minus)
    vega = _vega_european_digital_call(params, eps, N, seed, paths=paths)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }

# ==============================================================================
# 3. Barrier Option (Up-and-Out Call) Greeks
# ==============================================================================

def _delta_european_barrier_up_and_out_call(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None
) -> float:
    S0, K, T, r, q, sigma, H = (
        params["S0"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params.get("H", 120.0),
    )
    if paths is None:
        paths = generate_paths(T, N, seed)

    V_plus = MC_european_barrier_up_and_out_call_option(S0 + eps, K, T, r, q, sigma, H, seed, N, paths=paths)
    V_minus = MC_european_barrier_up_and_out_call_option(S0 - eps, K, T, r, q, sigma, H, seed, N, paths=paths)
    return (V_plus - V_minus) / (2 * eps)


def _gamma_european_barrier_up_and_out_call(
    params,
    eps: float = 1e-6,
    N: int = 10000,
    seed: int = 42,
    paths: np.ndarray | None = None,
    V_plus: float | None = None,
    V_minus: float | None = None,
) -> float:
    S0, K, T, r, q, sigma, H = (
        params["S0"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params.get("H", 120.0),
    )
    if paths is None:
        paths = generate_paths(T, N, seed)

    if V_plus is None:
        V_plus = MC_european_barrier_up_and_out_call_option(S0 + eps, K, T, r, q, sigma, H, seed, N, paths=paths)
    if V_minus is None:
        V_minus = MC_european_barrier_up_and_out_call_option(S0 - eps, K, T, r, q, sigma, H, seed, N, paths=paths)
    V_0 = MC_european_barrier_up_and_out_call_option(S0, K, T, r, q, sigma, H, seed, N, paths=paths)
    return (V_plus - 2 * V_0 + V_minus) / (eps ** 2)


def _vega_european_barrier_up_and_out_call(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None
) -> float:
    S0, K, T, r, q, sigma, H = (
        params["S0"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params.get("H", 120.0),
    )
    if paths is None:
        paths = generate_paths(T, N, seed)

    V_plus = MC_european_barrier_up_and_out_call_option(S0, K, T, r, q, sigma + eps, H, seed, N, paths=paths)
    V_minus = MC_european_barrier_up_and_out_call_option(S0, K, T, r, q, sigma - eps, H, seed, N, paths=paths)
    return (V_plus - V_minus) / (2 * eps)


def FDMC_geeks_european_barrier_up_and_out_call(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None
) -> dict[str, float]:
    if paths is None:
        paths = generate_paths(params["T"], N, seed)

    S0, K, T, r, q, sigma, H = (
        params["S0"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params.get("H", 120.0),
    )
    V_plus = MC_european_barrier_up_and_out_call_option(S0 + eps, K, T, r, q, sigma, H, seed, N, paths=paths)
    V_minus = MC_european_barrier_up_and_out_call_option(S0 - eps, K, T, r, q, sigma, H, seed, N, paths=paths)

    delta = (V_plus - V_minus) / (2 * eps)
    gamma = _gamma_european_barrier_up_and_out_call(params, eps, N, seed, paths=paths, V_plus=V_plus, V_minus=V_minus)
    vega = _vega_european_barrier_up_and_out_call(params, eps, N, seed, paths=paths)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }

# ==============================================================================
# 4. Multi-Product Greeks Suite
# ==============================================================================

def FDMC_all_greeks_european(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    """Calculates all Greeks for Call, Digital Call, and Barrier Up-and-Out options.

    Generates paths W_T once at the top level so that all 3 products and their
    respective Greeks share the exact same underlying simulation paths.
    """
    if paths is None:
        paths = generate_paths(params["T"], N, seed)

    return {
        "call": FDMC_geeks_european_call(params, eps, N, seed, paths=paths),
        "digital": FDMC_geeks_european_digital_call(params, eps, N, seed, paths=paths),
        "barrier": FDMC_geeks_european_barrier_up_and_out_call(params, eps, N, seed, paths=paths),
    }


if __name__ == "__main__":
    test_params = {
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
    print("FDMC All Greeks test:")
    print(FDMC_all_greeks_european(test_params))
