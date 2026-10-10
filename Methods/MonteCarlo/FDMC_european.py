import numpy as np
from pricers.europeans_MC import (
    MC_european_barrier_up_and_out_call_option,
    MC_european_call_option,
    MC_european_digital_call_option,
)
from Methods.sampling import estimate_from_samples, generate_paths


def _payoff_call(S_T: np.ndarray, params) -> np.ndarray:
    return np.maximum(S_T - params["K"], 0.0)


def _payoff_digital_call(S_T: np.ndarray, params) -> np.ndarray:
    return (S_T > params["K"]).astype(float)


def _payoff_call_terminal_knockout(S_T: np.ndarray, params) -> np.ndarray:
    return np.maximum(S_T - params["K"], 0.0) * (S_T <= params["H"])


def FDMC_samples_european(payoff_fn, params, eps: float, W_T: np.ndarray) -> dict[str, np.ndarray]:
    """Per-path central difference quotients with shared random numbers.

    Uses 5 payoff evaluations per path: S0 + eps, S0, S0 - eps, sigma + eps, sigma - eps.
    The mean of each array is the usual FDMC estimate; its variance is the per-path variance.
    """
    S0, T, r, q, sigma = params["S0"], params["T"], params["r"], params["q"], params["sigma"]
    discount = np.exp(-r * T)

    def discounted_payoff(s0: float, sig: float) -> np.ndarray:
        S_T = s0 * np.exp((r - q - 0.5 * sig**2) * T + sig * W_T)
        return discount * payoff_fn(S_T, params)

    V_plus = discounted_payoff(S0 + eps, sigma)
    V_minus = discounted_payoff(S0 - eps, sigma)
    V_0 = discounted_payoff(S0, sigma)
    V_sig_plus = discounted_payoff(S0, sigma + eps)
    V_sig_minus = discounted_payoff(S0, sigma - eps)

    return {
        "delta": (V_plus - V_minus) / (2 * eps),
        "gamma": (V_plus - 2 * V_0 + V_minus) / (eps**2),
        "vega": (V_sig_plus - V_sig_minus) / (2 * eps),
    }


def _FDMC_greeks(payoff_fn, params, eps, N, seed, paths, return_stats) -> dict[str, float]:
    return estimate_from_samples(
        lambda W_T: FDMC_samples_european(payoff_fn, params, eps, W_T),
        params["T"], N, seed, paths=paths, return_stats=return_stats,
    )


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
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Delta, Gamma and Vega by central differences (5 evaluations per path, shared random numbers).

    With ``return_stats`` also returns '{greek}_var' (variance of the per-path difference
    quotient) and '{greek}_se'.
    """
    return _FDMC_greeks(_payoff_call, params, eps, N, seed, paths, return_stats)

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
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Delta, Gamma and Vega by central differences (5 evaluations per path, shared random numbers).

    With ``return_stats`` also returns '{greek}_var' (variance of the per-path difference
    quotient) and '{greek}_se'.
    """
    return _FDMC_greeks(_payoff_digital_call, params, eps, N, seed, paths, return_stats)

# ==============================================================================
# 3. Call with Terminal Knock-Out Greeks
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
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None,
    return_stats: bool = False,
) -> dict[str, float]:
    """Delta, Gamma and Vega by central differences (5 evaluations per path, shared random numbers).

    With ``return_stats`` also returns '{greek}_var' (variance of the per-path difference
    quotient) and '{greek}_se'.
    """
    return _FDMC_greeks(_payoff_call_terminal_knockout, params, eps, N, seed, paths, return_stats)


# ==============================================================================
# 4. Multi-Product Greeks Suite
# ==============================================================================


def FDMC_all_greeks_european(
    params, eps: float = 1e-6, N: int = 10000, seed: int = 42, paths: np.ndarray | None = None
) -> dict[str, dict[str, float]]:
    """Calculates all Greeks for the call, digital call and call with terminal knock-out.

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
        "H": 105.25,
    }
    print("FDMC All Greeks test:")
    print(FDMC_all_greeks_european(test_params))
