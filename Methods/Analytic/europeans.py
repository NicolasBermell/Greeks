import QuantLib as ql
import numpy as np
from scipy.stats import norm

# Vanilla Call
def d1(S0, K, T, r, q, sigma):
    return (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def d2(S0, K, T, r, q, sigma):
    return d1(S0, K, T, r, q, sigma) - sigma * np.sqrt(T)

def delta_call(S0, K, T, r, q, sigma):
    return np.exp(-q * T) * norm.cdf(d1(S0, K, T, r, q, sigma))

def gamma_call(S0, K, T, r, q, sigma):
    return np.exp(-q * T) * norm.pdf(d1(S0, K, T, r, q, sigma)) / (S0 * sigma * np.sqrt(T))

def vega_call(S0, K, T, r, q, sigma):
    return S0 * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1(S0, K, T, r, q, sigma))

def greeks_european_call(params: dict[str, float]) -> dict[str, float]:
    r"""Computes the Greeks of a European call option using analytical Black-Scholes formulas.

    Args:
        params (dict[str, float]): Dictionary containing the option and market parameters:
            - "S0" (float): Initial underlying asset price ($S_0 > 0$).
            - "K" (float): Strike price of the option ($K > 0$).
            - "T" (float): Time to maturity in years ($T > 0$).
            - "r" (float): Annualized risk-free interest rate in decimal form (e.g., 0.05 for 5%).
            - "q" (float): Annualized continuous dividend yield in decimal form (e.g., 0.02 for 2%).
            - "sigma" (float): Annualized volatility in decimal form ($\sigma > 0$, e.g., 0.20 for 20%).

    Returns:
        dict[str, float]: Dictionary containing the analytical Greeks:
            - "delta" (float): First derivative of option price with respect to spot ($\partial V / \partial S_0$).
            - "gamma" (float): Second derivative of option price with respect to spot ($\partial^2 V / \partial S_0^2$).
            - "vega" (float): First derivative of option price with respect to volatility ($\partial V / \partial \sigma$).

    Raises:
        KeyError: If any of the required keys ("S0", "K", "T", "r", "q", "sigma") are missing from `params`.

    Example:
        >>> params = {
        ...     "S0": 100.0,
        ...     "K": 100.0,
        ...     "T": 1.0,
        ...     "r": 0.05,
        ...     "q": 0.02,
        ...     "sigma": 0.20,
        ... }
        >>> greeks = greeks_european_call(params)
        >>> print(greeks)
        {'delta': 0.5868511461347649, 'gamma': 0.01895057875500871, 'vega': 37.90115751001742}
    """

    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]

    results = {}
    results["delta"] = delta_call(S0, K, T, r, q, sigma)
    results["gamma"] = gamma_call(S0, K, T, r, q, sigma)
    results["vega"] = vega_call(S0, K, T, r, q, sigma)

    return results


# Digital Call
def _quantlib_european_digital_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> ql.VanillaOption:
    today = ql.Date(12, 9, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    days = int(round(T * 365))
    maturity = today + ql.Period(days, ql.Days)

    payoff = ql.CashOrNothingPayoff(ql.Option.Call, K, 1.0)
    exercise = ql.EuropeanExercise(maturity)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count))

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_ts, rate_ts, vol_ts)
    engine = ql.AnalyticEuropeanEngine(bsm_process)
    option.setPricingEngine(engine)
    return option

def greeks_european_digital_call(params: dict[str, float]) -> dict[str, float]:
    r"""Computes the Greeks of a digital (cash-or-nothing) call option using QuantLib.

    Args:
        params (dict[str, float]): Dictionary containing the option and market parameters:
            - "S0" (float): Initial underlying asset price ($S_0 > 0$).
            - "K" (float): Strike price of the option ($K > 0$).
            - "T" (float): Time to maturity in years ($T > 0$).
            - "r" (float): Annualized risk-free interest rate in decimal form (e.g., 0.05 for 5%).
            - "q" (float): Annualized continuous dividend yield in decimal form (e.g., 0.02 for 2%).
            - "sigma" (float): Annualized volatility in decimal form ($\sigma > 0$, e.g., 0.20 for 20%).

    Returns:
        dict[str, float]: Dictionary containing the QuantLib analytical Greeks:
            - "delta" (float): First derivative of option price with respect to spot ($\partial V / \partial S_0$).
            - "gamma" (float): Second derivative of option price with respect to spot ($\partial^2 V / \partial S_0^2$).
            - "vega" (float): First derivative of option price with respect to volatility ($\partial V / \partial \sigma$).

    Raises:
        KeyError: If any of the required keys ("S0", "K", "T", "r", "q", "sigma") are missing from `params`.

    Example:
        >>> params = {
        ...     "S0": 100.0,
        ...     "K": 100.0,
        ...     "T": 1.0,
        ...     "r": 0.05,
        ...     "q": 0.02,
        ...     "sigma": 0.20,
        ... }
        >>> greeks = greeks_digital_call(params)
        >>> print(greeks)
        {'delta': 0.01895057875500871, 'gamma': -0.00023688223443760922, 'vega': -0.47376446887521906}
    """

    S0, K, T, r, q, sigma = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"]

    results = {}
    option = _quantlib_european_digital_call_option(S0, K, T, r, q, sigma)
    results["delta"] = option.delta()
    results["gamma"] = option.gamma()
    results["vega"] = option.vega()

    return results

# Barrier Option (European Terminal Barrier Up-and-Out Call: (S_T - K)^+ * 1_{S_T <= H})
def price_barrier_up_and_out_call(S: float, K: float, T: float, r: float, q: float, sigma: float, H: float) -> float:
    r"""Exact analytical price of a European Terminal Barrier Up-and-Out Call option.

    Payoff at maturity T:
        Payoff = (S_T - K)^+ \cdot \mathbf{1}_{S_T \le H}
               = (S_T - K)^+ - (S_T - H)^+ - (H - K) \cdot \mathbf{1}_{S_T > H}

    This depends purely on the terminal price S_T at maturity (not on continuous path monitoring).

    Args:
        S (float): Current spot price ($S > 0$).
        K (float): Strike price ($K > 0$).
        T (float): Time to maturity in years ($T > 0$).
        r (float): Annualized risk-free interest rate.
        q (float): Annualized continuous dividend yield.
        sigma (float): Annualized volatility ($\sigma > 0$).
        H (float): Upper barrier level ($H > K$).

    Returns:
        float: Option price. Returns 0.0 if K >= H or if S <= 0, T <= 0, sigma <= 0.
    """
    if K >= H or S <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return 0.0

    vol_sqrt_t = sigma * np.sqrt(T)
    d1_k = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d1_h = (np.log(S / H) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2_k = d1_k - vol_sqrt_t
    d2_h = d1_h - vol_sqrt_t

    term1 = S * np.exp(-q * T) * (norm.cdf(d1_k) - norm.cdf(d1_h))
    term2 = K * np.exp(-r * T) * (norm.cdf(d2_k) - norm.cdf(d2_h))

    return float(term1 - term2)


def delta_barrier_up_and_out_call(S: float, K: float, T: float, r: float, q: float, sigma: float, H: float) -> float:
    r"""Exact analytical Delta ($\partial V / \partial S$) for European Terminal Barrier Up-and-Out Call.

    Delta = e^{-qT} [\Phi(d_1(S, K)) - \Phi(d_1(S, H))] - \frac{(H - K) e^{-rT} \phi(d_2(S, H))}{S \sigma \sqrt{T}}
    """
    if K >= H or S <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return 0.0

    vol_sqrt_t = sigma * np.sqrt(T)
    d1_k = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d1_h = (np.log(S / H) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2_h = d1_h - vol_sqrt_t

    delta = (
        np.exp(-q * T) * (norm.cdf(d1_k) - norm.cdf(d1_h))
        - (H - K) * np.exp(-r * T) * norm.pdf(d2_h) / (S * vol_sqrt_t)
    )
    return float(delta)


def gamma_barrier_up_and_out_call(S: float, K: float, T: float, r: float, q: float, sigma: float, H: float) -> float:
    r"""Exact analytical Gamma ($\partial^2 V / \partial S^2$) for European Terminal Barrier Up-and-Out Call.

    Gamma = \frac{e^{-qT}\phi(d_1(S, K))}{S \sigma \sqrt{T}}
          - \frac{e^{-qT}\phi(d_1(S, H))}{S \sigma \sqrt{T}}
          + \frac{(H - K) e^{-rT} \phi(d_2(S, H))}{S^2 \sigma \sqrt{T}} \left( 1 + \frac{d_2(S, H)}{\sigma \sqrt{T}} \right)
    """
    if K >= H or S <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return 0.0

    vol_sqrt_t = sigma * np.sqrt(T)
    d1_k = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d1_h = (np.log(S / H) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2_h = d1_h - vol_sqrt_t

    gamma = (
        np.exp(-q * T) * norm.pdf(d1_k) / (S * vol_sqrt_t)
        - np.exp(-q * T) * norm.pdf(d1_h) / (S * vol_sqrt_t)
        + (H - K) * np.exp(-r * T) * norm.pdf(d2_h) / (S**2 * vol_sqrt_t) * (1.0 + d2_h / vol_sqrt_t)
    )
    return float(gamma)


def vega_barrier_up_and_out_call(S: float, K: float, T: float, r: float, q: float, sigma: float, H: float) -> float:
    r"""Exact analytical Vega ($\partial V / \partial \sigma$) for European Terminal Barrier Up-and-Out Call.

    Vega = S e^{-qT} \sqrt{T} \phi(d_1(S, K))
         - S e^{-qT} \sqrt{T} \phi(d_1(S, H))
         + \frac{(H - K) e^{-rT} \phi(d_2(S, H)) d_1(S, H)}{\sigma}
    """
    if K >= H or S <= 0.0 or T <= 0.0 or sigma <= 0.0:
        return 0.0

    vol_sqrt_t = sigma * np.sqrt(T)
    d1_k = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d1_h = (np.log(S / H) + (r - q + 0.5 * sigma**2) * T) / vol_sqrt_t
    d2_h = d1_h - vol_sqrt_t

    vega = (
        S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1_k)
        - S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1_h)
        + (H - K) * np.exp(-r * T) * norm.pdf(d2_h) * d1_h / sigma
    )
    return float(vega)


def greeks_european_barrier_up_and_out_call(params: dict[str, float]) -> dict[str, float]:
    r"""Computes the Greeks of a terminal barrier up-and-out call option using exact analytical formulas.

    Args:
        params (dict[str, float]): Dictionary containing option and market parameters:
            - "S0" (float): Initial stock price ($S_0 > 0$).
            - "K" (float): Strike price ($K > 0$).
            - "T" (float): Time to maturity in years ($T > 0$).
            - "r" (float): Annualized risk-free interest rate in decimal form (e.g., 0.05 for 5%).
            - "q" (float): Annualized continuous dividend yield in decimal form (e.g., 0.02 for 2%).
            - "sigma" (float): Annualized volatility in decimal form ($\sigma > 0$, e.g., 0.20 for 20%).
            - "H" (float): Terminal barrier level ($H > K$).

    Returns:
        dict[str, float]: Dictionary containing computed analytical Greeks:
            - "delta" (float): First derivative w.r.t. spot price ($\partial V / \partial S_0$).
            - "gamma" (float): Second derivative w.r.t. spot price ($\partial^2 V / \partial S_0^2$).
            - "vega" (float): First derivative w.r.t. volatility ($\partial V / \partial \sigma$).
    """
    S0, K, T, r, q, sigma, H = (
        params["S0"],
        params["K"],
        params["T"],
        params["r"],
        params["q"],
        params["sigma"],
        params["H"],
    )

    return {
        "delta": delta_barrier_up_and_out_call(S0, K, T, r, q, sigma, H),
        "gamma": gamma_barrier_up_and_out_call(S0, K, T, r, q, sigma, H),
        "vega": vega_barrier_up_and_out_call(S0, K, T, r, q, sigma, H),
    }
