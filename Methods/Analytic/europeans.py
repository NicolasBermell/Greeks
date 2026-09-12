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

# Barrier Option (Up-and-Out Call)
import math

def delta_barrier_up_and_out_call(S: float, K: float, T: float, r: float, q: float, sigma: float, H: float) -> float:
    """Exact analytical Delta for European Up-and-Out Call."""
    if S >= H or K >= H:
        return 0.0
    return 2*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*(-q + r - 1/2*sigma**2)*math.exp(-T*r)/(S*sigma**2) - 2*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*(-q + r - 1/2*sigma**2)*math.exp(-T*r)/(S*sigma**2) - 1/2*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) + (1/2)*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) - 1/2*math.sqrt(2)*K*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/K)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) + (1/2)*math.sqrt(2)*K*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/H)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) - (H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q) + (H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q) + (H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q) - (H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q) - ((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/H)/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q) + ((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/K)/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q) + (1/2)*math.sqrt(2)*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*math.sqrt(T)*sigma) - 1/2*math.sqrt(2)*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*math.sqrt(T)*sigma) + (1/2)*math.sqrt(2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/K)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*math.sqrt(T)*sigma) - 1/2*math.sqrt(2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/H)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*math.sqrt(T)*sigma)

def gamma_barrier_up_and_out_call(S: float, K: float, T: float, r: float, q: float, sigma: float, H: float) -> float:
    """Exact analytical Gamma for European Up-and-Out Call."""
    if S >= H or K >= H:
        return 0.0
    return -2*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*(-q + r - 1/2*sigma**2)*math.exp(-T*r)/(S**2*sigma**2) + 2*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*(-q + r - 1/2*sigma**2)*math.exp(-T*r)/(S**2*sigma**2) - 4*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*(-q + r - 1/2*sigma**2)**2*math.exp(-T*r)/(S**2*sigma**4) + 4*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*(-q + r - 1/2*sigma**2)**2*math.exp(-T*r)/(S**2*sigma**4) + (1/2)*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*T*sigma**2) - 1/2*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*T*sigma**2) - 1/2*math.sqrt(2)*K*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/H)/(math.sqrt(T)*sigma))*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/H)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*T*sigma**2) + (1/2)*math.sqrt(2)*K*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/K)/(math.sqrt(T)*sigma))*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/K)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*T*sigma**2) + (1/2)*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*math.sqrt(T)*sigma) - 1/2*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*math.sqrt(T)*sigma) + (1/2)*math.sqrt(2)*K*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/K)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*math.sqrt(T)*sigma) - 1/2*math.sqrt(2)*K*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/H)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*math.sqrt(T)*sigma) + 2*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*(-q + r - 1/2*sigma**2)*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*math.sqrt(T)*sigma**3) - 2*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*(-q + r - 1/2*sigma**2)*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S**2*math.sqrt(T)*sigma**3) + (H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)**2*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q)/S - (H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)**2*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q)/S - (H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q)/S + (H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q)/S - 1/2*math.sqrt(2)*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*T*sigma**2) + (1/2)*math.sqrt(2)*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*T*sigma**2) + (1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/H)/(math.sqrt(T)*sigma))*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/H)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*T*sigma**2) - 1/2*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/K)/(math.sqrt(T)*sigma))*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/K)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*T*sigma**2) - math.sqrt(2)*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) + math.sqrt(2)*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) + (1/2)*math.sqrt(2)*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) - 1/2*math.sqrt(2)*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) + (1/2)*math.sqrt(2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/K)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma) - 1/2*math.sqrt(2)*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/H)/(math.sqrt(T)*sigma))**2)/(math.sqrt(math.pi)*S*math.sqrt(T)*sigma)

def vega_barrier_up_and_out_call(S: float, K: float, T: float, r: float, q: float, sigma: float, H: float) -> float:
    """Exact analytical Vega for European Up-and-Out Call."""
    if S >= H or K >= H:
        return 0.0
    return -K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*(-2/sigma - 4*(-q + r - 1/2*sigma**2)/sigma**3)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*r)*math.log(H/S) + K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*(-2/sigma - 4*(-q + r - 1/2*sigma**2)/sigma**3)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*r)*math.log(H/S) - 1/2*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*(math.sqrt(T)*sigma*(-1/sigma - 2*(-q + r - 1/2*sigma**2)/sigma**3) + math.sqrt(T)*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T) - math.log(H/S)/(math.sqrt(T)*sigma**2))*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H/S)/(math.sqrt(T)*sigma))**2)/math.sqrt(math.pi) + (1/2)*math.sqrt(2)*K*(H/S)**(2*(-q + r - 1/2*sigma**2)/sigma**2)*(math.sqrt(T)*sigma*(-1/sigma - 2*(-q + r - 1/2*sigma**2)/sigma**3) + math.sqrt(T)*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T) - math.log(H**2/(K*S))/(math.sqrt(T)*sigma**2))*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/math.sqrt(math.pi) + (1/2)*math.sqrt(2)*K*(math.sqrt(T)*sigma*(-1/sigma - 2*(-q + r - 1/2*sigma**2)/sigma**3) + math.sqrt(T)*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T) - math.log(S/H)/(math.sqrt(T)*sigma**2))*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/H)/(math.sqrt(T)*sigma))**2)/math.sqrt(math.pi) - 1/2*math.sqrt(2)*K*(math.sqrt(T)*sigma*(-1/sigma - 2*(-q + r - 1/2*sigma**2)/sigma**3) + math.sqrt(T)*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T) - math.log(S/K)/(math.sqrt(T)*sigma**2))*math.exp(-T*r)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.sqrt(T)*sigma + math.log(S/K)/(math.sqrt(T)*sigma))**2)/math.sqrt(math.pi) + S*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(-2/sigma - 4*(-q + r - 1/2*sigma**2)/sigma**3)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q)*math.log(H/S) - S*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(-2/sigma - 4*(-q + r - 1/2*sigma**2)/sigma**3)*((1/2)*math.erf((1/2)*math.sqrt(2)*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))) + 1/2)*math.exp(-T*q)*math.log(H/S) + (1/2)*math.sqrt(2)*S*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(math.sqrt(T)*sigma*(-1/sigma - 2*(-q + r - 1/2*sigma**2)/sigma**3) + math.sqrt(T)*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.log(H/S)/(math.sqrt(T)*sigma**2))*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H/S)/(math.sqrt(T)*sigma))**2)/math.sqrt(math.pi) - 1/2*math.sqrt(2)*S*(H/S)**(2 + 2*(-q + r - 1/2*sigma**2)/sigma**2)*(math.sqrt(T)*sigma*(-1/sigma - 2*(-q + r - 1/2*sigma**2)/sigma**3) + math.sqrt(T)*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.log(H**2/(K*S))/(math.sqrt(T)*sigma**2))*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(H**2/(K*S))/(math.sqrt(T)*sigma))**2)/math.sqrt(math.pi) - 1/2*math.sqrt(2)*S*(math.sqrt(T)*sigma*(-1/sigma - 2*(-q + r - 1/2*sigma**2)/sigma**3) + math.sqrt(T)*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.log(S/H)/(math.sqrt(T)*sigma**2))*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/H)/(math.sqrt(T)*sigma))**2)/math.sqrt(math.pi) + (1/2)*math.sqrt(2)*S*(math.sqrt(T)*sigma*(-1/sigma - 2*(-q + r - 1/2*sigma**2)/sigma**3) + math.sqrt(T)*(1 + (-q + r - 1/2*sigma**2)/sigma**2) - math.log(S/K)/(math.sqrt(T)*sigma**2))*math.exp(-T*q)*math.exp(-1/2*(math.sqrt(T)*sigma*(1 + (-q + r - 1/2*sigma**2)/sigma**2) + math.log(S/K)/(math.sqrt(T)*sigma))**2)/math.sqrt(math.pi)

def greeks_european_barrier_up_and_out_call(params: dict[str, float]) -> dict[str, float]:
    r"""Computes the Greeks of a barrier up-and-out call option using exact analytical formulas.

    Args:
        params (dict[str, float]): Dictionary containing option and market parameters:
            - "S0" (float): Initial stock price ($S_0 > 0$).
            - "K" (float): Strike price ($K > 0$).
            - "T" (float): Time to maturity in years ($T > 0$).
            - "r" (float): Annualized risk-free interest rate in decimal form (e.g., 0.05 for 5%).
            - "q" (float): Annualized continuous dividend yield in decimal form (e.g., 0.02 for 2%).
            - "sigma" (float): Annualized volatility in decimal form ($\sigma > 0$, e.g., 0.20 for 20%).
            - "H" (float): Up-and-out barrier level ($H > S_0$).

    Returns:
        dict[str, float]: Dictionary containing computed analytical Greeks:
            - "delta" (float): First derivative w.r.t. spot price ($\partial V / \partial S_0$).
            - "gamma" (float): Second derivative w.r.t. spot price ($\partial^2 V / \partial S_0^2$).
            - "vega" (float): First derivative w.r.t. volatility ($\partial V / \partial \sigma$).
    """
    S0, K, T, r, q, sigma, H = params["S0"], params["K"], params["T"], params["r"], params["q"], params["sigma"], params["H"]

    return {
        "delta": delta_barrier_up_and_out_call(S0, K, T, r, q, sigma, H),
        "gamma": gamma_barrier_up_and_out_call(S0, K, T, r, q, sigma, H),
        "vega": vega_barrier_up_and_out_call(S0, K, T, r, q, sigma, H),
    }
