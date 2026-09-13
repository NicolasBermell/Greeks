import QuantLib as ql


def get_american_option(
    params: dict[str, float],
    t_grid: int = 200,
    x_grid: int = 200,
    option_type: ql.Option.Type = ql.Option.Call,
) -> ql.VanillaOption:
    """Builds and returns a QuantLib American option configured with a finite difference PDE engine.

    Args:
        params (dict[str, float]): Market and contract parameters (S0, K, T, r, q, sigma).
        t_grid (int): Number of time steps in the PDE mesh.
        x_grid (int): Number of spatial asset price points in the PDE mesh.
        option_type (ql.Option.Type): ql.Option.Call or ql.Option.Put.

    Returns:
        ql.VanillaOption: Priced option with attached quotes and pricing engine.
    """
    S0, K, T, r, q, sigma = (
        float(params["S0"]),
        float(params["K"]),
        float(params["T"]),
        float(params["r"]),
        float(params["q"]),
        float(params["sigma"]),
    )

    calendar = ql.NullCalendar()
    day_counter = ql.Actual365Fixed()
    settlement_date = ql.Date(1, ql.January, 2026)
    ql.Settings.instance().evaluationDate = settlement_date
    maturity_date = settlement_date + ql.Period(max(1, int(round(T * 365))), ql.Days)

    # Market quotes wrapped in SimpleQuote to enable efficient bump & reprice
    spot_quote = ql.SimpleQuote(S0)
    rate_quote = ql.SimpleQuote(r)
    div_quote = ql.SimpleQuote(q)
    vol_quote = ql.SimpleQuote(sigma)

    spot_handle = ql.QuoteHandle(spot_quote)
    rate_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(settlement_date, ql.QuoteHandle(rate_quote), day_counter)
    )
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(settlement_date, ql.QuoteHandle(div_quote), day_counter)
    )
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(settlement_date, calendar, ql.QuoteHandle(vol_quote), day_counter)
    )

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_ts, rate_ts, vol_ts)
    payoff = ql.PlainVanillaPayoff(option_type, K)
    exercise = ql.AmericanExercise(settlement_date, maturity_date)
    american_option = ql.VanillaOption(payoff, exercise)

    fd_engine = ql.FdBlackScholesVanillaEngine(bsm_process, t_grid, x_grid)
    american_option.setPricingEngine(fd_engine)

    # Attach references to the option object to prevent garbage collection
    # and provide direct access for bump & reprice sensitivities
    american_option.spot_quote = spot_quote
    american_option.rate_quote = rate_quote
    american_option.div_quote = div_quote
    american_option.vol_quote = vol_quote
    american_option.bsm_process = bsm_process
    american_option.fd_engine = fd_engine

    return american_option


def get_american_digital_option(
    params: dict[str, float],
    t_grid: int = 200,
    x_grid: int = 200,
    cash_payoff: float = 1.0,
    option_type: ql.Option.Type = ql.Option.Call,
) -> ql.VanillaOption:
    """Builds and returns a QuantLib American digital (cash-or-nothing) option."""
    S0, K, T, r, q, sigma = (
        float(params["S0"]),
        float(params["K"]),
        float(params["T"]),
        float(params["r"]),
        float(params["q"]),
        float(params["sigma"]),
    )

    calendar = ql.NullCalendar()
    day_counter = ql.Actual365Fixed()
    settlement_date = ql.Date(1, ql.January, 2026)
    ql.Settings.instance().evaluationDate = settlement_date
    maturity_date = settlement_date + ql.Period(max(1, int(round(T * 365))), ql.Days)

    spot_quote = ql.SimpleQuote(S0)
    rate_quote = ql.SimpleQuote(r)
    div_quote = ql.SimpleQuote(q)
    vol_quote = ql.SimpleQuote(sigma)

    spot_handle = ql.QuoteHandle(spot_quote)
    rate_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(settlement_date, ql.QuoteHandle(rate_quote), day_counter)
    )
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(settlement_date, ql.QuoteHandle(div_quote), day_counter)
    )
    vol_ts = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(settlement_date, calendar, ql.QuoteHandle(vol_quote), day_counter)
    )

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_ts, rate_ts, vol_ts)
    payoff = ql.CashOrNothingPayoff(option_type, K, float(cash_payoff))
    exercise = ql.AmericanExercise(settlement_date, maturity_date)
    american_digital_option = ql.VanillaOption(payoff, exercise)

    fd_engine = ql.FdBlackScholesVanillaEngine(bsm_process, t_grid, x_grid)
    american_digital_option.setPricingEngine(fd_engine)

    american_digital_option.spot_quote = spot_quote
    american_digital_option.rate_quote = rate_quote
    american_digital_option.div_quote = div_quote
    american_digital_option.vol_quote = vol_quote
    american_digital_option.bsm_process = bsm_process
    american_digital_option.fd_engine = fd_engine

    return american_digital_option


def get_vega(option: ql.VanillaOption, vol_bump: float = 1e-4) -> float:
    """Computes Vega via central finite difference bump & reprice on the option's volatility quote."""
    if not hasattr(option, "vol_quote"):
        raise AttributeError("Option object does not have an attached vol_quote.")
    vol_quote = option.vol_quote
    vol_orig = vol_quote.value()
    vol_quote.setValue(vol_orig + vol_bump)
    npv_up = option.NPV()
    vol_quote.setValue(vol_orig - vol_bump)
    npv_down = option.NPV()
    vol_quote.setValue(vol_orig)
    return (npv_up - npv_down) / (2.0 * vol_bump)



def PDE_ql_geeks_american_call(
    params: dict[str, float],
    t_grid: int = 200,
    x_grid: int = 200,
    vol_bump: float = 1e-4,
) -> dict[str, float]:
    """Computes Delta, Gamma, and Vega for an American call option using 1D Finite Difference PDE."""
    am_option = get_american_option(params, t_grid, x_grid, option_type=ql.Option.Call)
    delta = am_option.delta()
    gamma = am_option.gamma()
    vega = get_vega(am_option, vol_bump)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }


def PDE_ql_geeks_american_put(
    params: dict[str, float],
    t_grid: int = 200,
    x_grid: int = 200,
    vol_bump: float = 1e-4,
) -> dict[str, float]:
    """Computes Delta, Gamma, and Vega for an American put option using 1D Finite Difference PDE."""
    am_option = get_american_option(params, t_grid, x_grid, option_type=ql.Option.Put)
    delta = am_option.delta()
    gamma = am_option.gamma()
    vega = get_vega(am_option, vol_bump)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }


def PDE_ql_geeks_american_digital_call(
    params: dict[str, float],
    t_grid: int = 200,
    x_grid: int = 200,
    vol_bump: float = 1e-4,
) -> dict[str, float]:
    """Computes Delta, Gamma, and Vega for an American digital call option using 1D Finite Difference PDE."""
    am_option = get_american_digital_option(params, t_grid, x_grid, option_type=ql.Option.Call)
    delta = am_option.delta()
    gamma = am_option.gamma()
    vega = get_vega(am_option, vol_bump)

    return {
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
    }

def PDE_ql_all_greeks_american(
    params: dict[str, float],
    t_grid: int = 200,
    x_grid: int = 200,
    vol_bump: float = 1e-4,
) -> dict[str, dict[str, float]]:
    """Calculates all Greeks for American Call, Put, and Digital Call options."""
    return {
        "call": PDE_ql_geeks_american_call(params, t_grid, x_grid, vol_bump),
        "put": PDE_ql_geeks_american_put(params, t_grid, x_grid, vol_bump),
        "digital": PDE_ql_geeks_american_digital_call(params, t_grid, x_grid, vol_bump),
    }


if __name__ == "__main__":
    example_american_call_ATM = {
        "S0": 100,
        "K": 100,
        "T": 1,
        "r": 0.05,
        "q": 0.02,
        "sigma": 0.20,
        "name": "Call American ATM",
    }

    print("--- American Call Option Greeks (PDE) ---")
    call_greeks = PDE_ql_geeks_american_call(example_american_call_ATM)
    for greek, val in call_greeks.items():
        print(f"{greek.capitalize():<6}: {val:.6f}")

    print("\n--- American Put Option Greeks (PDE) ---")
    put_greeks = PDE_ql_geeks_american_put(example_american_call_ATM)
    for greek, val in put_greeks.items():
        print(f"{greek.capitalize():<6}: {val:.6f}")

    print("\n--- American Digital Call Option Greeks (PDE) ---")
    digital_greeks = PDE_ql_geeks_american_digital_call(example_american_call_ATM)
    for greek, val in digital_greeks.items():
        print(f"{greek.capitalize():<6}: {val:.6f}")