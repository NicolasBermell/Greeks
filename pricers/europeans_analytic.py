import QuantLib as ql

# Vanilla Call
def european_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> ql.VanillaOption:
    today = ql.Date(12, 9, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    days = int(round(T * 365))
    maturity = today + ql.Period(days, ql.Days)

    payoff = ql.VanillaPayoff(ql.Option.Call, K)
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

# Digital Call
def european_digital_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float) -> ql.VanillaOption:
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

# Barrier Option (Up-and-Out Call)

def european_barrier_up_and_out_call_option(S0: float, K: float, T: float, r: float, q: float, sigma: float, H: float) -> ql.BarrierOption:
    today = ql.Date(12, 9, 2026)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()
    days = int(round(T * 365))
    maturity = today + ql.Period(days, ql.Days)

    option = ql.BarrierOption(ql.Barrier.UpOut, H, 0.0, ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(maturity))

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S0))
    rate_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    div_ts = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))
    vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(today, ql.NullCalendar(), sigma, day_count))

    bsm_process = ql.BlackScholesMertonProcess(spot_handle, div_ts, rate_ts, vol_ts)
    engine = ql.AnalyticBarrierEngine(bsm_process)
    option.setPricingEngine(engine)
    return option

    