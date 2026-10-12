
example_european_call_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Call European ATM"
}

example_european_call_1M = {
    "S0": 100,
    "K": 100,
    "T": 1 / 12,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Call European 1M"
}

example_european_digital_1M = {
    "S0": 100,
    "K": 100,
    "T": 1 / 12,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Digital European 1M"
}

# One trading day (T = 1/252): the extreme short-maturity case used in the replications and gap sweep
example_european_call_1D = {
    "S0": 100,
    "K": 100,
    "T": 1 / 252,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Call European 1D"
}

example_european_digital_1D = {
    "S0": 100,
    "K": 100,
    "T": 1 / 252,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Digital European 1D"
}

example_european_digital_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Digital European ATM"
}

# Call with terminal knock-out: (S_T - K)^+ * 1_{S_T <= H}, monitored at maturity only.
example_european_call_terminal_knockout = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "H": 105.25,
    "sigma": 0.20,
    "name": "Call with terminal knock-out"
}

# ==============================================================================
# Asian Options (Discrete Arithmetic Average, Monthly Frequency m=12)
# ==============================================================================

# Vanilla Asian Calls
example_asian_call_ITM = {
    "S0": 100,
    "K": 80,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.25,
    "type": "discrete_arithmetic_average",
    "frequency": 12,
    "name": "Call Asian ITM"
}

example_asian_call_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "type": "discrete_arithmetic_average", # can be "discrete_geometric_average", "continuous_arithmetic_average", "continuous_geometric_average"
    "frequency": 12, # 1 per month (np.inf for continuous)
    "name": "Call Asian ATM"
}

example_asian_call_OTM = {
    "S0": 100,
    "K": 125,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.25,
    "type": "discrete_arithmetic_average",
    "frequency": 12,
    "name": "Call Asian OTM"
}

# Digital Asian Calls (Cash-or-Nothing, Payoff: 1_{A > K})
example_asian_digital_ITM = {
    "S0": 100,
    "K": 80,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.25,
    "type": "discrete_arithmetic_average",
    "frequency": 12,
    "name": "Digital Asian ITM"
}

example_asian_digital_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "type": "discrete_arithmetic_average",
    "frequency": 12,
    "name": "Digital Asian ATM"
}

example_asian_digital_OTM = {
    "S0": 100,
    "K": 125,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.25,
    "type": "discrete_arithmetic_average",
    "frequency": 12,
    "name": "Digital Asian OTM"
}

# Barrier Asian Calls (Terminal Up-and-Out on Arithmetic Average, Payoff: (A - K)^+ * 1_{A <= H})
example_asian_barrier_up_and_out_ITM = {
    "S0": 100,
    "K": 80,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.25,
    "H": 135.0,
    "type": "discrete_arithmetic_average",
    "frequency": 12,
    "name": "Barrier Asian Up and Out ITM"
}

example_asian_barrier_up_and_out_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "H": 135.0,
    "type": "discrete_arithmetic_average",
    "frequency": 12,
    "name": "Barrier Asian Up and Out ATM"
}

example_asian_barrier_up_and_out_OTM = {
    "S0": 100,
    "K": 125,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.25,
    "H": 135.0,
    "type": "discrete_arithmetic_average",
    "frequency": 12,
    "name": "Barrier Asian Up and Out OTM"
}

asian_products = {
    "Call Asian ITM": example_asian_call_ITM,
    "Call Asian ATM": example_asian_call_ATM,
    "Call Asian OTM": example_asian_call_OTM,
    "Digital Asian ITM": example_asian_digital_ITM,
    "Digital Asian ATM": example_asian_digital_ATM,
    "Digital Asian OTM": example_asian_digital_OTM,
    "Barrier Asian Up and Out ITM": example_asian_barrier_up_and_out_ITM,
    "Barrier Asian Up and Out ATM": example_asian_barrier_up_and_out_ATM,
    "Barrier Asian Up and Out OTM": example_asian_barrier_up_and_out_OTM,
}
