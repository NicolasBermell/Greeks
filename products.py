
example_european_call_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Call European ATM"
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

example_european_barrier_up_and_out_95 = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "H": 105.25,
    "sigma": 0.20,
    "name": "Barrier European Up and Out 95"
}

example_american_call_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Call American ATM"
}

example_american_digital_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "name": "Digital American ATM"
}

example_american_barrier_up_and_out_95 = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "H": 105.25,
    "sigma": 0.20,
    "name": "Barrier American Up and Out 95"
}

example_asian_call_ATM = {
    "S0": 100,
    "K": 100,
    "T": 1,
    "r": 0.05,
    "q": 0.02,
    "sigma": 0.20,
    "type": "discrete_arithmetic_average", # can be "discrete_geometric_average", "continuous_arithmetic_average", "continuous_geometric_average"
    "frequency": 12, #1 per month (np.inf for continuous)
    "name": "Call Asian ATM"
}