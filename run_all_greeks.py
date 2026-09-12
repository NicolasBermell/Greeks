from pandas import DataFrame, concat
from Methods.MonteCarlo.FDMC import *
from products import *
from config import *


def run_all_FDMC():

    results = DataFrame()

    for N in FDMC_N_SIMS_ARRAY:
        for eps in FDMC_EPS_ARRAY:
            print(f"running FDMC N={N}, eps={eps}")
            FDMC_european_call = FDMC_geeks_european_call(example_european_call_ATM, eps=eps, N=N, seed=FDMC_SEED)
            FDMC_european_digital = FDMC_geeks_european_digital_call(example_european_digital_ATM, eps=eps, N=N, seed=FDMC_SEED)
            FDMC_european_barrier_up_and_out = FDMC_geeks_european_barrier_up_and_out_call(example_european_barrier_up_and_out_95, eps=eps, N=N, seed=FDMC_SEED)
            current_results = DataFrame({
                    example_european_call_ATM["name"]: FDMC_european_call,
                    example_european_digital_ATM["name"]: FDMC_european_digital,
                    example_european_barrier_up_and_out_95["name"]: FDMC_european_barrier_up_and_out,
                }).T
            current_results.index.name = "Product"
            current_results["Method"] = "FDMC"
            current_results["N"] = N
            current_results["eps"] = eps

            results = concat([results, current_results])
    
    results.to_csv(FDMC_RESULTS_FILE)
    

def run_all_european():
    run_all_FDMC()

def run_all_greeks():
    run_all_european()