from pandas import DataFrame, concat
from Methods.MonteCarlo.FDMC_european import *
from Methods.Malliavin.malliavin_european_standard import (
    malliavin_greeks_european_barrier_up_and_out_call,
    malliavin_greeks_european_call,
    malliavin_greeks_european_digital_call,
)
from Methods.Malliavin.malliavin_european_fournier import (
    malliavin_fournier_greeks_european_barrier_up_and_out_call,
    malliavin_fournier_greeks_european_call,
    malliavin_fournier_greeks_european_digital_call,
)
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


def run_all_malliavin():

    results = DataFrame()

    for N in MALLIAVIN_N_SIMS_ARRAY:
        print(f"running Standard Malliavin N={N}")
        malliavin_call = malliavin_greeks_european_call(example_european_call_ATM, N=N, seed=MALLIAVIN_SEED)
        malliavin_digital = malliavin_greeks_european_digital_call(example_european_digital_ATM, N=N, seed=MALLIAVIN_SEED)
        malliavin_barrier = malliavin_greeks_european_barrier_up_and_out_call(example_european_barrier_up_and_out_95, N=N, seed=MALLIAVIN_SEED)
        current_results = DataFrame({
            example_european_call_ATM["name"]: malliavin_call,
            example_european_digital_ATM["name"]: malliavin_digital,
            example_european_barrier_up_and_out_95["name"]: malliavin_barrier,
        }).T
        current_results.index.name = "Product"
        current_results["Method"] = "Malliavin"
        current_results["N"] = N

        results = concat([results, current_results])

    results.to_csv(MALLIAVIN_RESULTS_FILE)


def run_all_malliavin_fournier(
    cutoff_type: str = MALLIAVIN_FOURNIER_CUTOFF_TYPE,
    delta_strike: float = MALLIAVIN_FOURNIER_DELTA_STRIKE,
    barrier_delta: float = MALLIAVIN_FOURNIER_BARRIER_DELTA,
):

    results = DataFrame()

    for N in MALLIAVIN_N_SIMS_ARRAY:
        print(f"running Fournie Localized Malliavin N={N}, cutoff={cutoff_type}, delta={delta_strike}, barrier_delta={barrier_delta}")
        call_res = malliavin_fournier_greeks_european_call(
            example_european_call_ATM, N=N, seed=MALLIAVIN_SEED, cutoff_type=cutoff_type, delta_strike=delta_strike
        )
        dig_res = malliavin_fournier_greeks_european_digital_call(
            example_european_digital_ATM, N=N, seed=MALLIAVIN_SEED, cutoff_type=cutoff_type, delta_strike=delta_strike
        )
        bar_res = malliavin_fournier_greeks_european_barrier_up_and_out_call(
            example_european_barrier_up_and_out_95, N=N, seed=MALLIAVIN_SEED, cutoff_type=cutoff_type, delta_strike=barrier_delta
        )
        current_results = DataFrame({
            example_european_call_ATM["name"]: call_res,
            example_european_digital_ATM["name"]: dig_res,
            example_european_barrier_up_and_out_95["name"]: bar_res,
        }).T
        current_results.index.name = "Product"
        current_results["Method"] = "Malliavin Fournie"
        current_results["N"] = N
        current_results["cutoff_type"] = cutoff_type
        current_results["delta_strike"] = [delta_strike, delta_strike, barrier_delta]

        results = concat([results, current_results])

    results.to_csv(MALLIAVIN_FOURNIER_RESULTS_FILE)
    

def run_all_european():
    run_all_FDMC()
    run_all_malliavin()
    run_all_malliavin_fournier()

def run_all_greeks():
    run_all_european()


if __name__ == "__main__":
    run_all_greeks()