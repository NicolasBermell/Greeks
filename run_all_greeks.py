from pandas import DataFrame, concat
from Methods.MonteCarlo.FDMC_european import (
    FDMC_geeks_european_barrier_up_and_out_call,
    FDMC_geeks_european_call,
    FDMC_geeks_european_digital_call,
)
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
from Methods.sampling import GREEKS
from products import *
from config import *

# (product params, FDMC fn, standard Malliavin fn, Fournié fn, Fournié bandwidths)
EUROPEAN_PRODUCTS = [
    (example_european_call_ATM, FDMC_geeks_european_call, malliavin_greeks_european_call,
     malliavin_fournier_greeks_european_call, MALLIAVIN_FOURNIER_DELTA_STRIKE),
    (example_european_digital_ATM, FDMC_geeks_european_digital_call, malliavin_greeks_european_digital_call,
     malliavin_fournier_greeks_european_digital_call, MALLIAVIN_FOURNIER_DELTA_STRIKE),
    (example_european_call_terminal_knockout, FDMC_geeks_european_barrier_up_and_out_call,
     malliavin_greeks_european_barrier_up_and_out_call,
     malliavin_fournier_greeks_european_barrier_up_and_out_call, MALLIAVIN_FOURNIER_BARRIER_DELTA),
]

# Each Greek is followed by its per-path variance and standard error.
GREEK_COLUMNS = [c for g in GREEKS for c in (g, f"{g}_var", f"{g}_se")]


def _results_frame(rows: dict[str, dict[str, float]]) -> DataFrame:
    df = DataFrame(rows).T[GREEK_COLUMNS]
    df.index.name = "Product"
    return df


def run_all_FDMC():

    results = DataFrame()

    for N in FDMC_N_SIMS_ARRAY:
        for eps in FDMC_EPS_ARRAY:
            print(f"running FDMC N={N}, eps={eps}")
            current_results = _results_frame({
                params["name"]: fdmc_fn(params, eps=eps, N=N, seed=FDMC_SEED, return_stats=True)
                for params, fdmc_fn, _, _, _ in EUROPEAN_PRODUCTS
            })
            current_results["Method"] = "FDMC"
            current_results["N"] = N
            current_results["eps"] = eps

            results = concat([results, current_results])

    results.to_csv(FDMC_RESULTS_FILE)


def run_all_malliavin():

    results = DataFrame()

    for N in MALLIAVIN_N_SIMS_ARRAY:
        print(f"running Standard Malliavin N={N}")
        current_results = _results_frame({
            params["name"]: mall_fn(params, N=N, seed=MALLIAVIN_SEED, return_stats=True)
            for params, _, mall_fn, _, _ in EUROPEAN_PRODUCTS
        })
        current_results["Method"] = "Malliavin"
        current_results["N"] = N

        results = concat([results, current_results])

    results.to_csv(MALLIAVIN_RESULTS_FILE)


def run_all_malliavin_fournier(cutoff_type: str = MALLIAVIN_FOURNIER_CUTOFF_TYPE):

    results = DataFrame()

    for N in MALLIAVIN_N_SIMS_ARRAY:
        print(f"running Fournié Localized Malliavin N={N}, cutoff={cutoff_type}")
        current_results = _results_frame({
            params["name"]: four_fn(
                params, N=N, seed=MALLIAVIN_SEED, cutoff_type=cutoff_type, delta_strike=bandwidths, return_stats=True
            )
            for params, _, _, four_fn, bandwidths in EUROPEAN_PRODUCTS
        })
        current_results["Method"] = "Malliavin Fournié"
        current_results["N"] = N
        current_results["cutoff_type"] = cutoff_type
        current_results["delta_strike"] = "decoupled"

        results = concat([results, current_results])

    results.to_csv(MALLIAVIN_FOURNIER_RESULTS_FILE)


def run_all_asian():
    from run_asian_benchmarks import main as run_asian_main
    run_asian_main()


def run_all_european():
    run_all_FDMC()
    run_all_malliavin()
    run_all_malliavin_fournier()


def run_all_greeks():
    run_all_european()
    run_all_asian()


if __name__ == "__main__":
    run_all_greeks()
