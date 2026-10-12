from pandas import DataFrame, concat
from Methods.MonteCarlo.FDMC_european import (
    FDMC_greeks_european_call_terminal_knockout,
    FDMC_greeks_european_call,
    FDMC_greeks_european_digital_call,
)
from Methods.Malliavin.malliavin_european_standard import (
    malliavin_greeks_european_call_terminal_knockout,
    malliavin_greeks_european_call,
    malliavin_greeks_european_digital_call,
)
from Methods.Malliavin.malliavin_european_fournier import (
    malliavin_fournier_greeks_european_call_terminal_knockout,
    malliavin_fournier_greeks_european_call,
    malliavin_fournier_greeks_european_digital_call,
)
from Methods.sampling import GREEK_STAT_COLUMNS
from products import (
    example_european_call_ATM,
    example_european_call_terminal_knockout,
    example_european_digital_ATM,
)
from config import (
    FDMC_EPS_ARRAY,
    FDMC_N_SIMS_ARRAY,
    FDMC_RESULTS_FILE,
    FDMC_SEED,
    MALLIAVIN_FOURNIER_CUTOFF_TYPE,
    MALLIAVIN_FOURNIER_RESULTS_FILE,
    MALLIAVIN_N_SIMS_ARRAY,
    MALLIAVIN_RESULTS_FILE,
    MALLIAVIN_SEED,
)

# (product params, FDMC fn, standard Malliavin fn, Fournié fn). Fournié uses the default
# bandwidths of fournier_bandwidths(params), capped at (H - K) / 2 for the knock-out.
EUROPEAN_PRODUCTS = [
    (example_european_call_ATM, FDMC_greeks_european_call, malliavin_greeks_european_call,
     malliavin_fournier_greeks_european_call),
    (example_european_digital_ATM, FDMC_greeks_european_digital_call, malliavin_greeks_european_digital_call,
     malliavin_fournier_greeks_european_digital_call),
    (example_european_call_terminal_knockout, FDMC_greeks_european_call_terminal_knockout,
     malliavin_greeks_european_call_terminal_knockout,
     malliavin_fournier_greeks_european_call_terminal_knockout),
]

def _results_frame(rows: dict[str, dict[str, float]]) -> DataFrame:
    df = DataFrame(rows).T[GREEK_STAT_COLUMNS]
    df.index.name = "Product"
    return df


def run_all_FDMC():

    results = DataFrame()

    for N in FDMC_N_SIMS_ARRAY:
        for eps in FDMC_EPS_ARRAY:
            print(f"running FDMC N={N}, eps={eps}")
            current_results = _results_frame({
                params["name"]: fdmc_fn(params, eps=eps, N=N, seed=FDMC_SEED, return_stats=True)
                for params, fdmc_fn, _, _ in EUROPEAN_PRODUCTS
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
            for params, _, mall_fn, _ in EUROPEAN_PRODUCTS
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
            params["name"]: four_fn(params, N=N, seed=MALLIAVIN_SEED, cutoff_type=cutoff_type, return_stats=True)
            for params, _, _, four_fn in EUROPEAN_PRODUCTS
        })
        current_results["Method"] = "Malliavin Fournié"
        current_results["N"] = N
        current_results["cutoff_type"] = cutoff_type
        current_results["bandwidth"] = "default"

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
