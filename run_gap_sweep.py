"""Per-path variance of the Fournié estimators as a function of the gap.

For the call and the digital at T = 1 and T = 1/252, and for each gap
    l = multiplier * S0 * sigma * sqrt(T)     (GAP_SWEEP_MULTIPLIERS),
the same l is used for Delta, Gamma and Vega with the Gaussian cutoff. All gaps of a product use
the same paths, so the curves are smooth in l. Standard Malliavin (the limit l -> infinity) and
FDMC at eps = 0.01 are computed on the same paths as references.

Outputs:
    results/gap_sweep.csv             one row per (product, Greek, estimator, gap)
    results/figures/gap_sweep.png     variance relative to standard Malliavin vs l / (S0 sigma sqrt(T))
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import (
    FDMC_FIXED_EPS,
    FIGURES_DIR,
    GAP_SWEEP_FILE,
    GAP_SWEEP_MULTIPLIERS,
    GAP_SWEEP_N,
    GAP_SWEEP_SEED,
)
from Methods import payoffs
from Methods.Analytic.europeans import greeks_european_call, greeks_european_digital_call
from Methods.Malliavin import (
    malliavin_fournier_samples_european_call,
    malliavin_fournier_samples_european_digital_call,
    malliavin_samples_european,
)
from Methods.MonteCarlo.FDMC_european import FDMC_samples_european
from Methods.sampling import GREEKS, generate_paths
from products import (
    example_european_call_1D,
    example_european_call_ATM,
    example_european_digital_1D,
    example_european_digital_ATM,
)
from Utilities.plotting import plot_gap_sweep

# (params, payoff, Fournié samples fn, analytical Greeks fn)
SWEEP_PRODUCTS = [
    (example_european_call_ATM, payoffs.call, malliavin_fournier_samples_european_call, greeks_european_call),
    (example_european_call_1D, payoffs.call, malliavin_fournier_samples_european_call, greeks_european_call),
    (example_european_digital_ATM, payoffs.digital_call, malliavin_fournier_samples_european_digital_call,
     greeks_european_digital_call),
    (example_european_digital_1D, payoffs.digital_call, malliavin_fournier_samples_european_digital_call,
     greeks_european_digital_call),
]


def _summarize(samples: np.ndarray) -> dict[str, float]:
    """Mean, per-path variance and the standard error of that variance (from the 4th moment)."""
    n = len(samples)
    mean = float(np.mean(samples))
    centered = samples - mean
    var = float(np.sum(centered**2) / (n - 1))
    m4 = float(np.mean(centered**4))
    return {"mean": mean, "var_per_path": var, "var_se": float(np.sqrt(max(m4 - var**2, 0.0) / n))}


def run_gap_sweep() -> pd.DataFrame:
    rows = []
    for params, payoff, fournier_samples, analytic in SWEEP_PRODUCTS:
        print(f"Gap sweep: {params['name']} (N={GAP_SWEEP_N:_})")
        scale = params["S0"] * params["sigma"] * np.sqrt(params["T"])
        bench = analytic(params)
        paths = generate_paths(params["T"], GAP_SWEEP_N, seed=GAP_SWEEP_SEED)
        base = {"Product": params["name"], "T": params["T"], "scale": scale}

        references = {
            "Malliavin": malliavin_samples_european(payoff, params, paths),
            f"FDMC (ε={FDMC_FIXED_EPS:g})": FDMC_samples_european(payoff, params, FDMC_FIXED_EPS, paths),
        }
        for estimator, samples in references.items():
            for g in GREEKS:
                rows.append({**base, "Greek": g, "Estimator": estimator, "multiplier": np.nan, "gap": np.nan,
                             "benchmark": bench[g], **_summarize(samples[g])})

        for mult in GAP_SWEEP_MULTIPLIERS:
            samples = fournier_samples(params, paths, cutoff_type="gaussian", bandwidth=mult * scale)
            for g in GREEKS:
                rows.append({**base, "Greek": g, "Estimator": "Malliavin Fournié", "multiplier": mult,
                             "gap": mult * scale, "benchmark": bench[g], **_summarize(samples[g])})

    df = pd.DataFrame(rows)
    df["se_of_mean"] = np.sqrt(df["var_per_path"] / GAP_SWEEP_N)
    df["bias_z"] = (df["mean"] - df["benchmark"]) / df["se_of_mean"]
    standard = df[df["Estimator"] == "Malliavin"].set_index(["Product", "Greek"])["var_per_path"]
    df["var_vs_standard"] = df["var_per_path"].to_numpy() / standard.loc[list(zip(df["Product"], df["Greek"]))].to_numpy()
    return df


def main():
    df = run_gap_sweep()
    Path(GAP_SWEEP_FILE).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(GAP_SWEEP_FILE, index=False)
    Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    fig = plot_gap_sweep(df)
    fig.savefig(Path(FIGURES_DIR) / "gap_sweep.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {GAP_SWEEP_FILE} and {FIGURES_DIR}/gap_sweep.png")


if __name__ == "__main__":
    main()
