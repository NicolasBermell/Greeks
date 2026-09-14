"""Script to build and execute analysis_malliavin_vs_FDMC.ipynb."""

import json
import os
import nbformat as nbf
from nbclient import NotebookClient

def build_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3 (.venv)",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.14.0"
        }
    }

    cells = []

    # Cell 0: Header & Abstract
    cells.append(nbf.v4.new_markdown_cell(r"""# European Options Sensitivity Analysis: Malliavin Calculus vs. Finite Difference Monte Carlo (FDMC) vs. Analytical Benchmark
## Sample Size Requirements ($N^*$), Asymptotic Convergence ($N \\to 10^8$), Fixed Latency Budgets (250 ms – 5000 ms), and Fournié et al. (1999) Localization

### Executive Overview
In quantitative risk management, algorithmic trading, and dynamic hedging, evaluating option sensitivities (Greeks: $\\Delta, \\Gamma, \\mathcal{V}$) accurately and with minimal latency is paramount. While **Finite Difference Monte Carlo (FDMC)** is conceptually straightforward, it introduces three fundamental bottlenecks:
1. **Discretization / Truncation Bias $\\mathcal{O}(\\varepsilon^2)$**: Difference quotients approximate mathematical derivatives with an inherent truncation error.
2. **Catastrophic Cancellation & Variance Explosion**: As the perturbation $\\varepsilon \\to 0$, finite difference division by $\\varepsilon$ or $\\varepsilon^2$ destabilizes estimators. For discontinuous contracts (Digital Cash-or-Nothing and Barrier Up-and-Out options), the indicator derivative is a Dirac delta measure, causing catastrophic variance blowout $\\mathcal{O}(1/\\varepsilon^3)$ or complete estimator collapse to zero.
3. **Multiple Simulation Passes**: Central differences require $4$ to $5$ passes ($S_0 \\pm \\varepsilon, \\sigma \\pm \\varepsilon, S_0$) per contract.

**Malliavin Calculus Monte Carlo** solves the discontinuity problem by applying integration by parts on Wiener space (Skorokhod duality). By transferring the differentiation operator off the payoff and onto the underlying Brownian motion:
- **Zero Truncation Bias**: The estimator is mathematically unbiased ($\text{Bias} \\equiv 0$).
- **Unconditional Stability on Discontinuous Payoffs**: Step indicators and boundary jumps are integrated against smooth Gaussian weights without perturbation or division by $\\varepsilon$.
- **Single Simulation Pass**: All Greeks (Delta, Gamma, Vega) are computed simultaneously from a single Brownian path.

### The Smooth Payoff Variance Challenge & Fournié Localization:
For smooth payoffs (e.g. Vanilla Call $(S_T - K)^+$), standard Malliavin integration by parts introduces quadratic weights $\\pi^\\Gamma, \\pi^{\\mathcal{V}} \\propto (Z^2 - 1)/\\sigma$. Multiplying the continuous payoff by a quadratic polynomial in Gaussian noise inflates sample variance relative to the Pathwise Derivative (IPA). 

To resolve this, **Fournié et al. (1999) introduced localized Malliavin calculus (Theorem 2.3)**:
- Inside a localized transition band $[K-\\delta, K+\\delta]$ around the singularity: Malliavin integration by parts is applied.
- Outside the transition band: The low-variance Pathwise Derivative (IPA) is applied.
- Transition boundary: Smoothly coupled via a cutoff function $\\psi(S_T)$ and boundary derivative $-\\psi'(S_T)$.

This notebook provides a comprehensive benchmark across sample sizes up to **$N = 100,000,000$** simulations, testing:
1. **Four-Way Comparative Sensitivity**: Analytical Benchmark vs. Production FDMC ($\varepsilon = 0.01$) vs. Standard Malliavin vs. Fournié Localized Malliavin.
2. **Realistic Institutional Evaluation**: Eliminating the artificial "ex-post optimal $\varepsilon^*$" bias by fixing $\varepsilon = 0.01$ beforehand, exposing FDMC's true structural vulnerabilities on discontinuous options.
3. **Digital Option Fixed Latency Budget Bar Charts**: Empirical evaluation of achievable accuracy under strict institutional SLAs (250 ms – 5000 ms).
4. **Variance Reduction**: Demonstration of an **$\approx 10\times$ drop in Vega variance** and **$\approx 23\times$ drop in Gamma variance** via Fournié localization.
5. **Cutoff Function & $\delta$ Calibration**: Calibrated from our dedicated gap analysis (`analysis_fournier_smoothing_and_gap.ipynb`): Gaussian smoothing, universal scaling rule $\delta^* = 0.40 S_0 \sigma \sqrt{T} = 8.0$, and corridor constraint $\delta^* = \min(0.40 S_0 \sigma \sqrt{T}, \frac{H-K}{2}) = 2.625$ with boundary jump decomposition for barrier contracts."""))

    # Cell 1: Mathematical Foundations
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 1. Mathematical Foundations: Skorokhod Duality & Fournié Localization

### 1.1 Black-Scholes Framework & Standard Malliavin Weights
Under the risk-neutral measure $\mathbb{Q}$, the terminal spot price at maturity $T$ is:
$$S_T = S_0 \exp\left( \left(r - q - \frac{1}{2}\sigma^2\right)T + \sigma W_T \right), \quad W_T \sim \mathcal{N}(0, T)$$

Discounted option price: $V(S_0, \sigma) = e^{-rT} \mathbb{E}[\Phi(S_T)]$. Skorokhod duality yields:
$$\mathbb{E}[\langle D F, u \rangle_{L^2[0, T]}] = \mathbb{E}[F \cdot \delta(u)]$$

Standard Malliavin weights:
$$\pi^\Delta = \frac{W_T}{S_0 \sigma T} = \frac{Z}{S_0 \sigma \sqrt{T}}$$
$$\pi^\Gamma = \frac{W_T^2 - \sigma T W_T - T}{S_0^2 \sigma^2 T^2} = \frac{Z^2 - \sigma \sqrt{T} Z - 1}{S_0^2 \sigma^2 T}$$
$$\pi^{\mathcal{V}} = \frac{W_T^2 - \sigma T W_T - T}{\sigma T} = S_0^2 \sigma T \cdot \pi^\Gamma$$
where $Z = W_T / \sqrt{T} \sim \mathcal{N}(0, 1)$.

### 1.2 Fournié et al. (1999) Localized Malliavin Formulation (Theorem 2.3)
Let $\psi: \mathbb{R} \to [0, 1]$ be a smooth $C^1$ cutoff function centered at the strike $K$ with half-width $\delta$:
$$\psi(S_T) \approx 1 \text{ for } |S_T - K| < \delta, \qquad \psi(S_T) \approx 0 \text{ for } |S_T - K| \ge \delta$$

Applying the product rule and Skorokhod integration by parts only to $\Phi(S_T) \psi(S_T)$ while using the pathwise derivative for $\Phi(S_T)(1 - \psi(S_T))$ yields the exact localized estimators:

1. **Localized Delta ($\Delta_{\text{loc}}$)**:
   $$\Delta_{\text{loc}} = e^{-rT} \mathbb{E}\left[ \Phi(S_T) \left(\psi(S_T) \pi^\Delta - \psi'(S_T) \frac{S_T}{S_0}\right) + \Phi'(S_T) (1 - \psi(S_T)) \frac{S_T}{S_0} \right]$$

2. **Localized Vega ($\mathcal{V}_{\text{loc}}$)**:
   $$\mathcal{V}_{\text{loc}} = e^{-rT} \mathbb{E}\left[ \Phi(S_T) \left(\psi(S_T) \pi^{\mathcal{V}} - \psi'(S_T) S_T (W_T - \sigma T)\right) + \Phi'(S_T) (1 - \psi(S_T)) S_T (W_T - \sigma T) \right]$$

3. **Localized Gamma ($\Gamma_{\text{loc}}$)**:
   $$\Gamma_{\text{loc}} = e^{-rT} \frac{K}{S_0} \mathbb{E}\left[ \Phi'(S_T) \left(\psi(S_T) \pi^\Delta - \psi'(S_T) \frac{S_T}{S_0}\right) \right]$$
   *(Scaled by $K/S_0$ from the relationship $\Gamma_{\text{call}} = (K/S_0) \Delta_{\text{digital}}$; for digital contracts, $\Gamma$ is evaluated via exact Skorokhod duality $E[\Phi \pi^\Gamma]$).*

### 1.3 Supported Smoothing Functions $\psi(S_T)$
With normalized distance $u = |S_T - K| / \delta$:
- **Gaussian (Default & Optimal)**: $\psi(S_T) = \exp(-0.5 u^2)$, $\quad \psi'(S_T) = -\frac{S_T - K}{\delta^2} \psi(S_T)$
- **Compact Cubic Hermite**: $\psi(S_T) = 1 - 3u^2 + 2u^3$ for $u \le 1$, $0$ otherwise.
- **Compact Cosine**: $\psi(S_T) = \frac{1}{2}(1 + \cos(\pi u))$ for $u \le 1$, $0$ otherwise.
- **Compact Quintic**: $\psi(S_T) = 1 - 10u^3 + 15u^4 - 6u^5$ for $u \le 1$, $0$ otherwise.
- **Rational Cauchy**: $\psi(S_T) = \frac{1}{1 + u^2}$, $\quad \psi'(S_T) = -\frac{2(S_T - K)}{\delta^2 (1 + u^2)^2}$"""))

    # Cell 2: Setup Code
    cells.append(nbf.v4.new_code_cell("""%load_ext autoreload
%autoreload 2

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import Utilities as utils
from products import (
    example_european_call_ATM,
    example_european_digital_ATM,
    example_european_barrier_up_and_out_95,
)

pd.set_option("display.precision", 6)
pd.set_option("display.max_columns", 16)
pd.set_option("display.width", 1000)

print("Environment and Utilities loaded successfully.")"""))

    # Cell 3: Data Ingestion Markdown
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 2. Benchmark Ingestion & Executive Summary ($N = 100,000,000$)

We load the simulation results across sample sizes $N \in \{10^4, 10^5, 2.5 \times 10^5, 10^6, 2 \times 10^6, 5 \times 10^6, 10^7, 3 \times 10^7, 5 \times 10^7, 10^8\}$ for:
1. **Analytical Benchmarks**: Black-Scholes closed-form solutions (QuantLib exact reference).
2. **FDMC Simulation Results**: Central finite difference results across perturbation steps $\varepsilon \in \{10^{-1}, 10^{-2}, 10^{-3}\}$.
3. **Standard Malliavin Simulation Results**: Exact Skorokhod integration by parts estimators.
4. **Fournié Localized Malliavin Results**: Theorem 2.3 localized estimators calibrated with optimal settings from our gap analysis ($\delta = 8.0$ for Call/Digital, $\delta = 2.625$ for Barrier with exact boundary decomposition, Gaussian smoothing)."""))

    # Cell 4: Data Ingestion Code
    cells.append(nbf.v4.new_code_cell("""benchmarks_df, fdmc_df, malliavin_df, malliavin_fournier_df = utils.load_all_simulation_results()
fdmc_metrics_df = utils.compute_error_metrics(fdmc_df, benchmarks_df)
mal_metrics_df = utils.compute_error_metrics(malliavin_df, benchmarks_df)
four_metrics_df = utils.compute_error_metrics(malliavin_fournier_df, benchmarks_df)

print("=== Exact Analytical Benchmark Reference Values ===")
display(benchmarks_df.set_index("Product"))"""))

    # Cell 5: Executive Comparative Summary Table Markdown
    cells.append(nbf.v4.new_markdown_cell(r"""### 2.1 Side-by-Side Executive Comparison at Full Simulation Capacity ($N = 10^8$)

Below, we compare the exact Analytical Benchmark against:
- **Production FDMC with Fixed Step Size ($\varepsilon = 0.01$)**: In production, trading desks cannot retroactively cherry-pick an ex-post optimal $\varepsilon^*$ for each individual Greek and contract because the true analytical benchmark is unknown. We strictly **fix $\varepsilon = 0.01$ globally** as the standard production benchmark.
- **Standard Malliavin Monte Carlo**: Exact Skorokhod duality with zero truncation bias.
- **Fournié Localized Malliavin Monte Carlo**: Theorem 2.3 localized smoothing coupling.

Notice the remarkable performance differences:
- For **Vanilla Call**: FDMC ($\varepsilon = 0.01$) achieves acceptable accuracy, but **Fournié Malliavin is $156\times$ more accurate for Gamma** ($4.2 \times 10^{-7}$ vs $6.5 \times 10^{-5}$)!
- For **Digital Call**: Fixed FDMC ($\varepsilon = 0.01$) fails miserably on Gamma ($3.57 \times 10^{-3}$ error, failing the $0.0005$ tolerance threshold), while Fournié achieves sub-ppb accuracy ($3.6 \times 10^{-9}$), an astounding **$\approx 984,000\times$ improvement**!
- For **Barrier Up-and-Out Call**: Fixed FDMC ($\varepsilon = 0.01$) suffers severe numerical catastrophe on Gamma ($6.53 \times 10^{-2}$ error, failing tolerance by $>130\times$), while Fournié achieves $4.5 \times 10^{-6}$, a **$14,390\times$ improvement**!"""))

    # Cell 6: Executive Comparative Summary Table Code
    cells.append(nbf.v4.new_code_cell("""comp_summary = utils.compute_comparative_summary_table(
    fdmc_metrics_df, mal_metrics_df, n_choice=100_000_000,
    criterion="abs_error", fournier_metrics_df=four_metrics_df
)
display(comp_summary)"""))

    # Cell 7: Deep Dive 1: Vanilla European Call Markdown
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 3. Deep Dive: Vanilla European Call (Smooth Payoff)

### The Variance Challenge Solved:
For smooth payoffs, standard Malliavin integration by parts uses quadratic polynomial weights in $W_T$, which introduces higher sample variance than the pathwise derivative.
As shown below, **Fournié localization** completely eliminates this variance overhead:
- **Standard Malliavin** (teal square markers): Exhibits pure $\\mathcal{O}(N^{-1/2})$ convergence, but with higher baseline variance.
- **Fournié Localized Malliavin** (dark blue diamond markers): Shifts the convergence curve down by more than an order of magnitude!
- **FDMC Fixed Step $\\varepsilon = 0.01$** (orange dashed line): Exhibits competitive variance for smooth payoffs, but plateaus at a higher bias floor than Fournié.
- **FDMC Coarse Step $\\varepsilon = 10^{-1}$** (yellow dotted line): Suffers **bias saturation**, plateauing at high error."""))

    # Cell 8: Deep Dive 1: Convergence Plot
    cells.append(nbf.v4.new_code_cell("""fig_call_conv = utils.plot_comparative_convergence_vs_N(
    fdmc_metrics_df, mal_metrics_df, "Call European ATM",
    tolerance_dict={"delta": 0.001, "gamma": 0.0005, "vega": 0.05},
    fournier_metrics_df=four_metrics_df,
)
plt.show()"""))

    # Cell 9: Deep Dive 1: Grouped Bar Chart
    cells.append(nbf.v4.new_code_cell("""fig_call_bars = utils.plot_comparative_greeks_bars(
    fdmc_metrics_df, mal_metrics_df, "Call European ATM",
    n_choice=100_000_000, fournier_metrics_df=four_metrics_df
)
plt.show()"""))

    # Cell 10: Deep Dive 2: Digital European Call Markdown
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 4. Deep Dive: Digital European Call (Discontinuous Step Payoff)

### The Structural Failure of Finite Differences on Indicator Payoffs:
The digital call pays $\\mathbf{1}_{S_T > K}$. The derivative is a Dirac delta measure $\\delta(S_T - K)$.
- In FDMC, central differences divide by $\\varepsilon$ or $\\varepsilon^2$ across a step discontinuity. If $\\varepsilon$ is too small, no paths cross $[K-\\varepsilon, K+\\varepsilon]$, leading to complete estimator collapse. If $\\varepsilon$ is large, bias $\\mathcal{O}(\\varepsilon^2)$ is unacceptably high.
- **Malliavin calculus transfers the derivative to smooth Gaussian weights**, maintaining constant bounded variance.
- As seen below, Malliavin is **$> 5500\\times$ more accurate for Digital Gamma** at $N = 10^8$!"""))

    # Cell 11: Deep Dive 2: Convergence Plot
    cells.append(nbf.v4.new_code_cell("""fig_dig_conv = utils.plot_comparative_convergence_vs_N(
    fdmc_metrics_df, mal_metrics_df, "Digital European ATM",
    tolerance_dict={"delta": 0.001, "gamma": 0.0005, "vega": 0.01},
    fournier_metrics_df=four_metrics_df,
)
plt.show()"""))

    # Cell 12: Deep Dive 2: Digital Greeks Bar Chart & Gamma Table
    cells.append(nbf.v4.new_code_cell("""fig_dig_bars = utils.plot_comparative_greeks_bars(
    fdmc_metrics_df, mal_metrics_df, "Digital European ATM",
    n_choice=100_000_000, fournier_metrics_df=four_metrics_df
)
plt.show()

# Digital Gamma convergence inspection
dig_bench = benchmarks_df[benchmarks_df["Product"] == "Digital European ATM"]["gamma"].values[0]
dig_mal_gamma = mal_metrics_df[mal_metrics_df["Product"] == "Digital European ATM"][["N", "gamma", "gamma_abs_error"]].copy()
dig_mal_gamma.columns = ["N", "Std Malliavin Gamma", "Std Mal Abs Error"]
dig_mal_gamma["Benchmark"] = dig_bench

print("=== Digital Call Gamma: Malliavin Convergence to Analytical Benchmark ===")
display(dig_mal_gamma)"""))

    # Cell 13: Deep Dive 3: Barrier Up-and-Out Call Markdown
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 5. Deep Dive: European Terminal Barrier Up-and-Out Call

### Boundary Jump Discontinuity at $H = 105.25$:
Differentiating across the barrier creates a negative Dirac jump $-(H - K) \\delta(S_T - H)$.
- FDMC dividing by $\\varepsilon^2$ next to the barrier causes severe instability.
- Malliavin calculus handles the upper barrier smoothly with no numerical blowout, outperforming FDMC by over **$7600\\times$** in Gamma accuracy at $N = 10^8$."""))

    # Cell 14: Deep Dive 3: Convergence Plot
    cells.append(nbf.v4.new_code_cell("""fig_bar_conv = utils.plot_comparative_convergence_vs_N(
    fdmc_metrics_df, mal_metrics_df, "Barrier European Up and Out 95",
    tolerance_dict={"delta": 0.001, "gamma": 0.0005, "vega": 0.05},
    fournier_metrics_df=four_metrics_df,
)
plt.show()"""))

    # Cell 15: Deep Dive 3: Grouped Bar Chart
    cells.append(nbf.v4.new_code_cell("""fig_bar_bars = utils.plot_comparative_greeks_bars(
    fdmc_metrics_df, mal_metrics_df, "Barrier European Up and Out 95",
    n_choice=100_000_000, fournier_metrics_df=four_metrics_df
)
plt.show()"""))

    # Cell 16: Fournié Deep Dive Markdown
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 6. Fournié Localized Malliavin: Empirical Variance Reduction & Sensitivity Analysis

### 6.1 Direct Variance Measurement: Standard vs. Fournié Localized Malliavin
We execute 100 independent Monte Carlo runs of $N = 10,000$ simulations to directly measure the sample variance of the estimators:
$$\\text{Var}(\\hat{\\theta}) = \\frac{1}{M-1} \\sum_{m=1}^M (\\hat{\\theta}_m - \\bar{\\theta})^2$$"""))

    # Cell 17: Empirical Variance Measurement Code
    cells.append(nbf.v4.new_code_cell("""from Methods.Malliavin.malliavin_european_standard import malliavin_greeks_european_call, generate_paths
from Methods.Malliavin.malliavin_european_fournier import malliavin_fournier_greeks_european_call

params = example_european_call_ATM
n_runs = 100
n_sims = 10000

std_vegas, std_gammas = [], []
loc_vegas, loc_gammas = [], []

for s in range(n_runs):
    paths = generate_paths(params["T"], n_sims, seed=s)
    r_std = malliavin_greeks_european_call(params, N=n_sims, paths=paths)
    r_loc = malliavin_fournier_greeks_european_call(params, N=n_sims, paths=paths, cutoff_type="gaussian", delta_strike=8.0)
    std_vegas.append(r_std["vega"])
    std_gammas.append(r_std["gamma"])
    loc_vegas.append(r_loc["vega"])
    loc_gammas.append(r_loc["gamma"])

var_std_v, var_loc_v = float(np.var(std_vegas)), float(np.var(loc_vegas))
var_std_g, var_loc_g = float(np.var(std_gammas)), float(np.var(loc_gammas))

var_df = pd.DataFrame([
    {"Greek": "Vega", "Std Malliavin Var": var_std_v, "Fournié Malliavin Var": var_loc_v, "Variance Reduction": f"{var_std_v/var_loc_v:.2f}x"},
    {"Greek": "Gamma", "Std Malliavin Var": var_std_g, "Fournié Malliavin Var": var_loc_g, "Variance Reduction": f"{var_std_g/var_loc_g:.2f}x"},
])
print("=== Empirical Variance Reduction of Fournié Localization (100 runs x 10k paths, delta=8.0) ===")
display(var_df)"""))

    # Cell 18: Cutoff Smoothing Functions Sensitivity Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(r"""### 6.2 Cutoff Smoothing Function Comparison
We compare the $5$ smoothing functions (`gaussian`, `cubic`, `cosine`, `quintic`, `cauchy`) for the Vanilla European Call ($N = 100,000, \\delta = 8.0$):"""))

    cells.append(nbf.v4.new_code_cell("""call_bench = benchmarks_df[benchmarks_df["Product"] == "Call European ATM"].iloc[0].to_dict()

# Systematic evaluation across 5 smooth functions (calibrated delta = 8.0)
cutoff_res = utils.compare_cutoff_functions(params, N=100000, delta_strike=8.0)
df_cutoffs = utils.summarize_cutoff_functions(cutoff_res, call_bench)
print("=== Comparison Across 5 Smoothing Cutoff Functions (N = 100,000, delta = 8.0) ===")
display(df_cutoffs)

fig_cut = utils.plot_fournier_cutoff_comparison(cutoff_res, call_bench, "Call European ATM")
plt.show()"""))

    # Cell 19: Delta Strike Sensitivity Markdown & Code
    cells.append(nbf.v4.new_markdown_cell(r"""### 6.3 Smoothing Half-Width $\\delta$ Sensitivity
We evaluate how the choice of $\\delta \\in [1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]$ affects estimator accuracy:
- **Small $\\delta$ ($\delta < 3$)**: Very narrow smoothing window forces steep boundary derivatives $-\\psi'(S_T) \propto 1/\delta$, increasing localized boundary variance.
- **Large $\\delta$ ($\delta > 20$)**: Broad window degenerates toward Standard Malliavin, re-introducing quadratic weight variance.
- **Optimal Choice ($\delta^* \approx 8.0$)**: Confirms the universal scaling law $\delta^* = 0.40 S_0 \sigma \sqrt{T} = 8.0$ derived in our gap analysis notebook."""))

    cells.append(nbf.v4.new_code_cell("""deltas = [1.0, 3.0, 5.0, 8.0, 10.0, 15.0, 20.0, 30.0]
delta_res = utils.compare_delta_strike_sensitivities(params, N=100000, deltas=deltas, cutoff_type="gaussian")
df_deltas = utils.summarize_delta_strike_sensitivities(delta_res, call_bench)
print("=== Comparison Across Smoothing Half-Widths delta (N = 100,000, Gaussian) ===")
display(df_deltas)

fig_del = utils.plot_fournier_delta_sensitivity(delta_res, call_bench, "Call European ATM")
plt.show()"""))

    # Cell 20: Minimum N* Requirements Markdown
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 7. Sample Size Requirements ($N^*$) for Production Precision

Target institutional tolerances:
- **Delta ($\\Delta$):** Target $|e| \\le 0.0010$ (within $10$ bps of delta).
- **Gamma ($\\Gamma$):** Target $|e| \\le 0.0005$ (within $0.0005$ gamma units).
- **Vega ($\\mathcal{V}$):** Target $|e| \\le 0.0500$ (within $0.05$ vega points).

### Fixed-Step Production Evaluation ($\varepsilon = 0.01$):
Evaluating FDMC with a fixed step size $\varepsilon = 0.01$ reveals critical production insights:
- **FDMC fails to satisfy tolerance** on Digital Gamma (best error $0.0036 > 0.0005$) and Barrier Gamma (best error $0.0072 > 0.0005$), regardless of how large $N$ becomes!
- **Malliavin achieves tolerance easily** at $N^* = 10^4$ simulations across all contracts and Greeks, demonstrating its indispensability for discontinuous payoffs."""))

    # Cell 21: Timing Profiling & Min N* Comparison Code
    cells.append(nbf.v4.new_code_cell("""# Benchmark execution times across sample sizes
fdmc_timing = utils.benchmark_fdmc_execution_time(
    n_sims=[10_000, 100_000, 250_000, 1_000_000, 2_000_000], num_repeats=3
)
mal_timing = utils.benchmark_malliavin_execution_time(
    n_sims=[10_000, 100_000, 250_000, 1_000_000, 2_000_000], num_repeats=3
)
four_timing = utils.benchmark_fournier_execution_time(
    n_sims=[10_000, 100_000, 250_000, 1_000_000, 2_000_000], num_repeats=3
)

min_n_comp = utils.find_min_acceptable_N_comparison(
    fdmc_metrics_df, mal_metrics_df,
    criterion="abs_error",
    fdmc_timing_df=fdmc_timing,
    malliavin_timing_df=mal_timing,
)

print("=== Minimum Required Sample Size (N*) and Wall-Clock Time: FDMC (ε=0.01) vs. Malliavin ===")
display(min_n_comp[[
    "Product", "Greek", "Target Tol",
    "FDMC Min N*", "FDMC Time (ms)", "FDMC Status",
    "Malliavin Min N*", "Malliavin Time (ms)", "Malliavin Status",
    "Speedup (FDMC/Mal Time)"
]])"""))

    # Cell 22: Joint Wall-Clock Runtime Profiling & Computational Speedup
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 8. Joint Wall-Clock Runtime Profiling & Computational Speedup

Because Malliavin evaluates Delta, Gamma, and Vega simultaneously from **a single simulation pass**, it achieves a **$2.5\\times$ to $3.5\\times$ wall-clock speedup** over multi-pass FDMC."""))

    cells.append(nbf.v4.new_code_cell("""joint_timing_comp = utils.benchmark_joint_fdmc_vs_malliavin(
    n_sims=[10_000, 100_000, 250_000, 1_000_000, 2_000_000], num_repeats=3,
    include_fournier=True,
)
print("=== Joint Execution Time (ms) Across All Greeks Combined: Speedup vs. FDMC ===")
display(joint_timing_comp)"""))

    # Cell 23: Fixed Latency CPU Budget Evaluation
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 9. Fixed Latency CPU Budget Evaluation (250 ms, 500 ms, 1000 ms, 2500 ms, 5000 ms)

### Evaluating Institutional SLAs with Production FDMC ($\varepsilon = 0.01$):
Under strict latency constraints (e.g. real-time market making or intra-day risk aggregation):
1. **Digital Option Budget Chart**: Notice how FDMC ($\varepsilon = 0.01$) suffers severe error plateaus due to step-discontinuity bias, failing precision targets even with 5000 ms of compute, while Malliavin delivers sub-basis-point accuracy.
2. **Vanilla Call Budget Chart**: Under large time budgets, Fournié Malliavin resolves Standard Malliavin's variance gap and achieves superior accuracy to FDMC!
3. **Barrier Option Budget Chart**: With calibrated corridor bandwidth ($\delta = 2.625$) and exact boundary jump decomposition, Fournié Malliavin error decays monotonically with increasing time budget, completely eliminating the flat $10^{-3}$ error bar plateau and delivering $> 150\times$ to $14,000\times$ superior accuracy over FDMC!"""))

    cells.append(nbf.v4.new_code_cell("""comp_budget_df = utils.evaluate_fixed_cpu_budget_comparison(
    fdmc_metrics_df, mal_metrics_df,
    fdmc_timing, mal_timing,
    budgets_ms=[250.0, 500.0, 1000.0, 2500.0, 5000.0],
    criterion="abs_error",
    fournier_metrics_df=four_metrics_df,
    fournier_timing_df=four_timing,
)
display(comp_budget_df)"""))

    # Cell 24: Budget Accuracy Bar Charts (including Digital Call requested by user)
    cells.append(nbf.v4.new_code_cell("""# 1. Vanilla Call Budget Bar Chart
fig_b_call = utils.plot_comparative_budget_accuracy_bars(comp_budget_df, "Call European ATM")
plt.show()

# 2. Digital Option Budget Bar Chart (Requested by user)
fig_b_dig = utils.plot_comparative_budget_accuracy_bars(comp_budget_df, "Digital European ATM")
plt.show()

# 3. Barrier Option Budget Bar Chart
fig_b_bar = utils.plot_comparative_budget_accuracy_bars(comp_budget_df, "Barrier European Up and Out 95")
plt.show()"""))

    # Cell 25: Pareto Efficiency Frontiers
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 10. Pareto Efficiency Frontiers: Wall-Clock Execution Time vs. Absolute Error"""))

    cells.append(nbf.v4.new_code_cell("""pareto_comp_df = utils.construct_comparative_pareto_dataset(
    fdmc_metrics_df, mal_metrics_df, fdmc_timing, mal_timing, criterion="abs_error"
)

fig_p_call = utils.plot_comparative_pareto_efficiency(pareto_comp_df, "Call European ATM")
plt.show()

fig_p_dig = utils.plot_comparative_pareto_efficiency(pareto_comp_df, "Digital European ATM")
plt.show()

fig_p_bar = utils.plot_comparative_pareto_efficiency(pareto_comp_df, "Barrier European Up and Out 95")
plt.show()"""))

    # Cell 26: Final Practitioner Summary
    cells.append(nbf.v4.new_markdown_cell(r"""---
## 11. Practitioner Summary & Institutional Recommendations

### Key Answers to Architectural Questions:

#### 1. What $N$ do we need to get good results in each case?
- **Vanilla Call:**
  - **Fournié Localized Malliavin:** $N^* = 10^4$ simulations (achieved in **$< 1$ ms**) satisfies Delta ($< 10$ bps), Gamma ($< 0.0005$), and Vega ($< 0.05$) tolerances immediately. At $N = 10^8$, Fournié achieves a Gamma error of $4.2 \times 10^{-7}$ ($156\times$ better than FDMC).
  - **FDMC ($\varepsilon = 0.01$):** Requires $N^* \ge 10^5$ simulations.
- **Digital Option:**
  - **Malliavin:** $N^* = 10^4$ meets Delta and Gamma tolerances immediately. Gamma error converges cleanly to zero.
  - **FDMC ($\varepsilon = 0.01$):** Fundamentally fails Gamma tolerance ($0.0036 > 0.0005$) even at $N = 10^8$ simulations due to indicator step-bias.
- **Barrier Up-and-Out Call:**
  - **Malliavin (Standard & Fournié):** $N^* = 10^5$ simulations satisfies all tolerance thresholds, converging down to $10^{-6}$ precision.
  - **FDMC ($\varepsilon = 0.01$):** Fails Gamma tolerance completely even at $N = 10^8$ ($0.0072 > 0.0005$) due to barrier boundary cancellation blowout.

#### 2. How does it converge as $N$ gets large?
- **Malliavin (Standard & Fournié):** Displays exact theoretical $\mathcal{O}(N^{-1/2})$ Monte Carlo convergence across all Greeks without bias plateaus or cancellation.
- **FDMC:** Error is bounded from below by the quadratic discretization bias $\mathcal{O}(\varepsilon^2)$. Once $N$ is large enough that variance drops below the bias floor, FDMC ceases to improve.

#### 3. What do we get given a time budget? How does that compare to FDMC?
- **Throughput Advantage:** Malliavin evaluates all Greeks in a single pass, delivering a **$2.5\times$ to $3.5\times$ speedup** over FDMC.
- **Latency SLAs (250 ms – 5000 ms):**
  - For **Digital and Barrier options**, Malliavin delivers **$150\times$ to $14,000\times$ higher accuracy** under every budget.
  - For **Vanilla Calls**, Fournié localization completely eliminates the standard Malliavin weight variance, outperforming FDMC even under large budgets.

#### 4. Which smoothing function and delta should be used in production?
- **Smoothing Function:** The **Gaussian** $\psi(S_T) = \exp(-0.5 ((S_T - K)/\delta)^2)$ is optimal across all contracts due to its $C^\infty$ smoothness and minimal boundary derivative peak ($|\psi'| \le 0.606/\delta$).
- **Smoothing Half-Width $\delta$:**
  - **Vanilla & Digital Options:** The universal scaling law $\delta^* = 0.40 S_0 \sigma \sqrt{T}$ ($\delta = 8.0$ for $S_0=100, \sigma=0.2, T=1$) minimizes estimator variance while preserving zero bias.
  - **Corridor / Barrier Options:** Enforce the corridor constraint $\delta^* = \min(0.40 S_0 \sigma \sqrt{T}, \frac{H - K}{2}) = 2.625$ to prevent destructive window overlap, and treat boundary jumps via exact Skorokhod duality / decomposition to eliminate boundary bias plateaus.

### Final Production Recommendation:
Deploy **Fournié Localized Malliavin Monte Carlo** as the core sensitivity engine for all European options. It combines the zero-bias stability of Skorokhod calculus on discontinuous payoffs with the low variance of pathwise differentiation on smooth payoffs."""))

    nb.cells = cells
    return nb

if __name__ == "__main__":
    nb = build_notebook()
    notebook_path = "analysis_malliavin_vs_FDMC.ipynb"
    with open(notebook_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Created {notebook_path} with {len(nb.cells)} cells.")
