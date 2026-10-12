# Benchmark of Option Greeks: Malliavin Calculus vs. Finite Difference Monte Carlo (FDMC)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A comprehensive, production-grade quantitative benchmark evaluating numerical methods for computing financial Greeks (**Delta $\Delta$, Gamma $\Gamma$, and Vega $\mathcal{V}$**) under Black-Scholes dynamics.

This project investigates the stability, computational efficiency, and variance profiles of:
1. **Finite Difference Monte Carlo (FDMC)** with Common Random Numbers (CRN).
2. **Standard Malliavin Calculus** using exact global Skorokhod integration-by-parts weights.
3. **Fournié et al. (1999) Localized Malliavin Calculus** with per-Greek gaps.
4. **Analytical Benchmarks** (Black-Scholes closed forms; the call with terminal knock-out is priced as call spread minus digital).

---

## Table of Contents

- [1. Executive Summary & Core Research Questions](#1-executive-summary--core-research-questions)
- [2. Mathematical Foundations](#2-mathematical-foundations)
  - [2.1 Black-Scholes Model & Payoff Geometry](#21-black-scholes-model--payoff-geometry)
  - [2.2 Finite Difference Monte Carlo (FDMC) & The Pathwise Secret](#22-finite-difference-monte-carlo-fdmc--the-pathwise-secret)
  - [2.3 The Dirac Delta Breakdown on Gamma and Digitals](#23-the-dirac-delta-breakdown-on-gamma-and-digitals)
  - [2.4 Standard Malliavin Calculus & The Unbounded Wing Problem](#24-standard-malliavin-calculus--the-unbounded-wing-problem)
  - [2.5 Fournié Localization: Best of Both Worlds](#25-fournié-localization-best-of-both-worlds)
  - [2.6 Bandwidths (Gap $\ell$)](#26-bandwidths-gap-ell)
  - [2.7 The Call with Terminal Knock-Out](#27-the-call-with-terminal-knock-out)
- [3. Supported Contracts & Reference Benchmarks](#3-supported-contracts--reference-benchmarks)
- [4. Project Architecture & File Tree](#4-project-architecture--file-tree)
- [5. How to Run & Reproduce](#5-how-to-run--reproduce)
  - [5.1 Installation & Virtual Environment](#51-installation--virtual-environment)
  - [5.2 Running the Test Suite](#52-running-the-test-suite)
  - [5.3 Generating Analytical Benchmarks](#53-generating-analytical-benchmarks)
  - [5.4 Running Full Monte Carlo Simulations](#54-running-full-monte-carlo-simulations)
  - [5.5 Interactive Research Notebooks](#55-interactive-research-notebooks)
- [6. Key Findings & Institutional Decision Rules](#6-key-findings--institutional-decision-rules)

---

## 1. Executive Summary & Core Research Questions

In quantitative risk management, calculating sensitivities (Greeks) via standard Monte Carlo bumping poses major challenges:
- **Resimulation Overhead:** Computing $M$ Greeks with central differences requires $2M + 1$ distinct simulation passes.
- **Singular Payoffs:** For discontinuous or kinked payoffs, finite difference variance explodes as perturbation step $\varepsilon \to 0$.
- **Tails & Corridors:** While Malliavin calculus bypasses payoff differentiation by transferring derivatives onto Gaussian weights, standard Malliavin suffers from heavy-tailed variance in unbounded wings ($S_T \to \infty$).

This benchmark answers four critical questions:
1. **Why does Standard Malliavin underperform FDMC for Delta/Vega of a Vanilla Call?** Because FDMC with Common Random Numbers secretly computes the low-variance Pathwise Derivative (IPA), whereas Standard Malliavin multiplies the payoff by Brownian weights that grow as $S_T \ln S_T$.
2. **Why does FDMC collapse catastrophically for Gamma?** Because differentiating a step payoff creates a Dirac delta singularity $\delta(S_T - K)$, driving FDMC sample variance to explode as $\mathcal{O}(1/\varepsilon)$.
3. **How does Fournié Localization resolve both problems?** By using Pathwise differentiation in the smooth wings and Malliavin integration by parts only in a localized band around the strike, evaluating all Greeks in a **single simulation pass**.
4. **Why does Fournié not help on the call with terminal knock-out?** The payoff lives in the narrow interval $[K, H]$, so standard Malliavin already has small variance, while the cutoff terms $\Phi\psi'$, $\Phi\psi''$ scale like $1/\delta$, $1/\delta^2$ with $\delta \le (H-K)/2$.

---

## 2. Mathematical Foundations

### 2.1 Black-Scholes Model & Payoff Geometry

Under the risk-neutral measure $\mathbb{Q}$, the asset price follows geometric Brownian motion:
$$dS_t = (r - q) S_t dt + \sigma S_t dW_t \implies S_T = S_0 \exp\left[\left(r - q - \frac{1}{2}\sigma^2\right)T + \sigma \sqrt{T} Z\right], \quad Z \sim \mathcal{N}(0, 1)$$

The price of a European derivative with payoff $\Phi(S_T)$ is:
$$V(S_0, \sigma, T) = e^{-rT} \mathbb{E}[\Phi(S_T)]$$

---

### 2.2 Finite Difference Monte Carlo (FDMC) & The Pathwise Secret

Central finite difference for Delta with step size $\varepsilon$:
$$\hat{\Delta}_{\text{FDMC}} = \frac{e^{-rT} \left[ \Phi(S_T(S_0 + \varepsilon)) - \Phi(S_T(S_0 - \varepsilon)) \right]}{2\varepsilon}$$

When **Common Random Numbers (CRN)** are shared across paths:
$$S_T(S_0 \pm \varepsilon) = S_T \cdot \left(1 \pm \frac{\varepsilon}{S_0}\right)$$

For a continuous Call payoff $\Phi(S_T) = (S_T - K)^+$, for all paths where both perturbed prices lie on the same side of the strike ($>99.9\%$ of paths when $\varepsilon = 0.01$):
$$\frac{\Phi(S_T(S_0 + \varepsilon)) - \Phi(S_T(S_0 - \varepsilon))}{2\varepsilon} = \mathbf{1}_{S_T > K} \frac{S_T}{S_0}$$

> **Key Insight:** For continuous payoffs, FDMC with CRN evaluates the **Pathwise Derivative (Infinitesimal Perturbation Analysis - IPA)**. Its variance is low (per-path $\operatorname{Var} \approx 0.33$ at $\varepsilon = 0.01$) because it grows only linearly with $S_T$ and contains **no stochastic Brownian multipliers**.

---

### 2.3 The Dirac Delta Breakdown on Gamma and Digitals

For Gamma, the second derivative of $(S_T - K)^+$ is a **Dirac delta** $\delta(S_T - K)$. Taking the second difference quotient:
$$\hat{\Gamma}_{\text{FDMC}} = \frac{e^{-rT}\left[ \Phi(S_0 + \varepsilon) - 2\Phi(S_0) + \Phi(S_0 - \varepsilon) \right]}{\varepsilon^2}$$

- For paths outside a microscopic band $|S_T - K| \le \varepsilon \frac{S_T}{S_0}$, the numerator is identically zero.
- For the rare paths landing inside the band (probability $\mathcal{O}(\varepsilon)$), the numerator is $\mathcal{O}(\varepsilon)$, divided by $\varepsilon^2$, yielding a sample spike of **$\mathcal{O}(1/\varepsilon)$**.
- Therefore, the sample variance diverges:
  $$\operatorname{Var}(\hat{\Gamma}_{\text{FDMC}}) \sim \mathcal{O}\left(\frac{1}{\varepsilon}\right)$$

For $\varepsilon = 0.01$, sample spikes reach $100$. The measured per-path variance of the FDMC call Gamma is $1.21$, against $0.017$ for standard Malliavin, a factor of about $70$ in the number of paths needed (`results/efficiency.csv`). For the digital the per-path variance grows like $1/\varepsilon^3$ (about $3.6\times10^4$ at $\varepsilon = 0.01$).

---

### 2.4 Standard Malliavin Calculus & The Unbounded Wing Problem

Using Malliavin integration by parts (IBP), derivatives of the expectation are transferred onto the underlying Wiener measure:
$$e^{rT} \frac{\partial V}{\partial \theta} = \mathbb{E}[\Phi(S_T) \cdot \pi^\theta]$$

The exact 1D Black-Scholes Skorokhod weights are:
- **Delta ($\Delta$):**
  $$\pi^\Delta = \frac{W_T}{S_0 \sigma T}$$
- **Gamma ($\Gamma$):**
  $$\pi^\Gamma = \frac{W_T^2 - T - \sigma T W_T}{(S_0 \sigma T)^2}$$
- **Vega ($\mathcal{V}$):**
  $$\pi^\mathcal{V} = \frac{W_T^2 - \sigma T W_T - T}{\sigma T}$$

Because the weight $\pi$ does not differentiate $\Phi$, **Standard Malliavin completely eliminates Dirac delta singularities**, making it unbiased and smooth for Gamma.

#### The Unbounded Wing Penalty
However, Standard Malliavin integrates across the entire domain $[0, \infty)$. Deep in the money ($S_T \to \infty$):
- Payoff: $(S_T - K) \sim S_T$
- Weight: $\pi^\Delta \sim W_T \sim \ln S_T$
- Integrand: **$S_T \cdot W_T$**

Because $S_T$ and $W_T$ are positively correlated, their product produces heavy tails:
- **Pathwise / FDMC ($\varepsilon = 0.01$) per-path variance for Delta:** $0.33$
- **Standard Malliavin per-path variance for Delta:** $1.93$ ($5.9\times$ higher, so $5.9\times$ more paths for the same SE)

For Vega, multiplying by $W_T^2$ inflates the per-path variance $12.8\times$ ($68{,}900$ vs $5{,}390$). Measured over $R = 100$ runs of $N = 10^6$; see `results/efficiency.csv`.

---

### 2.5 Fournié Localization: Best of Both Worlds

Fournié et al. (1999) partition the payoff using a smooth cutoff function $\psi(S_T)$ localized around the strike $K$ with bandwidth $\delta$:
$$\Phi(S_T) = \underbrace{\Phi(S_T) \psi(S_T)}_{\text{Singular Strike Zone } [K-\delta, K+\delta]} + \underbrace{\Phi(S_T) (1 - \psi(S_T))}_{\text{Smooth Wings } S_T > K+\delta}$$

1. **In the wings ($S_T > K + \delta$):** Evaluates the **Pathwise derivative** (IPA), eliminating the $S_T \cdot W_T$ weight explosion.
2. **At the strike ($|S_T - K| \le \delta$):** Evaluates **Malliavin integration by parts**, eliminating FDMC's $1/\varepsilon$ Dirac explosion.
3. **Boundary terms ($-\psi', -\psi''$):** Correct for the localization boundary, guaranteeing **exact zero bias**.

The localized estimators are:
- **Delta ($\Delta$):**
  $$Y^\Delta_{\text{loc}} = e^{-rT} \left[ \Phi(S_T)\left(\psi \pi^\Delta - \psi' \frac{S_T}{S_0}\right) + \Phi'(S_T)(1 - \psi)\frac{S_T}{S_0} \right]$$
- **Gamma ($\Gamma$), general formula:**
  $$Y^\Gamma_{\text{loc}} = e^{-rT} \left[ \Phi\psi\, \pi^\Gamma + \Big(\Phi''(1 - \psi) - 2\Phi'\psi' - \Phi\psi''\Big)\left(\frac{S_T}{S_0}\right)^2 \right]$$

  $\Phi'$ and $\Phi''$ are the classical (almost-everywhere) derivatives; the Dirac masses of the distributional derivatives sit at the singularities, where $\psi = 1$ and $\psi' = 0$, so they drop out. The same formula is used for the call, the digital, and the call with terminal knock-out. The earlier call-only shortcut $\Gamma_{\text{call}} = (K/S_0)\cdot(\text{localized digital }\Delta)$ is still available as `gamma_formula="shortcut"` and is reported as a separate run in the replications.
- **Vega ($\mathcal{V}$):**
  $$Y^\mathcal{V}_{\text{loc}} = e^{-rT} \left[ \Phi(S_T)\left(\psi \pi^\mathcal{V} - \psi' S_T (W_T - \sigma T)\right) + \Phi'(S_T)(1 - \psi) S_T (W_T - \sigma T) \right]$$

---

### 2.6 Bandwidths (Gap $\ell$)

The code uses per-Greek default gaps ("Option A") $\ell_\Delta = \ell_\mathcal{V} = 0.20\,S_0\sigma\sqrt{T}$ and $\ell_\Gamma = 0.50\,S_0\sigma\sqrt{T}$ (`fournier_bandwidths`), all evaluated in one Monte Carlo pass, or a single "coupled" gap $0.40\,S_0\sigma\sqrt{T}$. These defaults are not optima. The gap sweep (Section 5.4c, `results/figures/gap_sweep.png`) shows:

| Greek | Call ($T = 1$ and $1/12$) | Digital ($T = 1$ and $1/12$) |
| :--- | :--- | :--- |
| $\Delta$ | No interior minimum: smaller is better, down to the pathwise (FDMC) variance as $\ell \to 0$ | Minimum near $\ell \approx 1.05$–$1.1\,S_0\sigma\sqrt{T}$ (1.5× below standard Malliavin); 2.3× worse at the default 0.2 |
| $\Gamma$ | Broad minimum near $\ell \approx 1.05$–$1.2\,S_0\sigma\sqrt{T}$ (23–28× below standard Malliavin); about 4× worse at the default 0.5 | Narrow minimum near $\ell \approx 1.05$–$1.2\,S_0\sigma\sqrt{T}$ (70–900× below standard); 1.4–2× worse than standard at the default 0.5 |
| $\mathcal{V}$ | No interior minimum, flat at the pathwise level below $\ell \approx 0.3$ | Minimum at small gaps (0.05–0.2), 9–31× below standard Malliavin |

The narrow digital Gamma minimum comes from a near-cancellation: with a Gaussian cutoff and $\ell = S_0\sigma\sqrt{T}$, $\psi'' = (Z^2 - 1)\psi/\ell^2$ almost matches the Gamma weight $(Z^2 - \sigma\sqrt{T} Z - 1)/\ell^2$ path by path, more closely as $T$ shrinks.

---

### 2.7 The Call with Terminal Knock-Out

The contract pays $(S_T - K)^+\mathbf{1}_{S_T \le H}$ with $S_0=100, K=100, H=105.25, T=1.0, \sigma=0.20$. The knock-out is checked only at maturity, so this is not a path-dependent barrier option, and its analytical value is two Black-Scholes calls and a digital: $(S_T-K)^+ - (S_T-H)^+ - (H-K)\mathbf{1}_{S_T>H}$.

Fournié localization is applied directly to the payoff, with one cutoff covering both singularities,
$$\psi = 1 - (1-\psi_K)(1-\psi_H),$$
so that $\psi = 1$ and $\psi' = 0$ at both $K$ and $H$. For compact cutoffs with $\delta \le (H-K)/2$ (cubic, cosine, quintic) this is exactly $\psi_K + \psi_H$. For the Gaussian cutoff used here ($\delta = 2.625$), $\psi_K + \psi_H$ equals $1 + e^{-2}$ at $K$ and gives a heavily biased estimator; the product form does not. The payoff derivatives are $\Phi' = \mathbf{1}_{K < S_T \le H}$ and $\Phi'' = 0$ almost everywhere.

Mean over $R = 100$ independent runs of $N = 10^6$ paths each (`results/replications_summary.csv`); the value in parentheses is the standard error of a single run:

| Greek | Analytical (`benchmarks.csv`) | Standard Malliavin | Fournié ($\delta = 2.625$, Gaussian) | Per-path variance, Fournié / Standard |
| :--- | :---: | :---: | :---: | :---: |
| **Delta ($\Delta$)** | $0.0015092$ | $0.0015092$ (SE $6.7 \times 10^{-6}$) | $0.0015124$ (SE $5.7 \times 10^{-5}$) | $74\times$ |
| **Gamma ($\Gamma$)** | $-0.00062971$ | $-0.00062966$ (SE $2.2 \times 10^{-6}$) | $-0.00063363$ (SE $8.4 \times 10^{-5}$) | $1{,}467\times$ |
| **Vega ($\mathcal{V}$)** | $-1.25941$ | $-1.25932$ (SE $4.4 \times 10^{-3}$) | $-1.25933$ (SE $4.2 \times 10^{-3}$) | $0.92\times$ |

Both estimators are unbiased within their standard errors. The same run with $\psi = \psi_K + \psi_H$ (Gaussian) is not: it gives $\Delta = -0.0117$, $\Gamma = +0.0099$ (both with the wrong sign, about $2{,}000$ and $1{,}300$ standard errors away) and $\mathcal{V} = -1.2671$ ($14$ standard errors away). Its $\pm1.96\,$SE interval never covers the analytical Delta or Gamma in the 100 runs. It is kept as the run "Malliavin Fournié (ψ_K+ψ_H)" to show this. Standard Malliavin is far better for Delta and Gamma, and the two are on par for Vega. Fournié also costs about $2.8\times$ more time per path (see `results/efficiency.csv`).

#### Why localization does not help here:
1. **Standard Malliavin is already cheap:** about $90\%$ of paths end outside $[100, 105.25]$, where the payoff is zero and so is the Malliavin sample. Inside the corridor the payoff is at most $5.25$ and the weight is small.
2. **The pathwise part is not small:** the corridor is narrower than $2\delta$ for any reasonable $\delta$, so localization replaces the small Malliavin weight by $\Phi\psi'$ and $\Phi\psi''$ terms of order $1/\delta$ and $1/\delta^2$, with $\delta \le (H-K)/2 = 2.625$.

---

## 3. Supported Contracts & Reference Benchmarks

All contracts are evaluated under institutional parameters ($S_0 = 100, r = 0.05, q = 0.02, \sigma = 0.20$):

| Product | Parameters | Singularity Topology | Benchmark Reference | Benchmark Formula |
| :--- | :--- | :--- | :--- | :--- |
| **Vanilla European Call (1Y)** | $K = 100, T = 1.0$ | Continuous kink at $K$ ($C^0$) | Analytical | Black-Scholes-Merton closed form |
| **Vanilla European Call (1M)** | $K = 100, T = 1/12$ | Sharp continuous kink ($C^0$) | Analytical | Black-Scholes-Merton ($T \to 0$ stress test) |
| **Digital European Call** | $K = 100, T = 1.0$ | Discontinuous step at $K$ ($C^{-1}$) | Analytical | Black-Scholes Cash-or-Nothing closed form |
| **Call with terminal knock-out** | $K = 100, H = 105.25, T = 1.0$ | Payoff $(S_T - K)^+\mathbf{1}_{S_T \le H}$, checked at maturity only; kink at $K$, jump at $H$ | Analytical | $(S_T-K)^+ - (S_T-H)^+ - (H-K)\mathbf{1}_{S_T>H}$: two Black-Scholes calls and a digital |

---

## 4. Project Architecture & File Tree

```text
Greeks/
├── Benchmarks/                           # Analytical closed-form reference engines
│   ├── __init__.py
│   ├── asian.py                          # Asian benchmark Greeks (conditional MC, with var/se)
│   └── europeans.py                      # Black-Scholes analytical Greeks (call, digital, terminal knock-out)
├── Methods/                              # Greek estimation engines
│   ├── __init__.py
│   ├── Analytic/                         # Analytic pricer formulas
│   │   ├── asians.py                     # Asian moment-matching Greeks
│   │   └── europeans.py
│   ├── Malliavin/                        # Malliavin calculus implementations
│   │   ├── __init__.py
│   │   ├── malliavin_european_standard.py # Exact Skorokhod weights (Delta, Gamma, Vega)
│   │   └── malliavin_european_fournier.py # Fournié localization, general Gamma, fournier_bandwidths()
│   ├── MonteCarlo/                       # Finite Difference Monte Carlo
│   │   ├── __init__.py
│   │   ├── FDMC_asian.py                 # Asian FDMC (standard and conditional MC), with var/se
│   │   └── FDMC_european.py              # Central FDMC with Common Random Numbers (CRN)
│   ├── payoffs.py                        # Payoffs and their a.e. derivatives, shared by all estimators
│   └── sampling.py                       # default_rng(seed) paths, S_T, chunking, per-path variance / SE
├── Utilities/                            # Core analysis, profiling, and plotting utilities
│   ├── __init__.py
│   ├── metrics.py                        # Metric loaders, error enrichment, convergence tests
│   ├── plotting.py                       # Matplotlib visualization routines
│   └── profiling.py                      # Wall-clock timers; single-run FDMC budget/Pareto tools (analysis_FDMC.ipynb)
├── pricers/                              # Asian pricers (moment matching, MC, conditional MC)
├── references/                           # Third-party reference notebook (P. Marchesi, 2025)
├── results/                              # Simulation datasets (up to N = 10^8)
│   ├── benchmarks.csv                    # Analytical truth values
│   ├── FDMC_results.csv                  # FDMC runs across 10 scales x 3 eps values
│   ├── malliavin_results.csv             # Standard Malliavin runs across 10 scales
│   ├── malliavin_fournier_results.csv    # Fournié Option A runs across 10 scales
│   ├── asian_*.csv                       # Asian benchmark, FDMC grid (with var/se) and moment matching
│   ├── replications_raw.csv              # R = 100 independent runs per estimator, product and N
│   ├── replications_summary.csv          # Bias, SE, RMSE, per-path variance, time per path
│   ├── efficiency.csv                    # Variance x time per path, N and CPU ratios vs FDMC
│   ├── gap_sweep.csv                     # Per-path variance vs Fournié gap
│   ├── figures/                          # Error vs N with bands (one PNG per product), gap sweep
│   └── summary.csv                       # Consolidated benchmark summary
├── tests/                                # Automated unit & regression test suite
│   ├── test_estimators.py                # Per-path statistics, seeding, general Fournié Gamma
│   ├── test_metrics.py                   # Metrics and convergence validation tests
│   ├── test_asian_greeks.py              # Asian pricers and Greeks
│   └── test_short_maturity_call.py       # 1-Month Call and decoupled bandwidth validation
├── notebooks/                            # Notebook sources (jupytext percent .py), one per notebook
├── analysis_malliavin_vs_FDMC.ipynb      # Main comparative notebook (executed, paired with notebooks/)
├── analysis_european_call_1M.ipynb       # 1M call and terminal knock-out notebook
├── analysis_smoothing_functions.ipynb    # Cutoff functions, bandwidths, Asian and basket prototypes
├── analysis_asian_moment_matching_vs_FDMC.ipynb # Asian Greeks: moment matching vs FDMC
├── analysis_FDMC.ipynb                   # FDMC-only analysis (single runs, eps sensitivity)
├── run_notebooks.py                      # Executes notebooks/*.py and writes the paired .ipynb
├── run_asian_benchmarks.py               # Asian moment matching, conditional MC benchmark, FDMC grid
├── run.py                                # End-to-end master pipeline entry point
├── run_all_greeks.py                     # Runner executing all Monte Carlo simulation scales
├── run_benchmarks.py                     # Runner computing analytical benchmarks
├── run_replications.py                   # Independent replications, timing and efficiency
├── run_gap_sweep.py                      # Per-path variance vs Fournié gap (call, digital; T = 1, 1/12)
├── products.py                           # Standardized contract parameter dictionaries
├── config.py                             # Simulation parameters, grid scales, and seed configs
├── pyproject.toml                        # Packaging configuration
└── README.md                             # Project documentation
```

---

## 5. How to Run & Reproduce

### 5.1 Installation & Virtual Environment

Clone the repository and install dependencies in an editable environment:
```bash
# Clone the repository
git clone https://github.com/NicolasBermell/Greeks.git
cd Greeks

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install requirements and package in editable mode (Python 3.10+)
python -m pip install -r requirements.txt
python -m pip install -e .
```

---

### 5.2 Running the Test Suite

Execute the complete automated unit test suite:
```bash
python -m unittest discover tests
```
*Expected output: All 31 tests pass in a few seconds.*

---

### 5.3 Generating Analytical Benchmarks

Compute and export the exact analytical reference values:
```bash
python run_benchmarks.py
```
*This populates `results/benchmarks.csv` with exact values for all European contracts.*

---

### 5.4 Running Full Monte Carlo Simulations

To execute the entire simulation matrix (10 scales from $N = 10^4$ to $N = 10^8$ across FDMC, Standard Malliavin, and Fournié Option A):
```bash
python run.py
```
*Alternatively, run individual modules via `run_all_greeks.py`.*

Every row of `FDMC_results.csv`, `malliavin_results.csv` and `malliavin_fournier_results.csv` stores, next to each Greek, its per-path variance (`delta_var`, …) and standard error (`delta_se` = $\sqrt{\text{var}/N}$). For FDMC the variance is that of the per-path difference quotient with shared random numbers. Paths come from `np.random.default_rng(seed)`, one generator per run, and large $N$ is processed in blocks of $2\times10^6$ paths.

### 5.4b Independent Replications, Timing and Efficiency

```bash
python run_replications.py
```
Runs $R = 100$ seeds at $N \in \{10^4, 10^5, 10^6\}$ for every estimator and product (plus the call at $T = 1/12$) (FDMC at $\varepsilon \in \{0.1, 0.01, 0.001\}$, standard Malliavin, Fournié, plus the old Fournié call-Gamma shortcut) and writes `replications_summary.csv` (bias, SE across runs, RMSE, coverage of the $\pm1.96\,$SE interval, per-path variance, time per path), `efficiency.csv`, and `results/figures/error_vs_N_*.png`. Beyond $10^6$ the error is extrapolated with $\sqrt{\text{bias}^2 + \text{Var}/N}$ rather than simulated.

The number of paths for a tolerance is $N^* = z^2\,\text{Var}/\text{tol}^2$, so $N^*_A / N^*_B = \text{Var}_A / \text{Var}_B$. Each run times all three Greeks together (FDMC: 5 payoff evaluations per path; standard Malliavin: 1; Fournié: 1 plus the cutoff evaluations), and efficiency is reported as $\text{Var} \times \text{time per path}$, the CPU time needed per unit of inverse squared error.

---

### 5.4c Fournié Gap Sweep

```bash
python run_gap_sweep.py
```
Per-path variance of the Fournié estimators against the gap $\ell$ (in units of $S_0\sigma\sqrt{T}$, same $\ell$ for all Greeks, Gaussian cutoff) for the call and the digital at $T = 1$ and $T = 1/12$, on $2\times10^6$ shared paths per product, with standard Malliavin and FDMC ($\varepsilon = 0.01$) on the same paths. Writes `results/gap_sweep.csv` and `results/figures/gap_sweep.png`. Main results (variance relative to standard Malliavin):
- Call Delta and Vega: no interior minimum; the variance falls to the FDMC (pathwise) level as $\ell \to 0$.
- Call Gamma: broad minimum at $\ell \approx 1$–$1.2\,S_0\sigma\sqrt{T}$, 23–28× below standard Malliavin; the default 0.5 is about 4× worse.
- Digital Delta and Gamma: worse than standard Malliavin at the default gaps (2.3× and 1.4–2×); near $\ell \approx S_0\sigma\sqrt{T}$, 1.5× better for Delta and 70–900× better for Gamma, but the Gamma minimum is narrow (a near-cancellation between $\psi''$ and the Gamma weight for the Gaussian cutoff).
- Digital Vega: best at small gaps, 9–31× below standard Malliavin.

### 5.5 Notebooks

Each notebook is written as a plain Python file in `notebooks/` (jupytext percent format, cells marked with `# %%`), which is the version to edit and review. The executed `.ipynb` at the repository root is paired with it.

1. **[analysis_malliavin_vs_FDMC.ipynb](analysis_malliavin_vs_FDMC.ipynb):** error vs. $N$ from $R = 100$ independent runs with bands and $1/\sqrt{N}$ extrapolation; per-path variance, time per path, efficiency and $N^*$; expected error under fixed time budgets.
2. **[analysis_european_call_1M.ipynb](analysis_european_call_1M.ipynb):** the short-maturity ($T = 1/12$) call (Fournié decomposition, bandwidth sweeps, Vega analysis, per-Greek bandwidths) and the call with terminal knock-out (the two ways to combine the cutoffs, the bias of $\psi_K + \psi_H$, variance vs. bandwidth).
3. **[analysis_smoothing_functions.ipynb](analysis_smoothing_functions.ipynb):** the 5 cutoff families, bandwidth grids, and prototypes for an Asian call (weight $W_{t_1}/(S_0\sigma t_1)$) and a 3-asset basket (weight $(L^{-1}\mathbf{1})\cdot W_T/(S_0 T)$), each checked against the pathwise Delta.
4. **[analysis_asian_moment_matching_vs_FDMC.ipynb](analysis_asian_moment_matching_vs_FDMC.ipynb):** Asian Greeks by moment matching vs. FDMC against the conditional MC benchmark.
5. **[analysis_FDMC.ipynb](analysis_FDMC.ipynb):** FDMC alone: $\varepsilon$ sensitivity and single-run budget/Pareto views.

To execute all notebooks (or some of them) from their sources:
```bash
python run_notebooks.py
python run_notebooks.py analysis_european_call_1M
```
To edit a notebook interactively, open the `.ipynb` in Jupyter with the jupytext extension installed, or edit `notebooks/<name>.py` and run `jupytext --sync <name>.ipynb`.

---

## 6. Key Findings & Institutional Decision Rules

### Summary Comparison Table

Paths needed relative to FDMC ($\varepsilon = 0.01$) for the same standard error, i.e. the ratio of per-path variances, from $R = 100$ runs at $N = 10^6$ (`results/efficiency.csv`). Lower is better; multiply by the time per path (FDMC $\approx 75$ ns, standard Malliavin $\approx 28$ ns, Fournié $\approx 79$ ns) for the CPU-time ratio. Fournié uses the default gaps (Section 2.6); for the digital, other gaps change the picture (Section 5.4c).

| Contract | Greek | Standard Malliavin | Fournié | Best (variance × time) |
| :--- | :---: | :---: | :---: | :--- |
| **Call** | $\Delta$ | $5.9$ | $1.17$ | FDMC (Fournié close) |
| **Call** | $\Gamma$ | $0.014$ | $0.0021$ ($0.0006$ with the call-only shortcut) | Fournié |
| **Call** | $\mathcal{V}$ | $12.8$ | $1.00$ | FDMC / Fournié |
| **Digital** | $\Delta$ | $0.0009$ | $0.0020$ | Standard Malliavin |
| **Digital** | $\Gamma$ | $1.4 \times 10^{-10}$ | $2.7 \times 10^{-10}$ | Standard Malliavin |
| **Digital** | $\mathcal{V}$ | $0.88$ | $0.10$ | Fournié |
| **Call with terminal knock-out** | $\Delta$ | $1.8 \times 10^{-6}$ | $1.3 \times 10^{-4}$ | Standard Malliavin |
| **Call with terminal knock-out** | $\Gamma$ | $5.0 \times 10^{-12}$ | $7.3 \times 10^{-9}$ | Standard Malliavin |
| **Call with terminal knock-out** | $\mathcal{V}$ | $0.61$ | $0.56$ | Standard Malliavin |

At $\varepsilon = 0.01$ the FDMC truncation bias is below the noise of these runs except for the knock-out Vega; the problem of FDMC on non-smooth payoffs is variance (per-path variance growing like $1/\varepsilon$ for a kink and $1/\varepsilon^3$ for a jump).

### Production Rules of Thumb:
1. **Unbounded payoffs ($S_T \to \infty$):** use **Fournié localization** for Gamma, with a gap near $S_0 \sigma \sqrt{T}$; for Delta and Vega it can only match the pathwise derivative (FDMC with shared random numbers), at a similar cost per path.
2. **Narrow knock-out intervals ($H - K \ll S_0 \sigma \sqrt{T}$):** Turn Fournié **OFF** and deploy **Pure Standard Malliavin**.
3. **Cost per path:** central differences need 5 payoff evaluations per path for $(\Delta, \Gamma, \mathcal{V})$ against 1 for Malliavin. Measured with all three Greeks together, standard Malliavin takes about $28$ ns per path, FDMC about $75$ ns and Fournié about $79$ ns (`results/efficiency.csv`). Compare methods by variance $\times$ time per path, not by path count alone.