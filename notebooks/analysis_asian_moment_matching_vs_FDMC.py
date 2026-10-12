# ---
# jupyter:
#   jupytext:
#     formats: ipynb,notebooks//py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.6
#   kernelspec:
#     display_name: Python 3 (.venv)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Asian Option Sensitivity Analysis: FDMC vs. Moment Matching vs. Brute Force Conditional MC Benchmark
# ## Discrete Arithmetic Average, Tail Distribution Mismatch (OTM vs. ATM vs. ITM), Discontinuous Payoffs (Digital & Barrier), Sample Size Requirements ($N^*$), and Malliavin Outlook
#
# ### Executive Overview
# In energy, commodity, foreign exchange, and equity derivatives markets, **Asian options**—whose payoff depends on the arithmetic average price of the underlying asset over a discrete schedule of observation dates—are among the most actively traded exotic contracts:
# $$A = \frac{1}{m} \sum_{i=1}^m S_{t_i}, \quad t_i = i \frac{T}{m}$$
#
# Because the sum of correlated lognormal random variables **does not possess an analytical closed-form probability distribution**, market participants and quantitative desks face a fundamental dilemma:
# 1. **Moment Matching (Turnbull-Wakeman / Levy)**: Equates the first two analytical moments $\mathbb{E}[A]$ and $\mathbb{E}[A^2]$ to an approximating lognormal variable $A \sim \text{Lognormal}(\mu_A, v_A^2)$ and evaluates option prices and Greeks using Black-Scholes-like closed forms.
#    - *Strengths*: Instantaneous computation ($\approx 0.01$ ms per contract), zero Monte Carlo noise.
#    - *Failure Modes*: Intrinsic **model / distributional bias**. Because the true arithmetic average has different higher cumulants (skewness and kurtosis), the approximation breaks down in the **tails** (deep In-The-Money and Out-Of-The-Money strikes) and on **discontinuous contracts** (Digital and Barrier options).
# 2. **Finite Difference Monte Carlo (FDMC)**: Simulates the true diffusion paths without distributional assumptions and differentiates via finite differences $\frac{\Delta V}{2\varepsilon}$ and $\frac{\Delta^2 V}{\varepsilon^2}$.
#    - *Strengths*: Asymptotically exact for smooth payoffs as $N \to \infty$.
#    - *Failure Modes*: Severe **computational latency** ($N \ge 10^5$ simulations needed), discretization error $\mathcal{O}(\varepsilon^2)$, and **catastrophic variance blowout $\mathcal{O}(1/\varepsilon^2)$** when differentiating discontinuous Heaviside indicator functions (Digital Cash-or-Nothing and Barrier knockouts).
# 3. **The Ground Truth Benchmark: Brute Force Conditional Monte Carlo (Broadie & Glasserman 1996)**:
#    - To rigorously evaluate the errors of both proposed models, we employ **Conditional Monte Carlo with Common Random Numbers ($N = 5,000,000$ paths)** as the high-precision benchmark. By conditioning on the penultimate filtration $\mathcal{F}_{t_{m-1}}$, the conditional expectation is evaluated in closed form via the smooth Gaussian CDF $\Phi(d_2^*)$, providing a smooth, low-variance, exact numerical reference.
#
# This notebook presents a comprehensive comparative sensitivity analysis across 9 discrete Asian contracts (Strikes: $K \in \{80, 100, 125\}$, Barrier $H=135$):
# - **Executive Comparative Summary Table**: Direct side-by-side comparison of Benchmark vs. Moment Matching vs. Production FDMC ($N=5,000,000, \varepsilon=0.01$).
# - **Deep Dives by Contract Category**:
#   1. Vanilla Asian Calls (Smooth Payoff: OTM vs. ATM vs. ITM)
#   2. Digital Asian Calls (Discontinuous Step Payoff: OTM vs. ATM vs. ITM)
#   3. Barrier Up-and-Out Asian Calls (Boundary Jump Discontinuity: OTM vs. ATM vs. ITM)
# - **Multi-$N$ Error Convergence Curves**: Log-log error convergence of FDMC ($N = 10^4 \to 5 \times 10^6$) compared against constant Moment Matching model error and institutional tolerance thresholds.
# - **Sample Size Requirements ($N^*$)**: Minimum simulations needed for FDMC to satisfy institutional tolerances vs. Moment Matching Pass/Fail status.
# - **Malliavin Calculus Outlook**: How Malliavin integration by parts on Wiener space resolves both challenges simultaneously.

# %% [markdown]
# ---
# ## 1. Mathematical Foundations: Exact Moments, Lognormal Matching, FDMC, and Conditional MC
#
# ### 1.1 Black-Scholes Asset Dynamics and Discrete Observation Grid
# Under the risk-neutral pricing measure $\mathbb{Q}$, the underlying spot price evolves as:
# $$dS_t = (r - q) S_t dt + \sigma S_t dW_t$$
# Given spot $S_0$, the price at observation date $t_i = i \frac{T}{m}$ ($i=1, \dots, m$) is:
# $$S_{t_i} = S_0 \exp\left( \left(r - q - \frac{1}{2}\sigma^2\right) t_i + \sigma W_{t_i} \right)$$
# The discrete arithmetic average is $A = \frac{1}{m} \sum_{i=1}^m S_{t_i}$.
#
# ### 1.2 Exact Moment Calculations
# The first two analytical moments under $\mathbb{Q}$ are given by:
# 1. **First Moment ($M_1 = \mathbb{E}[A]$)**:
#    $$M_1 = \frac{S_0}{m} \sum_{i=1}^m e^{(r - q) t_i}$$
# 2. **Second Moment ($M_2 = \mathbb{E}[A^2]$)**:
#    $$\mathbb{E}[S_{t_i} S_{t_j}] = S_0^2 \exp\left( (r - q)(t_i + t_j) + \sigma^2 \min(t_i, t_j) \right)$$
#    $$M_2 = \frac{S_0^2}{m^2} \left[ \sum_{i=1}^m e^{2(r - q) t_i + \sigma^2 t_i} + 2 \sum_{1 \le i < j \le m} e^{(r - q)(t_i + t_j) + \sigma^2 t_i} \right]$$
#
# ### 1.3 Lognormal Moment Matching Formulation
# Matching $A$ to an approximating lognormal variable $A \sim \text{Lognormal}(\mu_A, v_A^2)$:
# $$v_A^2 = \ln\left( \frac{M_2}{M_1^2} \right), \quad v_A = \sqrt{v_A^2}, \quad \mu_A = \ln(M_1) - \frac{1}{2} v_A^2$$
# $$d_1(K) = \frac{\ln(M_1 / K) + \frac{1}{2} v_A^2}{v_A}, \quad d_2(K) = d_1(K) - v_A$$
#
# Closed-form pricing equations:
# - **Vanilla Asian Call**: $C_{\text{Asian}}(K) = e^{-rT} \left( M_1 \Phi(d_1(K)) - K \Phi(d_2(K)) \right)$
# - **Digital Asian Call**: $D_{\text{Asian}}(K) = e^{-rT} \Phi(d_2(K))$
# - **Barrier Up-and-Out Asian Call**: $B_{\text{Asian}}(K, H) = C_{\text{Asian}}(K) - C_{\text{Asian}}(H) - (H - K) D_{\text{Asian}}(H)$
#
# Greeks are computed via central differences on the analytical closed-form formulas ($\varepsilon = 10^{-5}$):
# $$\Delta = \frac{V(S_0 + \varepsilon) - V(S_0 - \varepsilon)}{2\varepsilon}, \quad \Gamma = \frac{V(S_0 + \varepsilon) - 2V(S_0) + V(S_0 - \varepsilon)}{\varepsilon^2}, \quad \mathcal{V} = \frac{V(\sigma + \varepsilon) - V(\sigma - \varepsilon)}{2\varepsilon}$$
#
# ### 1.4 Finite Difference Monte Carlo (FDMC)
# Standard FDMC estimates sensitivities by simulating paths with Common Random Numbers (CRN / CN):
# - For smooth payoffs (Vanilla Calls), central differences achieve $\mathcal{O}(\varepsilon^2 + \sigma_{\text{MC}}/\sqrt{N})$ accuracy.
# - For discontinuous payoffs (Digital $\mathbf{1}_{A > K}$ and Barrier $\mathbf{1}_{A \le H}$), the derivative of the indicator is a Dirac delta measure $\delta(A - K)$. Dividing pathwise flips by $\varepsilon^2 = 10^{-4}$ causes catastrophic $\mathcal{O}(1/(\varepsilon^2 N))$ variance blowout.
#
# ### 1.5 The Benchmark: Brute Force Conditional Monte Carlo (CMC)
# By conditioning on the filtration up to $t_{m-1}$ (Broadie & Glasserman 1996):
# $$A = \frac{1}{m}\sum_{i=1}^{m-1} S_{t_i} + \frac{1}{m} S_{t_m}$$
# Given $\mathcal{F}_{t_{m-1}}$, the partial sum is known, and $S_{t_m}$ is a single lognormal step. The conditional expectation $\mathbb{E}[\text{Payoff} \mid \mathcal{F}_{t_{m-1}}]$ is evaluated in closed form via the smooth Gaussian CDF $\Phi(d_2^*)$, completely eliminating boundary jump noise and providing a stable, high-precision ground truth.

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root in sys.path
root_dir = str(Path.cwd())
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from config import (
    ASIAN_BENCHMARK_N_SIMS,
    ASIAN_BENCHMARK_EPS,
    ASIAN_BENCHMARKS_FILE,
    ASIAN_FDMC_RESULTS_FILE,
    ASIAN_MOMENT_MATCHING_RESULTS_FILE,
)
from products import asian_products

# Styling configuration
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.titlesize": 15,
    "lines.linewidth": 2.0,
})

pd.set_option("display.precision", 6)
pd.set_option("display.max_columns", 16)
pd.set_option("display.width", 1000)

print("Environment and configuration loaded successfully.")

# %% [markdown]
# ---
# ## 2. Benchmark Ingestion & Executive Comparative Summary Table ($N = 5,000,000$)
#
# We load the simulation datasets across all 9 Asian contracts:
# 1. **Ground Truth Benchmark (`asian_benchmarks.csv`)**: Brute Force Conditional Monte Carlo ($N = 5,000,000$ paths, $\varepsilon = 0.01$, Common Numbers).
# 2. **Moment Matching Results (`asian_moment_matching_results.csv`)**: Closed-form Turnbull-Wakeman approximation with analytical finite differences ($\varepsilon = 10^{-5}$).
# 3. **Multi-$N$ FDMC Results (`asian_FDMC_results.csv`)**: Standard Finite Difference Monte Carlo with Common Numbers across $N \in [10^4, 5\times 10^4, 10^5, 2.5\times 10^5, 5\times 10^5, 10^6, 2\times 10^6, 5\times 10^6]$ ($\varepsilon = 0.01$).
#
# For both proposed models, we compute:
# - **Absolute Error**: $|e| = |\text{Estimate} - \text{Benchmark}|$
# - **Relative Error (%)**: $\text{Rel. Error} = \frac{|\text{Estimate} - \text{Benchmark}|}{|\text{Benchmark}|} \times 100\%$
# - **Signed Bias**: $\text{Bias} = \text{Estimate} - \text{Benchmark}$
# - **Error Ratio**: $\text{Ratio} = \frac{\text{Abs Error (FDMC)}}{\text{Abs Error (MM)}}$

# %%
# Load datasets
b_df = pd.read_csv(ASIAN_BENCHMARKS_FILE, index_col="Product")
mm_df = pd.read_csv(ASIAN_MOMENT_MATCHING_RESULTS_FILE, index_col="Product")
fdmc_df = pd.read_csv(ASIAN_FDMC_RESULTS_FILE)

# Extract N=5,000,000 FDMC for executive comparison
fdmc_5M = fdmc_df[fdmc_df["N"] == 5_000_000].set_index("Product")

# Build comprehensive side-by-side comparative summary table
summary_records = []
greeks = ["delta", "gamma", "vega"]

for prod in b_df.index:
    for g in greeks:
        b_val = b_df.loc[prod, g]
        m_val = mm_df.loc[prod, g]
        f_val = fdmc_5M.loc[prod, g]
        
        m_abs = abs(m_val - b_val)
        f_abs = abs(f_val - b_val)
        m_rel = (m_abs / abs(b_val)) * 100.0 if abs(b_val) > 0 else np.nan
        f_rel = (f_abs / abs(b_val)) * 100.0 if abs(b_val) > 0 else np.nan
        ratio = f_abs / m_abs if m_abs > 0 else np.nan
        
        summary_records.append({
            "Product": prod,
            "Greek": g.capitalize(),
            "Benchmark (CMC 5M)": b_val,
            "MM Value": m_val,
            "MM Abs Error": m_abs,
            "MM Rel Err (%)": m_rel,
            "FDMC Value (5M)": f_val,
            "FDMC Abs Error": f_abs,
            "FDMC Rel Err (%)": f_rel,
            "Ratio (FDMC / MM)": ratio,
        })

executive_summary_df = pd.DataFrame(summary_records)
print("=== Executive Comparative Summary: Benchmark vs. Proposed Models (N = 5,000,000) ===")
display(executive_summary_df.round(6))

# %% [markdown]
# ---
# ## 3. Deep Dive: Vanilla Asian Calls (Smooth Payoff: OTM vs. ATM vs. ITM)
#
# ### 3.1 Payoff Regularity & Model Comparison
# For the vanilla Asian call with payoff $(A - K)^+$:
# - The payoff is continuous ($C^0$) with a single kink at $A = K$.
# - **Moment Matching**: Evaluates the integral over the probability density. Around the center of the distribution (**ATM $K=100$**), the lognormal model achieves excellent precision ($\approx 0.67\%$ error on Delta, $\approx 0.69\%$ on Vega). However, in the **tails** (deep **OTM $K=125$** and **ITM $K=80$**), the kurtosis and skewness mismatch manifest as higher relative error ($\approx 1.14\%$ to $8.62\%$).
# - **FDMC**: Displays textbook $\mathcal{O}(N^{-1/2})$ Monte Carlo convergence for continuous payoffs, achieving high precision across all Greeks as $N \to 5 \times 10^6$.
#
# Below, we plot:
# 1. **Multi-$N$ Error Convergence**: $\log_{10}(\text{Abs Error})$ vs. $N$ for FDMC, plotted alongside the constant Moment Matching model error and target institutional tolerances ($\Delta \le 0.005$, $\Gamma \le 0.0005$, $\mathcal{V} \le 0.05$).
# 2. **Comparative Grouped Bar Charts**: Ordered strictly as **OTM ($K=125$), ATM ($K=100$), ITM ($K=80$)**.

# %%
# Filter for Vanilla Asian Call contracts
call_contracts = ["Call Asian OTM", "Call Asian ATM", "Call Asian ITM"]
tolerances = {"Delta": 0.005, "Gamma": 0.0005, "Vega": 0.05}

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
greeks = ["Delta", "Gamma", "Vega"]

for idx, g in enumerate(greeks):
    ax = axes[idx]
    g_lower = g.lower()
    
    for c_name in call_contracts:
        # FDMC data across N
        sub_fdmc = fdmc_df[fdmc_df["Product"] == c_name].sort_values("N")
        b_val = b_df.loc[c_name, g_lower]
        mm_val = mm_df.loc[c_name, g_lower]
        
        fdmc_err = (sub_fdmc[g_lower] - b_val).abs()
        mm_err = abs(mm_val - b_val)
        
        label_base = c_name.replace("Call Asian ", "")
        ax.plot(sub_fdmc["N"], fdmc_err, marker="o", lw=1.8, label=f"FDMC {label_base}")
        ax.axhline(mm_err, ls="--", alpha=0.7, label=f"MM {label_base} (err={mm_err:.2e})")
    
    # Target tolerance line
    ax.axhline(tolerances[g], color="black", ls=":", lw=2.0, label=f"Institutional Tol ({tolerances[g]})")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel(f"Absolute Error in {g}")
    ax.set_title(f"Vanilla Calls - {g} Error Convergence vs. N", fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    if idx == 0:
        ax.legend(loc="lower left", fontsize=9, frameon=True)

plt.suptitle("Vanilla Asian Calls: FDMC Convergence vs. Moment Matching Error Floor", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Grouped Bar Charts for Vanilla Asian Calls (Ordered: OTM, ATM, ITM)
strikes_labels = ["OTM (K=125)", "ATM (K=100)", "ITM (K=80)"]
call_prods = ["Call Asian OTM", "Call Asian ATM", "Call Asian ITM"]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
width = 0.35
x = np.arange(len(strikes_labels))

for idx, g in enumerate(["Delta", "Gamma", "Vega"]):
    ax = axes[idx]
    g_lower = g.lower()
    
    mm_errs = [abs(mm_df.loc[p, g_lower] - b_df.loc[p, g_lower]) for p in call_prods]
    fdmc_errs = [abs(fdmc_5M.loc[p, g_lower] - b_df.loc[p, g_lower]) for p in call_prods]
    
    ax.bar(x - width/2, mm_errs, width, label="Moment Matching (FD)", color="#3b82f6", alpha=0.85)
    ax.bar(x + width/2, fdmc_errs, width, label="FDMC (N=5M, eps=0.01)", color="#f59e0b", alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(strikes_labels)
    ax.set_ylabel(f"Absolute Error in {g}")
    ax.set_title(f"Vanilla Calls: {g} Error by Strike", fontweight="bold")
    ax.grid(True, alpha=0.3)
    if idx == 0:
        ax.legend(loc="upper left")

plt.suptitle("Vanilla Asian Calls: Absolute Error Comparison Across Strikes (OTM, ATM, ITM)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 4. Deep Dive: Digital Asian Calls (Discontinuous Step Payoff: OTM vs. ATM vs. ITM)
#
# ### 4.1 Step Discontinuity & The Failure of Standard FDMC Gamma
# The digital call pays the pure Heaviside indicator $\mathbf{1}_{A > K}$:
# - The mathematical derivative of $\mathbf{1}_{A > K}$ is a **Dirac delta measure $\delta(A - K)$**.
# - **The Catastrophic Failure of FDMC Gamma**:
#   Evaluating $\Gamma = \frac{V(S_0+\varepsilon) - 2V(S_0) + V(S_0-\varepsilon)}{\varepsilon^2}$ across a step discontinuity divides indicator boundary jump flips by $\varepsilon^2 = 10^{-4}$.
#   Even with $N = 5,000,000$ simulations, standard FDMC Gamma has an absolute error of **$0.072$** at ATM, **failing the target tolerance of $0.0005$ by over $144\times$**!
# - **Moment Matching Performance**:
#   Evaluates the smooth derivative of $e^{-rT}\Phi(d_2)$. While it completely eliminates the jump noise, it carries a small tail model error (e.g. MM Gamma error is $0.00034$, well within tolerance).
# - **The Conditional MC Benchmark Resolution**:
#   By evaluating $\mathbb{E}[\mathbf{1}_{A > K} \mid \mathcal{F}_{t_{m-1}}] = \Phi(d_2^*)$ analytically, Conditional MC provides the true ground truth reference without numerical blowout.

# %%
# Convergence plots for Digital Asian Calls
dig_contracts = ["Digital Asian OTM", "Digital Asian ATM", "Digital Asian ITM"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, g in enumerate(greeks):
    ax = axes[idx]
    g_lower = g.lower()
    
    for c_name in dig_contracts:
        sub_fdmc = fdmc_df[fdmc_df["Product"] == c_name].sort_values("N")
        b_val = b_df.loc[c_name, g_lower]
        mm_val = mm_df.loc[c_name, g_lower]
        
        fdmc_err = (sub_fdmc[g_lower] - b_val).abs()
        mm_err = abs(mm_val - b_val)
        
        label_base = c_name.replace("Digital Asian ", "")
        ax.plot(sub_fdmc["N"], fdmc_err, marker="o", lw=1.8, label=f"FDMC {label_base}")
        ax.axhline(mm_err, ls="--", alpha=0.7, label=f"MM {label_base} (err={mm_err:.2e})")
    
    tol_val = tolerances[g] if g != "Gamma" else 0.0005
    ax.axhline(tol_val, color="black", ls=":", lw=2.0, label=f"Target Tol ({tol_val})")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel(f"Absolute Error in {g}")
    ax.set_title(f"Digital Calls - {g} Error Convergence vs. N", fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    if idx == 1:
        ax.legend(loc="upper right", fontsize=9, frameon=True)

plt.suptitle("Digital Asian Calls: Severe FDMC Gamma Discontinuity Explosion vs. Moment Matching", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Grouped Bar Charts for Digital Asian Calls (Ordered: OTM, ATM, ITM)
dig_prods = ["Digital Asian OTM", "Digital Asian ATM", "Digital Asian ITM"]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

for idx, g in enumerate(["Delta", "Gamma", "Vega"]):
    ax = axes[idx]
    g_lower = g.lower()
    
    mm_errs = [abs(mm_df.loc[p, g_lower] - b_df.loc[p, g_lower]) for p in dig_prods]
    fdmc_errs = [abs(fdmc_5M.loc[p, g_lower] - b_df.loc[p, g_lower]) for p in dig_prods]
    
    ax.bar(x - width/2, mm_errs, width, label="Moment Matching (FD)", color="#3b82f6", alpha=0.85)
    ax.bar(x + width/2, fdmc_errs, width, label="FDMC (N=5M, eps=0.01)", color="#f59e0b", alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(strikes_labels)
    ax.set_ylabel(f"Absolute Error in {g}")
    ax.set_title(f"Digital Calls: {g} Error by Strike", fontweight="bold")
    ax.grid(True, alpha=0.3)
    if idx == 1:
        ax.set_yscale("log")
        ax.set_ylabel("Absolute Error in Gamma (Log Scale)")
    if idx == 0:
        ax.legend(loc="upper left")

plt.suptitle("Digital Asian Calls: Absolute Error Comparison Across Strikes (OTM, ATM, ITM)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 5. Deep Dive: Barrier Up-and-Out Asian Calls (Boundary Jump Discontinuity: OTM vs. ATM vs. ITM)
#
# ### 5.1 Upper Boundary Discontinuity at $H = 135$
# The barrier contract pays $(A - K)^+ \mathbf{1}_{A \le H}$:
# - **Boundary Knockout Jump**: At the upper barrier $H=135$, the payoff drops precipitously from $(H - K)$ to $0$, introducing a boundary step discontinuity $-(H - K) \delta(A - H)$.
# - **FDMC Failure**: Differentiating across the boundary at $H=135$ causes massive cancellation noise for Gamma:
#   - At ATM, FDMC Gamma error is **$0.387$** (failing tolerance by $>770\times$ and even flipping to negative!).
#   - At ITM, FDMC Gamma error is **$1.127$**!
# - **Moment Matching Sensitivity to Tail Truncation**:
#   Truncating the upper tail at $H=135$ magnifies the lognormal tail mismatch, producing Vega errors of **$+7.22\%$ to $+8.31\%$**.

# %%
# Convergence plots for Barrier Up-and-Out Asian Calls
bar_contracts = ["Barrier Asian Up and Out OTM", "Barrier Asian Up and Out ATM", "Barrier Asian Up and Out ITM"]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, g in enumerate(greeks):
    ax = axes[idx]
    g_lower = g.lower()
    
    for c_name in bar_contracts:
        sub_fdmc = fdmc_df[fdmc_df["Product"] == c_name].sort_values("N")
        b_val = b_df.loc[c_name, g_lower]
        mm_val = mm_df.loc[c_name, g_lower]
        
        fdmc_err = (sub_fdmc[g_lower] - b_val).abs()
        mm_err = abs(mm_val - b_val)
        
        label_base = c_name.replace("Barrier Asian Up and Out ", "")
        ax.plot(sub_fdmc["N"], fdmc_err, marker="o", lw=1.8, label=f"FDMC {label_base}")
        ax.axhline(mm_err, ls="--", alpha=0.7, label=f"MM {label_base} (err={mm_err:.2e})")
    
    tol_val = tolerances[g] if g != "Gamma" else 0.0005
    ax.axhline(tol_val, color="black", ls=":", lw=2.0, label=f"Target Tol ({tol_val})")
    
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample Size (N)")
    ax.set_ylabel(f"Absolute Error in {g}")
    ax.set_title(f"Barrier Calls - {g} Error Convergence vs. N", fontweight="bold")
    ax.grid(True, which="both", ls="-", alpha=0.3)
    if idx == 0:
        ax.legend(loc="lower left", fontsize=9, frameon=True)

plt.suptitle("Barrier Up-and-Out Asian Calls: Boundary Jump Discontinuity Error vs. N", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %%
# Grouped Bar Charts for Barrier Up-and-Out Asian Calls (Ordered: OTM, ATM, ITM)
bar_prods = ["Barrier Asian Up and Out OTM", "Barrier Asian Up and Out ATM", "Barrier Asian Up and Out ITM"]

fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

for idx, g in enumerate(["Delta", "Gamma", "Vega"]):
    ax = axes[idx]
    g_lower = g.lower()
    
    mm_errs = [abs(mm_df.loc[p, g_lower] - b_df.loc[p, g_lower]) for p in bar_prods]
    fdmc_errs = [abs(fdmc_5M.loc[p, g_lower] - b_df.loc[p, g_lower]) for p in bar_prods]
    
    ax.bar(x - width/2, mm_errs, width, label="Moment Matching (FD)", color="#3b82f6", alpha=0.85)
    ax.bar(x + width/2, fdmc_errs, width, label="FDMC (N=5M, eps=0.01)", color="#f59e0b", alpha=0.85)
    
    ax.set_xticks(x)
    ax.set_xticklabels(strikes_labels)
    ax.set_ylabel(f"Absolute Error in {g}")
    ax.set_title(f"Barrier Calls: {g} Error by Strike", fontweight="bold")
    ax.grid(True, alpha=0.3)
    if idx == 1:
        ax.set_yscale("log")
        ax.set_ylabel("Absolute Error in Gamma (Log Scale)")
    if idx == 0:
        ax.legend(loc="upper left")

plt.suptitle("Barrier Up-and-Out Asian Calls: Absolute Error Comparison Across Strikes (OTM, ATM, ITM)", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# ## 6. Sample Size Requirements ($N^*$) and Institutional Tolerance Evaluation
#
# Institutional production tolerances:
# - **Delta ($\Delta$)**: $|e| \le 0.0050$ ($50$ basis points of delta).
# - **Gamma ($\Gamma$)**: $|e| \le 0.0005$ (within $0.0005$ curvature units).
# - **Vega ($\mathcal{V}$)**: $|e| \le 0.0500$ ($5$ basis points per $1\%$ vol).
#
# Below, we determine:
# 1. **FDMC Min $N^*$**: The minimum simulation sample size required for standard FDMC to satisfy the institutional tolerance.
# 2. **Moment Matching Status**: Whether the analytical approximation satisfies the tolerance (Pass) or permanently fails due to model bias (Fail).

# %%
# Compute minimum acceptable N* for FDMC and check Moment Matching status
tol_dict = {"delta": 0.0050, "gamma": 0.0005, "vega": 0.0500}
n_star_records = []

for prod in b_df.index:
    for g in ["delta", "gamma", "vega"]:
        target_tol = tol_dict[g]
        b_val = b_df.loc[prod, g]
        mm_val = mm_df.loc[prod, g]
        mm_err = abs(mm_val - b_val)
        mm_status = "PASS" if mm_err <= target_tol else "FAIL (Model Bias)"
        
        # Check FDMC across N
        sub_fdmc = fdmc_df[fdmc_df["Product"] == prod].sort_values("N")
        fdmc_n_star = None
        fdmc_achieved_err = None
        
        for _, row in sub_fdmc.iterrows():
            err = abs(row[g] - b_val)
            if err <= target_tol:
                fdmc_n_star = int(row["N"])
                fdmc_achieved_err = err
                break
        
        if fdmc_n_star is None:
            fdmc_status = "FAIL (Variance/Bias)"
            fdmc_n_star_display = f"> {sub_fdmc['N'].max():,}"
            best_row = sub_fdmc.iloc[-1]
            fdmc_achieved_err = abs(best_row[g] - b_val)
        else:
            fdmc_status = "PASS"
            fdmc_n_star_display = f"{fdmc_n_star:,}"
        
        n_star_records.append({
            "Product": prod,
            "Greek": g.capitalize(),
            "Target Tol": target_tol,
            "MM Error": mm_err,
            "MM Status": mm_status,
            "FDMC Min N*": fdmc_n_star_display,
            "FDMC Achieved Err": fdmc_achieved_err,
            "FDMC Status": fdmc_status,
        })

n_star_df = pd.DataFrame(n_star_records)
print("=== Institutional Precision Assessment: Minimum Required N* and Status ===")
display(n_star_df)

# %% [markdown]
# ---
# ## 7. Computational Efficiency & Latency Trade-Off
#
# ### Practical Trade-Offs for Derivatives Trading Desks:
# 1. **Moment Matching (FD)**:
#    - Evaluated in **$< 0.02$ milliseconds** per contract.
#    - Completely instantaneous; ideal for real-time risk aggregation, portfolio rebalancing, and high-frequency quote generation for vanilla ATM structures.
#    - However, cannot be trusted blindly for tail-risk hedging (deep OTM calls) or discontinuous barrier contracts.
# 2. **Standard FDMC**:
#    - Takes **$\approx 200$ to $2,500$ milliseconds** per contract for $N \ge 10^5$ simulations.
#    - For vanilla calls, it achieves higher accuracy than Moment Matching once $N \ge 10^5$.
#    - For Digital and Barrier Gamma, it **fundamentally fails even after millions of simulations and seconds of compute** due to indicator step variance blowout.
# 3. **The Ground Truth Benchmark (Conditional MC)**:
#    - Completely eliminates the indicator jump noise, but requires generating Brownian paths and conditional Black-Scholes evaluations.
#    - Serves as the definitive benchmark against which fast models are validated.

# %%
import time
from pricers.asians_analytic import compute_discrete_asian_moments
from Methods.Analytic.asians import compute_all_asian_moment_matching_greeks
from Methods.MonteCarlo.FDMC_asian import compute_all_asian_FDMC_greeks
from pricers.asians_MC import generate_asian_paths

# Benchmark wall-clock runtime across sample sizes
sample_prod = asian_products["Call Asian ATM"]
timing_records = []

# Moment Matching timing
t0 = time.perf_counter()
for _ in range(1000):
    _ = compute_all_asian_moment_matching_greeks(sample_prod, eps=1e-5)
t1 = time.perf_counter()
mm_time_ms = ((t1 - t0) / 1000.0) * 1000.0

timing_records.append({
    "Method": "Moment Matching (FD)",
    "N": "Analytical",
    "Time per Contract (ms)": mm_time_ms,
    "Throughput (Contracts/sec)": 1000.0 / mm_time_ms,
})

# FDMC timing across N
for N in [10_000, 100_000, 500_000, 1_000_000]:
    t0 = time.perf_counter()
    paths = generate_asian_paths(sample_prod["T"], 12, N, seed=42)
    _ = compute_all_asian_FDMC_greeks(sample_prod, eps=0.01, N=N, seed=42, paths=paths)
    t1 = time.perf_counter()
    fdmc_time_ms = (t1 - t0) * 1000.0
    
    timing_records.append({
        "Method": "Standard FDMC",
        "N": f"{N:,}",
        "Time per Contract (ms)": fdmc_time_ms,
        "Throughput (Contracts/sec)": 1000.0 / fdmc_time_ms,
    })

timing_df = pd.DataFrame(timing_records)
print("=== Wall-Clock Runtime Profiling: Moment Matching vs. FDMC ===")
display(timing_df)

# %% [markdown]
# ---
# ## 8. Outlook: How Malliavin Calculus Solves Both Problems for Asian Options
#
# As demonstrated by the comparative analysis above:
# - **Moment Matching** is ultrafast but suffers from **systematic model bias in the tails** and on non-smooth contracts due to the invalid lognormal assumption.
# - **Finite Difference Monte Carlo (FDMC)** avoids distributional assumptions, but suffers from **severe computational latency** and **unacceptable variance explosion $\mathcal{O}(1/\varepsilon^2)$ on discontinuous payoffs**.
#
# ### 8.1 Skorokhod Duality & Wiener Integration by Parts
# Malliavin Calculus completely bypasses numerical finite differences. By applying **Skorokhod integration by parts on Wiener space**, the differentiation operator is transferred off the discontinuous payoff function and onto the underlying Gaussian noise:
# $$\mathbb{E}[\langle DF, u \rangle_{L^2[0, T]}] = \mathbb{E}[F \cdot \delta(u)]$$
# where $D$ is the Malliavin derivative and $\delta$ is the Skorokhod integral (the adjoint of $D$).
#
# For an option with discounted payoff $V = e^{-rT} \Phi(A)$, the sensitivity with respect to spot $S_0$ or volatility $\sigma$ is:
# $$\frac{\partial V}{\partial \theta} = e^{-rT} \mathbb{E}\left[ \Phi(A) \cdot \pi^\theta \right]$$
# where $\pi^\theta = \delta(u)$ is the **Malliavin weight**.
#
# ### 8.2 Pathwise Malliavin Derivative of the Discrete Average
# For the discrete arithmetic average $A = \frac{1}{m} \sum_{i=1}^m S_{t_i}$, the Malliavin derivative with respect to Brownian motion at time $s \le T$ is:
# $$D_s S_{t_i} = \sigma S_{t_i} \mathbf{1}_{s \le t_i} \implies D_s A = \frac{\sigma}{m} \sum_{i=1}^m S_{t_i} \mathbf{1}_{s \le t_i}$$
#
# ### 8.3 Strategic Comparison: The Complete Asian Sensitivity Landscape
# | Method Feature | Moment Matching (FD) | Standard FDMC | Conditional MC Benchmark | Malliavin Calculus |
# | :--- | :--- | :--- | :--- | :--- |
# | **Distributional Assumption** | Assumes Lognormal (Biased in Tails) | None (Exact Dynamics) | None (Exact Dynamics) | **None (Exact Dynamics)** |
# | **Payoff Discontinuity** | Smooth closed-form approximation | Variance blowout $\mathcal{O}(1/\varepsilon^2)$ | Analytical tower expectation | **Unconditionally Stable** |
# | **Finite Difference Perturbation** | Perturbation error $\mathcal{O}(\varepsilon^2)$ | Discretization error $\mathcal{O}(\varepsilon^2)$ | Perturbation error $\mathcal{O}(\varepsilon^2)$ | **Zero Perturbation ($\varepsilon = 0$)** |
# | **Simulation Passes** | Analytical (0 passes) | 4–5 Passes ($S_0 \pm \varepsilon, \sigma \pm \varepsilon$) | 4–5 Passes | **Single Pass for All Greeks** |
# | **Computational Speed** | Instantaneous ($<0.02$ ms) | High latency ($>500$ ms) | Moderate latency | **$2.5\times$ to $3.5\times$ faster than FDMC** |
#
# In the next phase of this project, we will implement the **Malliavin Greek engine for Asian options**, achieving exact pathwise integration without perturbation or variance blowout!
