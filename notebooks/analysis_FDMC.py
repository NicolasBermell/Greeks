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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # European Options Sensitivity Analysis: Finite Difference Monte Carlo (FDMC)
# ## Precision, Sample Size Requirements, and Joint CPU Budget Performance (250 ms, 500 ms, 1000 ms, 5000 ms)
#
# ### 1. Objective & Scope
# This notebook provides a dedicated evaluation of **Finite Difference Monte Carlo (FDMC)** with Common Random Numbers (CRN) for computing Greeks ($\Delta, \Gamma, \mathcal{V}$) across European options:
# 1. **Vanilla European Call**: Smooth $C^1$ convex payoff $(S_T - K)^+$.
# 2. **Digital European Call**: Discontinuous step payoff $\mathbf{1}_{S_T > K}$ with Dirac delta derivative $\delta(S_T - K)$.
# 3. **European Terminal Barrier Up-and-Out Call**: Piecewise payoff $(S_T - K)^+ \mathbf{1}_{S_T \le H} = (S_T - K)^+ - (S_T - H)^+ - (H - K)\mathbf{1}_{S_T > H}$ with an embedded cash jump drop at barrier $H$.
#
# ### Why Absolute Error?
# We evaluate performance using **Absolute Error**:
# $$\text{Absolute Error} = |G_{\text{FDMC}} - G_{\text{bench}}|$$
# Because Greeks such as $\Gamma$ or barrier sensitivities can have values close to zero (e.g. $\Gamma \approx 0.0006$), percentage relative errors can inflate misleadingly (e.g. $100\%$ error on a difference of $0.0002$). Absolute error directly reflects the physical hedging error in financial units (e.g. shares of underlying for $\Delta$, dollar vega per $1\%$ vol move for $\mathcal{V}$).
#
# ### Joint Greek Computation Architecture
# In production systems, Greeks are computed **jointly**:
# - A single vector of standard Brownian motion increments $W_T \sim \mathcal{N}(0, T)$ is generated once for all $N$ paths.
# - The spot-perturbed prices $V(S_0 + \varepsilon)$ and $V(S_0 - \varepsilon)$ are computed once and shared across both $\Delta$ and $\Gamma$.
# - The base price $V(S_0)$ is evaluated for $\Gamma$.
# - Volatility-perturbed prices $V(\sigma + \varepsilon)$ and $V(\sigma - \varepsilon)$ reuse the exact same paths $W_T$.
# - Consequently, all three Greeks are produced in only 5 pricing evaluations with zero redundant random number generation.
#
# ### Three Core Research Questions Addressed:
# 1. **Step Size Selection ($\varepsilon^*$):** Which perturbation step $\varepsilon$ minimizes absolute error across sample sizes?
# 2. **Minimum Acceptable Simulations ($N^*$):** How many simulations are required to achieve an acceptable production tolerance, and how much joint wall-clock time does that take?
# 3. **Fixed Joint CPU Budget Evaluation (250 ms, 500 ms, 1000 ms, 5000 ms):** How well does FDMC perform when computing all Greeks jointly under realistic latency SLA budgets?
#

# %% [markdown]
# ### 2. Mathematical Formulations: Central Finite Differences (CRN)
#
# Using Common Random Numbers (CRN), identical Brownian paths $W_T^{(i)}$ are recycled across the unperturbed and perturbed states:
#
# - **Delta ($\Delta = \frac{\partial V}{\partial S_0}$):**
#   $$\Delta_{\text{FDMC}} = \frac{V(S_0 + \varepsilon, \sigma) - V(S_0 - \varepsilon, \sigma)}{2\varepsilon}, \quad \text{Bias: } \mathcal{O}(\varepsilon^2), \quad \text{Roundoff: } \mathcal{O}\left(\frac{\epsilon_{\text{mach}}}{\varepsilon}\right)$$
#
# - **Gamma ($\Gamma = \frac{\partial^2 V}{\partial S_0^2}$):**
#   $$\Gamma_{\text{FDMC}} = \frac{V(S_0 + \varepsilon, \sigma) - 2V(S_0, \sigma) + V(S_0 - \varepsilon, \sigma)}{\varepsilon^2}, \quad \text{Bias: } \mathcal{O}(\varepsilon^2), \quad \text{Roundoff: } \mathcal{O}\left(\frac{\epsilon_{\text{mach}}}{\varepsilon^2}\right)$$
#
# - **Vega ($\mathcal{V} = \frac{\partial V}{\partial \sigma}$):**
#   $$\mathcal{V}_{\text{FDMC}} = \frac{V(S_0, \sigma + \varepsilon) - V(S_0, \sigma - \varepsilon)}{2\varepsilon}, \quad \text{Bias: } \mathcal{O}(\varepsilon^2)$$
#

# %%
# %load_ext autoreload
# %autoreload 2

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import Utilities as utils
importlib.reload(utils)
importlib.reload(utils.metrics)
importlib.reload(utils.profiling)
importlib.reload(utils.plotting)

pd.set_option("display.precision", 6)
pd.set_option("display.max_columns", 10)
pd.set_option("display.width", 1000)

print("Environment and Utilities loaded successfully.")


# %% [markdown]
# ---
# ## 1. Data Ingestion & Analytical Benchmark Reference
#
# We load the simulation results computed across sample sizes $N \in \{10^4, 10^5, 10^6, 10^7\}$ and perturbation steps $\varepsilon \in \{10^{-1}, 10^{-2}, \dots, 10^{-8}\}$ alongside exact analytical benchmarks.
#

# %%
benchmarks_df, fdmc_df = utils.load_simulation_results()
metrics_df = utils.compute_error_metrics(fdmc_df, benchmarks_df)

print("=== Analytical Benchmark Reference Values ===")
display(benchmarks_df.set_index("Product"))


# %% [markdown]
# ---
# ## 2. Executive Summary: Optimal $\varepsilon^*$ and Absolute Error Performance
#
# For each product and sensitivity, we determine the optimal perturbation step $\varepsilon^*$ that minimizes **Absolute Error** at maximum simulation capacity ($N = 10^7$).
#

# %%
summary_table = utils.compute_summary_table(metrics_df, n_choice=10_000_000, criterion="abs_error")
display(summary_table)


# %% [markdown]
# ---
# ## 3. Deep Dive: Vanilla European Call (Smooth Payoff)
#
# The European call has a smooth $C^1$ payoff. Below, we plot **Absolute Error vs. Sample Size $N$** with separate curves for each perturbation step $\varepsilon$.
#
# ### Diagnostic Guide for Reading the Error vs. $N$ Curves:
# 1. **Bias Saturation (Horizontal Lines):** Coarse steps (e.g. $\varepsilon = 10^{-1}$) plateau as $N$ increases because the error is dominated by the deterministic Taylor truncation bias $\mathcal{O}(\varepsilon^2)$.
# 2. **Catastrophic Cancellation (Elevated Lines):** For $\Gamma$, fine steps $\varepsilon \le 10^{-6}$ explode by orders of magnitude due to $\mathcal{O}(\epsilon_{\text{mach}} / \varepsilon^2)$ floating-point subtraction error.
# 3. **The Optimal Step (Highlighted Curve $\mathbf{(\star\ Best)}$):** The curve that tracks the theoretical $\mathcal{O}(N^{-1/2})$ convergence down to the smallest absolute error.
#

# %%
fig_call_conv_eps = utils.plot_convergence_by_eps(
    metrics_df, "Call European ATM", metric="abs_error",
    tolerance_dict={"delta": 0.001, "gamma": 0.0005, "vega": 0.05}
)
plt.show()


# %%
fig_call_sens = utils.plot_epsilon_sensitivity(metrics_df, "Call European ATM", metric="abs_error")
plt.show()


# %%
gamma_pivot = utils.generate_error_pivot(metrics_df, "Call European ATM", "gamma", metric="abs_error")
print("European Call Gamma Absolute Error (|FDMC - Benchmark|):")
display(gamma_pivot.map(lambda x: f"{x:.2e}" if (x < 0.001 or x > 1) else f"{x:.5f}"))


# %% [markdown]
# ---
# ## 4. Deep Dive: Digital European Call (Discontinuous Step Payoff)
#
# The digital call pays $\mathbf{1}_{S_T > K}$. The derivative is a Dirac delta measure $\delta(S_T - K)$.
# - At fine step sizes ($\varepsilon \le 10^{-6}$), no simulated paths fall inside the band $[K-\varepsilon, K+\varepsilon]$, causing **complete estimator collapse** (Delta and Gamma evaluate to identically zero, giving an absolute error equal to the true benchmark!).
# - At coarse step sizes ($\varepsilon = 10^{-1}, 10^{-2}$), finite differences smooth the Dirac spike, achieving stable estimates.
# - **Vega** converges with exceptional accuracy because volatility perturbation shifts the probability distribution globally rather than locally bumping around the strike.
#

# %%
fig_dig_conv_eps = utils.plot_convergence_by_eps(
    metrics_df, "Digital European ATM", metric="abs_error",
    tolerance_dict={"delta": 0.001, "gamma": 0.0005, "vega": 0.01}
)
plt.show()


# %%
dig_delta_pivot = utils.generate_error_pivot(metrics_df, "Digital European ATM", "delta", metric="value")
print("Digital Call Delta FDMC Values (Analytical Benchmark = 0.018951):")
display(dig_delta_pivot.map(lambda x: f"{x:.6f}"))


# %% [markdown]
# ---
# ## 5. Deep Dive: European Terminal Barrier Up-and-Out Call
#
# The contract pays $(S_T - K)^+ \cdot \mathbf{1}_{S_T \le H}$ at maturity.
# ### Mathematical Payoff Decomposition:
# $$\text{Payoff} = \underbrace{(S_T - K)^+ - (S_T - H)^+}_{\text{Bull Call Spread (Smooth)}} - \underbrace{(H - K) \cdot \mathbf{1}_{S_T > H}}_{\text{Short Digital Position (Discontinuous)}}$$
#
# - Differentiating at $H$ produces a Dirac delta term $-\frac{(H - K) e^{-rT} \phi(d_2(S, H))}{S \sigma \sqrt{T}}$ with jump magnitude $(H - K) = 5.25$.
# - **Vega** achieves high-precision convergence ($|e| < 0.005$) because volatility perturbation smooths the barrier boundary.
# - **Delta & Gamma** inherit digital jump sensitivity, requiring coarse $\varepsilon$ or specialized regularization.
#

# %%
fig_bar_conv_eps = utils.plot_convergence_by_eps(
    metrics_df, "Call with terminal knock-out", metric="abs_error",
    tolerance_dict={"delta": 0.001, "gamma": 0.0005, "vega": 0.05}
)
plt.show()


# %% [markdown]
# ---
# ## 6. Determining Minimum Acceptable Simulations ($N^*$) and Joint Wall-Clock Time
#
# To establish practical production guidelines, we define standard financial tolerances in absolute error:
# - **Delta ($\Delta$):** Target $|e| \le 0.0010$ (within $10$ bps of delta / $0.1$ cents per share).
# - **Gamma ($\Gamma$):** Target $|e| \le 0.0005$ (within $0.0005$ gamma units).
# - **Vega ($\mathcal{V}$):** Target $|e| \le 0.0500$ (within $0.05$ vega points).
#
# ### Critical Safeguard: Filtering Out Catastrophic Step Sizes ($\varepsilon$):
# When searching for the optimal step size $\varepsilon^*$ and determining the minimum required $N^*$, automated algorithms must NOT blindly pick catastrophic step sizes:
# 1. **Zero-Crossing Estimator Collapse:** For discontinuous options (Digital, Barrier), when $\varepsilon \le 10^{-4}$, practically zero simulated paths cross the strike/barrier. The estimator evaluates to **identically zero** ($0.000000$). Because the analytical benchmark is small (e.g. $-0.00024$), a naive error minimization would spuriously declare $\varepsilon = 10^{-6}$ "optimal" simply because $|0 - (-0.00024)|$ is small!
# 2. **Catastrophic Roundoff / Variance Explosion:** For Gamma, dividing by $\varepsilon^2$ ($10^{-8}$ to $10^{-12}$) leads to catastrophic floating-point cancellation and massive variance explosion for $\varepsilon \le 10^{-3}$.
# Our utility `find_min_acceptable_N(..., exclude_catastrophic=True)` and `evaluate_fixed_cpu_budget(..., exclude_catastrophic=True)` enforces valid non-catastrophic step domains ($
# arepsilon \ge 0.01$ for vanilla Gamma, and $
# arepsilon \ge 0.1$ for discontinuous Gamma/Delta), guaranteeing that only physically meaningful estimators are selected.
#

# %%
# Benchmark joint runtime across all Greeks combined
joint_timing = utils.benchmark_fdmc_execution_time(
    n_sims=[10_000, 100_000, 250_000, 1_000_000, 2_000_000], num_repeats=3
)

min_n_table = utils.find_min_acceptable_N(metrics_df, criterion="abs_error", timing_df=joint_timing)
display(min_n_table)


# %% [markdown]
# ---
# ## 7. Fixed Joint CPU Budget Evaluation: Computing All Greeks Jointly (250 ms – 5000 ms)
#
# In financial production risk engines, computing sensitivities one-by-one is inefficient and never done in practice. The engine generates Brownian paths $W_T$ once and simultaneously evaluates the entire Greek surface ($\Delta, \Gamma, \mathcal{V}$).
#
# Here, we evaluate four realistic institutional latency SLA budgets for computing **all Greeks jointly**:
# 1. **250 ms Budget (Quarter-Second SLA):** High-frequency intraday limit monitoring and real-time interactive trading desk pricing.
# 2. **500 ms Budget (Half-Second SLA):** Standard market-maker quoting and client RFQ response.
# 3. **1000 ms Budget (1-Second SLA):** Portfolio rebalancing, trade validation, and scenario sensitivity sweeps.
# 4. **5000 ms Budget (5-Second SLA):** End-of-day batch processing, comprehensive stress testing, and official regulatory reporting.
#
# Below, we report the measured joint execution times, the achievable simulation sample size $N$, and the resulting absolute error across these four operational budgets.
#

# %%
print("=== Joint Execution Time Across All Greeks Combined (Delta, Gamma, Vega) ===")
display(joint_timing)


# %%
# Evaluate achievable N and absolute error under joint budgets (250 ms, 500 ms, 1000 ms, 5000 ms)
budget_df = utils.evaluate_fixed_cpu_budget(
    metrics_df, joint_timing, budgets_ms=[250.0, 500.0, 1000.0, 5000.0], criterion="abs_error"
)
display(budget_df)


# %%
fig_budget_bars = utils.plot_budget_accuracy_bars(budget_df, "Call European ATM")
plt.show()

fig_budget_bars_bar = utils.plot_budget_accuracy_bars(budget_df, "Call with terminal knock-out")
plt.show()


# %%
pareto_df = utils.construct_pareto_dataset(metrics_df, joint_timing, criterion="abs_error")
fig_pareto = utils.plot_pareto_efficiency(pareto_df, "Call European ATM", criterion="abs_error")
plt.show()


# %% [markdown]
# ---
# ## 8. Practitioner Summary & FDMC Operational Guidelines
#
# ### 1. Optimal Step Size Selection ($\varepsilon^*$):
# - **Smooth Payoffs (Vanilla European Call):**
#   - Delta and Vega: $\varepsilon^* = 10^{-2}$ provides optimal bias-variance balance with absolute errors $< 0.0001$.
#   - Gamma: Must operate within $\varepsilon^* \in [10^{-2}, 10^{-1}]$; finer steps $\varepsilon \le 10^{-4}$ trigger catastrophic floating-point cancellation ($> 900\%$ error).
# - **Discontinuous Payoffs (Digital & Barrier Call):**
#   - Delta and Gamma: Must employ coarse perturbation $\varepsilon^* \approx 10^{-1}$ to avoid zero-crossing / Dirac delta collapse. Catastrophic steps $\varepsilon \le 10^{-3}$ collapse to zero and must be excluded.
#   - Vega: $\varepsilon^* = 10^{-2}$ to $10^{-3}$ converges reliably because volatility shifts the terminal distribution smoothly across the discontinuity.
#
# ### 2. Minimum Acceptable Sample Size ($N^*$) and Joint Latency:
# - **European Call:**
#   - Delta ($|e| \le 0.0010$): Satisfied at **$N = 100,000$** paths (joint time $\approx 7.2$ ms).
#   - Gamma ($|e| \le 0.0005$): Satisfied at **$N = 100,000$** paths (joint time $\approx 7.2$ ms).
#   - Vega ($|e| \le 0.0500$): Satisfied at **$N = 250,000$** paths (joint time $\approx 18.1$ ms).
#   - **Joint SLA Conclusion:** Because all Greeks share pre-generated Brownian paths $W_T$ and reused pricing runs, running **$N = 250,000$** paths delivers the entire Greek profile $(\Delta, \Gamma, \mathcal{V})$ within acceptable financial tolerances in just **$\approx 18.1$ ms** of total wall-clock time!
# - **Digital European Call:**
#   - Delta: Satisfied at $N=10,000$ with $\varepsilon^* = 0.01$ (error $7.4 \times 10^{-5}$) in $\approx 1.1$ ms.
#   - Gamma: With catastrophic zero-collapsed $\varepsilon$ excluded, requires coarse $\varepsilon^* = 0.10$ and reaches tolerance at $N = 2,000,000$ (error $9.25 \times 10^{-7}$) in $\approx 220$ ms.
#   - Vega: Satisfied at $N=10,000$ with $\varepsilon^* = 0.001$ (error $0.0018$) in $\approx 1.1$ ms.
# - **Barrier Up-and-Out Call:**
#   - Delta: Satisfied at $N=10,000$ with $\varepsilon^* = 0.10$ (error $0.00015$) in $\approx 0.97$ ms.
#   - Gamma: With catastrophic zero-collapsed $\varepsilon$ excluded, optimal $\varepsilon^* = 0.10$; at $N=10^7$ absolute error is $0.00146$, which exceeds the target tolerance of $0.0005$.
#   - Vega: Satisfied at $N=10,000$ (error $0.036$) in $\approx 0.97$ ms.
#
# ### 3. Performance Under Joint CPU Budgets (250 ms, 500 ms, 1000 ms, 5000 ms):
# - **250 ms Budget (Real-Time Intraday SLA):**
#   - Achieves **$N \approx 3.46 \times 10^6$** paths for all Greeks jointly.
#   - European Call: $|e|_\Delta \approx 0.00029$, $|e|_\Gamma \approx 0.00011$, $|e|_\mathcal{V} \approx 0.0174$.
#   - Easily surpasses all target tolerances with a $14\times$ safety margin over the minimum required sample size ($N=250k$).
# - **500 ms Budget (Standard Quoting SLA):**
#   - Achieves **$N \approx 6.92 \times 10^6$** paths jointly.
#   - European Call: $|e|_\Delta \approx 0.00024$, $|e|_\Gamma \approx 0.00002$, $|e|_\mathcal{V} \approx 0.0145$.
# - **1000 ms Budget (1-Second Scenario & Rehedging SLA):**
#   - Achieves **$N \approx 13.84 \times 10^6$** paths jointly (exceeds our maximum grid of $N=10^7$).
#   - European Call: $|e|_\Delta \approx 0.00021$, $|e|_\Gamma \approx 0.000007$, $|e|_\mathcal{V} \approx 0.0132$.
# - **5000 ms Budget (5-Second Batch & Stress-Testing SLA):**
#   - Achieves **$N \approx 69.21 \times 10^6$** paths (deep Monte Carlo capacity).
#   - Absolute errors hit the fundamental precision floor of finite difference discretization and double-precision arithmetic.
#
# ### 4. Structural Limitations & Methodological Outlook:
# - For vanilla contracts, joint FDMC with CRN is fast, robust, and computationally inexpensive.
# - For discontinuous payoffs (Digital, Barrier), increasing the joint CPU budget from 250 ms to 5000 ms fails to resolve the structural deficiency in Gamma (absolute error plateaus around $\approx 0.0015$ due to the Dirac jump drop).
# - This demonstrates the fundamental boundary of Finite Differences and justifies investigating **Likelihood Ratio Method (LRM)** and **Malliavin Calculus** in subsequent notebooks.
#
