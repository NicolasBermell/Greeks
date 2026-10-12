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
# # European Greeks: Finite Differences (FDMC) vs. Malliavin vs. Fournié Localization
# ## Bias, Per-Path Variance, Cost per Path and Efficiency for a Call, a Digital and a Call with Terminal Knock-Out
#
# ### Overview
# **Finite differences (FDMC)** re-price with bumped inputs and take difference quotients (central differences, shared random numbers, $\varepsilon = 0.01$ unless stated). They have three costs:
# 1. **Truncation bias** of order $\varepsilon^2$.
# 2. **Variance that grows as $\varepsilon \to 0$** when the payoff is not smooth: for Gamma the per-path variance grows like $1/\varepsilon$ for a kink (call) and like $1/\varepsilon^3$ for a jump (digital, knock-out).
# 3. **Five payoff evaluations per path** for $(\Delta, \Gamma, \mathcal{V})$: $S_0 \pm \varepsilon$, $S_0$, $\sigma \pm \varepsilon$.
#
# **Malliavin** estimators move the derivative from the payoff onto a weight in $W_T$ (integration by parts). They are unbiased, need no bump, and use one payoff evaluation per path for all three Greeks. For smooth, unbounded payoffs their weights inflate the variance compared with the pathwise derivative.
#
# **Fournié et al. (1999) localization** uses the Malliavin weight only near the singularities, through a cutoff $\psi$, and the pathwise derivative elsewhere.
#
# All comparisons of accuracy are based on $R = 100$ independent runs per estimator (one `np.random.default_rng(seed)` per run) at $N \in \{10^4, 10^5, 10^6\}$, with the per-path variance stored for every run. The single-run results up to $N = 10^8$ are shown for reference.

# %% [markdown]
# ---
# ## 1. Mathematical Foundations: Skorokhod Duality & Fournié Localization
#
# ### 1.1 Black-Scholes Framework & Standard Malliavin Weights
# Under the risk-neutral measure $\mathbb{Q}$, the terminal spot price at maturity $T$ is:
# $$S_T = S_0 \exp\left( \left(r - q - \frac{1}{2}\sigma^2\right)T + \sigma W_T \right), \quad W_T \sim \mathcal{N}(0, T)$$
#
# Discounted option price: $V(S_0, \sigma) = e^{-rT} \mathbb{E}[\Phi(S_T)]$. Skorokhod duality yields:
# $$\mathbb{E}[\langle D F, u \rangle_{L^2[0, T]}] = \mathbb{E}[F \cdot \delta(u)]$$
#
# Standard Malliavin weights:
# $$\pi^\Delta = \frac{W_T}{S_0 \sigma T} = \frac{Z}{S_0 \sigma \sqrt{T}}$$
# $$\pi^\Gamma = \frac{W_T^2 - \sigma T W_T - T}{S_0^2 \sigma^2 T^2} = \frac{Z^2 - \sigma \sqrt{T} Z - 1}{S_0^2 \sigma^2 T}$$
# $$\pi^{\mathcal{V}} = \frac{W_T^2 - \sigma T W_T - T}{\sigma T} = S_0^2 \sigma T \cdot \pi^\Gamma$$
# where $Z = W_T / \sqrt{T} \sim \mathcal{N}(0, 1)$.
#
# ### 1.2 Fournié et al. (1999) Localized Malliavin Formulation (Theorem 2.3)
# Let $\psi: \mathbb{R} \to [0, 1]$ be a smooth $C^1$ cutoff function centered at the strike $K$ with half-width $\delta$:
# $$\psi(S_T) \approx 1 \text{ for } |S_T - K| < \delta, \qquad \psi(S_T) \approx 0 \text{ for } |S_T - K| \ge \delta$$
#
# Applying the product rule and Skorokhod integration by parts only to $\Phi(S_T) \psi(S_T)$ while using the pathwise derivative for $\Phi(S_T)(1 - \psi(S_T))$ yields the exact localized estimators:
#
# 1. **Localized Delta ($\Delta_{\text{loc}}$)**:
#    $$\Delta_{\text{loc}} = e^{-rT} \mathbb{E}\left[ \Phi(S_T) \left(\psi(S_T) \pi^\Delta - \psi'(S_T) \frac{S_T}{S_0}\right) + \Phi'(S_T) (1 - \psi(S_T)) \frac{S_T}{S_0} \right]$$
#
# 2. **Localized Vega ($\mathcal{V}_{\text{loc}}$)**:
#    $$\mathcal{V}_{\text{loc}} = e^{-rT} \mathbb{E}\left[ \Phi(S_T) \left(\psi(S_T) \pi^{\mathcal{V}} - \psi'(S_T) S_T (W_T - \sigma T)\right) + \Phi'(S_T) (1 - \psi(S_T)) S_T (W_T - \sigma T) \right]$$
#
# 3. **Localized Gamma ($\Gamma_{\text{loc}}$)**, general formula for any payoff:
#    $$\Gamma_{\text{loc}} = e^{-rT} \mathbb{E}\left[ \Phi\,\psi\,\pi^\Gamma + \Big(\Phi''(1 - \psi) - 2\Phi'\psi' - \Phi\psi''\Big)\left(\frac{S_T}{S_0}\right)^2 \right]$$
#    $\Phi'$ and $\Phi''$ are the classical (almost-everywhere) derivatives. The Dirac masses of the distributional derivatives sit at the singularities, where $\psi = 1$ and $\psi' = 0$, so they drop out. For the call with terminal knock-out the cutoff must satisfy this at both $K$ and $H$: $\psi = 1 - (1-\psi_K)(1-\psi_H)$ does, $\psi_K + \psi_H$ does not for the Gaussian cutoff (Section 5).
#    *(The earlier call-only shortcut $\Gamma_{\text{call}} = (K/S_0)\,\Delta^{\text{loc}}_{\text{digital}}$ is kept as a separate run, "Malliavin Fournié (Γ shortcut)", for comparison.)*
#
# ### 1.3 Supported Smoothing Functions $\psi(S_T)$
# With normalized distance $u = |S_T - K| / \delta$:
# - **Gaussian (Default & Optimal)**: $\psi(S_T) = \exp(-0.5 u^2)$, $\quad \psi'(S_T) = -\frac{S_T - K}{\delta^2} \psi(S_T)$
# - **Compact Cubic Hermite**: $\psi(S_T) = 1 - 3u^2 + 2u^3$ for $u \le 1$, $0$ otherwise.
# - **Compact Cosine**: $\psi(S_T) = \frac{1}{2}(1 + \cos(\pi u))$ for $u \le 1$, $0$ otherwise.
# - **Compact Quintic**: $\psi(S_T) = 1 - 10u^3 + 15u^4 - 6u^5$ for $u \le 1$, $0$ otherwise.
# - **Rational Cauchy**: $\psi(S_T) = \frac{1}{1 + u^2}$, $\quad \psi'(S_T) = -\frac{2(S_T - K)}{\delta^2 (1 + u^2)^2}$
#
# ### 1.4 Option A: Decoupled Bandwidths per Greek ($\delta_\Delta \ne \delta_\Gamma \ne \delta_\mathcal{V}$)
# A critical insight established in our dedicated gap analysis is that **the optimal Fournié smoothing bandwidth depends directly on the singularity order of each Greek**:
# - **Delta ($\Delta$) and Vega ($\mathcal{V}$): Continuous Kink ($C^0$)**
#   The payoff $g(S_T) = (S_T - K)^+$ is continuous. As $S_T \to K$, $g(S_T) \to 0$, which naturally suppresses the boundary derivative spike $-\psi'(S_T) \sim 1/\delta$. Therefore, Delta and Vega achieve minimal variance with a narrower bandwidth that maximizes the domain of the pathwise derivative:
#   $$\delta^*_\Delta = \delta^*_\mathcal{V} \approx 0.20 \times S_0 \sigma \sqrt{T} = 4.0 \quad (T = 1.0)$$
# - **Gamma ($\Gamma$): Discontinuous Step ($C^{-1}$)**
#   Gamma differentiates the digital step indicator $g'(S_T) = \mathbf{1}_{S_T > K}$, which jumps by $+1$ at $K$. The boundary term $-\psi'(S_T) \frac{S_T}{S_0}$ has no $(S_T - K)$ dampener and blows up as $\mathcal{O}(1/\delta)$ if $\delta$ is too small. Hence, Gamma strictly requires a wider bandwidth:
#   $$\delta^*_\Gamma \approx 0.50 \times S_0 \sigma \sqrt{T} = 10.0 \quad (T = 1.0)$$
#
# **Option A vs. Option B:**
# Rather than one compromise bandwidth for all Greeks (Option B, e.g. $\delta = 8.0$), **Option A uses one bandwidth per Greek** in the same Monte Carlo pass. Section 7 measures the difference.

# %%
# %load_ext autoreload
# %autoreload 2

import importlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import Utilities as utils
from products import (
    example_european_call_ATM,
    example_european_digital_ATM,
    example_european_call_terminal_knockout,
)

pd.set_option("display.precision", 6)
pd.set_option("display.max_columns", 16)
pd.set_option("display.width", 1000)

print("Environment and Utilities loaded successfully.")

# %% [markdown]
# ---
# ## 2. Data
#
# 1. **Analytical benchmarks**: Black-Scholes closed forms (QuantLib for the digital). The call with terminal knock-out is two calls and a digital: $(S_T-K)^+ - (S_T-H)^+ - (H-K)\mathbf{1}_{S_T>H}$.
# 2. **Single-run sweeps** (`FDMC_results.csv`, `malliavin_results.csv`, `malliavin_fournier_results.csv`): one seed, $N$ from $10^4$ to $10^8$, with the per-path variance and standard error of every Greek.
# 3. **Replications** (`replications_summary.csv`, `efficiency.csv`): $R = 100$ seeds at $N \in \{10^4, 10^5, 10^6\}$, with bias, RMSE, coverage, per-path variance and time per path.
#
# Fournié settings: Gaussian cutoff, bandwidths $\delta_\Delta = \delta_\mathcal{V} = 4$, $\delta_\Gamma = 10$ for the call and the digital, and $\delta = (H-K)/2 = 2.625$ for the knock-out with $\psi = 1 - (1-\psi_K)(1-\psi_H)$.

# %%
benchmarks_df, fdmc_df, malliavin_df, malliavin_fournier_df = utils.load_all_simulation_results()
fdmc_metrics_df = utils.compute_error_metrics(fdmc_df, benchmarks_df)
mal_metrics_df = utils.compute_error_metrics(malliavin_df, benchmarks_df)
four_metrics_df = utils.compute_error_metrics(malliavin_fournier_df, benchmarks_df)

print("=== Exact Analytical Benchmark Reference Values ===")
display(benchmarks_df.set_index("Product"))

# Independent replications (run_replications.py): R seeds per (estimator, product, N)
rep_summary = pd.read_csv("./results/replications_summary.csv")
efficiency_df = pd.read_csv("./results/efficiency.csv")

# %% [markdown]
# ### 2.1 Single Run at $N = 10^8$
#
# Absolute errors of one run at $N = 10^8$, FDMC at the fixed production step $\varepsilon = 0.01$ (in practice the analytical value is unknown, so $\varepsilon$ cannot be tuned per Greek after the fact). A single run's error is one draw of a random variable whose size is about one standard error, so ratios of these errors between methods are noisy. Use the replications and the per-path variances (Section 6) for comparisons.

# %%
comp_summary = utils.compute_comparative_summary_table(
    fdmc_metrics_df, mal_metrics_df, n_choice=100_000_000,
    criterion="abs_error", fournier_metrics_df=four_metrics_df
)
display(comp_summary)

# %% [markdown]
# ---
# ## 3. Deep Dive: Vanilla European Call (Smooth Payoff)
#
# ### The Variance Challenge Solved:
# For smooth payoffs, standard Malliavin integration by parts uses quadratic polynomial weights in $W_T$, which introduces higher sample variance than the pathwise derivative.
# The error plots below come from $R = 100$ independent runs (one `np.random.default_rng(seed)` per run) at $N \in \{10^4, 10^5, 10^6\}$:
# - **Solid line**: RMSE against the analytical value over the $R$ runs.
# - **Band**: 10th–90th percentile of $|\hat\theta - \theta|$ across runs.
# - **Dashed line**: extrapolation $\sqrt{\text{bias}^2 + \text{Var}/N}$ to $N = 10^8$, using the measured per-path variance (the bias term is kept only when significant, e.g. FDMC truncation bias).
# - **FDMC** uses the fixed production step $\varepsilon = 0.01$; the Fournié Gamma uses the general localized formula, with the old call-only shortcut shown dotted.

# %%
fig_call_conv = utils.plot_replication_error_vs_N(rep_summary, "Call European ATM")
plt.show()

# %%
fig_call_bars = utils.plot_comparative_greeks_bars(
    fdmc_metrics_df, mal_metrics_df, "Call European ATM",
    n_choice=100_000_000, fournier_metrics_df=four_metrics_df
)
plt.show()

# %% [markdown]
# ---
# ## 4. Deep Dive: Digital European Call (Discontinuous Step Payoff)
#
# ### The Structural Failure of Finite Differences on Indicator Payoffs:
# The digital call pays $\mathbf{1}_{S_T > K}$. The derivative is a Dirac delta measure $\delta(S_T - K)$.
# - In FDMC, central differences divide by $\varepsilon$ or $\varepsilon^2$ across a step discontinuity. If $\varepsilon$ is too small, no paths cross $[K-\varepsilon, K+\varepsilon]$, leading to complete estimator collapse. If $\varepsilon$ is large, bias $\mathcal{O}(\varepsilon^2)$ is unacceptably high.
# - **Malliavin calculus transfers the derivative to smooth Gaussian weights**, so its variance does not depend on any step size.
# - At $\varepsilon = 0.01$ the per-path variance of the FDMC digital Gamma is about $3.6 \times 10^{4}$, against about $5 \times 10^{-6}$ for standard Malliavin (`efficiency.csv`).

# %%
fig_dig_conv = utils.plot_replication_error_vs_N(rep_summary, "Digital European ATM")
plt.show()

# %%
fig_dig_bars = utils.plot_comparative_greeks_bars(
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
display(dig_mal_gamma)

# %% [markdown]
# ---
# ## 5. Deep Dive: Call with Terminal Knock-Out
#
# Payoff $(S_T - K)^+ \mathbf{1}_{S_T \le H}$, checked at maturity only (not a path-dependent barrier).
#
# ### Jump Discontinuity at $H = 105.25$:
# Differentiating across the barrier creates a negative Dirac jump $-(H - K) \delta(S_T - H)$.
# - FDMC divides a jump of size $H - K$ by $\varepsilon^2$, so its Gamma has a per-path variance of about $10^{6}$ at $\varepsilon = 0.01$.
# - Standard Malliavin has a per-path Gamma variance of about $5 \times 10^{-6}$: about $90\%$ of paths end outside $[K, H]$, where the payoff and the sample are zero.
# - Fournié with the unbiased cutoff is unbiased but has a larger variance than standard Malliavin for Delta and Gamma. Fournié with $\psi_K + \psi_H$ (dotted line) is biased: its error stops decreasing with $N$.

# %%
fig_bar_conv = utils.plot_replication_error_vs_N(rep_summary, "Call with terminal knock-out")
plt.show()

# %%
fig_bar_bars = utils.plot_comparative_greeks_bars(
    fdmc_metrics_df, mal_metrics_df, "Call with terminal knock-out",
    n_choice=100_000_000, fournier_metrics_df=four_metrics_df
)
plt.show()

# %% [markdown]
# ---
# ## 6. Per-Path Variance, Time per Path and Efficiency
#
# Every estimator is a mean of i.i.d. per-path quantities, so for a tolerance $\text{tol}$ at confidence $z$ the required number of paths is
# $$N^* = \frac{z^2\,\text{Var}}{\text{tol}^2}, \qquad \frac{N^*_{\text{Malliavin}}}{N^*_{\text{FDMC}}} = \frac{\text{Var}_{\text{Malliavin}}}{\text{Var}_{\text{FDMC}}}.$$
# All three Greeks are timed together per method (FDMC: 5 payoff evaluations per path; standard Malliavin: 1; Fournié: 1 plus the cutoff evaluations). Efficiency is $\text{Var} \times \text{time per path}$, the CPU time needed per unit of inverse squared error (lower is better).
# - `n_ratio_vs_FDMC`: paths needed relative to FDMC ($\varepsilon = 0.01$) for the same SE.
# - `efficiency_ratio_vs_FDMC`: CPU time relative to FDMC for the same SE.
# - `n_star`, `time_star_s`: paths and seconds for a 95% half-width of 1% of the analytical value. They ignore bias; `bias_exceeds_tol` flags FDMC rows where the truncation bias alone exceeds that tolerance.

# %%
display(efficiency_df)

# %% [markdown]
# ---
# ## 7. Fournié Localized Malliavin: Empirical Variance Reduction & Decoupled Bandwidths (Option A)
#
# ### Direct Variance Measurement: Standard vs. Coupled Fournié vs. Option A Decoupled
# We execute 100 independent Monte Carlo runs of $N = 10,000$ simulations to directly measure the sample variance of the estimators:
# $$\text{Var}(\hat{\theta}) = \frac{1}{M-1} \sum_{m=1}^M (\hat{\theta}_m - \bar{\theta})^2$$
#
# We compare:
# 1. **Standard Malliavin**: Pure Skorokhod duality with polynomial weights.
# 2. **Coupled Fournié ($\delta = 8.0$)**: Uniform compromise bandwidth.
# 3. **Option A Decoupled Fournié**: $\delta^*_\Delta = 4.0$, $\delta^*_\Gamma = 10.0$, $\delta^*_\mathcal{V} = 4.0$ tailored to each Greek's singularity order.

# %%
from Methods.Malliavin.malliavin_european_standard import malliavin_greeks_european_call, generate_paths
from Methods.Malliavin.malliavin_european_fournier import malliavin_fournier_greeks_european_call

params = example_european_call_ATM
n_runs = 100
n_sims = 10000

std_deltas, std_gammas, std_vegas = [], [], []
coup_deltas, coup_gammas, coup_vegas = [], [], []
decoup_deltas, decoup_gammas, decoup_vegas = [], [], []

decoupled_deltas = {"delta": 4.0, "gamma": 10.0, "vega": 4.0}

for s in range(n_runs):
    paths = generate_paths(params["T"], n_sims, seed=s)
    r_std = malliavin_greeks_european_call(params, N=n_sims, paths=paths)
    r_coup = malliavin_fournier_greeks_european_call(params, N=n_sims, paths=paths, cutoff_type="gaussian", bandwidth=8.0)
    r_decoup = malliavin_fournier_greeks_european_call(params, N=n_sims, paths=paths, cutoff_type="gaussian", bandwidth=decoupled_deltas)
    
    std_deltas.append(r_std["delta"])
    std_gammas.append(r_std["gamma"])
    std_vegas.append(r_std["vega"])
    
    coup_deltas.append(r_coup["delta"])
    coup_gammas.append(r_coup["gamma"])
    coup_vegas.append(r_coup["vega"])
    
    decoup_deltas.append(r_decoup["delta"])
    decoup_gammas.append(r_decoup["gamma"])
    decoup_vegas.append(r_decoup["vega"])

var_std_d, var_std_g, var_std_v = float(np.var(std_deltas)), float(np.var(std_gammas)), float(np.var(std_vegas))
var_coup_d, var_coup_g, var_coup_v = float(np.var(coup_deltas)), float(np.var(coup_gammas)), float(np.var(coup_vegas))
var_decoup_d, var_decoup_g, var_decoup_v = float(np.var(decoup_deltas)), float(np.var(decoup_gammas)), float(np.var(decoup_vegas))

var_df = pd.DataFrame([
    {
        "Greek": "Delta (Δ)",
        "Std Malliavin Var": var_std_d,
        "Coupled Fournié (δ=8) Var": var_coup_d,
        "Option A Decoupled Var": var_decoup_d,
        "Redux vs Std (Coupled)": f"{var_std_d/var_coup_d:.2f}x",
        "Redux vs Std (Option A)": f"{var_std_d/var_decoup_d:.2f}x",
        "Option A Gain vs Coupled": f"+{(var_coup_d/var_decoup_d - 1)*100:.1f}%",
    },
    {
        "Greek": "Gamma (Γ)",
        "Std Malliavin Var": var_std_g,
        "Coupled Fournié (δ=8) Var": var_coup_g,
        "Option A Decoupled Var": var_decoup_g,
        "Redux vs Std (Coupled)": f"{var_std_g/var_coup_g:.2f}x",
        "Redux vs Std (Option A)": f"{var_std_g/var_decoup_g:.2f}x",
        "Option A Gain vs Coupled": f"+{(var_coup_g/var_decoup_g - 1)*100:.1f}%",
    },
    {
        "Greek": "Vega (V)",
        "Std Malliavin Var": var_std_v,
        "Coupled Fournié (δ=8) Var": var_coup_v,
        "Option A Decoupled Var": var_decoup_v,
        "Redux vs Std (Coupled)": f"{var_std_v/var_coup_v:.2f}x",
        "Redux vs Std (Option A)": f"{var_std_v/var_decoup_v:.2f}x",
        "Option A Gain vs Coupled": f"+{(var_coup_v/var_decoup_v - 1)*100:.1f}%",
    },
])
print("=== Empirical Variance Reduction: Standard vs. Coupled Fournié vs. Option A Decoupled ===")
display(var_df)

# %% [markdown]
# ---
# ## 8. Fixed Time Budgets
#
# Given a CPU budget $B$ and the measured time per path $t$ (all three Greeks together), an estimator gets $N = B / t$ paths and its expected error is
# $$\text{RMSE}(B) = \sqrt{\text{bias}^2 + \text{Var}\cdot t / B}.$$
# The bias term is kept only when it is significant in the replications. The table gives the expected RMSE for budgets of 250 ms, 1 s and 5 s. It replaces the earlier budget and Pareto charts, which were built from the error of a single run at each $N$.

# %%
budget_df = utils.expected_error_under_budget(efficiency_df, budgets_ms=[250.0, 1000.0, 5000.0])
budget_df = budget_df[~budget_df["Estimator"].isin(["FDMC (ε=0.1)", "FDMC (ε=0.001)"])]
display(budget_df.pivot_table(index=["Product", "Greek", "Estimator"], columns="budget_ms", values="expected_rmse"))

# %% [markdown]
# ---
# ## 9. Wall-Clock Time vs. $N$
#
# Total time for all three Greeks at several $N$ (median of 3 repeats). FDMC evaluates the payoff 5 times per path, standard Malliavin once, and Fournié once plus the cutoff evaluations, so FDMC and Fournié cost about the same per path and standard Malliavin less (see `time_per_path_ns` in Section 6).

# %%
joint_timing_comp = utils.benchmark_joint_fdmc_vs_malliavin(
    n_sims=[10_000, 100_000, 250_000, 1_000_000, 2_000_000], num_repeats=3,
    include_fournier=True,
)
display(joint_timing_comp)

# %%
fig_runtime = utils.plot_runtime_vs_simulations(joint_timing_comp)
plt.show()

# %% [markdown]
# ---
# ## 10. Summary
#
# All statements below come from the per-path variances, biases and timings in Section 6 ($R = 100$ runs, $N = 10^6$, FDMC at $\varepsilon = 0.01$).
#
# **Call (smooth, unbounded payoff)**
# - Delta and Vega: FDMC is effectively the pathwise derivative and has the lowest variance. Standard Malliavin needs about $6\times$ (Delta) and $13\times$ (Vega) more paths; per path it is about $2.7\times$ cheaper, so about $2\times$ and $5\times$ more CPU time. Fournié is close to FDMC.
# - Gamma: FDMC's per-path variance grows like $1/\varepsilon$. Standard Malliavin needs about $70\times$ fewer paths than FDMC, Fournié (general formula) about $7\times$ fewer again; the call-only shortcut a further $3.5\times$ fewer.
#
# **Digital (jump)**
# - FDMC Gamma is unusable at $\varepsilon = 0.01$ (per-path variance $\sim 10^4$, growing like $1/\varepsilon^3$); Delta needs about $10^3\times$ more paths than standard Malliavin.
# - Standard Malliavin has the lowest variance for Delta and Gamma; Fournié has the lowest for Vega.
#
# **Call with terminal knock-out**
# - Standard Malliavin is the best choice for all three Greeks once cost per path is included.
# - Fournié needs a cutoff with $\psi = 1$ and $\psi' = 0$ at both $K$ and $H$. With $\psi_K + \psi_H$ and a Gaussian cutoff the estimator is biased (Delta and Gamma even have the wrong sign).
#
# **Bias of FDMC**: at $\varepsilon = 0.01$ the truncation bias is below the noise of $R = 100$ runs at $N = 10^6$ except for the knock-out Vega; at $\varepsilon = 0.1$ it is large for Vega. The main problem of FDMC on non-smooth payoffs is variance, not bias.
