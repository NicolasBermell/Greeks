# Benchmark of Option Greeks: Malliavin Calculus vs. Standard Methods

A comprehensive benchmark and quantitative report analyzing numerical methods for computing financial Greeks ($\Delta$, $\Gamma$, $\mathcal{V}$) across vanilla, path-dependent, and early-exercise contracts under Black-Scholes dynamics.

The core focus of this project is to evaluate the stability, computational efficiency, and variance profile of **Malliavin Calculus** (both Global Skorokhod weights and Fournié et al. localized estimators) against traditional industrial approaches: Finite Difference Monte Carlo (FDMC), Pathwise differentiation, and 1D Finite Difference PDE solvers.

---

## 1. Project Overview & Scope

Computing sensitivities for non-smooth or early-exercise payoffs poses severe numerical challenges:
- **Discontinuous Payoffs (e.g., Digitals):** Pathwise derivatives degenerate into Dirac deltas, while finite differences suffer from variance explosion near the strike.
- **Path-Dependent / Early-Exercise (e.g., Asians, Bermudans, Americans):** Bumping parameters requires re-evaluating backward regressions or fine-mesh projections, introducing structural noise.

Malliavin Calculus bypasses payoff differentiation by transferring derivatives onto the underlying probability space via stochastic integration by parts (IBP), yielding an expectation of the unperturbed payoff weighted by a Skorokhod integral.

### Supported Products & Method Matrix

| Product | Benchmark Reference ($G_{\text{ref}}$) | Standard MC / Deterministic | Malliavin Implementations |
| :--- | :--- | :--- | :--- |
| **European Call** | QuantLib `AnalyticEuropeanEngine` | Pathwise MC, FDMC (CRN) | Global Skorokhod weight, Fournié $C^2$ localization |
| **Digital Call** | QuantLib `AnalyticEuropeanEngine` (CashOrNothing) | FDMC (CRN), Likelihood Ratio (LRM) | Global weight, Fournié $C^1$ step-regularization |
| **Barrier Option** (Knock-Out) | QuantLib `AnalyticBarrierEngine` / 1D PDE | FDMC (fine grid), Brownian Bridge MC | Malliavin boundary/exit-time bridge |
| **Asian Option** (Arithmetic) | QuantLib `TurnbullWakemanAsianEngine` / Brute-Force MC | Pathwise MC, FDMC (CRN) | Functional Malliavin ($u_t$ path-dependent process) |
| **Bermudan Option** | QuantLib `FdBlackScholesVanillaEngine` (1D PDE) | LSMC + FDMC (bumping backward regressions) | LSMC + Local Malliavin weights at stopping time $\tau$ |
| **American Option** ($q > 0$) | QuantLib `FdBlackScholesVanillaEngine` (PSOR 1D PDE) | LSMC + FDMC | LSMC + Local Malliavin at $\tau$ (+ European control variate) |

---

## 2. Core Malliavin Formulations (Black-Scholes 1D)

Under $dS_t = r S_t dt + \sigma S_t dW_t$, the price is given by $V = e^{-rT} \mathbb{E}[f(S_T)]$. Using Malliavin IBP with directional process $u_t \equiv 1$:

- **Delta ($\Delta = \frac{\partial V}{\partial S_0}$):**
  $$\mathcal{W}_\Delta = \frac{W_T}{S_0 \sigma T} \implies \Delta = e^{-rT} \mathbb{E}\left[ f(S_T) \cdot \frac{W_T}{S_0 \sigma T} \right]$$

- **Gamma ($\Gamma = \frac{\partial^2 V}{\partial S_0^2}$):**
  $$\mathcal{W}_\Gamma = \frac{W_T^2 - T - \sigma T W_T}{(S_0 \sigma T)^2} \implies \Gamma = e^{-rT} \mathbb{E}\left[ f(S_T) \cdot \mathcal{W}_\Gamma \right]$$

- **Vega ($\mathcal{V} = \frac{\partial V}{\partial \sigma}$):**
  $$\mathcal{W}_{\mathcal{V}} = \frac{W_T^2 - T - \sigma T W_T}{\sigma T} \implies \mathcal{V} = e^{-rT} \mathbb{E}\left[ f(S_T) \cdot \mathcal{W}_{\mathcal{V}} \right]$$

- **Fournié et al. (1999) Localization:**
  To reduce the $\mathcal{O}(T^{-2})$ variance of $\mathcal{W}_\Gamma$ for short expiries, the payoff is decomposed into a smooth component $G_\delta$ and a localized residual $F_\delta$ with support on $[K-\delta, K+\delta]$:
  $$\Gamma = e^{-rT} \mathbb{E}\left[ G''_\delta(S_T) \left(\frac{S_T}{S_0}\right)^2 \right] + e^{-rT} \mathbb{E}\left[ F_\delta(S_T) \cdot \mathcal{W}_\Gamma \right]$$

- **American / Bermudan LSMC Integration:**
  Given optimal exercise dates $\tau^{(i)}$ identified via backward induction, sensitivities are evaluated forward in a single pass without re-fitting regression coefficients:
  $$\Delta^{\text{Amer}} \approx \frac{1}{N} \sum_{i=1}^N e^{-r\tau^{(i)}} f\big(S_{\tau^{(i)}}^{(i)}\big) \cdot \frac{W_{\tau^{(i)}}^{(i)}}{S_0 \sigma \tau^{(i)}}$$

---

## 3. Analysis Framework & Performance Metrics

Experiments are structured around two complementary evaluations:

1. **Fixed CPU Budget Evaluation (100 ms vs. 1000 ms):**
   - Each method dynamically adjusts its sample size $N$ to exhaust an allocated CPU time limit.
   - Compares residual Relative Error ($\%$) and Mean Squared Error (MSE).
2. **Target Error Horizon (Pareto Curves):**
   - Measures CPU time (ms) required to achieve target relative errors (e.g., $1.0\%$ and $0.5\%$).
   - Plots log-log efficiency curves ($\ln(\text{Time})$ vs. $\ln(\text{Error})$).
3. **Stress Tests:**
   - Expiry limits: $T \to 0$ (instability of raw Skorokhod weights).
   - Moneyness variations: Deep Out-of-the-Money vs. At-the-Money ($S_0 \approx K$).

---

## 4. Repository Structure

```text
├── Benchmarks/               # Reference benchmark engines across contract types
│   ├── americans.py
│   ├── asian.py
│   ├── bermudan.py
│   └── europeans.py
├── Methods/                  # Greek computation engines & numerical schemes
│   ├── Analytic/             # Closed-form analytic solutions
│   ├── LSMC/                 # Longstaff-Schwartz pricer with Malliavin integration
│   ├── Malliavin/            # Skorokhod weights & Fournié localization estimators
│   ├── MonteCarlo/           # Standard MC, Pathwise, and FDMC (CRN)
│   └── PDE 1D/               # 1D Finite Difference PDE solvers
│       ├── americans.py
│       ├── asian.py
│       ├── bermudan.py
│       └── europeans.py
├── other/                    # Exploratory research notebooks
│   └── MalliavinDerivative.ipynb
├── results/                  # Exported benchmark and simulation runs
├── Utilities/                # Shared utilities and helper routines
├── analysis.ipynb            # Visualization, Pareto curves, and pivot tables
├── README.md
├── requirements.txt          # Project dependencies
├── run_all_greeks.py         # Runner for Greek computations across all methods
├── run_benchmarks.py         # Runner for analytical/PDE benchmarks
└── run.py                    # Main pipeline entry point
```