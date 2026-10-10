import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ----------------------------------------------------------------------
# 1. Parámetros Financieros y de Mercado
# ----------------------------------------------------------------------
S0 = 100.0       # Spot inicial
K = 100.0        # Strike (At-the-Money)
r = 0.05         # Tipo libre de riesgo
sigma = 0.20     # Volatilidad
T = 1.0          # Vencimiento (1 año)
N_sim = 80_000   # Número de trayectorias para Monte Carlo

# Delta analítica exacta de Black-Scholes para la Call
d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
delta_teorica = norm.cdf(d1)

# ----------------------------------------------------------------------
# 2. Simulación de Monte Carlo y Pesos de Malliavin / Score Function
# ----------------------------------------------------------------------
np.random.seed(42)
Z = np.random.standard_normal(N_sim)
W_T = np.sqrt(T) * Z

# Trayectorias del activo al vencimiento
drift = (r - 0.5 * sigma**2) * T
S_T = S0 * np.exp(drift + sigma * W_T)

# Payoff de la Call
payoff = np.maximum(S_T - K, 0.0)

# Peso de Malliavin / Score Function evaluado en cada senda: W_T / (S0 * sigma * T)
W_delta = W_T / (S0 * sigma * T)

# Estimador de la Delta por Malliavin / Likelihood Ratio Method
deltas_por_senda = np.exp(-r * T) * payoff * W_delta
delta_estimada = np.mean(deltas_por_senda)

print(f"--- RESULTADOS MONTE CARLO ---")
print(f"Delta Analítica BS Exacta: {delta_teorica:.4f}")
print(f"Delta Estimada Malliavin : {delta_estimada:.4f}")
print(f"Error Absoluto          : {abs(delta_estimada - delta_teorica):.5f}")

# ----------------------------------------------------------------------
# 3. Gráficos Explicativos
# ----------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Panel 1: Qué le pasa a la densidad cuando S0 sube un poco
s_grid = np.linspace(40, 200, 500)

def lognormal_pdf(s, s_init):
    mu = np.log(s_init) + (r - 0.5 * sigma**2) * T
    scale = sigma * np.sqrt(T)
    return (1.0 / (s * scale * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(s) - mu) / scale)**2)

pdf_base = lognormal_pdf(s_grid, S0)
pdf_bumped = lognormal_pdf(s_grid, S0 + 2.0)

# Score function analítica sobre la rejilla
score_grid = (np.log(s_grid / S0) - (r - 0.5 * sigma**2) * T) / (S0 * (sigma**2) * T)

ax1.plot(s_grid, pdf_base, label=f"Densidad $p(s; S_0={S0:.0f})$", color="royalblue", lw=2)
ax1.plot(s_grid, pdf_bumped, label=f"Densidad $p(s; S_0={S0+2:.0f})$", color="darkorange", lw=1.5, linestyle="--")
ax1.axvline(K, color="black", linestyle=":", label=f"Strike $K={K:.0f}$")

# Eje secundario para la función Score
ax1_twin = ax1.twinx()
ax1_twin.plot(s_grid, score_grid, color="crimson", lw=1.5, alpha=0.8, label=r"Score $\frac{\partial \ln p}{\partial S_0}$")
ax1_twin.axhline(0, color="crimson", linestyle=":", alpha=0.5)
ax1_twin.set_ylabel("Valor de la Score Function", color="crimson")
ax1_twin.tick_params(axis='y', labelcolor="crimson")

ax1.set_title("1. Desplazamiento de la Densidad vs. Score Function", fontsize=12, fontweight='bold')
ax1.set_xlabel("Precio Terminal $S_T$")
ax1.set_ylabel("Densidad de Probabilidad")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Panel 2: Descomposición de la simulación en Monte Carlo
# Tomamos una submuestra de 2,000 puntos para visualización clara
sample_idx = np.random.choice(N_sim, size=2000, replace=False)
s_sample = S_T[sample_idx]
contrib_sample = deltas_por_senda[sample_idx]

scatter = ax2.scatter(s_sample, contrib_sample, c=W_delta[sample_idx], 
                      cmap="coolwarm", alpha=0.6, edgecolors="none", s=25)
cbar = plt.colorbar(scatter, ax=ax2)
cbar.set_label(r"Peso $\mathcal{W}_\Delta$ asignado a la senda", rotation=90)

ax2.axvline(K, color="black", linestyle=":", label=f"Strike $K={K:.0f}$")
ax2.axhline(0, color="gray", lw=1)
ax2.axhline(delta_estimada, color="green", lw=2, linestyle="--", 
            label=f"Media Total (Delta = {delta_estimada:.3f})")

ax2.set_title(r"2. Contribución por Senda: $e^{-rT} \cdot \phi(S_T) \cdot \mathcal{W}_\Delta$", 
              fontsize=12, fontweight='bold')
ax2.set_xlabel("Precio Terminal $S_T$ de la senda")
ax2.set_ylabel("Aporte marginal a la Delta")
ax2.set_ylim(-0.5, 2.5)
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()