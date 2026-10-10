import numpy as np
import matplotlib.pyplot as plt

# ======================================================================
# CONFIGURACIÓN
# ======================================================================
# False: 2 paneles (Panel 1: Teoría/Densidad, Panel 2: Pesos de Malliavin en eje Y)
# True : 3 paneles (Panel 1: Densidad, Panel 2: Pesos puros, Panel 3: Aporte a Delta)
MODO_3_PANELES = False

# ----------------------------------------------------------------------
# 1. Parámetros Financieros y de Mercado (Cartera: 1 Acción)
# ----------------------------------------------------------------------
S0 = 100.0       # Spot inicial
r = 0.05         # Tipo libre de riesgo
sigma = 0.20     # Volatilidad
T = 1.0          # Vencimiento (1 año)
N_sim = 80_000   # Número de trayectorias para Monte Carlo

# Para una cartera de 1 acción, la Delta teórica exacta es d(S0)/d(S0) = 1.0
delta_teorica = 1.0

# ----------------------------------------------------------------------
# 2. Simulación de Monte Carlo y Pesos de Malliavin / Score Function
# ----------------------------------------------------------------------
np.random.seed(42)
Z = np.random.standard_normal(N_sim)
W_T = np.sqrt(T) * Z

# Trayectorias del activo al vencimiento
drift = (r - 0.5 * sigma**2) * T
S_T = S0 * np.exp(drift + sigma * W_T)

# Cartera = 1 acción -> Payoff = S_T
payoff = S_T

# Peso de Malliavin / Score Function: W_T / (S0 * sigma * T)
W_delta = W_T / (S0 * sigma * T)

# Estimador de la Delta por Malliavin: e^(-rT) * S_T * W_delta
deltas_por_senda = np.exp(-r * T) * payoff * W_delta
delta_estimada = np.mean(deltas_por_senda)
media_pesos = np.mean(W_delta)

# Punto neutro donde el peso se anula: S* = S0 * exp((r - 0.5*sigma^2)*T)
S_star = S0 * np.exp((r - 0.5 * sigma**2) * T)

print("--- RESULTADOS MONTE CARLO (CARTERA: 1 ACCIÓN) ---")
print(f"Media de los Pesos W_delta : {media_pesos:.6f}  (Teórica: 0.0)")
print(f"Delta Teórica (1 acción)   : {delta_teorica:.4f}")
print(f"Delta Estimada Malliavin   : {delta_estimada:.4f}")
print(f"Error Absoluto             : {abs(delta_estimada - delta_teorica):.5f}")

# ----------------------------------------------------------------------
# 3. Gráficos Explicativos
# ----------------------------------------------------------------------
s_grid = np.linspace(40, 200, 500)

def lognormal_pdf(s, s_init):
    mu = np.log(s_init) + (r - 0.5 * sigma**2) * T
    scale = sigma * np.sqrt(T)
    return (1.0 / (s * scale * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((np.log(s) - mu) / scale)**2)

pdf_base = lognormal_pdf(s_grid, S0)
pdf_bumped = lognormal_pdf(s_grid, S0 + 2.0)
score_grid = (np.log(s_grid / S0) - (r - 0.5 * sigma**2) * T) / (S0 * (sigma**2) * T)

# Submuestra para visualización clara
sample_idx = np.random.choice(N_sim, size=2500, replace=False)
s_sample = S_T[sample_idx]
w_sample = W_delta[sample_idx]
contrib_sample = deltas_por_senda[sample_idx]

if not MODO_3_PANELES:
    # ------------------ VISTA DE 2 PANELES ------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Panel 1: Desplazamiento de Densidad vs Score Function
    ax1.plot(s_grid, pdf_base, label=f"Densidad $p(s; S_0={S0:.0f})$", color="royalblue", lw=2)
    ax1.plot(s_grid, pdf_bumped, label=f"Densidad $p(s; S_0={S0+2:.0f})$", color="darkorange", lw=1.5, linestyle="--")
    ax1.axvline(S_star, color="purple", linestyle=":", lw=1.5, label=fr"Punto neutro $S^*={S_star:.2f}$ ($\mathcal{{W}}_\Delta=0$)")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(s_grid, score_grid, color="crimson", lw=1.5, alpha=0.8, label=r"Score / Peso $\mathcal{W}_\Delta(s)$")
    ax1_twin.axhline(0, color="crimson", linestyle=":", alpha=0.5)
    ax1_twin.set_ylabel(r"Score Function / Peso $\mathcal{W}_\Delta$", color="crimson")
    ax1_twin.tick_params(axis='y', labelcolor="crimson")

    ax1.set_title("1. Desplazamiento de la Densidad vs. Función de Peso (Score)", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Precio Terminal $S_T$")
    ax1.set_ylabel("Densidad de Probabilidad")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    # Panel 2: Pesos de Malliavin asignados a cada senda
    scatter = ax2.scatter(s_sample, w_sample, c=contrib_sample, 
                          cmap="coolwarm", alpha=0.65, edgecolors="none", s=25)
    cbar = plt.colorbar(scatter, ax=ax2)
    cbar.set_label(r"Aporte marginal a la Delta: $e^{-rT} \cdot S_T \cdot \mathcal{W}_\Delta$", rotation=90)

    ax2.plot(s_grid, score_grid, color="darkblue", lw=2, label=r"Curva teórica $\mathcal{W}_\Delta(S_T)$")
    ax2.axvline(S_star, color="purple", linestyle=":", lw=1.5, label=fr"Punto neutro $S^*={S_star:.2f}$")
    ax2.axhline(0, color="black", lw=1, linestyle="--", label=fr"Media Pesos $\mathbb{{E}}[\mathcal{{W}}_\Delta] = {media_pesos:.4f} \approx 0$")

    ax2.text(0.04, 0.95, 
             fr"$\mathbf{{Cartera:\ 1\ Acci\acute{{o}}n}}$" + "\n" +
             fr"$\mathbb{{E}}[\mathcal{{W}}_\Delta] \approx {media_pesos:.4f}$ (pesos puros = 0)" + "\n" +
             fr"$\Delta = \mathbb{{E}}[e^{{-rT}} S_T \mathcal{{W}}_\Delta] = {delta_estimada:.3f}$" + "\n" +
             fr"(Teórica = ${delta_teorica:.2f}$)",
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.85))

    ax2.set_title(r"2. Pesos de Malliavin $\mathcal{W}_\Delta$ por Senda (Cartera: 1 Acción)", 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel("Precio Terminal $S_T$ de la senda")
    ax2.set_ylabel(r"Peso de Malliavin $\mathcal{W}_\Delta$ asignado")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

else:
    # ------------------ VISTA DE 3 PANELES ------------------
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1: Densidad y Score
    ax1.plot(s_grid, pdf_base, label=f"Densidad $p(s; S_0={S0:.0f})$", color="royalblue", lw=2)
    ax1.plot(s_grid, pdf_bumped, label=f"Densidad $p(s; S_0={S0+2:.0f})$", color="darkorange", lw=1.5, linestyle="--")
    ax1.axvline(S_star, color="purple", linestyle=":", lw=1.5, label=fr"$S^*={S_star:.1f}$ ($\mathcal{{W}}_\Delta=0$)")

    ax1_twin = ax1.twinx()
    ax1_twin.plot(s_grid, score_grid, color="crimson", lw=1.5, alpha=0.8, label=r"Score $\mathcal{W}_\Delta(s)$")
    ax1_twin.axhline(0, color="crimson", linestyle=":", alpha=0.5)
    ax1_twin.set_ylabel(r"Score Function / Peso $\mathcal{W}_\Delta$", color="crimson")
    ax1_twin.tick_params(axis='y', labelcolor="crimson")

    ax1.set_title("1. Desplazamiento Densidad vs Score", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Precio Terminal $S_T$")
    ax1.set_ylabel("Densidad de Probabilidad")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Pesos de Malliavin puros
    scatter2 = ax2.scatter(s_sample, w_sample, c=w_sample, cmap="coolwarm", alpha=0.6, edgecolors="none", s=20)
    ax2.plot(s_grid, score_grid, color="black", lw=1.5, label=r"Curva $\mathcal{W}_\Delta(S_T)$")
    ax2.axhline(0, color="black", lw=1, linestyle="--", label=r"$\mathbb{E}[\mathcal{W}_\Delta] = 0$")
    ax2.axvline(S_star, color="purple", linestyle=":", lw=1.5)
    ax2.set_title(r"2. Pesos de Malliavin $\mathcal{W}_\Delta$ por Senda", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Precio Terminal $S_T$")
    ax2.set_ylabel(r"Peso $\mathcal{W}_\Delta = W_T / (S_0 \sigma T)$")
    ax2.text(0.05, 0.90, fr"$\mathbb{{E}}[\mathcal{{W}}_\Delta] = {media_pesos:.4f} \approx 0$" + "\n(Media de pesos = 0)", 
             transform=ax2.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(True, alpha=0.3)
    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label(r"Peso $\mathcal{W}_\Delta$", rotation=90)

    # Panel 3: Cartera de 1 acción ponderada por los pesos
    scatter3 = ax3.scatter(s_sample, contrib_sample, c=w_sample, cmap="coolwarm", alpha=0.6, edgecolors="none", s=20)
    ax3.axhline(0, color="gray", lw=1)
    ax3.axhline(delta_estimada, color="forestgreen", lw=2, linestyle="--", label=f"Media Delta = {delta_estimada:.3f}")
    ax3.axvline(S_star, color="purple", linestyle=":", lw=1.5)
    ax3.set_title(r"3. Cartera 1 Acción: $e^{-rT} \cdot S_T \cdot \mathcal{W}_\Delta$", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Precio Terminal $S_T$")
    ax3.set_ylabel("Aporte marginal a la Delta")
    ax3.text(0.05, 0.90, f"Delta Teórica: {delta_teorica:.2f}\nDelta MC: {delta_estimada:.4f}", 
             transform=ax3.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax3.legend(loc="lower right", fontsize=9)
    ax3.grid(True, alpha=0.3)
    cbar3 = plt.colorbar(scatter3, ax=ax3)
    cbar3.set_label(r"Peso $\mathcal{W}_\Delta$", rotation=90)

plt.tight_layout()
plt.show()