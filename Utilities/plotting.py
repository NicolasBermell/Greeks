"""Plotting and visualization utilities for Greeks benchmarking analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from config import FDMC_FIXED_EPS

# Set publication style defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
})

GREEK_LABELS = {
    "delta": r"$\Delta$ (Delta)",
    "gamma": r"$\Gamma$ (Gamma)",
    "vega": r"$\mathcal{V}$ (Vega)",
}


def plot_epsilon_sensitivity(
    metrics_df: pd.DataFrame,
    product: str,
    greeks: list[str] = ["delta", "gamma", "vega"],
    metric: str = "abs_error",
    figsize: tuple[int, int] = (17, 5),
) -> plt.Figure:
    """Plots log-log sensitivity curves of Error vs. eps across sample sizes N.

    Visualizes the fundamental FDMC bias-variance / numerical cancellation trade-off.

    Args:
        metrics_df: DataFrame with error metrics.
        product: Contract name (e.g., 'Call European ATM').
        greeks: List of Greeks to plot.
        metric: 'abs_error' or 'rel_error'.
        figsize: Figure dimensions.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    prod_data = metrics_df[metrics_df["Product"] == product].copy()
    n_values = sorted(prod_data["N"].unique())
    palette = sns.color_palette("mako", n_colors=len(n_values))
    ylabel = "Absolute Error" if metric == "abs_error" else "Relative Error (%)"

    fig, axes = plt.subplots(1, len(greeks), figsize=figsize, sharey=False)
    if len(greeks) == 1:
        axes = [axes]

    for ax, greek in zip(axes, greeks):
        err_col = f"{greek}_{metric}"

        for idx, n in enumerate(n_values):
            sub = prod_data[prod_data["N"] == n].sort_values("eps", ascending=False)
            valid_sub = sub[sub[err_col] > 0]

            ax.plot(
                valid_sub["eps"],
                valid_sub[err_col],
                marker="o",
                label=f"N = {n:_}",
                color=palette[idx],
                alpha=0.85,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Perturbation Step Size $\varepsilon$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{product}\nSensitivity for {GREEK_LABELS.get(greek, greek)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title="Sample Size $N$", frameon=True)
        ax.invert_xaxis()

    plt.tight_layout()
    return fig


def plot_convergence_by_eps(
    metrics_df: pd.DataFrame,
    product: str,
    greeks: list[str] = ["delta", "gamma", "vega"],
    metric: str = "abs_error",
    eps_subset: list[float] | None = None,
    tolerance_dict: dict[str, float] | None = None,
    figsize: tuple[int, int] = (18, 5.5),
) -> plt.Figure:
    """Plots log-log curves of Error vs. sample size N for each eps curve.

    Puts sample size N on the x-axis, allowing direct visual comparison of how different
    perturbation step sizes eps perform and converge as N increases.

    Args:
        metrics_df: DataFrame with error metrics.
        product: Contract name (e.g., 'Call European ATM').
        greeks: List of Greeks to plot.
        metric: 'abs_error' or 'rel_error'.
        eps_subset: Optional list of eps values to include. Defaults to all available eps.
        tolerance_dict: Optional target tolerances per Greek to display as dashed threshold lines.
        figsize: Figure dimensions.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    prod_data = metrics_df[metrics_df["Product"] == product].copy()
    all_eps = sorted(prod_data["eps"].unique(), reverse=True)
    if eps_subset is not None:
        all_eps = [e for e in all_eps if e in eps_subset]

    markers = ["o", "s", "^", "v", "D", "<", ">", "p"]
    palette = sns.color_palette("tab10", n_colors=len(all_eps))
    ylabel = "Absolute Error" if metric == "abs_error" else "Relative Error (%)"

    fig, axes = plt.subplots(1, len(greeks), figsize=figsize, sharey=False)
    if len(greeks) == 1:
        axes = [axes]

    max_n = prod_data["N"].max()

    for ax, greek in zip(axes, greeks):
        err_col = f"{greek}_{metric}"

        # Determine best eps at max N for highlighting
        max_n_data = prod_data[prod_data["N"] == max_n]
        best_eps = None
        if not max_n_data.empty:
            valid_max = max_n_data.dropna(subset=[err_col])
            valid_max = valid_max[valid_max[err_col] > 0]
            if not valid_max.empty:
                best_idx = valid_max[err_col].idxmin()
                best_eps = valid_max.loc[best_idx, "eps"]

        for idx, eps_val in enumerate(all_eps):
            sub = prod_data[prod_data["eps"] == eps_val].sort_values("N")
            valid_sub = sub[sub[err_col] > 0]

            is_best = (eps_val == best_eps)
            lw = 2.8 if is_best else 1.6
            alpha = 1.0 if is_best else 0.75
            marker = markers[idx % len(markers)]
            color = palette[idx]

            eps_exp = int(np.round(np.log10(eps_val)))
            label = rf"$\varepsilon=10^{{{eps_exp}}}$"
            if is_best:
                label += r" $\mathbf{(\star\ Best)}$"

            if not valid_sub.empty:
                ax.plot(
                    valid_sub["N"],
                    valid_sub[err_col],
                    marker=marker,
                    markersize=8 if is_best else 6,
                    label=label,
                    color=color,
                    linewidth=lw,
                    alpha=alpha,
                )

        # Plot target tolerance threshold if provided
        if tolerance_dict and greek in tolerance_dict:
            tol_val = tolerance_dict[greek]
            ax.axhline(
                tol_val,
                color="crimson",
                linestyle="-.",
                linewidth=1.5,
                alpha=0.85,
                label=f"Acceptable Tol ({tol_val:g})",
            )

        # Plot O(1/sqrt(N)) reference slope normalized to first valid point of best eps
        n_vals = sorted(prod_data["N"].unique())
        if best_eps is not None:
            best_sub = prod_data[prod_data["eps"] == best_eps].sort_values("N")
            first_valid = best_sub[best_sub[err_col] > 0]
            if not first_valid.empty:
                ref_y0 = first_valid.iloc[0][err_col]
                ref_n0 = first_valid.iloc[0]["N"]
                ref_line = ref_y0 * np.sqrt(ref_n0 / np.array(n_vals))
                ax.plot(
                    n_vals,
                    ref_line,
                    linestyle=":",
                    color="black",
                    alpha=0.5,
                    linewidth=1.5,
                    label=r"$\mathcal{O}(N^{-1/2})$ Reference",
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sample Size $N$ (Number of Simulations)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{product}\n{ylabel} vs. $N$ by $\\varepsilon$ — {GREEK_LABELS.get(greek, greek)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(title=r"Step Size $\varepsilon$", frameon=True, fontsize=8.5, ncol=1)

    plt.tight_layout()
    return fig


def _format_bar_val(val: float) -> str:
    """Formats Greek values cleanly for bar chart labels."""
    if abs(val) == 0:
        return "0.00"
    elif abs(val) < 0.001:
        return f"{val:.2e}"
    elif abs(val) < 0.1:
        return f"{val:.4f}"
    elif abs(val) < 10:
        return f"{val:.3f}"
    else:
        return f"{val:.2f}"


def plot_comparative_greeks_bars(
    fdmc_metrics_df: pd.DataFrame,
    malliavin_metrics_df: pd.DataFrame,
    product: str,
    n_choice: int | None = None,
    fdmc_eps: float | dict[str, float] | str = FDMC_FIXED_EPS,
    fournier_metrics_df: pd.DataFrame | None = None,
    figsize: tuple[int, int] = (14, 5.5),
) -> plt.Figure:
    """Plots a 3-way or 4-way grouped bar chart across two scales (Delta/Gamma and Vega)."""
    from Utilities.metrics import compute_comparative_summary_table

    if n_choice is None:
        n_choice = int(malliavin_metrics_df["N"].max())

    comp_table = compute_comparative_summary_table(
        fdmc_metrics_df,
        malliavin_metrics_df,
        n_choice=n_choice,
        fdmc_eps=fdmc_eps,
        fournier_metrics_df=fournier_metrics_df,
    )
    sub = comp_table[comp_table["Product"] == product]

    sub_dg = sub[sub["Greek"].isin(["delta", "gamma"])].copy()
    sub_v = sub[sub["Greek"] == "vega"].copy()

    has_four = "Fournié Malliavin" in sub.columns
    width = 0.19 if has_four else 0.25

    fdmc_cols = [c for c in sub.columns if c.startswith("FDMC (")]
    fdmc_col = fdmc_cols[0] if fdmc_cols else "FDMC"

    fig, (ax_dg, ax_v) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [2.2, 1.2]})

    panels = [
        (ax_dg, sub_dg, [r"$\Delta$", r"$\Gamma$"], r"$\Delta$ (Delta) & $\Gamma$ (Gamma)"),
        (ax_v, sub_v, [r"$\mathcal{V}$"], r"$\mathcal{V}$ (Vega)"),
    ]

    handles, legend_labels = [], []

    for ax, sub_p, labels, p_name in panels:
        if sub_p.empty:
            continue
        x = np.arange(len(sub_p))
        b_vals = sub_p["Benchmark"].tolist()
        f_vals = sub_p[fdmc_col].tolist()
        m_vals = sub_p["Std Malliavin"].tolist()
        four_vals = sub_p["Fournié Malliavin"].tolist() if has_four else []

        all_vals = b_vals + f_vals + m_vals + four_vals
        y_min = min(all_vals) if all_vals else 0.0
        y_max = max(all_vals) if all_vals else 1.0

        y_max_plot = max(y_max, 0.0)
        y_min_plot = min(y_min, 0.0)
        base_span = max(y_max_plot - y_min_plot, 1e-4)

        bottom_lim = y_min_plot - 0.22 * base_span if y_min_plot < 0 else 0.0
        top_lim = y_max_plot + 0.28 * base_span if y_max_plot > 0 else 0.0
        ax.set_ylim(bottom_lim, top_lim)
        total_span = top_lim - bottom_lim

        rects1 = ax.bar(x - (1.5 * width if has_four else width), b_vals, width, label="Analytical Benchmark", color="#2b5c8f")
        rects2 = ax.bar(x - (0.5 * width if has_four else 0), f_vals, width, label=f"{fdmc_col}" if "N=" in fdmc_col else f"{fdmc_col} ($N={n_choice:_}$)", color="#e07a5f")
        rects3 = ax.bar(x + (0.5 * width if has_four else width), m_vals, width, label=f"Std Malliavin ($N={n_choice:_}$)", color="#2a9d8f")

        bar_groups = [(rects1, b_vals), (rects2, f_vals), (rects3, m_vals)]
        if has_four:
            rects4 = ax.bar(x + 1.5 * width, four_vals, width, label=f"Fournié Malliavin ($N={n_choice:_}$)", color="#3d5a80")
            bar_groups.append((rects4, four_vals))

        if not handles:
            h, l = ax.get_legend_handles_labels()
            handles, legend_labels = h, l

        for rects, vals in bar_groups:
            for rect, val in zip(rects, vals):
                txt = _format_bar_val(val)
                y_pos = rect.get_y() + rect.get_height()
                if val >= 0:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        y_pos + 0.025 * total_span,
                        txt,
                        ha="center",
                        va="bottom",
                        fontsize=8.0,
                        fontweight="bold",
                        color="#1f2937",
                    )
                else:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        y_pos - 0.025 * total_span,
                        txt,
                        ha="center",
                        va="top",
                        fontsize=8.0,
                        fontweight="bold",
                        color="#1f2937",
                    )

        if y_min < 0:
            ax.axhline(0, color="#666666", linestyle="--", linewidth=0.8, alpha=0.7)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=12)
        ax.set_title(p_name, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    ax_dg.set_ylabel("Greek Value (Delta / Gamma)", fontsize=10)
    ax_v.set_ylabel("Greek Value (Vega)", fontsize=10)

    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(handles),
        frameon=True,
        fontsize=9,
    )
    fig.suptitle(f"{product}: Benchmark vs. Simulation Sensitivity Comparison ($N={n_choice:_}$)", fontsize=12, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.91])
    return fig


def plot_cutoff_shapes_and_derivatives(
    K: float = 100.0,
    deltas: list[float] = [3.0, 10.0, 25.0],
    cutoff_types: list[str] = ["gaussian", "cubic", "cosine", "quintic", "cauchy"],
    include_second_deriv: bool = False,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """Plots the cutoff smoothing function psi(S), its derivative psi'(S), and optionally psi''(S)."""
    from Methods.Malliavin.malliavin_european_fournier import get_cutoff

    num_rows = 3 if include_second_deriv else 2
    if figsize is None:
        figsize = (16, 12 if include_second_deriv else 9)

    S_vals = np.linspace(K - 35.0, K + 35.0, 1000)
    fig, axes = plt.subplots(num_rows, len(deltas), figsize=figsize, sharex=True)

    colors = {
        "gaussian": "#1d3557",
        "cubic": "#e63946",
        "cosine": "#457b9d",
        "quintic": "#2a9d8f",
        "cauchy": "#e76f51",
    }

    for col_idx, d in enumerate(deltas):
        ax_psi = axes[0, col_idx] if num_rows > 1 else axes[col_idx]
        ax_psi_prime = axes[1, col_idx]

        for ctype in cutoff_types:
            if include_second_deriv:
                psi, psi_prime, psi_double_prime = get_cutoff(S_vals, K, bandwidth=d, cutoff_type=ctype, return_second_deriv=True)
            else:
                psi, psi_prime = get_cutoff(S_vals, K, bandwidth=d, cutoff_type=ctype)
            color = colors.get(ctype, "gray")

            ax_psi.plot(S_vals, psi, label=ctype.capitalize(), color=color, linewidth=2.0)
            ax_psi_prime.plot(S_vals, psi_prime, label=ctype.capitalize(), color=color, linewidth=2.0)
            if include_second_deriv:
                axes[2, col_idx].plot(S_vals, psi_double_prime, label=ctype.capitalize(), color=color, linewidth=2.0)

        ax_psi.axvline(K, color="black", linestyle=":", alpha=0.5, label="Strike K" if col_idx == 0 else "")
        ax_psi.axvline(K - d, color="gray", linestyle="--", alpha=0.4)
        ax_psi.axvline(K + d, color="gray", linestyle="--", alpha=0.4)

        ax_psi_prime.axvline(K, color="black", linestyle=":", alpha=0.5)
        ax_psi_prime.axvline(K - d, color="gray", linestyle="--", alpha=0.4)
        ax_psi_prime.axvline(K + d, color="gray", linestyle="--", alpha=0.4)

        ax_psi.set_title(f"Cutoff Profile $\\psi(S_T)$ [$\\delta = {d}$]")
        ax_psi.set_ylabel("$\\psi(S_T)$")
        ax_psi.grid(True, alpha=0.3)
        if col_idx == 0:
            ax_psi.legend(frameon=True, fontsize=9)

        ax_psi_prime.set_title(f"Boundary 1st Deriv $\\psi'(S_T)$ [$\\delta = {d}$]")
        ax_psi_prime.set_ylabel("$\\psi'(S_T)$")
        ax_psi_prime.grid(True, alpha=0.3)
        if not include_second_deriv:
            ax_psi_prime.set_xlabel("Asset Spot Price $S_T$")

        if include_second_deriv:
            ax_pp = axes[2, col_idx]
            ax_pp.axvline(K, color="black", linestyle=":", alpha=0.5)
            ax_pp.axvline(K - d, color="gray", linestyle="--", alpha=0.4)
            ax_pp.axvline(K + d, color="gray", linestyle="--", alpha=0.4)
            ax_pp.set_title(f"Boundary 2nd Deriv $\\psi''(S_T)$ [$\\delta = {d}$]")
            ax_pp.set_xlabel("Asset Spot Price $S_T$")
            ax_pp.set_ylabel("$\\psi''(S_T)$")
            ax_pp.grid(True, alpha=0.3)

    title_desc = "Profiles, 1st & 2nd Derivatives" if include_second_deriv else "Profiles & Boundary Derivatives"
    fig.suptitle(f"Fournié Localized Cutoff {title_desc} Around Strike $K={K}$", fontsize=14)
    plt.tight_layout()
    return fig


def plot_error_vs_gap_grid(
    grid_df: pd.DataFrame,
    product: str = "Call European ATM",
    figsize: tuple[int, int] = (17, 5),
) -> plt.Figure:
    """Plots Absolute Error vs. Gap Half-Width delta for each Greek across all 5 smoothing functions."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    greeks = ["delta", "gamma", "vega"]
    cutoff_types = grid_df["cutoff_type"].unique()

    colors = {
        "gaussian": "#1d3557",
        "cubic": "#e63946",
        "cosine": "#457b9d",
        "quintic": "#2a9d8f",
        "cauchy": "#e76f51",
    }
    markers = {
        "gaussian": "o",
        "cubic": "s",
        "cosine": "^",
        "quintic": "D",
        "cauchy": "v",
    }

    for ax, greek in zip(axes, greeks):
        err_col = f"{greek}_abs_error"
        for ctype in cutoff_types:
            sub = grid_df[grid_df["cutoff_type"] == ctype].sort_values("bandwidth")
            ax.plot(
                sub["bandwidth"],
                sub[err_col],
                marker=markers.get(ctype, "o"),
                color=colors.get(ctype, "gray"),
                label=ctype.capitalize(),
                linewidth=1.8,
                markersize=6,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Smoothing Gap Half-Width $\delta$ from Strike")
        ax.set_ylabel("Absolute Error vs. Analytical Benchmark")
        ax.set_title(f"{GREEK_LABELS.get(greek, greek)} Error vs. Gap $\\delta$")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=True, fontsize=8.5)

    fig.suptitle(f"{product}: Greek Error vs. Smoothing Gap Size $\\delta$ Across All Cutoff Functions", fontsize=13)
    plt.tight_layout()
    return fig


def plot_variance_convexity_curve(
    grid_df: pd.DataFrame,
    product: str = "Call European ATM",
    greeks: list[str] = ["gamma", "vega"],
    figsize: tuple[int, int] = (15, 5),
) -> plt.Figure:
    """Plots sample estimator variance vs. Gap Size delta, illustrating the U-shaped convex variance curve."""
    fig, axes = plt.subplots(1, len(greeks), figsize=figsize, sharey=False)
    if len(greeks) == 1:
        axes = [axes]

    cutoff_types = grid_df["cutoff_type"].unique()
    colors = {
        "gaussian": "#1d3557",
        "cubic": "#e63946",
        "cosine": "#457b9d",
        "quintic": "#2a9d8f",
        "cauchy": "#e76f51",
    }
    markers = {
        "gaussian": "o",
        "cubic": "s",
        "cosine": "^",
        "quintic": "D",
        "cauchy": "v",
    }

    for ax, greek in zip(axes, greeks):
        var_col = f"{greek}_var"

        for ctype in cutoff_types:
            sub = grid_df[grid_df["cutoff_type"] == ctype].sort_values("bandwidth")
            ax.plot(
                sub["bandwidth"],
                sub[var_col],
                marker=markers.get(ctype, "o"),
                color=colors.get(ctype, "gray"),
                label=ctype.capitalize(),
                linewidth=2.0,
                markersize=6.5,
            )

            # Mark the minimum variance point
            min_idx = sub[var_col].idxmin()
            opt_d = sub.loc[min_idx, "bandwidth"]
            opt_v = sub.loc[min_idx, var_col]
            if ctype == "gaussian":
                ax.scatter([opt_d], [opt_v], color="gold", s=140, zorder=6, edgecolors="black", linewidth=1.5,
                           label=f"Gaussian Min Var ($\\delta^*={opt_d}$)")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Smoothing Gap Half-Width $\delta$ from Strike")
        ax.set_ylabel(f"Sample Estimator Variance $\\mathrm{{Var}}(\\hat{{{greek}}})$")
        ax.set_title(f"{GREEK_LABELS.get(greek, greek)} Estimator Variance: Convex Gap Curve")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=True, fontsize=8.5)

    fig.suptitle(f"{product}: Estimator Variance Convexity — Trade-off Between Boundary Spikes ($\\delta \\to 0$) and Malliavin Weights ($\\delta \\to \\infty$)", fontsize=13)
    plt.tight_layout()
    return fig


def plot_gap_function_heatmap(
    grid_df: pd.DataFrame,
    greek: str = "gamma",
    metric: str = "var",
    product: str = "Call European ATM",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Displays a 2D heatmap of (Cutoff Function x Gap Size delta) for the chosen Greek and metric."""
    col = f"{greek}_{metric}"
    pivot = grid_df.pivot(index="cutoff_type", columns="bandwidth", values=col)

    fig, ax = plt.subplots(figsize=figsize)
    # Use log10 scale for heatmap colors to highlight subtle differences across orders of magnitude
    log_pivot = np.log10(pivot.clip(lower=1e-12))

    sns.heatmap(
        log_pivot,
        annot=pivot.map(lambda v: f"{v:.1e}" if v < 0.01 else f"{v:.3f}"),
        fmt="",
        cmap="viridis_r",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": f"$\\log_{{10}}(\\text{{{metric.capitalize()}}})$"},
    )

    ax.set_title(f"{product}: 2D Heatmap of {GREEK_LABELS.get(greek, greek)} {metric.capitalize()} Across (Function $\\times$ Gap $\\delta$)")
    ax.set_xlabel(r"Smoothing Half-Width $\delta$ from Strike")
    ax.set_ylabel("Smoothing Function")

    plt.tight_layout()
    return fig


# Colors match the single-seed comparison plots; markers/line styles carry identity too.
REPLICATION_STYLES = {
    "FDMC": {"color": "#e76f51", "marker": "o", "linestyle": "-"},
    "Malliavin": {"color": "#2a9d8f", "marker": "s", "linestyle": "-"},
    "Malliavin Fournié": {"color": "#3d5a80", "marker": "D", "linestyle": "-"},
    "Malliavin Fournié (Γ shortcut)": {"color": "#3d5a80", "marker": "D", "linestyle": ":", "mfc": "none"},
    "Malliavin Fournié (ψ_K+ψ_H)": {"color": "#3d5a80", "marker": "D", "linestyle": ":", "mfc": "none"},
}


def plot_replication_error_vs_N(
    summary_df: pd.DataFrame,
    product: str,
    greeks: list[str] = ["delta", "gamma", "vega"],
    fdmc_eps: float = FDMC_FIXED_EPS,
    extrapolate_to: float = 1e8,
    figsize: tuple[int, int] = (18, 5.5),
) -> plt.Figure:
    """Error vs N from independent replications, one panel per Greek.

    Solid line: RMSE over the R runs. Band: 10th-90th percentile of |estimate - benchmark|.
    Dashed line: extrapolation sqrt(bias^2 + Var/N) beyond the largest simulated N, using the
    per-path variance; the bias term is only kept when it is significant (|bias| > 3 bias_se).
    """
    sub = summary_df[summary_df["Product"] == product]
    if sub.empty:
        raise ValueError(f"Product '{product}' not found. Available: {list(summary_df['Product'].unique())}")
    is_fdmc = sub["Method"] == "FDMC"
    sub = sub[~is_fdmc | np.isclose(sub["eps"].fillna(-1.0), fdmc_eps)]
    R = int(sub["R"].max())

    fig, axes = plt.subplots(1, len(greeks), figsize=figsize, sharey=False)
    if len(greeks) == 1:
        axes = [axes]

    for ax, greek in zip(axes, greeks):
        g = sub[sub["Greek"] == greek]
        for method, style in REPLICATION_STYLES.items():
            m = g[g["Method"] == method].sort_values("N")
            if m.empty:
                continue
            label = f"FDMC (ε={fdmc_eps:g})" if method == "FDMC" else method
            ax.fill_between(m["N"], m["abs_err_q10"], m["abs_err_q90"], color=style["color"], alpha=0.15, linewidth=0)
            ax.plot(
                m["N"], m["rmse"], marker=style["marker"], linestyle=style["linestyle"], color=style["color"],
                markerfacecolor=style.get("mfc", style["color"]), label=label, zorder=4,
            )
            last = m.iloc[-1]
            if extrapolate_to and extrapolate_to > last["N"]:
                n_ext = np.geomspace(last["N"], extrapolate_to, 30)
                bias = last["bias"] if abs(last["bias"]) > 3 * last["bias_se"] else 0.0
                ax.plot(
                    n_ext, np.sqrt(bias**2 + last["var_per_path"] / n_ext),
                    linestyle="--", linewidth=1.4, color=style["color"], alpha=0.8, zorder=3,
                )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sample Size $N$")
        ax.set_ylabel("Error vs. analytical")
        ax.set_title(f"{product}\n{GREEK_LABELS.get(greek, greek)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=True, fontsize=8)

    fig.suptitle(
        f"RMSE over R = {R} independent runs (band: 10th–90th percentile of |error|; dashed: "
        r"$\sqrt{\mathrm{bias}^2 + \mathrm{Var}/N}$ extrapolation)",
        fontsize=11,
    )
    plt.tight_layout()
    return fig


# Colors identify the maturity (not a method); line style and marker repeat the encoding.
GAP_SWEEP_STYLES = {
    1.0: {"color": "#2a78d6", "linestyle": "-", "marker": "o", "label": "T = 1"},
    1.0 / 12.0: {"color": "#d55181", "linestyle": "--", "marker": "s", "label": "T = 1/12"},
}


def plot_gap_sweep(sweep_df: pd.DataFrame, greeks: list[str] = ["delta", "gamma", "vega"]) -> plt.Figure:
    """Per-path variance of Fournié relative to standard Malliavin vs the gap l / (S0 sigma sqrt(T)).

    One row per payoff (call, digital), one column per Greek, one curve per maturity. Below 1,
    localization needs fewer paths than standard Malliavin. The dotted lines are FDMC (eps = 0.01)
    on the same scale, the vertical line the default gap for that Greek, and the filled marker
    the minimum of each curve.
    """
    from Methods.Malliavin.malliavin_european_fournier import BANDWIDTH_MULTIPLIERS

    families = [("Call", "Call European"), ("Digital", "Digital European")]
    fig, axes = plt.subplots(len(families), len(greeks), figsize=(18, 10), squeeze=False)

    for row, (family, prefix) in enumerate(families):
        fam = sweep_df[sweep_df["Product"].str.startswith(prefix)]
        for col, greek in enumerate(greeks):
            ax = axes[row, col]
            g = fam[fam["Greek"] == greek]
            four = g[g["Estimator"] == "Malliavin Fournié"]
            y_lo, y_hi = four["var_vs_standard"].min(), max(four["var_vs_standard"].max(), 1.0)
            y_lo, y_hi = min(y_lo, 1.0) / 2.0, y_hi * 2.0

            for T, style in GAP_SWEEP_STYLES.items():
                curve = four[np.isclose(four["T"], T)].sort_values("multiplier")
                if curve.empty:
                    continue
                ax.plot(curve["multiplier"], curve["var_vs_standard"], color=style["color"],
                        linestyle=style["linestyle"], marker=style["marker"], markersize=5, label=style["label"])
                best = curve.loc[curve["var_vs_standard"].idxmin()]
                ax.plot(best["multiplier"], best["var_vs_standard"], marker=style["marker"], markersize=11,
                        color=style["color"], markeredgecolor="white", markeredgewidth=1.5, zorder=5)
                ax.annotate(f"{best['multiplier']:g}", (best["multiplier"], best["var_vs_standard"]),
                            textcoords="offset points", xytext=(0, -16) if T == 1.0 else (0, 9),
                            ha="center", fontsize=9, color=style["color"])

                fdmc = g[np.isclose(g["T"], T) & g["Estimator"].str.startswith("FDMC")]["var_vs_standard"].iloc[0]
                if y_lo <= fdmc <= y_hi:
                    ax.axhline(fdmc, color=style["color"], linestyle=":", linewidth=1.4, alpha=0.9,
                               label=f"FDMC ε=0.01, {style['label']}")
                else:
                    ax.annotate(f"FDMC, {style['label']}: {fdmc:.2g}× (off scale)", xy=(0.02, 0.97 if T == 1.0 else 0.90),
                                xycoords="axes fraction", fontsize=8.5, color=style["color"], va="top")

            ax.axhline(1.0, color="#52514e", linewidth=1.2, label="standard Malliavin")
            ax.axvline(BANDWIDTH_MULTIPLIERS[greek], color="#9a9a96", linestyle="-.", linewidth=1.2,
                       label=f"default gap ({BANDWIDTH_MULTIPLIERS[greek]:g})")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(y_lo, y_hi)
            ax.set_xlabel(r"gap $\ell \,/\, (S_0 \sigma \sqrt{T})$")
            ax.set_ylabel("Var per path / Var(standard Malliavin)")
            ax.set_title(f"{family}: {GREEK_LABELS.get(greek, greek)}")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend(frameon=True, fontsize=8, loc="lower right")

    fig.suptitle("Fournié localization (Gaussian cutoff, same gap for all Greeks): per-path variance vs gap", fontsize=13)
    plt.tight_layout()
    return fig
