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


def plot_convergence_vs_N(
    metrics_df: pd.DataFrame,
    product: str,
    greeks: list[str] = ["delta", "gamma", "vega"],
    metric: str = "abs_error",
    figsize: tuple[int, int] = (17, 5),
) -> plt.Figure:
    """Plots log-log convergence of Error vs. sample size N at optimal eps*.

    Args:
        metrics_df: DataFrame with error metrics.
        product: Contract name.
        greeks: List of Greeks to plot.
        metric: 'abs_error' or 'rel_error'.
        figsize: Figure dimensions.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    from Utilities.metrics import extract_optimal_eps

    opt_df = extract_optimal_eps(metrics_df, criterion=metric)
    prod_opt = opt_df[opt_df["Product"] == product]
    err_col = "min_abs_error" if metric == "abs_error" else "min_rel_error_pct"
    ylabel = "Min Absolute Error" if metric == "abs_error" else "Min Relative Error (%)"

    fig, axes = plt.subplots(1, len(greeks), figsize=figsize, sharey=False)
    if len(greeks) == 1:
        axes = [axes]

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for ax, greek, color in zip(axes, greeks, colors):
        sub = prod_opt[prod_opt["Greek"] == greek].sort_values("N")

        n_arr = sub["N"].values
        err_arr = sub[err_col].values

        ax.plot(
            n_arr,
            err_arr,
            marker="s",
            color=color,
            linewidth=2.2,
            label=r"Empirical FDMC ($\varepsilon^*$)",
        )

        if len(n_arr) > 1 and err_arr[0] > 0:
            theoretical_ref = err_arr[0] * np.sqrt(n_arr[0] / n_arr)
            ax.plot(
                n_arr,
                theoretical_ref,
                linestyle="--",
                color="gray",
                alpha=0.8,
                label=r"$\mathcal{O}(N^{-1/2})$ Reference",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sample Size $N$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{product}\nConvergence for {GREEK_LABELS.get(greek, greek)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=True)

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


def plot_error_heatmaps(
    metrics_df: pd.DataFrame,
    product: str,
    greek: str = "delta",
    metric: str = "abs_error",
    figsize: tuple[int, int] = (9, 6),
) -> plt.Figure:
    """Plots a 2D heatmap of Error over (N x eps) parameter space.

    Args:
        metrics_df: DataFrame with error metrics.
        product: Contract name.
        greek: Greek name ('delta', 'gamma', 'vega').
        metric: 'abs_error' or 'rel_error'.
        figsize: Figure dimensions.

    Returns:
        plt.Figure: The generated matplotlib figure.
    """
    from Utilities.metrics import generate_error_pivot

    pivot = generate_error_pivot(metrics_df, product, greek, metric=metric)
    ylabel = "Absolute Error" if metric == "abs_error" else "Relative Error (%)"

    fig, ax = plt.subplots(figsize=figsize)
    log_pivot = np.log10(pivot.clip(lower=1e-8))

    sns.heatmap(
        log_pivot,
        annot=pivot.map(lambda x: f"{x:.2e}" if (x < 0.001 or x > 100) else f"{x:.4f}"),
        fmt="",
        cmap="YlOrRd",
        cbar_kws={"label": rf"$\log_{{10}}(\text{{{ylabel}}})$"},
        ax=ax,
        linewidths=0.5,
    )

    ax.set_title(f"{product} - {GREEK_LABELS.get(greek, greek)}\n{ylabel} Landscape")
    ax.set_xlabel(r"Perturbation Step $\varepsilon$")
    ax.set_ylabel("Monte Carlo Simulations ($N$)")
    plt.tight_layout()
    return fig


def plot_greeks_comparison_bars(
    metrics_df: pd.DataFrame,
    product: str,
    n_choice: int | None = None,
    criterion: str = "abs_error",
    figsize: tuple[int, int] = (10, 5),
) -> plt.Figure:
    """Plots a side-by-side bar chart of Benchmark vs best FDMC values for all three Greeks."""
    from Utilities.metrics import extract_optimal_eps

    if n_choice is None:
        n_choice = int(metrics_df["N"].max())

    opt_df = extract_optimal_eps(metrics_df, criterion=criterion)
    sub = opt_df[(opt_df["Product"] == product) & (opt_df["N"] == n_choice)]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(sub))
    width = 0.35

    greeks = sub["Greek"].tolist()
    labels = [GREEK_LABELS.get(g, g) for g in greeks]
    bench_vals = sub["benchmark_value"].tolist()
    fdmc_vals = sub["fdmc_value"].tolist()
    abs_errs = sub["min_abs_error"].tolist()

    rects1 = ax.bar(x - width / 2, bench_vals, width, label="Analytical Benchmark", color="#2b5c8f")
    rects2 = ax.bar(x + width / 2, fdmc_vals, width, label=f"FDMC ($N={n_choice:_}$)", color="#e07a5f")

    for i in range(len(x)):
        err_text = f"|err|: {abs_errs[i]:.2e}" if abs_errs[i] < 0.001 else f"|err|: {abs_errs[i]:.4f}"
        y_pos = max(bench_vals[i], fdmc_vals[i])
        y_offset = abs(y_pos) * 0.05 + 0.001
        ax.text(
            x[i],
            y_pos + y_offset,
            err_text,
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
            color="#333",
        )

    ax.set_ylabel("Greek Value")
    ax.set_title(f"{product}: Analytical Benchmark vs. Optimal FDMC")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_cross_product_comparison(
    metrics_df: pd.DataFrame,
    greek: str = "delta",
    n_choice: int | None = None,
    metric: str = "abs_error",
    figsize: tuple[int, int] = (12, 5),
) -> plt.Figure:
    """Compares the error of a Greek across all contracts."""
    from Utilities.metrics import extract_optimal_eps

    if n_choice is None:
        n_choice = int(metrics_df["N"].max())

    opt_df = extract_optimal_eps(metrics_df, criterion=metric)
    sub = opt_df[(opt_df["Greek"] == greek) & (opt_df["N"] == n_choice)].copy()
    err_col = "min_abs_error" if metric == "abs_error" else "min_rel_error_pct"
    ylabel = "Absolute Error" if metric == "abs_error" else "Relative Error (%)"

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#2a9d8f", "#e76f51", "#f4a261"]

    bars = ax.bar(sub["Product"], sub[err_col], color=colors[: len(sub)], width=0.5)

    for bar, eps in zip(bars, sub["optimal_eps"]):
        height = bar.get_height()
        val_str = f"{height:.2e}" if height < 0.001 else f"{height:.4f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height * 1.05,
            f"{val_str}\n(opt ε={eps})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    ax.set_ylabel(ylabel)
    ax.set_yscale("log")
    ax.set_title(f"Cross-Product Comparison for {GREEK_LABELS.get(greek, greek)} (N = {n_choice:_})")
    ax.grid(axis="y", which="both", alpha=0.3)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    return fig


def plot_malliavin_convergence_vs_N(
    malliavin_metrics_df: pd.DataFrame,
    product: str,
    greeks: list[str] = ["delta", "gamma", "vega"],
    metric: str = "abs_error",
    tolerance_dict: dict[str, float] | None = None,
    figsize: tuple[int, int] = (17, 5),
) -> plt.Figure:
    """Plots log-log convergence of Malliavin Error vs. sample size N alongside theoretical O(1/sqrt(N))."""
    prod_data = malliavin_metrics_df[malliavin_metrics_df["Product"] == product].sort_values("N")
    ylabel = "Absolute Error" if metric == "abs_error" else "Relative Error (%)"

    fig, axes = plt.subplots(1, len(greeks), figsize=figsize, sharey=False)
    if len(greeks) == 1:
        axes = [axes]

    colors = {"delta": "#1f77b4", "gamma": "#e76f51", "vega": "#2a9d8f"}

    for ax, greek in zip(axes, greeks):
        err_col = f"{greek}_{metric}"
        valid = prod_data[prod_data[err_col] > 0]
        n_vals = valid["N"].values
        err_vals = valid[err_col].values

        color = colors.get(greek, "#2a9d8f")

        ax.plot(
            n_vals,
            err_vals,
            marker="s",
            markersize=8,
            linewidth=2.5,
            color=color,
            label=f"Malliavin {GREEK_LABELS.get(greek, greek)}",
        )

        # Annotate points
        for n, err in zip(n_vals, err_vals):
            val_str = f"{err:.2e}" if err < 0.001 else f"{err:.4f}"
            ax.annotate(
                val_str,
                (n, err),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
                color="#222",
            )

        # Plot theoretical 1/sqrt(N) reference line
        if len(err_vals) > 0:
            ref_y0 = err_vals[0]
            ref_n0 = n_vals[0]
            ref_line = ref_y0 * np.sqrt(ref_n0 / n_vals)
            ax.plot(
                n_vals,
                ref_line,
                linestyle=":",
                color="black",
                alpha=0.6,
                linewidth=1.8,
                label=r"$\mathcal{O}(N^{-1/2})$ Monte Carlo Rate",
            )

        # Plot tolerance threshold if provided
        if tolerance_dict and greek in tolerance_dict:
            tol_val = tolerance_dict[greek]
            ax.axhline(
                tol_val,
                color="crimson",
                linestyle="-.",
                linewidth=1.5,
                alpha=0.85,
                label=f"Tolerance ({tol_val:g})",
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sample Size $N$ (Number of Simulations)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{product}\nMalliavin Convergence — {GREEK_LABELS.get(greek, greek)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=True, fontsize=8.5)

    plt.tight_layout()
    return fig


def plot_comparative_convergence_vs_N(
    fdmc_metrics_df: pd.DataFrame,
    malliavin_metrics_df: pd.DataFrame,
    product: str,
    greeks: list[str] = ["delta", "gamma", "vega"],
    metric: str = "abs_error",
    fdmc_eps: float | dict[str, float] | str = FDMC_FIXED_EPS,
    tolerance_dict: dict[str, float] | None = None,
    fournier_metrics_df: pd.DataFrame | None = None,
    figsize: tuple[int, int] = (18, 5.5),
) -> plt.Figure:
    """Plots comparative log-log convergence vs N for FDMC, Standard Malliavin, and Fournié Malliavin."""
    from Utilities.metrics import extract_fixed_or_optimal_eps

    fdmc_sub = fdmc_metrics_df[fdmc_metrics_df["Product"] == product]
    mal_sub = malliavin_metrics_df[malliavin_metrics_df["Product"] == product].sort_values("N")
    four_sub = (
        fournier_metrics_df[fournier_metrics_df["Product"] == product].sort_values("N")
        if fournier_metrics_df is not None
        else None
    )

    opt_fdmc = extract_fixed_or_optimal_eps(fdmc_sub, fdmc_eps=fdmc_eps, criterion=metric)
    ylabel = "Absolute Error" if metric == "abs_error" else "Relative Error (%)"

    is_opt = isinstance(fdmc_eps, str) and fdmc_eps.lower() == "optimal"
    fdmc_curve_label = r"FDMC (Tuned $\varepsilon^*$ Envelope)" if is_opt else (f"FDMC (Fixed $\\varepsilon={fdmc_eps}$)" if isinstance(fdmc_eps, (int, float)) else "FDMC (Fixed $\\varepsilon$)")

    fig, axes = plt.subplots(1, len(greeks), figsize=figsize, sharey=False)
    if len(greeks) == 1:
        axes = [axes]

    for ax, greek in zip(axes, greeks):
        err_col = f"{greek}_{metric}"

        # 1. Plot Standard Malliavin
        valid_mal = mal_sub[mal_sub[err_col] > 0]
        ax.plot(
            valid_mal["N"],
            valid_mal[err_col],
            marker="s",
            markersize=7,
            linewidth=2.4,
            color="#2a9d8f",
            label="Std Malliavin (Exact Weights)",
            zorder=4,
        )

        # 2. Plot Fournié Localized Malliavin if available
        if four_sub is not None:
            valid_four = four_sub[four_sub[err_col] > 0]
            ax.plot(
                valid_four["N"],
                valid_four[err_col],
                marker="D",
                markersize=6,
                linewidth=2.4,
                color="#3d5a80",
                label="Fournié Malliavin (Localized)",
                zorder=5,
            )

        # 3. Plot FDMC at chosen (fixed or optimal) eps
        opt_g = opt_fdmc[opt_fdmc["Greek"] == greek].sort_values("N")
        if not opt_g.empty:
            ax.plot(
                opt_g["N"],
                opt_g[f"min_{metric}" if metric == "abs_error" else "min_rel_error_pct"],
                marker="o",
                markersize=6,
                linewidth=2.0,
                linestyle="--",
                color="#e76f51",
                label=fdmc_curve_label,
                zorder=3,
            )

        # 4. Plot a coarse eps (e.g. 0.1) if different from target eps to illustrate bias saturation plateau
        if not (isinstance(fdmc_eps, (int, float)) and np.isclose(fdmc_eps, 0.1)):
            coarse = fdmc_sub[fdmc_sub["eps"] == 0.1].sort_values("N")
            if not coarse.empty and (coarse[err_col] > 0).any():
                ax.plot(
                    coarse["N"],
                    coarse[err_col],
                    marker="^",
                    markersize=5,
                    linewidth=1.2,
                    linestyle=":",
                    color="#f4a261",
                    alpha=0.75,
                    label=r"FDMC ($\varepsilon = 10^{-1}$ Bias Plateau)",
                    zorder=2,
                )

        # 5. Plot target tolerance
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

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Sample Size $N$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{product}\n{GREEK_LABELS.get(greek, greek)}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=True, fontsize=8)

    plt.tight_layout()
    return fig


def plot_comparative_greeks_bars(
    fdmc_metrics_df: pd.DataFrame,
    malliavin_metrics_df: pd.DataFrame,
    product: str,
    n_choice: int | None = None,
    fdmc_eps: float | dict[str, float] | str = FDMC_FIXED_EPS,
    fournier_metrics_df: pd.DataFrame | None = None,
    figsize: tuple[int, int] = (13, 5.5),
) -> plt.Figure:
    """Plots a 3-way or 4-way grouped bar chart: Benchmark vs FDMC vs Malliavin (vs Fournié)."""
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

    fig, ax = plt.subplots(figsize=figsize)
    greeks = sub["Greek"].tolist()
    labels = [GREEK_LABELS.get(g, g) for g in greeks]
    x = np.arange(len(sub))

    has_four = "Fournié Malliavin" in sub.columns
    width = 0.20 if has_four else 0.26

    bench_vals = sub["Benchmark"].tolist()
    fdmc_cols = [c for c in sub.columns if c.startswith("FDMC (")]
    fdmc_col = fdmc_cols[0] if fdmc_cols else "FDMC"
    fdmc_vals = sub[fdmc_col].tolist()
    mal_vals = sub["Std Malliavin"].tolist()

    rects1 = ax.bar(x - (1.5 * width if has_four else width), bench_vals, width, label="Analytical Benchmark", color="#2b5c8f")
    rects2 = ax.bar(x - (0.5 * width if has_four else 0), fdmc_vals, width, label=f"{fdmc_col} ($N={n_choice:_}$)" if "N=" not in fdmc_col else fdmc_col, color="#e07a5f")
    rects3 = ax.bar(x + (0.5 * width if has_four else width), mal_vals, width, label=f"Std Malliavin ($N={n_choice:_}$)", color="#2a9d8f")

    if has_four:
        four_vals = sub["Fournié Malliavin"].tolist()
        rects4 = ax.bar(x + 1.5 * width, four_vals, width, label=f"Fournié Malliavin ($N={n_choice:_}$)", color="#3d5a80")

    ax.set_ylabel("Greek Value")
    ax.set_title(f"{product}: Benchmark vs. Simulation Sensitivity Comparison ($N={n_choice:_}$)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend(frameon=True, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def plot_fournier_cutoff_comparison(
    cutoff_results_dict: dict[str, dict[str, float]],
    benchmarks: dict[str, float],
    product: str = "Call European ATM",
    figsize: tuple[int, int] = (15, 5),
) -> plt.Figure:
    """Plots comparative sensitivity absolute error across the 5 smoothing cutoff functions."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    greeks = ["delta", "gamma", "vega"]
    ctypes = list(cutoff_results_dict.keys())
    x = np.arange(len(ctypes))

    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

    for ax, greek in zip(axes, greeks):
        bench_val = benchmarks[greek]
        vals = [cutoff_results_dict[c][greek] for c in ctypes]
        errs = [abs(v - bench_val) for v in vals]

        bars = ax.bar(x, errs, color=palette[:len(ctypes)], alpha=0.85, edgecolor="black", linewidth=1.1)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(ctypes, rotation=15, ha="right", fontsize=9.5)
        ax.set_ylabel("Absolute Error vs. Benchmark")
        ax.set_title(f"{GREEK_LABELS.get(greek, greek)}\nBenchmark = {bench_val:.4f}")
        ax.grid(axis="y", which="both", alpha=0.3)

        for bar, err in zip(bars, errs):
            val_str = f"{err:.2e}" if err < 0.001 else f"{err:.4f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                err * 1.15,
                val_str,
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    fig.suptitle(f"{product}: Fournié Malliavin Accuracy Across Cutoff Smoothing Functions ($N=10^5, \\delta=10$)", fontsize=13)
    plt.tight_layout()
    return fig


def plot_fournier_delta_sensitivity(
    delta_results_dict: dict[float, dict[str, float]],
    benchmarks: dict[str, float],
    product: str = "Call European ATM",
    figsize: tuple[int, int] = (15, 5),
) -> plt.Figure:
    """Plots Greek absolute error vs smoothing half-width delta from strike."""
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=False)
    greeks = ["delta", "gamma", "vega"]
    deltas = sorted(delta_results_dict.keys())

    line_colors = {"delta": "#1f77b4", "gamma": "#e76f51", "vega": "#2a9d8f"}

    for ax, greek in zip(axes, greeks):
        bench_val = benchmarks[greek]
        errs = [abs(delta_results_dict[d][greek] - bench_val) for d in deltas]

        ax.plot(
            deltas,
            errs,
            marker="o",
            markersize=8,
            linewidth=2.5,
            color=line_colors.get(greek, "blue"),
            label=f"Fournié {GREEK_LABELS.get(greek, greek)}",
        )

        for d, err in zip(deltas, errs):
            val_str = f"{err:.2e}" if err < 0.001 else f"{err:.4f}"
            ax.annotate(
                val_str,
                (d, err),
                textcoords="offset points",
                xytext=(0, 7),
                ha="center",
                fontsize=8,
            )

        ax.set_yscale("log")
        ax.set_xlabel(r"Smoothing Half-Width $\delta$ from Strike")
        ax.set_ylabel("Absolute Error vs. Benchmark")
        ax.set_title(f"{GREEK_LABELS.get(greek, greek)}\nBenchmark = {bench_val:.4f}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(frameon=True)

    fig.suptitle(f"{product}: Fournié Malliavin Sensitivity to Smoothing Half-Width $\\delta$ ($N=10^5$, Gaussian)", fontsize=13)
    plt.tight_layout()
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
                psi, psi_prime, psi_double_prime = get_cutoff(S_vals, K, delta=d, cutoff_type=ctype, return_second_deriv=True)
            else:
                psi, psi_prime = get_cutoff(S_vals, K, delta=d, cutoff_type=ctype)
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
            sub = grid_df[grid_df["cutoff_type"] == ctype].sort_values("delta_strike")
            ax.plot(
                sub["delta_strike"],
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
            sub = grid_df[grid_df["cutoff_type"] == ctype].sort_values("delta_strike")
            ax.plot(
                sub["delta_strike"],
                sub[var_col],
                marker=markers.get(ctype, "o"),
                color=colors.get(ctype, "gray"),
                label=ctype.capitalize(),
                linewidth=2.0,
                markersize=6.5,
            )

            # Mark the minimum variance point
            min_idx = sub[var_col].idxmin()
            opt_d = sub.loc[min_idx, "delta_strike"]
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
    pivot = grid_df.pivot(index="cutoff_type", columns="delta_strike", values=col)

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


