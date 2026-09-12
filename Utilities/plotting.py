"""Plotting and visualization utilities for Greeks benchmarking analysis."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
