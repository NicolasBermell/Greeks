"""Utilities package for Greek sensitivity benchmark and analysis."""

from Utilities.metrics import (
    compute_error_metrics,
    compute_summary_table,
    extract_optimal_eps,
    find_min_acceptable_N,
    generate_error_pivot,
    load_simulation_results,
)
from Utilities.plotting import (
    plot_convergence_by_eps,
    plot_convergence_vs_N,
    plot_cross_product_comparison,
    plot_epsilon_sensitivity,
    plot_error_heatmaps,
    plot_greeks_comparison_bars,
)
from Utilities.profiling import (
    benchmark_fdmc_execution_time,
    benchmark_per_greek_execution_time,
    construct_pareto_dataset,
    evaluate_fixed_cpu_budget,
    plot_budget_accuracy_bars,
    plot_pareto_efficiency,
)

__all__ = [
    "load_simulation_results",
    "compute_error_metrics",
    "extract_optimal_eps",
    "generate_error_pivot",
    "compute_summary_table",
    "find_min_acceptable_N",
    "plot_epsilon_sensitivity",
    "plot_convergence_vs_N",
    "plot_convergence_by_eps",
    "plot_error_heatmaps",
    "plot_greeks_comparison_bars",
    "plot_cross_product_comparison",
    "benchmark_fdmc_execution_time",
    "benchmark_per_greek_execution_time",
    "construct_pareto_dataset",
    "evaluate_fixed_cpu_budget",
    "plot_budget_accuracy_bars",
    "plot_pareto_efficiency",
]
