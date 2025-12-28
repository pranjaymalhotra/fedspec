"""
Plots package initialization.
"""
from .plot_frobenius_gap import (
    plot_frobenius_gap,
    plot_frobenius_gap_multiple_alpha,
    plot_rank_evolution
)
from .plot_accuracy import (
    plot_accuracy_comparison,
    plot_accuracy_vs_communication,
    plot_all_metrics
)

__all__ = [
    "plot_frobenius_gap",
    "plot_frobenius_gap_multiple_alpha",
    "plot_rank_evolution",
    "plot_accuracy_comparison",
    "plot_accuracy_vs_communication",
    "plot_all_metrics"
]
