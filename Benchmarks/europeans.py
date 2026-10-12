"""Analytical reference Greeks for the European products."""

from Methods.Analytic.europeans import (
    greeks_european_call,
    greeks_european_call_terminal_knockout,
    greeks_european_digital_call,
)


def benchmark_greeks_european_call(params: dict[str, float]) -> dict[str, float]:
    return greeks_european_call(params)


def benchmark_greeks_european_digital(params: dict[str, float]) -> dict[str, float]:
    return greeks_european_digital_call(params)


def benchmark_greeks_european_call_terminal_knockout(params: dict[str, float]) -> dict[str, float]:
    return greeks_european_call_terminal_knockout(params)
