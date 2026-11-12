"""Simple time-series momentum with no use of the forecast return."""

import numpy as np


def momentum_positions(returns: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Position for return i uses only returns strictly before i."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("returns must be a finite one-dimensional array")
    if lookback < 1:
        raise ValueError("lookback must be positive")
    positions = np.zeros(values.size, dtype=float)
    for index in range(lookback, values.size):
        positions[index] = float(np.sign(values[index - lookback:index].sum()))
    return positions


def net_strategy_returns(
    returns: np.ndarray,
    positions: np.ndarray,
    *,
    cost_bps: float = 2.0,
) -> np.ndarray:
    """Apply already-lagged positions and a one-way turnover charge."""
    values = np.asarray(returns, dtype=float)
    weights = np.asarray(positions, dtype=float)
    if values.ndim != 1 or weights.shape != values.shape:
        raise ValueError("returns and positions must be equal-length vectors")
    if not np.isfinite(values).all() or not np.isfinite(weights).all():
        raise ValueError("inputs must be finite")
    if (np.abs(weights) > 1).any() or cost_bps < 0:
        raise ValueError("positions must be in [-1, 1] and cost_bps non-negative")
    turnover = np.abs(np.diff(np.concatenate(([0.0], weights))))
    return weights * values - turnover * cost_bps / 10_000.0

