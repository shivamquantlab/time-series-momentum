"""Simple time-series momentum with no use of the forecast return."""

import numpy as np


def momentum_positions(returns: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Position for return i uses only returns strictly before i."""
    values = np.asarray(returns, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("returns must be a finite one-dimensional array")
    if (
        isinstance(lookback, (bool, np.bool_))
        or not isinstance(lookback, (int, np.integer))
        or lookback < 1
    ):
        raise ValueError("lookback must be a positive integer")
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
    if values.ndim != 1 or values.size == 0 or weights.shape != values.shape:
        raise ValueError("returns and positions must be non-empty equal-length vectors")
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (values <= -1).any():
        raise ValueError("returns and positions must be finite; simple returns must exceed -100%")
    try:
        cost = float(cost_bps)
    except (TypeError, ValueError) as error:
        raise ValueError("cost_bps must be finite and non-negative") from error
    if (np.abs(weights) > 1).any() or not np.isfinite(cost) or cost < 0:
        raise ValueError("positions must be in [-1, 1] and cost_bps finite and non-negative")
    turnover = np.abs(np.diff(np.concatenate(([0.0], weights))))
    return weights * values - turnover * cost / 10_000.0

