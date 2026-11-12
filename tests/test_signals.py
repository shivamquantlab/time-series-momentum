import numpy as np
import pytest

from time_series_momentum.signals import momentum_positions, net_strategy_returns


def test_position_for_period_uses_only_strictly_prior_returns():
    returns = np.array([0.01, 0.02, -0.5, 0.03])
    positions = momentum_positions(returns, lookback=2)
    assert positions[2] == 1.0
    altered_current = returns.copy()
    altered_current[2] = 10.0
    assert momentum_positions(altered_current, lookback=2)[2] == positions[2]


def test_turnover_cost_is_deducted_from_net_returns():
    actual = net_strategy_returns(
        np.array([0.10, 0.10, -0.10]), np.array([1.0, -1.0, -1.0]), cost_bps=5
    )
    np.testing.assert_allclose(actual, [0.0995, -0.101, 0.1])


def test_lookback_must_be_positive():
    with pytest.raises(ValueError, match="lookback"):
        momentum_positions(np.array([0.1]), lookback=0)

