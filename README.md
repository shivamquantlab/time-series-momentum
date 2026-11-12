# Time-Series Momentum

## Research question
Does a sign-of-trailing-return rule demonstrate the difference between a lagged signal and a same-period look-ahead signal?

## Methodology and formulas
For lookback L, position p_t=sign(sum(r_(t-L)...r_(t-1))) is set before applying it to r_t. Net return subtracts turnover times the assumed cost in basis points divided by 10,000. The first L observations remain flat because there is not enough history.

## Assumptions and limitations
- The experiment creates five synthetic positive/negative-drift blocks plus seeded Gaussian noise; it is not historical market data.
- Positions are bounded to -1, 0, or 1 and use no volatility scaling or cross-asset portfolio construction.
- A 2 bp one-way turnover cost is illustrative. Borrow, financing, market impact, taxes, and execution delay are excluded.
- No survivorship issue arises in the single synthetic series, but a real multi-asset study needs point-in-time membership and delisting returns.
- Synthetic wealth curves are mechanics diagnostics, not investment performance.

## Reproducibility
With Python 3.11+, run python -m pip install -e ".[test]", then python -m pytest and momentum-demo --seed 89. Reports include a regenerated chart and summary; all returns are generated locally.

