"""Seeded synthetic trend regimes and a strictly lagged momentum demo."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .signals import momentum_positions, net_strategy_returns


def run(output_dir: Path, seed: int = 89) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    drift = np.tile(np.concatenate((np.full(60, 0.0008), np.full(40, -0.0006))), 5)
    returns = drift + rng.normal(0, 0.01, drift.size)
    positions = momentum_positions(returns, lookback=20)
    net = net_strategy_returns(returns, positions, cost_bps=2.0)
    gross = positions * returns
    wealth_net = np.cumprod(1.0 + net)
    wealth_gross = np.cumprod(1.0 + gross)
    summary = {
        "seed": seed,
        "lookback_periods": 20,
        "cost_bps_per_turnover_unit": 2.0,
        "observations": int(returns.size),
        "synthetic_gross_terminal_wealth": float(wealth_gross[-1]),
        "synthetic_net_terminal_wealth": float(wealth_net[-1]),
        "active_fraction": float(np.mean(positions != 0)),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(wealth_gross, label="Gross", color="#94a3b8")
    ax.plot(wealth_net, label="After assumed turnover cost", color="#be123c")
    ax.set(title="Synthetic time-series momentum mechanics", xlabel="Synthetic return interval", ylabel="Growth of 1 unit")
    ax.legend(frameon=False)
    ax.text(0.01, 0.98, "SYNTHETIC • not market data or investment performance",
            transform=ax.transAxes, va="top", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "synthetic_momentum.png", dpi=150)
    plt.close(fig)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=89)
    args = parser.parse_args()
    run(args.output_dir, seed=args.seed)
    print(f"Wrote synthetic momentum report to {args.output_dir}")


if __name__ == "__main__":
    main()

