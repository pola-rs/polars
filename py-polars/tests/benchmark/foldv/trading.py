"""Replay sequential, non-overlapping trades using the available bankroll.

Each row supplies an entry price and the realized fractional return of a completed
trade. Buy whole shares using at most 25% of the current bankroll. Skip trades
below $100 notional; executed trades pay $1 plus 0.1% commission in total. After
each trade, profit above $2,000 is withdrawn, so the bankroll stays in the range
where its size limits the next trade. This models capital allocation, not signal
generation or overlapping positions.

Rows must be in execution order; inputs must be finite and non-null, prices
positive, and returns >= -1. State/output is Float64 USD, initially $1,000.
Position size and eligibility depend on previous realized gains and losses, and
withdrawals remove gains as they happen, so neither cum_sum nor cum_prod computes
this state sequence directly.

Related implementation: Backtrader's cash-dependent sizers and commission model:
https://www.backtrader.com/docu/sizers/sizers/
https://www.backtrader.com/docu/commission-schemes/commission-schemes/
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

INITIAL_STATE = 1_000.0
WITHDRAWAL_THRESHOLD = 2_000.0


def foldv() -> pl.Expr:
    def trade(cash: pl.Expr) -> pl.Expr:
        notional = (cash * 0.25 / pl.col("price")).floor() * pl.col("price")
        traded = cash + notional * pl.col("return") - 1.0 - notional * 0.001
        return (
            pl.when(notional >= 100.0)
            .then(pl.min_horizontal(traded, WITHDRAWAL_THRESHOLD))
            .otherwise(cash)
        )

    return pl.foldv(trade, initial_state=INITIAL_STATE).alias("cash")


def best_effort() -> pl.Expr:
    def replay(columns: Sequence[pl.Series]) -> pl.Series:
        cash = INITIAL_STATE
        result = []
        for price, trade_return in zip(*columns, strict=True):
            notional = math.floor(cash * 0.25 / price) * price
            if notional >= 100.0:
                traded = cash + notional * trade_return - 1.0 - notional * 0.001
                cash = min(traded, WITHDRAWAL_THRESHOLD)
            result.append(cash)
        return pl.Series(result, dtype=pl.Float64)

    return pl.map_batches(
        ["price", "return"], replay, return_dtype=pl.Float64, is_elementwise=False
    ).alias("cash")


def sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "price": [10.0, 10.0, 10.0, 300.0, 10.0],
            "return": [0.5, 0.5, -0.2, 0.5, 3.0],
        }
    )


def generate(rows: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {
            "price": rng.uniform(5.0, 200.0, rows),
            "return": rng.uniform(-0.10, 0.12, rows),
        }
    )


if __name__ == "__main__":
    print(sample().with_columns(foldv(), best_effort().alias("baseline")))
