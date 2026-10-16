"""Allocate stock to all-or-nothing orders, with deliveries before each order.

Inputs are non-null, nonnegative Int64 unit counts in event order. Initial stock
is 10 units; available stock must fit Int64. Rejected orders consume no stock and
are not backlogged. Output is the remaining stock after each event.

Rejecting an order changes which later orders can be fulfilled. A cumulative sum
followed by clipping incorrectly consumes stock for rejected orders.

Related inventory replay implementation (different fulfilment policy):
https://stockpyl.readthedocs.io/en/latest/tutorial/tutorial_sim.html
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

INITIAL_STATE = 10


def foldv() -> pl.Expr:
    def allocate(stock: pl.Expr, demand: pl.Expr) -> pl.Expr:
        available = stock + pl.col("delivery")
        fulfilled = (demand <= available).cast(pl.Int64)
        return available - demand * fulfilled

    return pl.col("demand").foldv(allocate, initial_state=INITIAL_STATE).alias("stock")


def best_effort() -> pl.Expr:
    def replay(columns: Sequence[pl.Series]) -> pl.Series:
        stock = INITIAL_STATE
        result = []
        for delivery, demand in zip(*columns, strict=True):
            stock += delivery
            if demand <= stock:
                stock -= demand
            result.append(stock)
        return pl.Series(result, dtype=pl.Int64)

    return pl.map_batches(
        ["delivery", "demand"], replay, return_dtype=pl.Int64, is_elementwise=False
    ).alias("stock")


def sample() -> pl.DataFrame:
    return pl.DataFrame({"delivery": [0, 0, 4, 0, 0], "demand": [7, 8, 5, 2, 1]})


def generate(rows: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    return pl.DataFrame(
        {"delivery": rng.integers(0, 6, rows), "demand": rng.integers(1, 12, rows)}
    )


if __name__ == "__main__":
    print(sample().with_columns(foldv(), best_effort().alias("baseline")))
