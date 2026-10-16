"""Compare battery capacities against the same household solar and demand history.

Candidates have 5, 10, 15, and 20 kWh capacity, each initially storing 2.5 kWh.
Each fold carries its own state, with 5 kW charge/discharge limits on the household
side. Each direction is 90% efficient. Intervals are ordered, non-overlapping, and
have positive duration.
Inputs are finite, non-null Float64: solar/load in kW and duration in hours.

Both implementations calculate the unconstrained energy change with native
Polars expressions, then carry stored energy through the lower/upper bounds.
Clipping cum_sum is incorrect: surplus lost at capacity cannot cover a later
deficit, and unmet demand cannot be charged against future energy.

Related implementation: SAM's behind-the-meter battery dispatch model:
https://samrepo.nlr.gov/help/battery_storage_btm.html
"""

from __future__ import annotations

import numpy as np

import polars as pl

INITIAL_STATE = 2.5
CAPACITY = 10.0
CAPACITIES = (5.0, 10.0, 15.0, 20.0)
NAMES = [f"capacity_{capacity:g}_kwh" for capacity in CAPACITIES]


def energy_change() -> pl.Expr:
    net = pl.col("solar_kw") - pl.col("load_kw")
    # Signed power: positive charges the battery; negative discharges it.
    power = net.clip(0.0, 5.0) * 0.9 + net.clip(-5.0, 0.0) / 0.9
    return power * pl.col("hours")


def foldv(capacity: float = CAPACITY) -> pl.Expr:
    return (
        energy_change()
        .foldv(
            lambda stored, change: (stored + change).clip(0.0, capacity),
            initial_state=INITIAL_STATE,
        )
        .alias("stored_kwh")
    )


def best_effort(capacity: float = CAPACITY) -> pl.Expr:
    def replay(changes: pl.Series) -> pl.Series:
        stored = INITIAL_STATE
        result = []
        for change in changes:
            stored = min(max(stored + change, 0.0), capacity)
            result.append(stored)
        return pl.Series(result, dtype=pl.Float64)

    return (
        energy_change()
        .map_batches(replay, return_dtype=pl.Float64, is_elementwise=False)
        .alias("stored_kwh")
    )


def sample() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "solar_kw": [10.0, 10.0, 0.0, 0.0, 2.0],
            "load_kw": [0.0, 0.0, 9.0, 9.0, 0.0],
            "hours": [1.0] * 5,
        }
    )


def generate(rows: int, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    daylight = np.maximum(np.sin((np.arange(rows) % 96 - 24) * np.pi / 48), 0.0)
    return pl.DataFrame(
        {
            "solar_kw": daylight * rng.uniform(4.0, 9.0, rows),
            "load_kw": rng.uniform(0.5, 4.0, rows),
            "hours": np.full(rows, 0.25),
        }
    )


def foldv_sizing() -> list[pl.Expr]:
    """Compare four capacities with independent folds sharing energy preprocessing."""
    return [
        foldv(capacity).alias(name)
        for capacity, name in zip(CAPACITIES, NAMES, strict=True)
    ]


def best_effort_sizing() -> pl.Expr:
    """Update all four states in one pass through one Polars batch UDF."""
    cap_a, cap_b, cap_c, cap_d = CAPACITIES

    def replay(changes: pl.Series) -> pl.Series:
        a = b = c = d = INITIAL_STATE
        out_a: list[float] = []
        out_b: list[float] = []
        out_c: list[float] = []
        out_d: list[float] = []
        for change in changes:
            a = min(max(a + change, 0.0), cap_a)
            b = min(max(b + change, 0.0), cap_b)
            c = min(max(c + change, 0.0), cap_c)
            d = min(max(d + change, 0.0), cap_d)
            out_a.append(a)
            out_b.append(b)
            out_c.append(c)
            out_d.append(d)
        return pl.DataFrame(
            dict(zip(NAMES, [out_a, out_b, out_c, out_d], strict=True)),
            schema=dict.fromkeys(NAMES, pl.Float64),
        ).to_struct()

    return (
        energy_change()
        .map_batches(
            replay,
            return_dtype=pl.Struct(dict.fromkeys(NAMES, pl.Float64)),
            is_elementwise=False,
        )
        .struct.unnest()
    )


if __name__ == "__main__":
    print(sample().select(foldv_sizing()))
