from __future__ import annotations

import math
import random
from bisect import bisect_left, bisect_right

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal


def test_bin_quantiles() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    result = s.bin_quantiles([0.25, 0.75], labels=["a", "b", "c"])

    expected = pl.Series("a", ["a", "b", "b", "c", "c"], dtype=pl.Enum(["a", "b", "c"]))
    assert_series_equal(result, expected)


def test_bin_quantiles_uniform() -> None:
    s = pl.Series("a", [-2, -1, 0, 1, 2])

    result = s.bin_quantiles(2, labels=["low", "high"])

    assert result.to_list() == ["low", "low", "high", "high", "high"]


def test_bin_quantiles_uniform_is_computed_exactly() -> None:
    s = pl.Series("a", list(range(91)))

    result = s.bin_quantiles(10, labels=False, include_intervals=True)
    breaks = sorted(set(result.struct["right"].drop_nulls().to_list()))

    # Positions are `((i + 1) * (len - 1)) / n_bins` in integer arithmetic. Expanding to
    # `(i + 1) / n_bins` probabilities instead would put breakpoint 7 at 62, because
    # `7 / 10` is 0.69999999999999996 and `0.7 * 90` floors to 62.
    assert breaks == [9 * (i + 1) for i in range(9)]
    assert breaks[6] == 63


def test_bin_quantiles_uniform_differs_from_expanded_probabilities() -> None:
    s = pl.Series("a", list(range(91)))

    uniform = s.bin_quantiles(10, labels=False)
    expanded = s.bin_quantiles([i / 10 for i in range(1, 10)], labels=False)

    # Deliberate: the two forms are not required to agree, because only the integer
    # form can be evaluated without going through inexact probabilities.
    assert uniform.to_list() != expanded.to_list()


def test_bin_quantiles_include_intervals() -> None:
    s = pl.Series("a", [1, 2, 3, 4, 5])

    result = s.bin_quantiles([0.25, 0.75], labels=False, include_intervals=True)

    # floor(0.25 * 4) == 1 and floor(0.75 * 4) == 3, so the breakpoints are the values
    # at those sorted positions: 2 and 4.
    expected = pl.Series(
        "a",
        [
            {"bin": 0, "left": None, "right": 2},
            {"bin": 1, "left": 2, "right": 4},
            {"bin": 1, "left": 2, "right": 4},
            {"bin": 2, "left": 4, "right": None},
            {"bin": 2, "left": 4, "right": None},
        ],
        dtype=pl.Struct({"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}),
    )
    assert_series_equal(result, expected)


@pytest.mark.parametrize("quantiles", [[-0.1], [1.5]])
def test_bin_quantiles_out_of_range_raises(quantiles: list[float]) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match=r"between 0\.0 and 1\.0"):
        lf.select(pl.col("a").bin_quantiles(quantiles, labels=False)).collect_schema()


@pytest.mark.parametrize("n_bins", [1, 2, 4, 7, 100, 300])
@pytest.mark.parametrize("right_closed", [False, True])
def test_bin_quantiles_matches_sorted_oracle(n_bins: int, right_closed: bool) -> None:
    rng = random.Random(n_bins)
    values = [None if rng.random() < 0.1 else rng.randint(-5, 5) for _ in range(200)]
    s = pl.concat(
        [pl.Series("a", values[i : i + 40]) for i in range(0, 200, 40)], rechunk=False
    )
    ordered = sorted(v for v in values if v is not None)
    span = len(ordered) - 1
    breaks = [ordered[i * span // n_bins] for i in range(1, n_bins)]
    find = bisect_left if right_closed else bisect_right
    expected = [None if v is None else find(breaks, v) for v in values]

    result = s.bin_quantiles(
        n_bins, labels=False, include_intervals=True, right_closed=right_closed
    )

    assert result.struct["bin"].to_list() == expected
    assert result.struct["right"].to_list() == [
        None if b is None else (breaks + [None])[b] for b in expected
    ]


def test_bin_quantiles_decimal_matches_physical() -> None:
    s = pl.Series("a", [3, None, 1, 4, 1, 5, 9, 2, 6], dtype=pl.Int64)
    expected = s.bin_quantiles([0.3, 0.6], labels=False)
    assert_series_equal(
        s.cast(pl.Decimal(10, 1)).bin_quantiles([0.3, 0.6], labels=False), expected
    )


@pytest.mark.parametrize("right_closed", [False, True])
def test_bin_quantiles_floats_explicit_matches_sorted_oracle(
    right_closed: bool,
) -> None:
    rng = random.Random(7)
    specials = [math.nan, -0.0, 0.0, math.inf, -math.inf]
    values = [
        None
        if rng.random() < 0.1
        else rng.choice(specials)
        if rng.random() < 0.3
        else rng.uniform(-5, 5)
        for _ in range(300)
    ]
    s = pl.concat(
        [pl.Series("a", values[i : i + 50]) for i in range(0, 300, 50)], rechunk=False
    )
    probs = [0.0, 0.1, 0.5, 0.9, 0.97, 1.0]

    def key(v: float) -> tuple[bool, float]:
        # Polars' total order: NaN is the largest value and equal to itself.
        return (True, 0.0) if math.isnan(v) else (False, v)

    ordered = sorted((v for v in values if v is not None), key=key)
    span = len(ordered) - 1
    breaks = [ordered[math.floor(span * q)] for q in probs]
    find = bisect_left if right_closed else bisect_right
    keys = [key(b) for b in breaks]
    expected = [None if v is None else find(keys, key(v)) for v in values]

    result = s.bin_quantiles(probs, labels=False, right_closed=right_closed)

    assert result.to_list() == expected
