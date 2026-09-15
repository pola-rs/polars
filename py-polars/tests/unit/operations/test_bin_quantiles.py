from __future__ import annotations

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
