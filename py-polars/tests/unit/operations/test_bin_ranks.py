from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.testing import assert_series_equal


def test_bin_ranks() -> None:
    s = pl.Series("a", [10, 20, 30, 40])

    result = s.bin_ranks([0.5], labels=["low", "high"])

    expected = pl.Series(
        "a", ["low", "low", "high", "high"], dtype=pl.Enum(["low", "high"])
    )
    assert_series_equal(result, expected)


def test_bin_ranks_fractions_give_requested_shares() -> None:
    s = pl.Series("a", list(range(10)))

    result = s.bin_ranks([0.2, 0.5, 0.8], labels=False)

    # Bin `i` holds `ranks[i + 1] - ranks[i]` of the values: 20%, 30%, 30%, 20%.
    assert result.to_list() == [0, 0, 1, 1, 1, 2, 2, 2, 3, 3]


def test_bin_ranks_uniform_sizes() -> None:
    s = pl.Series("a", list(range(14)))

    result = s.bin_ranks(4, labels=False)

    # 14 over 4 bins is 4 + 4 + 3 + 3: the earlier bins take the remainder.
    assert result.value_counts().sort("a")["count"].to_list() == [4, 4, 3, 3]


@pytest.mark.parametrize(
    ("length", "n_bins", "expected"),
    [
        (14, 4, [4, 4, 3, 3]),
        (10, 3, [4, 3, 3]),
        (12, 4, [3, 3, 3, 3]),
        (3, 2, [2, 1]),
    ],
)
def test_bin_ranks_uniform_sizes_parametrized(
    length: int, n_bins: int, expected: list[int]
) -> None:
    s = pl.Series("a", list(range(length)))

    result = s.bin_ranks(n_bins, labels=False)

    assert result.value_counts().sort("a")["count"].to_list() == expected


def test_bin_ranks_splits_ties() -> None:
    # Every value is identical, so no value-based binning could separate them at all.
    s = pl.Series("a", [7] * 14)

    result = s.bin_ranks(4, labels=False)

    assert result.value_counts().sort("a")["count"].to_list() == [4, 4, 3, 3]


def test_bin_ranks_boundaries_are_values_not_ranks() -> None:
    s = pl.Series("a", [10, 20, 30, 40])

    result = s.bin_ranks(2, labels=False, include_intervals=True)

    expected = pl.Series(
        "a",
        [
            {"bin": 0, "left": None, "right": 30},
            {"bin": 0, "left": None, "right": 30},
            {"bin": 1, "left": 30, "right": None},
            {"bin": 1, "left": 30, "right": None},
        ],
        dtype=pl.Struct({"bin": pl.UInt32, "left": pl.Int64, "right": pl.Int64}),
    )
    assert_series_equal(result, expected)


def test_bin_ranks_has_no_right_closed() -> None:
    s = pl.Series("a", [1, 2])

    # Bins are delimited by position, so there is no value boundary to close on.
    with pytest.raises(TypeError, match="unexpected keyword argument 'right_closed'"):
        s.bin_ranks(2, labels=False, right_closed=True)  # type: ignore[call-arg]


def test_bin_ranks_trailing_fraction_of_one_gives_an_empty_bin() -> None:
    s = pl.Series("a", [1, 2, 3, 4])

    result = s.bin_ranks([0.5, 1.0], labels=False, include_intervals=True)

    # `1.0` puts the second cut past the last element, so the third bin is empty and
    # the boundary value for it is null.
    assert result.struct["bin"].to_list() == [0, 0, 1, 1]
    assert result.struct["right"].to_list() == [3, 3, None, None]


def test_bin_ranks_more_bins_than_rows() -> None:
    s = pl.Series("a", [1, 2, 3])

    assert s.bin_ranks(5, labels=False).to_list() == [0, 1, 2]


@pytest.mark.parametrize("ranks", [[-0.1], [1.5]])
def test_bin_ranks_out_of_range_raises(ranks: list[float]) -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    with pytest.raises(ComputeError, match=r"between 0\.0 and 1\.0"):
        lf.select(pl.col("a").bin_ranks(ranks, labels=False)).collect_schema()


def test_bin_ranks_enum_uses_declaration_order() -> None:
    dtype = pl.Enum(["zebra", "apple", "mango"])
    s = pl.Series("a", ["mango", "zebra", "apple"], dtype=dtype)

    # Positions in declaration order: zebra is 0, apple is 1, mango is 2.
    assert s.bin_ranks(3, labels=False).to_list() == [2, 0, 1]


def test_bin_ranks_categorical_uses_lexical_order() -> None:
    s = pl.Series("a", ["mango", "zebra", "apple"], dtype=pl.Categorical)

    # Positions in lexical order: apple is 0, mango is 1, zebra is 2.
    assert s.bin_ranks(3, labels=False).to_list() == [1, 2, 0]
