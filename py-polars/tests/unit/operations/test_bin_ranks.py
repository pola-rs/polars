from __future__ import annotations

import math
import random
from typing import Any

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


def _chunked(values: list[Any], dtype: pl.DataType) -> pl.Series:
    parts = [values[i : i + 40] for i in range(0, len(values), 40)]
    return pl.concat([pl.Series("a", p, dtype=dtype) for p in parts], rechunk=False)


def _check_against_ordinal_rank(s: pl.Series, n_bins: int) -> None:
    """Compares with the bins of the ordinal ranks, which break ties in input order."""
    n = s.len() - s.null_count()
    k, r = divmod(n, n_bins)
    cuts = pl.Series([i * k + min(i, r) for i in range(1, n_bins)], dtype=pl.UInt32)
    rank = s.rank("ordinal") - 1
    bins = pl.select(
        pl.when(rank.is_not_null()).then(cuts.search_sorted(rank, side="right"))
    ).to_series()
    boundaries = s.drop_nulls().sort().gather(cuts.filter(cuts < n))
    boundaries = pl.concat(
        [boundaries, pl.repeat(None, n_bins + 1, dtype=s.dtype, eager=True)]
    )

    result = s.bin_ranks(n_bins, labels=False, include_intervals=True)

    assert_series_equal(result.struct["bin"], bins.cast(pl.UInt32), check_names=False)
    assert_series_equal(
        result.struct["right"],
        boundaries.gather(bins.fill_null(n_bins)),
        check_names=False,
    )


@pytest.mark.parametrize("n_bins", [1, 2, 3, 4, 7, 50, 300])
@pytest.mark.parametrize("dtype", [pl.Int64(), pl.Float64()])
def test_bin_ranks_matches_ordinal_rank(n_bins: int, dtype: pl.DataType) -> None:
    rng = random.Random(n_bins)
    choices = [math.nan, -0.0, 0.0, 1.0, 2.5] if dtype.is_float() else range(6)
    values = [None if rng.random() < 0.1 else rng.choice(choices) for _ in range(200)]
    _check_against_ordinal_rank(_chunked(values, dtype), n_bins)


def test_bin_ranks_splits_ties_in_input_order() -> None:
    s = pl.Series("a", [5, 1, 5, 5, 0, 5, 9])

    # The sorted order is 0 1 5 5 5 5 9; the cuts at ranks 3 and 5 fall inside the run
    # of 5s, which are handed out in input order.
    result = s.bin_ranks([3 / 7, 5 / 7], labels=False)

    assert_series_equal(result, pl.Series("a", [0, 0, 1, 1, 0, 2, 2], dtype=pl.UInt32))


@pytest.mark.parametrize(
    "dtype",
    [
        pl.Date(),
        pl.Duration("ms"),
        pl.Decimal(10, 2),
        pl.Datetime("us", "Europe/Amsterdam"),
        pl.Time(),
    ],
)
def test_bin_ranks_temporal_and_decimal_match_physical(dtype: pl.DataType) -> None:
    rng = random.Random(0)
    ints = [None if rng.random() < 0.1 else rng.randint(0, 20) for _ in range(100)]
    s = pl.Series("a", ints, dtype=pl.Int64)

    result = s.cast(dtype).bin_ranks(6, labels=False)

    assert_series_equal(result, s.bin_ranks(6, labels=False))


@pytest.mark.parametrize("n_bins", [3, 1000])
def test_bin_ranks_large_matches_ordinal_rank(n_bins: int) -> None:
    # Large enough for the binning passes to split into parallel ranges.
    df = pl.select(v=pl.int_range(300_000)).select(
        pl.when(pl.col("v").hash(1) % 20 != 0).then(pl.col("v").hash(2) % 50)
    )
    s = pl.concat([part.to_series() for part in df.iter_slices(100_000)], rechunk=False)
    assert s.n_chunks() == 3
    _check_against_ordinal_rank(s, n_bins)


def test_bin_ranks_many_small_chunks() -> None:
    # One range of rows spans all the single-row chunks.
    rng = random.Random(5)
    values = [
        None if rng.random() < 0.1 else rng.randint(0, 2000) for _ in range(20_000)
    ]
    s = pl.concat(
        [pl.Series("a", values[i : i + 1], dtype=pl.Int64) for i in range(len(values))],
        rechunk=False,
    )
    assert s.n_chunks() == len(values)

    result = s.bin_ranks(1000, labels=False, include_intervals=True)

    expected = s.rechunk().bin_ranks(1000, labels=False, include_intervals=True)
    assert_series_equal(result, expected)


def test_bin_ranks_cut_at_rank_zero() -> None:
    s = pl.Series("a", [3, 1, None, 1, 2])

    # The cut at rank 0 leaves the first bin empty.
    result = s.bin_ranks([0.0, 0.5], labels=False)

    assert_series_equal(result, pl.Series("a", [2, 1, None, 1, 2], dtype=pl.UInt32))


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize(
    "spec", [1, 3, 7, 300, [0.0, 0.25, 0.25, 0.5], [0.4, 1.0, 1.0]]
)
def test_bin_ranks_sorted_flag_matches_unflagged(
    descending: bool, nulls_last: bool, spec: int | list[float]
) -> None:
    rng = random.Random(9)
    ints = sorted((rng.randint(0, 20) for _ in range(150)), reverse=descending)
    nulls: list[int | None] = [None] * 11
    values = [*ints, *nulls] if nulls_last else [*nulls, *ints]
    flagged = _chunked(values, pl.Int64()).set_sorted(descending=descending)
    unflagged = _chunked(values, pl.Int64())
    assert unflagged.flags["SORTED_ASC"] is False

    result = flagged.bin_ranks(spec, labels=False, include_intervals=True)

    expected = unflagged.bin_ranks(spec, labels=False, include_intervals=True)
    assert_series_equal(result, expected)
