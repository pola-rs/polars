import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_chunks_align_16830() -> None:
    n = 2
    df = pl.DataFrame(
        {"index_1": np.repeat(np.arange(10), n), "index_2": np.repeat(np.arange(10), n)}
    )
    df = pl.concat([df[0:10], df[10:]], rechunk=False)
    df = df.filter(df["index_1"] == 0)  # filter chunks
    df = df.with_columns(
        index_2=pl.Series(values=[0] * n)
    )  # set a chunk of different size
    df.set_sorted("index_2")  # triggers `select_chunk`.


def test_first_last_non_null_empty_leading_chunk_28495() -> None:
    def with_empty_leading_chunk(values: list[int | None]) -> pl.Series:
        s = pl.concat(
            [
                pl.Series("a", [0], dtype=pl.Int64),
                pl.Series("a", values, dtype=pl.Int64),
            ],
            rechunk=False,
        )
        s = s.filter(pl.Series([False] + [True] * len(values)))
        assert [len(c) for c in s.get_chunks()] == [0, len(values)]
        return s

    nulls_first = with_empty_leading_chunk([None, None, 1, 2, 3]).set_sorted()
    assert nulls_first.arg_min() == 2
    assert nulls_first.arg_max() == 4
    assert nulls_first.min() == 1
    assert nulls_first.max() == 3

    nulls_last = with_empty_leading_chunk([3, 2, 1, None, None]).set_sorted(
        descending=True
    )
    assert nulls_last.arg_min() == 2
    assert nulls_last.arg_max() == 0
    assert nulls_last.min() == 1
    assert nulls_last.max() == 3


def _repeat(value: object, n: int, dtype: pl.DataType) -> pl.Series:
    """A single chunk that repeats `value` n times."""
    return pl.select(pl.repeat(value, n, dtype=dtype).alias("a")).to_series()


@pytest.mark.parametrize(
    ("dtype", "value", "other"),
    [
        (pl.Int64, 3, 5),
        (pl.Float64, 1.5, 2.5),
        (pl.Boolean, True, False),
        (pl.String, "abc", "ab"),
        (pl.List(pl.Int64), [1, 2], [3]),
        (pl.Struct({"x": pl.Int64}), {"x": 1}, {"x": 2}),
    ],
)
def test_several_repeated_chunks_read_as_one_element(
    dtype: pl.DataType, value: object, other: object
) -> None:
    # A column the streaming engine hands back is one chunk per morsel, so a repeat
    # reaches an op as several chunks that each repeat it. Every op that answers such a
    # column off the one element has to answer the same way however many chunks it
    # arrives in -- and has to stop where the chunks repeat *different* elements.
    n = 12
    half = n // 2
    same = pl.concat(
        [_repeat(value, half, dtype), _repeat(value, n - half, dtype)], rechunk=False
    )
    differ = pl.concat(
        [_repeat(value, half, dtype), _repeat(other, n - half, dtype)], rechunk=False
    )
    assert same.n_chunks() == 2
    assert differ.n_chunks() == 2

    for s in (same, differ):
        flat = pl.Series("a", s.to_list(), dtype=dtype)
        df, flat_df = pl.DataFrame({"a": s}), pl.DataFrame({"a": flat})
        for expr in (
            pl.col("a").sort(),
            pl.col("a").arg_sort(),
            pl.col("a").reverse(),
            pl.col("a").unique(maintain_order=True),
            pl.col("a").n_unique(),
            pl.col("a").unique_counts(),
            pl.col("a").value_counts(sort=True),
            pl.col("a").is_unique(),
            pl.col("a").is_duplicated(),
            pl.col("a").is_first_distinct(),
            pl.col("a").is_last_distinct(),
            pl.col("a").arg_unique(),
            pl.col("a").rle(),
            pl.col("a").rle_id(),
            pl.col("a").hash(seed=7),
            pl.col("a").is_in(pl.Series("x", [value], dtype=dtype)),
        ):
            assert_frame_equal(df.select(expr), flat_df.select(expr))
