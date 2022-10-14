from __future__ import annotations

import sys
import typing
from datetime import date, datetime, timedelta
from decimal import Decimal
from io import BytesIO
from typing import TYPE_CHECKING, Any, Iterator, cast

import numpy as np
import pyarrow as pa
import pytest

import polars as pl
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.exceptions import NoRowsReturned, TooManyRowsReturned
from polars.testing import assert_frame_equal, assert_series_equal, columns

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
else:
    from backports.zoneinfo import ZoneInfo


if TYPE_CHECKING:
    from polars.internals.type_aliases import JoinStrategy


def test_version() -> None:
    pl.__version__


def test_null_count() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", None]})
    assert df.null_count().shape == (1, 2)


def test_init_empty() -> None:
    # test various flavours of empty init
    for empty in (None, (), [], {}, pa.Table.from_arrays([])):
        df = pl.DataFrame(empty)
        assert df.shape == (0, 0)
        assert df.is_empty()

    # note: cannot use df (empty or otherwise) in boolean context
    empty_df = pl.DataFrame()
    with pytest.raises(ValueError, match="ambiguous"):
        not empty_df


def test_special_char_colname_init() -> None:
    from string import punctuation

    cols = [(c.name, c.dtype) for c in columns(punctuation)]
    df = pl.DataFrame(columns=cols)

    assert len(cols) == len(df.columns)
    assert len(df.rows()) == 0
    assert df.is_empty()


def test_comparisons() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})

    # Constants
    assert_frame_equal(df == 2, pl.DataFrame({"a": [False, True], "b": [False, False]}))
    assert_frame_equal(df != 2, pl.DataFrame({"a": [True, False], "b": [True, True]}))
    assert_frame_equal(df < 3.0, pl.DataFrame({"a": [True, True], "b": [False, False]}))
    assert_frame_equal(df >= 2, pl.DataFrame({"a": [False, True], "b": [True, True]}))
    assert_frame_equal(df <= 2, pl.DataFrame({"a": [True, True], "b": [False, False]}))

    with pytest.raises(pl.ComputeError):
        df > "2"  # noqa: B015

    # Series
    s = pl.Series([3, 1])
    assert_frame_equal(df >= s, pl.DataFrame({"a": [False, True], "b": [True, True]}))

    # DataFrame
    other = pl.DataFrame({"a": [1, 2], "b": [2, 3]})
    assert_frame_equal(
        df == other, pl.DataFrame({"a": [True, True], "b": [False, False]})
    )

    # DataFrame columns mismatch
    with pytest.raises(ValueError):
        df == pl.DataFrame({"a": [1, 2], "c": [3, 4]})  # noqa: B015
    with pytest.raises(ValueError):
        df == pl.DataFrame({"b": [3, 4], "a": [1, 2]})  # noqa: B015

    # DataFrame shape mismatch
    with pytest.raises(ValueError):
        df == pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})  # noqa: B015

    # Type mismatch
    with pytest.raises(pl.ComputeError):
        df == pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})  # noqa: B015


def test_selection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})

    # get_column by name
    assert df.get_column("a").to_list() == [1, 2, 3]

    # select columns by mask
    assert df[:2, :1].shape == (2, 1)
    assert df[:2, "a"].shape == (2, 1)  # type: ignore[comparison-overlap]

    # column selection by string(s) in first dimension
    assert df["a"].to_list() == [1, 2, 3]
    assert df["b"].to_list() == [1.0, 2.0, 3.0]
    assert df["c"].to_list() == ["a", "b", "c"]

    # row selection by integers(s) in first dimension
    assert df[0].frame_equal(pl.DataFrame({"a": [1], "b": [1.0], "c": ["a"]}))
    assert df[-1].frame_equal(pl.DataFrame({"a": [3], "b": [3.0], "c": ["c"]}))

    # row, column selection when using two dimensions
    assert df[:, 0].to_list() == [1, 2, 3]
    assert df[:, 1].to_list() == [1.0, 2.0, 3.0]
    assert df[:2, 2].to_list() == ["a", "b"]

    assert df[[1, 2]].frame_equal(
        pl.DataFrame({"a": [2, 3], "b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert df[[-1, -2]].frame_equal(
        pl.DataFrame({"a": [3, 2], "b": [3.0, 2.0], "c": ["c", "b"]})
    )

    assert df[["a", "b"]].columns == ["a", "b"]
    assert df[[1, 2], [1, 2]].frame_equal(
        pl.DataFrame({"b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert typing.cast(str, df[1, 2]) == "b"
    assert typing.cast(float, df[1, 1]) == 2.0
    assert typing.cast(int, df[2, 0]) == 3

    assert df[[0, 1], "b"].shape == (2, 1)  # type: ignore[comparison-overlap]
    assert df[[2], ["a", "b"]].shape == (1, 2)
    assert df.to_series(0).name == "a"
    assert (df["a"] == df["a"]).sum() == 3
    assert (df["c"] == df["a"].cast(str)).sum() == 0
    assert df[:, "a":"b"].shape == (3, 2)  # type: ignore[misc]
    assert df[:, "a":"c"].columns == ["a", "b", "c"]  # type: ignore[misc]
    expect = pl.DataFrame({"c": ["b"]})
    assert df[1, [2]].frame_equal(expect)
    expect = pl.DataFrame({"b": [1.0, 3.0]})
    assert df[[0, 2], [1]].frame_equal(expect)
    assert typing.cast(str, df[0, "c"]) == "a"
    assert typing.cast(str, df[1, "c"]) == "b"
    assert typing.cast(str, df[2, "c"]) == "c"
    assert typing.cast(int, df[0, "a"]) == 1

    # more slicing
    expect = pl.DataFrame({"a": [3, 2, 1], "b": [3.0, 2.0, 1.0], "c": ["c", "b", "a"]})
    assert df[::-1].frame_equal(expect)
    expect = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["a", "b"]})
    assert df[:-1].frame_equal(expect)

    expect = pl.DataFrame({"a": [1, 3], "b": [1.0, 3.0], "c": ["a", "c"]})
    assert df[::2].frame_equal(expect)

    # only allow boolean values in column position
    df = pl.DataFrame(
        {
            "a": [1, 2],
            "b": [2, 3],
            "c": [3, 4],
        }
    )

    assert df[:, [False, True, True]].columns == ["b", "c"]
    assert df[:, pl.Series([False, True, True])].columns == ["b", "c"]
    assert df[:, pl.Series([False, False, False])].columns == []


def test_mixed_sequence_selection() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.select(["a", pl.col("b"), pl.lit("c")])
    expected = pl.DataFrame({"a": [1, 2], "b": [3, 4], "literal": ["c", "c"]})
    assert_frame_equal(result, expected)


def test_from_arrow() -> None:
    tbl = pa.table(
        {
            "a": pa.array([1, 2], pa.timestamp("s")),
            "b": pa.array([1, 2], pa.timestamp("ms")),
            "c": pa.array([1, 2], pa.timestamp("us")),
            "d": pa.array([1, 2], pa.timestamp("ns")),
            "decimal1": pa.array([1, 2], pa.decimal128(2, 1)),
        }
    )
    expected_schema = {
        "a": pl.Datetime("ms"),
        "b": pl.Datetime("ms"),
        "c": pl.Datetime("us"),
        "d": pl.Datetime("ns"),
        "decimal1": pl.Float64,
    }
    expected_data = [
        (
            datetime(1970, 1, 1, 0, 0, 1),
            datetime(1970, 1, 1, 0, 0, 0, 1000),
            datetime(1970, 1, 1, 0, 0, 0, 1),
            datetime(1970, 1, 1, 0, 0),
            1.0,
        ),
        (
            datetime(1970, 1, 1, 0, 0, 2),
            datetime(1970, 1, 1, 0, 0, 0, 2000),
            datetime(1970, 1, 1, 0, 0, 0, 2),
            datetime(1970, 1, 1, 0, 0),
            2.0,
        ),
    ]

    df = cast(pl.DataFrame, pl.from_arrow(tbl))
    assert df.schema == expected_schema
    assert df.rows() == expected_data

    empty_tbl = tbl[:0]  # no rows
    df = cast(pl.DataFrame, pl.from_arrow(empty_tbl))
    assert df.schema == expected_schema
    assert df.rows() == []


def test_dataclasses_and_namedtuple() -> None:
    from dataclasses import dataclass
    from typing import NamedTuple

    @dataclass
    class TradeDC:
        timestamp: datetime
        ticker: str
        price: Decimal
        size: int | None = None

    class TradeNT(NamedTuple):
        timestamp: datetime
        ticker: str
        price: Decimal
        size: int | None = None

    raw_data = [
        (datetime(2022, 9, 8, 14, 30, 45), "AAPL", Decimal("157.5"), 125),
        (datetime(2022, 9, 9, 10, 15, 12), "FLSY", Decimal("10.0"), 1500),
        (datetime(2022, 9, 7, 15, 30), "MU", Decimal("55.5"), 400),
    ]

    for TradeClass in (TradeDC, TradeNT):
        trades = [TradeClass(*values) for values in raw_data]

        for DF in (pl.DataFrame, pl.from_records):
            df = DF(data=trades)  # type: ignore[operator]
            assert df.schema == {
                "timestamp": pl.Datetime("us"),
                "ticker": pl.Utf8,
                "price": pl.Float64,
                "size": pl.Int64,
            }
            assert df.rows() == raw_data

        # in conjunction with 'columns' override (rename/downcast)
        df = pl.DataFrame(
            data=trades,
            columns=[  # type: ignore[arg-type]
                ("ts", pl.Datetime("ms")),
                ("tk", pl.Utf8),
                ("pc", pl.Float32),
                ("sz", pl.UInt16),
            ],
        )
        assert df.schema == {
            "ts": pl.Datetime("ms"),
            "tk": pl.Utf8,
            "pc": pl.Float32,
            "sz": pl.UInt16,
        }
        assert df.rows() == raw_data


def test_dataframe_membership_operator() -> None:
    # cf. issue #4032
    df = pl.DataFrame({"name": ["Jane", "John"], "age": [20, 30]})
    assert "name" in df
    assert "phone" not in df


def test_sort() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    assert df.sort("a").frame_equal(pl.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]}))
    assert df.sort(["a", "b"]).frame_equal(
        pl.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]})
    )


def test_replace() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    s = pl.Series("c", [True, False, True])
    df.replace("a", s)
    assert df.frame_equal(pl.DataFrame({"a": [True, False, True], "b": [1, 2, 3]}))


def test_assignment() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [2, 3, 4]})
    df = df.with_column(pl.col("foo").alias("foo"))
    # make sure that assignment does not change column order
    assert df.columns == ["foo", "bar"]
    df = df.with_column(
        pl.when(pl.col("foo") > 1).then(9).otherwise(pl.col("foo")).alias("foo")
    )
    assert df["foo"].to_list() == [1, 9, 9]


def test_insert_at_idx() -> None:
    df = (
        pl.DataFrame({"z": [3, 4, 5]})
        .insert_at_idx(0, pl.Series("x", [1, 2, 3]))
        .insert_at_idx(-1, pl.Series("y", [2, 3, 4]))
    )
    expected_df = pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [3, 4, 5]})
    assert_frame_equal(expected_df, df)


def test_replace_at_idx() -> None:
    df = (
        pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [3, 4, 5]})
        .replace_at_idx(0, pl.Series("a", [4, 5, 6]))
        .replace_at_idx(-2, pl.Series("b", [5, 6, 7]))
        .replace_at_idx(-1, pl.Series("c", [6, 7, 8]))
    )
    expected_df = pl.DataFrame({"a": [4, 5, 6], "b": [5, 6, 7], "c": [6, 7, 8]})
    assert_frame_equal(expected_df, df)


def test_to_series() -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [3, 4, 5]})

    assert_series_equal(df.to_series(), df["x"])
    assert_series_equal(df.to_series(0), df["x"])
    assert_series_equal(df.to_series(-3), df["x"])

    assert_series_equal(df.to_series(1), df["y"])
    assert_series_equal(df.to_series(-2), df["y"])

    assert_series_equal(df.to_series(2), df["z"])
    assert_series_equal(df.to_series(-1), df["z"])


def test_take_every() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
    expected_df = pl.DataFrame({"a": [1, 3], "b": ["w", "y"]})
    assert_frame_equal(expected_df, df.take_every(2))


def test_slice() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    expected = pl.DataFrame({"a": [2, 3], "b": ["b", "c"]})
    for slice_params in (
        [1, 10],  # slice > len(df)
        [1, 2],  # slice == len(df)
        [1],  # optional len
    ):
        assert df.slice(*slice_params).frame_equal(expected)

    for py_slice in (
        slice(1, 2),
        slice(0, 2, 2),
        slice(3, -3, -1),
        slice(1, None, -2),
        slice(-1, -3, -1),
        slice(-3, None, -3),
    ):
        # confirm frame slice matches python slice
        assert df[py_slice].rows() == df.rows()[py_slice]


def test_head_tail_limit() -> None:
    df = pl.DataFrame({"a": range(10), "b": range(10)})
    assert df.head(5).height == 5
    assert df.limit(5).height == 5
    assert df.tail(5).height == 5

    assert not df.head(5).frame_equal(df.tail(5))
    # check if it doesn't fail when out of bounds
    assert df.head(100).height == 10
    assert df.limit(100).height == 10
    assert df.tail(100).height == 10

    # limit is an alias of head
    assert df.head(5).frame_equal(df.limit(5))


def test_drop_nulls() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, None, 8],
            "ham": ["a", "b", "c"],
        }
    )

    result = df.drop_nulls()
    expected = pl.DataFrame(
        {
            "foo": [1, 3],
            "bar": [6, 8],
            "ham": ["a", "c"],
        }
    )
    assert result.frame_equal(expected)

    # below we only drop entries if they are null in the column 'foo'
    result = df.drop_nulls("foo")
    assert result.frame_equal(df)


def test_pipe() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, None, 8]})

    def _multiply(data: pl.DataFrame, mul: int) -> pl.DataFrame:
        return data * mul

    result = df.pipe(_multiply, mul=3)

    assert result.frame_equal(df * 3)


def test_explode() -> None:
    df = pl.DataFrame({"letters": ["c", "a"], "nrs": [[1, 2], [1, 3]]})
    out = df.explode("nrs")
    assert out["letters"].to_list() == ["c", "c", "a", "a"]
    assert out["nrs"].to_list() == [1, 2, 1, 3]


def test_groupby() -> None:
    df = pl.DataFrame(
        {
            "a": ["a", "b", "a", "b", "b", "c"],
            "b": [1, 2, 3, 4, 5, 6],
            "c": [6, 5, 4, 3, 2, 1],
        }
    )

    assert df.groupby("a").apply(lambda df: df[["c"]].sum()).sort("c")["c"][0] == 1

    with pytest.deprecated_call():
        # TODO: find a way to avoid indexing into GroupBy
        for subdf in df.groupby("a"):  # type: ignore[attr-defined]
            # TODO: add __next__() to GroupBy
            if subdf["a"][0] == "b":
                assert subdf.shape == (3, 3)

    # Use lazy API in eager groupby
    assert df.groupby("a").agg([pl.sum("b")]).shape == (3, 2)
    # test if it accepts a single expression
    assert df.groupby("a").agg(pl.sum("b")).shape == (3, 2)

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["a", "a", "b", "b", "b"],
            "c": [None, 1, None, 1, None],
        }
    )

    # check if this query runs and thus column names propagate
    df.groupby("b").agg(pl.col("c").forward_fill()).explode("c")

    # get a specific column
    result = df.groupby("b").agg(pl.count("a"))
    assert result.shape == (2, 2)
    assert result.columns == ["b", "a"]

    # make sure all the methods below run
    assert df.groupby("b").first().shape == (2, 3)
    assert df.groupby("b").last().shape == (2, 3)
    assert df.groupby("b").max().shape == (2, 3)
    assert df.groupby("b").min().shape == (2, 3)
    assert df.groupby("b").count().shape == (2, 2)
    assert df.groupby("b").mean().shape == (2, 3)
    assert df.groupby("b").n_unique().shape == (2, 3)
    assert df.groupby("b").median().shape == (2, 3)
    # assert df.groupby("b").quantile(0.5).shape == (2, 3)
    assert df.groupby("b").agg_list().shape == (2, 3)

    # Invalid input: `by` not specified as a sequence
    with pytest.raises(TypeError):
        df.groupby("a", "b")  # type: ignore[arg-type]


BAD_AGG_PARAMETERS = [[("b", "sum")], [("b", ["sum"])], {"b": "sum"}, {"b": ["sum"]}]
GOOD_AGG_PARAMETERS: list[pl.Expr | list[pl.Expr]] = [
    [pl.col("b").sum()],
    pl.col("b").sum(),
]


@pytest.mark.parametrize("lazy", [True, False])
def test_groupby_agg_input_types(lazy: bool) -> None:
    df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [1, 2, 3, 4]})
    df_or_lazy: pl.DataFrame | pl.LazyFrame = df.lazy() if lazy else df

    for bad_param in BAD_AGG_PARAMETERS:
        with pytest.raises(TypeError):
            result = df_or_lazy.groupby("a").agg(bad_param)  # type: ignore[arg-type]
            if lazy:
                result.collect()  # type: ignore[union-attr]

    expected = pl.DataFrame({"a": [1, 2], "b": [3, 7]})

    for good_param in GOOD_AGG_PARAMETERS:
        result = df_or_lazy.groupby("a", maintain_order=True).agg(good_param)
        if lazy:
            result = result.collect()  # type: ignore[union-attr]
        assert_frame_equal(result, expected)


@pytest.mark.parametrize("lazy", [True, False])
def test_groupby_rolling_agg_input_types(lazy: bool) -> None:
    df = pl.DataFrame({"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]})
    df_or_lazy: pl.DataFrame | pl.LazyFrame = df.lazy() if lazy else df

    for bad_param in BAD_AGG_PARAMETERS:
        with pytest.raises(TypeError):
            result = df_or_lazy.groupby_rolling(
                index_column="index_column", period="2i"
            ).agg(
                bad_param  # type: ignore[arg-type]
            )
            if lazy:
                result.collect()  # type: ignore[union-attr]

    expected = pl.DataFrame({"index_column": [0, 1, 2, 3], "b": [1, 4, 4, 3]})

    for good_param in GOOD_AGG_PARAMETERS:
        result = df_or_lazy.groupby_rolling(
            index_column="index_column", period="2i"
        ).agg(good_param)
        if lazy:
            result = result.collect()  # type: ignore[union-attr]
        assert_frame_equal(result, expected)


@pytest.mark.parametrize("lazy", [True, False])
def test_groupby_dynamic_agg_input_types(lazy: bool) -> None:
    df = pl.DataFrame({"index_column": [0, 1, 2, 3], "b": [1, 3, 1, 2]})
    df_or_lazy: pl.DataFrame | pl.LazyFrame = df.lazy() if lazy else df

    for bad_param in BAD_AGG_PARAMETERS:
        with pytest.raises(TypeError):
            result = df_or_lazy.groupby_dynamic(
                index_column="index_column", every="2i", closed="right"
            ).agg(
                bad_param  # type: ignore[arg-type]
            )
            if lazy:
                result.collect()  # type: ignore[union-attr]

    expected = pl.DataFrame({"index_column": [0, 0, 2], "b": [1, 4, 2]})

    for good_param in GOOD_AGG_PARAMETERS:
        result = df_or_lazy.groupby_dynamic(
            index_column="index_column", every="2i", closed="right"
        ).agg(good_param)
        if lazy:
            result = result.collect()  # type: ignore[union-attr]
        assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "stack,exp_shape,exp_columns",
    [
        ([pl.Series("stacked", [-1, -1, -1])], (3, 3), ["a", "b", "stacked"]),
        (
            [pl.Series("stacked2", [-1, -1, -1]), pl.Series("stacked3", [-1, -1, -1])],
            (3, 4),
            ["a", "b", "stacked2", "stacked3"],
        ),
    ],
)
@pytest.mark.parametrize("in_place", [True, False])
def test_hstack_list_of_series(
    stack: list[pl.Series],
    exp_shape: tuple[int, int],
    exp_columns: list[str],
    in_place: bool,
) -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"]})
    if in_place:
        df.hstack(stack, in_place=True)
        assert df.shape == exp_shape
        assert df.columns == exp_columns
    else:
        df_out = df.hstack(stack, in_place=False)
        assert df_out.shape == exp_shape
        assert df_out.columns == exp_columns


@pytest.mark.parametrize("in_place", [True, False])
def test_hstack_dataframe(in_place: bool) -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"]})
    df2 = pl.DataFrame({"c": [2, 1, 3], "d": ["a", "b", "c"]})
    expected = pl.DataFrame(
        {"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [2, 1, 3], "d": ["a", "b", "c"]}
    )
    if in_place:
        df.hstack(df2, in_place=True)
        assert df.frame_equal(expected)
    else:
        df_out = df.hstack(df2, in_place=False)
        assert df_out.frame_equal(expected)


@pytest.mark.parametrize("in_place", [True, False])
def test_vstack(in_place: bool) -> None:
    df1 = pl.DataFrame({"foo": [1, 2], "bar": [6, 7], "ham": ["a", "b"]})
    df2 = pl.DataFrame({"foo": [3, 4], "bar": [8, 9], "ham": ["c", "d"]})

    expected = pl.DataFrame(
        {"foo": [1, 2, 3, 4], "bar": [6, 7, 8, 9], "ham": ["a", "b", "c", "d"]}
    )

    out = df1.vstack(df2, in_place=in_place)
    if in_place:
        assert df1.frame_equal(expected)
    else:
        assert out.frame_equal(expected)


def test_extend() -> None:
    with pl.StringCache():
        df1 = pl.DataFrame(
            {
                "foo": [1, 2],
                "bar": [True, False],
                "ham": ["a", "b"],
                "cat": ["A", "B"],
                "dates": [datetime(2021, 1, 1), datetime(2021, 2, 1)],
            }
        ).with_columns(
            [
                pl.col("cat").cast(pl.Categorical),
            ]
        )
        df2 = pl.DataFrame(
            {
                "foo": [3, 4],
                "bar": [True, None],
                "ham": ["c", "d"],
                "cat": ["C", "B"],
                "dates": [datetime(2022, 9, 1), datetime(2021, 2, 1)],
            }
        ).with_columns(
            [
                pl.col("cat").cast(pl.Categorical),
            ]
        )

        df1.extend(df2)
        expected = pl.DataFrame(
            {
                "foo": [1, 2, 3, 4],
                "bar": [True, False, True, None],
                "ham": ["a", "b", "c", "d"],
                "cat": ["A", "B", "C", "B"],
                "dates": [
                    datetime(2021, 1, 1),
                    datetime(2021, 2, 1),
                    datetime(2022, 9, 1),
                    datetime(2021, 2, 1),
                ],
            }
        ).with_column(
            pl.col("cat").cast(pl.Categorical),
        )
        assert df1.frame_equal(expected)


def test_drop() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    df = df.drop(name="a")  # type: ignore[call-arg]
    assert df.shape == (3, 2)
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    df = df.drop(columns="a")
    assert df.shape == (3, 2)
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", "c"], "c": [1, 2, 3]})
    s = df.drop_in_place("a")
    assert s.name == "a"


def test_file_buffer() -> None:
    f = BytesIO()
    f.write(b"1,2,3,4,5,6\n7,8,9,10,11,12")
    f.seek(0)
    df = pl.read_csv(f, has_header=False)
    assert df.shape == (2, 6)

    f = BytesIO()
    f.write(b"1,2,3,4,5,6\n7,8,9,10,11,12")
    f.seek(0)
    # check if not fails on TryClone and Length impl in file.rs
    with pytest.raises(pl.ArrowError):
        pl.read_parquet(f)


def test_read_missing_file() -> None:
    with pytest.raises(FileNotFoundError, match="fake_parquet_file"):
        pl.read_parquet("fake_parquet_file")

    with pytest.raises(FileNotFoundError, match="fake_csv_file"):
        pl.read_csv("fake_csv_file")

    with pytest.raises(FileNotFoundError, match="fake_csv_file"):
        with open("fake_csv_file") as f:
            pl.read_csv(f)


def test_melt() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    melted = df.melt(id_vars="A", value_vars=["B", "C"])
    assert all(melted["value"] == [1, 3, 5, 2, 4, 6])

    melted = df.melt(id_vars="A", value_vars="B")
    assert all(melted["value"] == [1, 3, 5])
    n = 3
    for melted in [df.melt(), df.lazy().melt().collect()]:
        assert melted["variable"].to_list() == ["A"] * n + ["B"] * n + ["C"] * n
        assert melted["value"].to_list() == [
            "a",
            "b",
            "c",
            "1",
            "3",
            "5",
            "2",
            "4",
            "6",
        ]

    for melted in [
        df.melt(value_name="foo", variable_name="bar"),
        df.lazy().melt(value_name="foo", variable_name="bar").collect(),
    ]:
        assert melted["bar"].to_list() == ["A"] * n + ["B"] * n + ["C"] * n
        assert melted["foo"].to_list() == [
            "a",
            "b",
            "c",
            "1",
            "3",
            "5",
            "2",
            "4",
            "6",
        ]


def test_shift() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    a = df.shift(1)
    b = pl.DataFrame(
        {"A": [None, "a", "b"], "B": [None, 1, 3]},
    )
    assert a.frame_equal(b, null_equal=True)


def test_to_dummies() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    dummies = df.to_dummies()
    assert dummies["A_a"].to_list() == [1, 0, 0]
    assert dummies["A_b"].to_list() == [0, 1, 0]
    assert dummies["A_c"].to_list() == [0, 0, 1]


def test_custom_groupby() -> None:
    df = pl.DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})

    out = (
        df.lazy()
        .groupby("b")
        .agg([pl.col("a").apply(lambda x: x.sum(), return_dtype=pl.Int64)])
        .collect()
    )
    assert out.shape == (3, 2)


def test_multiple_columns_drop() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    out = df.drop(["a", "b"])
    assert out.columns == ["c"]


def test_concat() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})

    df2 = pl.concat([df, df])
    assert df2.shape == (6, 3)
    assert df2.n_chunks() == 1  # the default is to rechunk

    assert pl.concat([df, df], rechunk=False).n_chunks() == 2

    # check if a remains unchanged
    a = pl.from_records(((1, 2), (1, 2)))
    _ = pl.concat([a, a, a])
    assert a.shape == (2, 2)

    with pytest.raises(ValueError):
        _ = pl.concat([])

    with pytest.raises(ValueError):
        pl.concat([df, df], how="rubbish")  # type: ignore[call-overload]


def test_arg_where() -> None:
    s = pl.Series([True, False, True, False])
    assert pl.arg_where(s, eager=True).cast(int).series_equal(pl.Series([0, 2]))


def test_get_dummies() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    res = pl.get_dummies(df)
    expected = pl.DataFrame(
        {"a_1": [1, 0, 0], "a_2": [0, 1, 0], "a_3": [0, 0, 1]}
    ).with_columns(pl.all().cast(pl.UInt8))
    assert res.frame_equal(expected)

    df = pl.DataFrame(
        {"i": [1, 2, 3], "category": ["dog", "cat", "cat"]},
        columns={"i": pl.Int32, "category": pl.Categorical},
    )
    expected = pl.DataFrame(
        {
            "i": [1, 2, 3],
            "category_cat": [0, 1, 1],
            "category_dog": [1, 0, 0],
        },
        columns={"i": pl.Int32, "category_cat": pl.UInt8, "category_dog": pl.UInt8},
    )
    result = pl.get_dummies(df, columns=["category"])
    assert result.frame_equal(expected)


def test_to_pandas(df: pl.DataFrame) -> None:
    # pyarrow cannot deal with unsigned dictionary integer yet.
    # pyarrow cannot convert a time64 w/ non-zero nanoseconds
    df = df.drop(["cat", "time"])
    df.to_arrow()
    df.to_pandas()
    # test shifted df
    df.shift(2).to_pandas()
    df = pl.DataFrame({"col": pl.Series([True, False, True])})
    df.shift(2).to_pandas()


def test_from_arrow_table() -> None:
    data = {"a": [1, 2], "b": [1, 2]}
    tbl = pa.table(data)

    df = cast(pl.DataFrame, pl.from_arrow(tbl))
    df.frame_equal(pl.DataFrame(data))


def test_df_stats(df: pl.DataFrame) -> None:
    df.var()
    df.std()
    df.min()
    df.max()
    df.sum()
    df.mean()
    df.median()
    df.quantile(0.4, "nearest")


def test_df_fold() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})

    assert df.fold(lambda s1, s2: s1 + s2).series_equal(pl.Series("a", [4.0, 5.0, 9.0]))
    assert df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2)).series_equal(
        pl.Series("a", [1.0, 1.0, 3.0])
    )

    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.fold(lambda s1, s2: s1 + s2)
    out.series_equal(pl.Series("", ["foo11", "bar22", "233"]))

    df = pl.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    # just check dispatch. values are tested on rust side.
    assert len(df.sum(axis=1)) == 3
    assert len(df.mean(axis=1)) == 3
    assert len(df.min(axis=1)) == 3
    assert len(df.max(axis=1)) == 3

    df_width_one = df[["a"]]
    assert df_width_one.fold(lambda s1, s2: s1).series_equal(df["a"])


def test_row_tuple() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})

    # return row by index
    assert df.row(0) == ("foo", 1, 1.0)
    assert df.row(1) == ("bar", 2, 2.0)
    assert df.row(-1) == ("2", 3, 3.0)

    # return row by predicate
    assert df.row(by_predicate=pl.col("a") == "bar") == ("bar", 2, 2.0)
    assert df.row(by_predicate=pl.col("b").is_in([2, 4, 6])) == ("bar", 2, 2.0)

    # expected error conditions
    with pytest.raises(TooManyRowsReturned):
        df.row(by_predicate=pl.col("b").is_in([1, 3, 5]))

    with pytest.raises(NoRowsReturned):
        df.row(by_predicate=pl.col("a") == "???")

    # cannot set both 'index' and 'by_predicate'
    with pytest.raises(ValueError):
        df.row(0, by_predicate=pl.col("a") == "bar")

    # must call 'by_predicate' by keyword
    with pytest.raises(TypeError):
        df.row(None, pl.col("a") == "bar")  # type: ignore[misc]

    # cannot pass predicate into 'index'
    with pytest.raises(TypeError):
        df.row(pl.col("a") == "bar")  # type: ignore[arg-type]

    # at least one of 'index' and 'by_predicate' must be set
    with pytest.raises(ValueError):
        df.row()


def test_df_apply() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.apply(lambda x: len(x), None).to_series()
    assert out.sum() == 9


def test_column_names() -> None:
    tbl = pa.table(
        {
            "a": pa.array([1, 2, 3, 4, 5], pa.decimal128(38, 2)),
            "b": pa.array([1, 2, 3, 4, 5], pa.int64()),
        }
    )
    for a in (tbl, tbl[:0]):
        df = cast(pl.DataFrame, pl.from_arrow(a))
        assert df.columns == ["a", "b"]


def test_lazy_functions() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.select([pl.count("a")])
    assert list(out["a"]) == [3]
    assert pl.count(df["a"]) == 3
    out = df.select(
        [
            pl.var("b").alias("1"),
            pl.std("b").alias("2"),
            pl.max("b").alias("3"),
            pl.min("b").alias("4"),
            pl.sum("b").alias("5"),
            pl.mean("b").alias("6"),
            pl.median("b").alias("7"),
            pl.n_unique("b").alias("8"),
            pl.first("b").alias("9"),
            pl.last("b").alias("10"),
        ]
    )
    expected = 1.0
    assert np.isclose(out.to_series(0), expected)
    assert np.isclose(pl.var(df["b"]), expected)  # type: ignore[arg-type]
    expected = 1.0
    assert np.isclose(out.to_series(1), expected)
    assert np.isclose(pl.std(df["b"]), expected)  # type: ignore[arg-type]
    expected = 3
    assert np.isclose(out.to_series(2), expected)
    assert np.isclose(pl.max(df["b"]), expected)
    expected = 1
    assert np.isclose(out.to_series(3), expected)
    assert np.isclose(pl.min(df["b"]), expected)
    expected = 6
    assert np.isclose(out.to_series(4), expected)
    assert np.isclose(pl.sum(df["b"]), expected)
    expected = 2
    assert np.isclose(out.to_series(5), expected)
    assert np.isclose(pl.mean(df["b"]), expected)
    expected = 2
    assert np.isclose(out.to_series(6), expected)
    assert np.isclose(pl.median(df["b"]), expected)
    expected = 3
    assert np.isclose(out.to_series(7), expected)
    assert np.isclose(pl.n_unique(df["b"]), expected)
    expected = 1
    assert np.isclose(out.to_series(8), expected)
    assert np.isclose(pl.first(df["b"]), expected)
    expected = 3
    assert np.isclose(out.to_series(9), expected)
    assert np.isclose(pl.last(df["b"]), expected)


def test_multiple_column_sort() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [2, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.sort([pl.col("b"), pl.col("c").reverse()])
    assert list(out["c"]) == [2.0, 1.0, 3.0]
    assert list(out["b"]) == [2, 2, 3]

    df = pl.DataFrame({"a": np.arange(1, 4), "b": ["a", "a", "b"]})

    df.sort("a", reverse=True).frame_equal(
        pl.DataFrame({"a": [3, 2, 1], "b": ["b", "a", "a"]})
    )
    df.sort("b", reverse=True).frame_equal(
        pl.DataFrame({"a": [3, 1, 2], "b": ["b", "a", "a"]})
    )
    df.sort(["b", "a"], reverse=[False, True]).frame_equal(
        pl.DataFrame({"a": [2, 1, 3], "b": ["a", "a", "b"]})
    )


def test_describe() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.8, 3.0],
            "b": [4, 5, None],
            "c": [True, False, True],
            "d": [None, "b", "c"],
            "e": ["usd", "eur", None],
            "f": [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
        }
    )
    df = df.with_column(pl.col("e").cast(pl.Categorical))
    expected = pl.DataFrame(
        {
            "describe": ["count", "null_count", "mean", "std", "min", "max", "median"],
            "a": [3.0, 0.0, 2.2666667, 1.101514, 1.0, 3.0, 2.8],
            "b": [3.0, 1.0, 4.5, 0.7071067811865476, 4.0, 5.0, 4.5],
            "c": [3.0, 0.0, None, None, 0.0, 1.0, None],
            "d": ["3", "1", None, None, "b", "c", None],
            "e": ["3", "1", None, None, None, None, None],
            "f": ["3", "0", None, None, "2020-01-01", "2022-01-01", None],
        }
    )
    pl.testing.assert_frame_equal(df.describe(), expected)


def test_duration_arithmetic() -> None:
    pl.Config.with_columns_kwargs = True

    df = pl.DataFrame(
        {"a": [datetime(2022, 1, 1, 0, 0, 0), datetime(2022, 1, 2, 0, 0, 0)]}
    )
    d1 = pl.duration(days=3, microseconds=987000)
    d2 = pl.duration(days=6, milliseconds=987)

    assert_frame_equal(
        df.with_columns(
            b=(df["a"] + d1),
            c=(pl.col("a") + d2),
        ),
        pl.DataFrame(
            {
                "a": [
                    datetime(2022, 1, 1, 0, 0, 0),
                    datetime(2022, 1, 2, 0, 0, 0),
                ],
                "b": [
                    datetime(2022, 1, 4, 0, 0, 0, 987000),
                    datetime(2022, 1, 5, 0, 0, 0, 987000),
                ],
                "c": [
                    datetime(2022, 1, 7, 0, 0, 0, 987000),
                    datetime(2022, 1, 8, 0, 0, 0, 987000),
                ],
            }
        ),
    )


def test_string_cache_eager_lazy() -> None:
    # tests if the global string cache is really global and not interfered by the lazy
    # execution. first the global settings was thread-local and this breaks with the
    # parallel execution of lazy
    with pl.StringCache():
        df1 = pl.DataFrame(
            {"region_ids": ["reg1", "reg2", "reg3", "reg4", "reg5"]}
        ).select([pl.col("region_ids").cast(pl.Categorical)])
        df2 = pl.DataFrame(
            {"seq_name": ["reg4", "reg2", "reg1"], "score": [3.0, 1.0, 2.0]}
        ).select([pl.col("seq_name").cast(pl.Categorical), pl.col("score")])

        expected = pl.DataFrame(
            {
                "region_ids": ["reg1", "reg2", "reg3", "reg4", "reg5"],
                "score": [2.0, 1.0, None, 3.0, None],
            }
        ).with_column(pl.col("region_ids").cast(pl.Categorical))

        assert df1.join(
            df2, left_on="region_ids", right_on="seq_name", how="left"
        ).frame_equal(expected, null_equal=True)


def test_assign() -> None:
    # check if can assign in case of a single column
    df = pl.DataFrame({"a": [1, 2, 3]})
    # test if we can assign in case of single column
    df = df.with_column(pl.col("a") * 2)
    assert list(df["a"]) == [2, 4, 6]


def test_to_numpy() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert df.to_numpy().shape == (3, 2)


def test_argsort_by(df: pl.DataFrame) -> None:
    idx_df = df.select(
        pl.argsort_by(["int_nulls", "floats"], reverse=[False, True]).alias("idx")
    )
    assert (idx_df["idx"] == [1, 0, 2]).all()

    idx_df = df.select(
        pl.argsort_by(["int_nulls", "floats"], reverse=False).alias("idx")
    )
    assert (idx_df["idx"] == [1, 0, 2]).all()

    df = pl.DataFrame({"x": [0, 0, 0, 1, 1, 2], "y": [9, 9, 8, 7, 6, 6]})
    for expr, expected in (
        (pl.argsort_by(["x", "y"]), [2, 0, 1, 4, 3, 5]),
        (pl.argsort_by(["x", "y"], reverse=[True, True]), [5, 3, 4, 0, 1, 2]),
        (pl.argsort_by(["x", "y"], reverse=[True, False]), [5, 4, 3, 2, 0, 1]),
        (pl.argsort_by(["x", "y"], reverse=[False, True]), [0, 1, 2, 3, 4, 5]),
    ):
        assert (df.select(expr.alias("idx"))["idx"] == expected).all()


def test_literal_series() -> None:
    df = pl.DataFrame(
        {
            "a": np.array([21.7, 21.8, 21], dtype=np.float32),
            "b": np.array([1, 3, 2], dtype=np.int8),
            "c": ["reg1", "reg2", "reg3"],
            "d": np.array(
                [datetime(2022, 8, 16), datetime(2022, 8, 17), datetime(2022, 8, 18)],
                dtype="<M8[ns]",
            ),
        }
    )
    out = (
        df.lazy()
        .with_column(pl.Series("e", [2, 1, 3], pl.Int32))
        .with_column(pl.col("e").cast(pl.Float32))
        .collect()
    )
    expected_schema = {
        "a": pl.Float32,
        "b": pl.Int8,
        "c": pl.Utf8,
        "d": pl.Datetime("ns"),
        "e": pl.Float32,
    }
    assert_frame_equal(
        pl.DataFrame(
            [
                (21.7, 1, "reg1", datetime(2022, 8, 16, 0), 2),
                (21.8, 3, "reg2", datetime(2022, 8, 17, 0), 1),
                (21.0, 2, "reg3", datetime(2022, 8, 18, 0), 3),
            ],
            columns=expected_schema,  # type: ignore[arg-type]
        ),
        out,
        atol=0.00001,
    )


def test_to_html(df: pl.DataFrame) -> None:
    # check if it does not panic/ error
    df._repr_html_()


def test_rows() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
    assert df.rows() == [(1, 1), (2, 2)]
    assert df.reverse().rows() == [(2, 2), (1, 1)]


def test_rename(df: pl.DataFrame) -> None:
    out = df.rename({"strings": "bars", "int": "foos"})
    # check if wel can select these new columns
    _ = out[["foos", "bars"]]


def test_write_csv() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3, 4, 5],
            "bar": [6, 7, 8, 9, 10],
            "ham": ["a", "b", "c", "d", "e"],
        }
    )
    expected = "foo,bar,ham\n1,6,a\n2,7,b\n3,8,c\n4,9,d\n5,10,e\n"

    # if no file argument is supplied, write_csv() will return the string
    s = df.write_csv()
    assert s == expected

    # otherwise it will write to the file/iobuffer
    file = BytesIO()
    df.write_csv(file)
    file.seek(0)
    s = file.read().decode("utf8")
    assert s == expected


def test_from_rows() -> None:
    df = pl.from_records([[1, 2, "foo"], [2, 3, "bar"]], orient="row")
    assert df.frame_equal(
        pl.DataFrame(
            {"column_0": [1, 2], "column_1": [2, 3], "column_2": ["foo", "bar"]}
        )
    )

    df = pl.from_records(
        [[1, datetime.fromtimestamp(100)], [2, datetime.fromtimestamp(2398754908)]],
        orient="row",
    )
    assert df.dtypes == [pl.Int64, pl.Datetime]


def test_repeat_by() -> None:
    df = pl.DataFrame({"name": ["foo", "bar"], "n": [2, 3]})

    out = df.select(pl.col("n").repeat_by("n"))
    s = out["n"]
    assert s[0].to_list() == [2, 2]
    assert s[1].to_list() == [3, 3, 3]


def test_join_dates() -> None:
    dts_in = pl.date_range(
        datetime(2021, 6, 24),
        datetime(2021, 6, 24, 10, 0, 0),
        interval=timedelta(hours=1),
        closed="left",
    )
    dts = (
        dts_in.cast(int)
        .apply(lambda x: x + np.random.randint(1_000 * 60, 60_000 * 60))
        .cast(pl.Datetime)
    )

    # some df with sensor id, (randomish) datetime and some value
    df = pl.DataFrame(
        {
            "sensor": ["a"] * 5 + ["b"] * 5,
            "datetime": dts,
            "value": [2, 3, 4, 1, 2, 3, 5, 1, 2, 3],
        }
    )
    out = df.join(df, on="datetime")
    assert len(out) == len(df)


def test_asof_cross_join() -> None:
    left = pl.DataFrame({"a": [-10, 5, 10], "left_val": ["a", "b", "c"]})
    right = pl.DataFrame({"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]})

    # only test dispatch of asof join
    out = left.join_asof(right, on="a")
    assert out.shape == (3, 3)

    left.lazy().join_asof(right.lazy(), on="a").collect()
    assert out.shape == (3, 3)

    # only test dispatch of cross join
    out = left.join(right, how="cross")
    assert out.shape == (15, 4)

    left.lazy().join(right.lazy(), how="cross").collect()
    assert out.shape == (15, 4)


def test_str_concat() -> None:
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, 4],
            "name": ["ham", "spam", "foo", None],
        }
    )
    out = df.with_column((pl.lit("Dr. ") + pl.col("name")).alias("graduated_name"))
    assert out["graduated_name"][0] == "Dr. ham"
    assert out["graduated_name"][1] == "Dr. spam"


def dot_product() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [2, 2, 2, 2]})

    assert df["a"].dot(df["b"]) == 20
    assert typing.cast(int, df.select([pl.col("a").dot("b")])[0, "a"]) == 20


def test_hash_rows() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": [2, 2, 2, 2]})
    assert df.hash_rows().dtype == pl.UInt64
    assert df["a"].hash().dtype == pl.UInt64
    assert df.select([pl.col("a").hash().alias("foo")])["foo"].dtype == pl.UInt64


def test_reproducible_hash_with_seeds() -> None:
    """
    Test the reproducibility of DataFrame.hash_rows, Series.hash, and Expr.hash.

    cf. issue #3966, hashes must always be reproducible across sessions when using
    the same seeds.

    """
    df = pl.DataFrame({"s": [1234, None, 5678]})
    seeds = (11, 22, 33, 44)
    expected = pl.Series(
        "s",
        [
            15801072432137883943,
            988796329533502010,
            9604537446374444741,
        ],
        dtype=pl.UInt64,
    )

    result = df.hash_rows(*seeds)
    assert_series_equal(expected, result, check_names=False, check_exact=True)
    result = df["s"].hash(*seeds)
    assert_series_equal(expected, result, check_names=False, check_exact=True)
    result = df.select([pl.col("s").hash(*seeds)])["s"]
    assert_series_equal(expected, result, check_names=False, check_exact=True)


def test_create_df_from_object() -> None:
    class Foo:
        def __init__(self, value: int) -> None:
            self._value = value

        def __eq__(self, other: Any) -> bool:
            return issubclass(other.__class__, self.__class__) and (
                self._value == other._value
            )

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}({self._value})"

    # from miscellaneous object
    df = pl.DataFrame({"a": [Foo(1), Foo(2)]})
    assert df["a"].dtype == pl.Object
    assert df.rows() == [(Foo(1),), (Foo(2),)]

    # from mixed-type input
    df = pl.DataFrame({"x": [["abc", 12, 34.5]], "y": [1]})
    assert df.schema == {"x": pl.Object, "y": pl.Int64}
    assert df.rows() == [(["abc", 12, 34.5], 1)]


def test_hashing_on_python_objects() -> None:
    # see if we can do a groupby, drop_duplicates on a DataFrame with objects.
    # this requires that the hashing and aggregations are done on python objects

    df = pl.DataFrame({"a": [1, 1, 3, 4], "b": [1, 1, 2, 2]})

    class Foo:
        def __hash__(self) -> int:
            return 0

        def __eq__(self, other: Any) -> bool:
            return True

    df = df.with_column(pl.col("a").apply(lambda x: Foo()).alias("foo"))
    assert df.groupby(["foo"]).first().shape == (1, 3)
    assert df.unique().shape == (3, 3)


def test_unique_unit_rows() -> None:
    df = pl.DataFrame({"a": [1], "b": [None]})

    # 'unique' one-row frame should be equal to the original frame
    assert df.frame_equal(df.unique(subset="a"))
    for col in df.columns:
        assert df.n_unique(subset=[col]) == 1


def test_panic() -> None:
    # may contain some tests that yielded a panic in polars or arrow
    # https://github.com/pola-rs/polars/issues/1110
    a = pl.DataFrame(
        {
            "col1": ["a"] * 500 + ["b"] * 500,
        }
    )
    a.filter(pl.col("col1") != "b")


def test_h_agg() -> None:
    df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})

    assert_series_equal(
        df.sum(axis=1, null_strategy="ignore"), pl.Series("a", [2, 2, 6])
    )
    assert_series_equal(
        df.sum(axis=1, null_strategy="propagate"), pl.Series("a", [2, None, 6])
    )
    assert_series_equal(
        df.mean(axis=1, null_strategy="propagate"), pl.Series("a", [1.0, None, 3.0])
    )


def test_slicing() -> None:
    # https://github.com/pola-rs/polars/issues/1322
    n = 20

    df = pl.DataFrame(
        {
            "d": ["u", "u", "d", "c", "c", "d", "d"] * n,
            "v1": [None, "help", None, None, None, None, None] * n,
        }
    )

    assert (df.filter(pl.col("d") != "d").select([pl.col("v1").unique()])).shape == (
        2,
        1,
    )


def test_apply_list_return() -> None:
    df = pl.DataFrame({"start": [1, 2], "end": [3, 5]})
    out = df.apply(lambda r: pl.Series(range(r[0], r[1] + 1))).to_series()
    assert out.to_list() == [[1, 2, 3], [2, 3, 4, 5]]


def test_apply_dataframe_return() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["c", "d", None]})

    out = df.apply(lambda row: (row[0] * 10, "foo", True, row[-1]))
    expected = pl.DataFrame(
        {
            "column_0": [10, 20, 30],
            "column_1": ["foo", "foo", "foo"],
            "column_2": [True, True, True],
            "column_3": ["c", "d", None],
        }
    )
    assert out.frame_equal(expected, null_equal=True)


def test_groupby_cat_list() -> None:
    grouped = (
        pl.DataFrame(
            [
                pl.Series("str_column", ["a", "b", "b", "a", "b"]),
                pl.Series("int_column", [1, 1, 2, 2, 3]),
            ]
        )
        .with_column(pl.col("str_column").cast(pl.Categorical).alias("cat_column"))
        .groupby("int_column", maintain_order=True)
        .agg([pl.col("cat_column")])["cat_column"]
    )

    out = grouped.explode()
    assert out.dtype == pl.Categorical
    assert out[0] == "a"

    # test if we can also correctly fmt the categorical in list
    assert (
        str(grouped)
        == """shape: (3,)
Series: 'cat_column' [list]
[
	["a", "b"]
	["b", "a"]
	["b"]
]"""  # noqa: E101, W191
    )


def test_groupby_agg_n_unique_floats() -> None:
    # tests proper dispatch
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    for dtype in [pl.Float32, pl.Float64]:
        out = df.groupby("a", maintain_order=True).agg(
            [pl.col("b").cast(dtype).n_unique()]
        )
        assert out["b"].to_list() == [2, 1]


def test_select_by_dtype(df: pl.DataFrame) -> None:
    out = df.select(pl.col(pl.Utf8))
    assert out.columns == ["strings", "strings_nulls"]
    out = df.select(pl.col([pl.Utf8, pl.Boolean]))
    assert out.columns == ["strings", "strings_nulls", "bools", "bools_nulls"]


def test_with_row_count() -> None:
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    out = df.with_row_count()
    assert out["row_nr"].to_list() == [0, 1, 2]

    out = df.lazy().with_row_count().collect()
    assert out["row_nr"].to_list() == [0, 1, 2]


def test_filter_with_all_expansion() -> None:
    df = pl.DataFrame(
        {
            "b": [1, 2, None],
            "c": [1, 2, None],
            "a": [None, None, None],
        }
    )
    out = df.filter(~pl.fold(True, lambda acc, s: acc & s.is_null(), pl.all()))
    assert out.shape == (2, 3)


def test_transpose() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1, 2, 3]})
    expected = pl.DataFrame(
        {
            "column": ["a", "b"],
            "column_0": [1, 1],
            "column_1": [2, 2],
            "column_2": [3, 3],
        }
    )
    out = df.transpose(include_header=True)
    assert expected.frame_equal(out)

    out = df.transpose(include_header=False, column_names=["a", "b", "c"])
    expected = pl.DataFrame(
        {
            "a": [1, 1],
            "b": [2, 2],
            "c": [3, 3],
        }
    )
    assert expected.frame_equal(out)

    out = df.transpose(
        include_header=True, header_name="foo", column_names=["a", "b", "c"]
    )
    expected = pl.DataFrame(
        {
            "foo": ["a", "b"],
            "a": [1, 1],
            "b": [2, 2],
            "c": [3, 3],
        }
    )
    assert expected.frame_equal(out)

    def name_generator() -> Iterator[str]:
        base_name = "my_column_"
        count = 0
        while True:
            yield f"{base_name}{count}"
            count += 1

    out = df.transpose(include_header=False, column_names=name_generator())
    expected = pl.DataFrame(
        {
            "my_column_0": [1, 1],
            "my_column_1": [2, 2],
            "my_column_2": [3, 3],
        }
    )
    assert expected.frame_equal(out)


def test_extension() -> None:
    class Foo:
        def __init__(self, value: Any) -> None:
            self.value = value

        def __repr__(self) -> str:
            return f"foo({self.value})"

    foos = [Foo(1), Foo(2), Foo(3)]
    # I believe foos, stack, and sys.getrefcount have a ref
    base_count = 3
    assert sys.getrefcount(foos[0]) == base_count

    df = pl.DataFrame({"groups": [1, 1, 2], "a": foos})
    assert sys.getrefcount(foos[0]) == base_count + 1
    del df
    assert sys.getrefcount(foos[0]) == base_count

    df = pl.DataFrame({"groups": [1, 1, 2], "a": foos})
    assert sys.getrefcount(foos[0]) == base_count + 1

    out = df.groupby("groups", maintain_order=True).agg(pl.col("a").list().alias("a"))
    assert sys.getrefcount(foos[0]) == base_count + 2
    s = out["a"].explode()
    assert sys.getrefcount(foos[0]) == base_count + 3
    del s
    assert sys.getrefcount(foos[0]) == base_count + 2

    assert out["a"].explode().to_list() == foos
    assert sys.getrefcount(foos[0]) == base_count + 2
    del out
    assert sys.getrefcount(foos[0]) == base_count + 1
    del df
    assert sys.getrefcount(foos[0]) == base_count


def test_groupby_order_dispatch() -> None:
    df = pl.DataFrame({"x": list("bab"), "y": range(3)})
    expected = pl.DataFrame({"x": ["b", "a"], "count": [2, 1]})
    assert df.groupby("x", maintain_order=True).count().frame_equal(expected)
    expected = pl.DataFrame({"x": ["b", "a"], "y": [[0, 2], [1]]})
    assert df.groupby("x", maintain_order=True).agg_list().frame_equal(expected)


def test_partitioned_groupby_order() -> None:
    # check if group ordering is maintained.
    # we only have 30 groups, so this triggers a partitioned group by
    df = pl.DataFrame({"x": [chr(v) for v in range(33, 63)], "y": range(30)})
    out = df.groupby("x", maintain_order=True).agg(pl.all().list())
    assert_series_equal(out["x"], df["x"])


def test_schema() -> None:
    df = pl.DataFrame(
        {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
    )
    expected = {"foo": pl.Int64, "bar": pl.Float64, "ham": pl.Utf8}
    assert df.schema == expected


def test_df_schema_unique() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(Exception):
        df.columns = ["a", "a"]

    with pytest.raises(Exception):
        df.rename({"b": "a"})


def test_empty_projection() -> None:
    assert pl.DataFrame({"a": [1, 2], "b": [3, 4]}).select([]).shape == (0, 0)


def test_with_column_renamed() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.rename({"b": "c"})
    expected = pl.DataFrame({"a": [1, 2], "c": [3, 4]})
    assert result.frame_equal(expected)


def test_rename_swap() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [5, 4, 3, 2, 1],
        }
    )

    out = df.rename({"a": "b", "b": "a"})
    expected = pl.DataFrame(
        {
            "b": [1, 2, 3, 4, 5],
            "a": [5, 4, 3, 2, 1],
        }
    )
    assert out.frame_equal(expected)


def test_rename_same_name() -> None:
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, 4, 5],
            "groups": ["A", "A", "B", "C", "B"],
        }
    ).lazy()
    df = df.rename({"groups": "groups"})
    df = df.select(["groups"])
    assert df.collect().to_dict(False) == {"groups": ["A", "A", "B", "C", "B"]}
    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, 4, 5],
            "groups": ["A", "A", "B", "C", "B"],
            "test": [1, 2, 3, 4, 5],
        }
    ).lazy()
    df = df.rename({"nrs": "nrs", "groups": "groups"})
    df = df.select(["groups"])
    df.collect()
    assert df.collect().to_dict(False) == {"groups": ["A", "A", "B", "C", "B"]}


def test_fill_null() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, None]})
    assert df.fill_null(4).frame_equal(pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert df.fill_null(strategy="max").frame_equal(
        pl.DataFrame({"a": [1, 2], "b": [3, 3]})
    )


def test_fill_nan() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, float("nan")]})
    assert df.fill_nan(4).frame_equal(pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert df.fill_nan(None).frame_equal(pl.DataFrame({"a": [1, 2], "b": [3, None]}))
    assert df["b"].fill_nan(5.0).to_list() == [3.0, 5.0]
    df = pl.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [datetime(1, 2, 2), datetime(2, 2, 2), datetime(3, 2, 2)],
        }
    )
    assert df.fill_nan(2.0).dtypes == [pl.Float64, pl.Datetime]


def test_shift_and_fill() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ["a", "b", "c"],
        }
    )
    result = df.shift_and_fill(periods=1, fill_value=0)
    expected = pl.DataFrame(
        {
            "foo": [0, 1, 2],
            "bar": [0, 6, 7],
            "ham": ["0", "a", "b"],
        }
    )
    assert result.frame_equal(expected)


def test_is_duplicated() -> None:
    df = pl.DataFrame({"foo": [1, 2, 2], "bar": [6, 7, 7]})
    assert df.is_duplicated().series_equal(pl.Series("", [False, True, True]))


def test_is_unique() -> None:
    df = pl.DataFrame({"foo": [1, 2, 2], "bar": [6, 7, 7]})

    assert df.is_unique().series_equal(pl.Series("", [True, False, False]))
    assert df.unique(maintain_order=True).rows() == [(1, 6), (2, 7)]
    assert df.n_unique() == 2


def test_n_unique_subsets() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 3, 4, 5],
            "b": [0.5, 0.5, 1.0, 2.0, 3.0, 3.0],
            "c": [True, True, True, False, True, True],
        }
    )
    # omitting 'subset' counts unique rows
    assert df.n_unique() == 5

    # providing it counts unique col/expr subsets
    assert df.n_unique(subset=["b", "c"]) == 4
    assert df.n_unique(subset=pl.col("c")) == 2
    assert (
        df.n_unique(subset=[(pl.col("a") // 2), (pl.col("c") | (pl.col("b") >= 2))])
        == 3
    )


def test_sample() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]})

    assert df.sample(n=2, seed=0).shape == (2, 3)
    assert df.sample(frac=0.4, seed=0).shape == (1, 3)


def test_shrink_to_fit() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]})

    assert df.shrink_to_fit(in_place=True) is df
    assert df.shrink_to_fit(in_place=False) is not df
    assert df.shrink_to_fit(in_place=False).frame_equal(df)


def test_arithmetic() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    df_mul = df * 2
    expected = pl.DataFrame({"a": [2, 4], "b": [6, 8]})
    assert df_mul.frame_equal(expected)

    df_div = df / 2
    expected = pl.DataFrame({"a": [0.5, 1.0], "b": [1.5, 2.0]})
    assert df_div.frame_equal(expected)

    df_plus = df + 2
    expected = pl.DataFrame({"a": [3, 4], "b": [5, 6]})
    assert df_plus.frame_equal(expected)

    df_minus = df - 2
    expected = pl.DataFrame({"a": [-1, 0], "b": [1, 2]})
    assert df_minus.frame_equal(expected)

    df_mod = df % 2
    expected = pl.DataFrame({"a": [1.0, 0.0], "b": [1.0, 0.0]})
    assert df_mod.frame_equal(expected)

    df2 = pl.DataFrame({"c": [10]})

    out = df + df2
    expected = pl.DataFrame({"a": [11.0, None], "b": [None, None]}).with_column(
        pl.col("b").cast(pl.Float64)
    )
    assert out.frame_equal(expected, null_equal=True)

    out = df - df2
    expected = pl.DataFrame({"a": [-9.0, None], "b": [None, None]}).with_column(
        pl.col("b").cast(pl.Float64)
    )
    assert out.frame_equal(expected, null_equal=True)

    out = df / df2
    expected = pl.DataFrame({"a": [0.1, None], "b": [None, None]}).with_column(
        pl.col("b").cast(pl.Float64)
    )
    assert out.frame_equal(expected, null_equal=True)

    out = df * df2
    expected = pl.DataFrame({"a": [10.0, None], "b": [None, None]}).with_column(
        pl.col("b").cast(pl.Float64)
    )
    assert out.frame_equal(expected, null_equal=True)

    out = df % df2
    expected = pl.DataFrame({"a": [1.0, None], "b": [None, None]}).with_column(
        pl.col("b").cast(pl.Float64)
    )
    assert out.frame_equal(expected, null_equal=True)

    # cannot do arithmetic with a sequence
    with pytest.raises(ValueError, match="Operation not supported"):
        _ = df + [1]  # type: ignore[operator]


def test_add_string() -> None:
    df = pl.DataFrame({"a": ["hi", "there"], "b": ["hello", "world"]})
    result = df + " hello"
    expected = pl.DataFrame(
        {"a": ["hi hello", "there hello"], "b": ["hello hello", "world hello"]}
    )
    assert result.frame_equal(expected)


def test_get_item() -> None:
    """Test all the methods to use [] on a dataframe."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3, 4, 5, 6]})

    # expression
    assert df.select(pl.col("a")).frame_equal(pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}))

    # tuple. The first element refers to the rows, the second element to columns
    assert df[:, :].frame_equal(df)

    # str, always refers to a column name
    assert df["a"].series_equal(pl.Series("a", [1.0, 2.0, 3.0, 4.0]))

    # int, always refers to a row index (zero-based): index=1 => second row
    assert df[1].frame_equal(pl.DataFrame({"a": [2.0], "b": [4]}))

    # range, refers to rows
    assert df[range(1, 3)].frame_equal(pl.DataFrame({"a": [2.0, 3.0], "b": [4, 5]}))

    # slice. Below an example of taking every second row
    assert df[1::2].frame_equal(pl.DataFrame({"a": [2.0, 4.0], "b": [4, 6]}))

    # numpy array: assumed to be row indices if integers, or columns if strings

    # numpy array: positive idxs.
    for np_dtype in (
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
    ):
        assert df[np.array([1, 0, 3, 2, 3, 0], dtype=np_dtype)].frame_equal(
            pl.DataFrame({"a": [2.0, 1.0, 4.0, 3.0, 4.0, 1.0], "b": [4, 3, 6, 5, 6, 3]})
        )

    # numpy array: positive and negative idxs.
    for np_dtype in (np.int8, np.int16, np.int32, np.int64):
        assert df[np.array([-1, 0, -3, -2, 3, -4], dtype=np_dtype)].frame_equal(
            pl.DataFrame({"a": [4.0, 1.0, 2.0, 3.0, 4.0, 1.0], "b": [6, 3, 4, 5, 6, 3]})
        )

    # note that we cannot use floats (even if they could be casted to integer without
    # loss)
    with pytest.raises(ValueError):
        _ = df[np.array([1.0])]

    # sequences (lists or tuples; tuple only if length != 2)
    # if strings or list of expressions, assumed to be column names
    # if bools, assumed to be a row mask
    # if integers, assumed to be row indices
    assert df[["a", "b"]].frame_equal(df)
    assert df.select([pl.col("a"), pl.col("b")]).frame_equal(df)
    assert df[[1, -4, -1, 2, 1]].frame_equal(
        pl.DataFrame({"a": [2.0, 1.0, 4.0, 3.0, 2.0], "b": [4, 3, 6, 5, 4]})
    )

    # pl.Series: strings for column selections.
    assert df[pl.Series("", ["a", "b"])].frame_equal(df)

    # pl.Series: positive idxs for row selection.
    for pl_dtype in (
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    ):
        assert df[pl.Series("", [1, 0, 3, 2, 3, 0], dtype=pl_dtype)].frame_equal(
            pl.DataFrame({"a": [2.0, 1.0, 4.0, 3.0, 4.0, 1.0], "b": [4, 3, 6, 5, 6, 3]})
        )

    # pl.Series: positive and negative idxs for row selection.
    for pl_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        assert df[pl.Series("", [-1, 0, -3, -2, 3, -4], dtype=pl_dtype)].frame_equal(
            pl.DataFrame({"a": [4.0, 1.0, 2.0, 3.0, 4.0, 1.0], "b": [6, 3, 4, 5, 6, 3]})
        )

    # Boolean masks not supported
    with pytest.raises(ValueError):
        df[np.array([True, False, True])]
    with pytest.raises(ValueError):
        df[[True, False, True], [False, True]]  # type: ignore[index]
    with pytest.raises(ValueError):
        df[pl.Series([True, False, True]), "b"]


@pytest.mark.parametrize("as_series,inner_dtype", [(True, pl.Series), (False, list)])
def test_to_dict(as_series: bool, inner_dtype: Any) -> None:
    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
            "optional": [28, 300, None, 2, -30],
        }
    )

    s = df.to_dict(as_series=as_series)
    assert isinstance(s, dict)
    for v in s.values():
        assert isinstance(v, inner_dtype)
        assert len(v) == len(df)


def test_df_broadcast() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    out = df.with_column(pl.Series([[1, 2]]))
    assert out.shape == (3, 2)


def test_product() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2, 3],
            "flt": [-1.0, 12.0, 9.0],
            "bool_0": [True, False, True],
            "bool_1": [True, True, True],
        }
    )
    out = df.product()
    expected = pl.DataFrame({"int": [6], "flt": [-108.0], "bool_0": [0], "bool_1": [1]})
    assert out.frame_equal(expected)


def test_first_last_expression(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(pl.first())
    assert out.columns == ["A"]

    out = df.select(pl.last())
    assert out.columns == ["cars"]


def test_empty_is_in() -> None:
    assert pl.DataFrame({"foo": ["a", "b", "c", "d"]}).filter(
        pl.col("foo").is_in([])
    ).shape == (0, 1)


def test_groupby_slice_expression_args() -> None:
    df = pl.DataFrame({"groups": ["a"] * 10 + ["b"] * 20, "vals": range(30)})

    out = (
        df.groupby("groups", maintain_order=True)
        .agg([pl.col("vals").slice(pl.count() * 0.1, (pl.count() // 5))])
        .explode("vals")
    )

    expected = pl.DataFrame(
        {"groups": ["a", "a", "b", "b", "b", "b"], "vals": [1, 2, 12, 13, 14, 15]}
    )
    assert out.frame_equal(expected)


def test_join_suffixes() -> None:
    df_a = pl.DataFrame({"A": [1], "B": [1]})
    df_b = pl.DataFrame({"A": [1], "B": [1]})

    join_strategies: list[JoinStrategy] = ["left", "inner", "outer", "cross"]
    for how in join_strategies:
        # no need for an assert, we error if wrong
        df_a.join(df_b, on="A", suffix="_y", how=how)["B_y"]

    df_a.join_asof(df_b, on="A", suffix="_y")["B_y"]


def test_preservation_of_subclasses_after_groupby_statements() -> None:
    """Group by operations should preserve inherited dataframe classes."""

    class SubClassedDataFrame(pl.DataFrame):
        pass

    # A group by operation should preserve the subclass
    subclassed_df = SubClassedDataFrame({"a": [1, 2], "b": [3, 4]})
    groupby = subclassed_df.groupby("a")
    assert isinstance(groupby.agg(pl.count()), SubClassedDataFrame)

    # Round-trips to GBSelection and back should also preserve subclass
    assert isinstance(
        groupby.agg(pl.col("a").count().alias("count")), SubClassedDataFrame
    )

    # Round-trips to PivotOps and back should also preserve subclass
    assert isinstance(
        groupby.pivot(pivot_column="a", values_column="b").first(),
        SubClassedDataFrame,
    )


def test_explode_empty() -> None:
    df = (
        pl.DataFrame({"x": ["a", "a", "b", "b"], "y": [1, 1, 2, 2]})
        .groupby("x")
        .agg(pl.col("y").take([]))
    )
    assert df.explode("y").shape == (0, 2)

    df = pl.DataFrame({"x": ["1", "2", "4"], "y": [["a", "b", "c"], ["d"], []]})
    assert df.explode("y").frame_equal(
        pl.DataFrame({"x": ["1", "1", "1", "2", "4"], "y": ["a", "b", "c", "d", None]})
    )


def test_asof_by_multiple_keys() -> None:
    lhs = pl.DataFrame(
        {
            "a": [-20, -19, 8, 12, 14],
            "by": [1, 1, 2, 2, 2],
            "by2": [1, 1, 2, 2, 2],
        }
    )

    rhs = pl.DataFrame(
        {
            "a": [-19, -15, 3, 5, 13],
            "by": [1, 1, 2, 2, 2],
            "by2": [1, 1, 2, 2, 2],
        }
    )

    assert (
        lhs.join_asof(rhs, on="a", by=["by", "by2"], strategy="backward")
        .select(["a", "by"])
        .frame_equal(pl.DataFrame({"a": [-20, -19, 8, 12, 14], "by": [1, 1, 2, 2, 2]}))
    )


@typing.no_type_check
def test_partition_by() -> None:
    df = pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "N": [1, 2, 2, 4, 2],
            "bar": ["k", "l", "m", "m", "l"],
        }
    )

    assert [
        a.to_dict(False) for a in df.partition_by(["foo", "bar"], maintain_order=True)
    ] == [
        {"foo": ["A"], "N": [1], "bar": ["k"]},
        {"foo": ["A"], "N": [2], "bar": ["l"]},
        {"foo": ["B", "B"], "N": [2, 4], "bar": ["m", "m"]},
        {"foo": ["C"], "N": [2], "bar": ["l"]},
    ]

    assert [a.to_dict(False) for a in df.partition_by("foo", maintain_order=True)] == [
        {"foo": ["A", "A"], "N": [1, 2], "bar": ["k", "l"]},
        {"foo": ["B", "B"], "N": [2, 4], "bar": ["m", "m"]},
        {"foo": ["C"], "N": [2], "bar": ["l"]},
    ]

    df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})
    assert df.partition_by(["a", "b"], as_dict=True)["one", 1].to_dict(False) == {
        "a": ["one"],
        "b": [1],
    }
    assert df.partition_by(["a"], as_dict=True)["one"].to_dict(False) == {
        "a": ["one", "one"],
        "b": [1, 3],
    }


@typing.no_type_check
def test_list_of_list_of_struct() -> None:
    expected = [{"list_of_list_of_struct": [[{"a": 1}, {"a": 2}]]}]
    pa_df = pa.Table.from_pylist(expected)

    df = pl.from_arrow(pa_df)
    assert df.rows() == [([[{"a": 1}, {"a": 2}]],)]
    assert df.to_dicts() == expected

    df = pl.from_arrow(pa_df[:0])
    assert df.to_dicts() == []


def test_concat_to_empty() -> None:
    assert pl.concat([pl.DataFrame([]), pl.DataFrame({"a": [1]})]).to_dict(False) == {
        "a": [1]
    }


def test_fill_null_limits() -> None:
    assert pl.DataFrame(
        {
            "a": [1, None, None, None, 5, 6, None, None, None, 10],
            "b": ["a", None, None, None, "b", "c", None, None, None, "d"],
            "c": [True, None, None, None, False, True, None, None, None, False],
        }
    ).select(
        [
            pl.all().fill_null(strategy="forward", limit=2),
            pl.all().fill_null(strategy="backward", limit=2).suffix("_backward"),
        ]
    ).to_dict(
        False
    ) == {
        "a": [1, 1, 1, None, 5, 6, 6, 6, None, 10],
        "b": ["a", "a", "a", None, "b", "c", "c", "c", None, "d"],
        "c": [True, True, True, None, False, True, True, True, None, False],
        "a_backward": [1, None, 5, 5, 5, 6, None, 10, 10, 10],
        "b_backward": ["a", None, "b", "b", "b", "c", None, "d", "d", "d"],
        "c_backward": [
            True,
            None,
            False,
            False,
            False,
            True,
            None,
            False,
            False,
            False,
        ],
    }


def test_selection_regex_and_multicol() -> None:
    test_df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [9, 10, 11, 12],
            "foo": [13, 14, 15, 16],
        }
    )

    # Selection only
    test_df.select(
        [
            pl.col(["a", "b", "c"]).suffix("_list"),
            pl.all().exclude("foo").suffix("_wild"),
            pl.col("^\\w$").suffix("_regex"),
        ]
    )

    # Multi * Single
    assert test_df.select(pl.col(["a", "b", "c"]) * pl.col("foo")).to_dict(False) == {
        "a": [13, 28, 45, 64],
        "b": [65, 84, 105, 128],
        "c": [117, 140, 165, 192],
    }
    assert test_df.select(pl.all().exclude("foo") * pl.col("foo")).to_dict(False) == {
        "a": [13, 28, 45, 64],
        "b": [65, 84, 105, 128],
        "c": [117, 140, 165, 192],
    }

    assert test_df.select(pl.col("^\\w$") * pl.col("foo")).to_dict(False) == {
        "a": [13, 28, 45, 64],
        "b": [65, 84, 105, 128],
        "c": [117, 140, 165, 192],
    }

    # Multi * Multi
    assert test_df.select(pl.col(["a", "b", "c"]) * pl.col(["a", "b", "c"])).to_dict(
        False
    ) == {"a": [1, 4, 9, 16], "b": [25, 36, 49, 64], "c": [81, 100, 121, 144]}
    assert test_df.select(pl.all().exclude("foo") * pl.all().exclude("foo")).to_dict(
        False
    ) == {"a": [1, 4, 9, 16], "b": [25, 36, 49, 64], "c": [81, 100, 121, 144]}
    assert test_df.select(pl.col("^\\w$") * pl.col("^\\w$")).to_dict(False) == {
        "a": [1, 4, 9, 16],
        "b": [25, 36, 49, 64],
        "c": [81, 100, 121, 144],
    }


def test_with_columns() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [0.5, 4, 10, 13],
            "c": [True, True, False, True],
        }
    )
    srs_named = pl.Series("f", [3, 2, 1, 0])
    srs_unnamed = pl.Series(values=[3, 2, 1, 0])

    expected = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [0.5, 4, 10, 13],
            "c": [True, True, False, True],
            "d": [0.5, 8.0, 30.0, 52.0],
            "e": [False, False, True, False],
            "f": [3, 2, 1, 0],
        }
    )

    # as exprs list
    dx = df.with_columns(
        [(pl.col("a") * pl.col("b")).alias("d"), ~pl.col("c").alias("e"), srs_named]
    )
    assert_frame_equal(dx, expected)

    # as **kwargs (experimental feature: requires opt-in)
    pl.Config.with_columns_kwargs = True

    dx = df.with_columns(
        d=pl.col("a") * pl.col("b"),
        e=~pl.col("c"),
        f=srs_unnamed,
    )
    assert_frame_equal(dx, expected)

    # mixed
    dx = df.with_columns(
        [(pl.col("a") * pl.col("b")).alias("d")],
        e=~pl.col("c"),
        f=srs_unnamed,
    )
    assert_frame_equal(dx, expected)

    # at least one of exprs/**named_exprs required
    with pytest.raises(ValueError):
        _ = df.with_columns()


def test_len_compute(df: pl.DataFrame) -> None:
    df = df.with_column(pl.struct(["list_bool", "cat"]).alias("struct"))
    filtered = df.filter(pl.col("bools"))
    for col in filtered.columns:
        assert len(filtered[col]) == 1

    taken = df[[1, 2], :]
    for col in taken.columns:
        assert len(taken[col]) == 2


def test_filter_sequence() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    assert df.filter([True, False, True])["a"].to_list() == [1, 3]
    assert df.filter(np.array([True, False, True]))["a"].to_list() == [1, 3]


def test_indexing_set() -> None:
    df = pl.DataFrame({"bool": [True, True], "str": ["N/A", "N/A"], "nr": [1, 2]})

    df[0, "bool"] = False
    df[0, "nr"] = 100
    df[0, "str"] = "foo"

    assert df.to_dict(False) == {
        "bool": [False, True],
        "str": ["foo", "N/A"],
        "nr": [100, 2],
    }


def test_set() -> None:
    # Setting a dataframe using indices is deprecated.
    # We keep these tests because we only generate a warning.
    np.random.seed(1)
    df = pl.DataFrame(
        {"foo": np.random.rand(10), "bar": np.arange(10), "ham": ["h"] * 10}
    )
    with pytest.raises(
        TypeError,
        match=r"'DataFrame' object does not support "
        r"'Series' assignment by index. Use "
        r"'DataFrame.with_columns'",
    ):
        df["new"] = np.random.rand(10)

    with pytest.raises(
        ValueError,
        match=r"Not allowed to set 'DataFrame' by "
        r"boolean mask in the row position. "
        r"Consider using 'DataFrame.with_columns'",
    ):
        df[df["ham"] > 0.5, "ham"] = "a"
    with pytest.raises(
        ValueError,
        match=r"Not allowed to set 'DataFrame' by "
        r"boolean mask in the row position. "
        r"Consider using 'DataFrame.with_columns'",
    ):
        df[[True, False], "ham"] = "a"

    # set 2D
    df = pl.DataFrame({"b": [0, 0]})
    df[["A", "B"]] = [[1, 2], [1, 2]]

    with pytest.raises(ValueError):
        df[["C", "D"]] = 1
    with pytest.raises(ValueError):
        df[["C", "D"]] = [1, 1]
    with pytest.raises(ValueError):
        df[["C", "D"]] = [[1, 2, 3], [1, 2, 3]]

    # set tuple
    df = pl.DataFrame({"b": [0, 0]})
    df[0, "b"] = 1
    assert df[0, "b"] == 1

    df[0, 0] = 2
    assert df[0, "b"] == 2

    # row and col selection have to be int or str
    with pytest.raises(ValueError):
        df[:, [1]] = 1  # type: ignore[index]
    with pytest.raises(ValueError):
        df[True, :] = 1  # type: ignore[index]

    # needs to be a 2 element tuple
    with pytest.raises(ValueError):
        df[(1, 2, 3)] = 1

    # we cannot index with any type, such as bool
    with pytest.raises(ValueError):
        df[True] = 1  # type: ignore[index]


def test_union_with_aliases_4770() -> None:
    lf = pl.DataFrame(
        {
            "a": [1, None],
            "b": [3, 4],
        }
    ).lazy()

    lf = pl.concat(
        [
            lf.select([pl.col("a").alias("x")]),
            lf.select([pl.col("b").alias("x")]),
        ]
    ).filter(pl.col("x").is_not_null())

    assert lf.collect()["x"].to_list() == [1, 3, 4]


@pytest.mark.skip(reason="locale issues, see #5177")
def test_init_with_timezone() -> None:
    for tu in DTYPE_TEMPORAL_UNITS | frozenset([None]):
        df = pl.DataFrame(
            data={
                "d1": [datetime(2022, 10, 12, 12, 30)],
                "d2": [datetime(2022, 10, 12, 12, 30)],
            },
            columns=[
                ("d1", pl.Datetime(tu, "America/New_York")),  # type: ignore[arg-type]
                ("d2", pl.Datetime(tu, "Asia/Tokyo")),  # type: ignore[arg-type]
            ],
        )
        # note: setting timezone doesn't change the underlying/physical value...
        assert (df["d1"].to_physical() == df["d2"].to_physical()).all()

        # ...but (as expected) it _does_ change the interpretation of that value
        assert df.rows() == [
            (
                datetime(2022, 10, 12, 8, 30, tzinfo=ZoneInfo("America/New_York")),
                datetime(2022, 10, 12, 21, 30, tzinfo=ZoneInfo("Asia/Tokyo")),
            )
        ]
