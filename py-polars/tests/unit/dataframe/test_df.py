from __future__ import annotations

import contextlib
import sys
import textwrap
import typing
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from io import BytesIO
from operator import floordiv, truediv
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, cast

import numpy as np
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal, assert_equal

import polars as pl
import polars.selectors as cs
from polars.datatypes import DTYPE_TEMPORAL_UNITS, FLOAT_DTYPES, INTEGER_DTYPES
from polars.exceptions import ComputeError, TimeZoneAwareConstructorWarning
from polars.testing import (
    assert_frame_equal,
    assert_frame_not_equal,
    assert_series_equal,
)
from polars.testing.parametric import columns
from polars.utils._construction import iterable_to_pydf

if TYPE_CHECKING:
    from polars.type_aliases import IndexOrder, JoinStrategy, UniqueKeepStrategy

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
else:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo


def test_version() -> None:
    pl.__version__


def test_null_count() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", None]})
    assert df.null_count().shape == (1, 2)
    assert df.null_count().row(0) == (0, 1)
    assert df.null_count().row(np.int64(0)) == (0, 1)  # type: ignore[call-overload]


def test_init_empty() -> None:
    # test various flavours of empty init
    for empty in (None, (), [], {}, pa.Table.from_arrays([])):
        df = pl.DataFrame(empty)
        assert df.shape == (0, 0)
        assert df.is_empty()

    # note: cannot use df (empty or otherwise) in boolean context
    empty_df = pl.DataFrame()
    with pytest.raises(TypeError, match="ambiguous"):
        not empty_df


def test_special_char_colname_init() -> None:
    from string import punctuation

    with pl.StringCache():
        cols = [(c.name, c.dtype) for c in columns(punctuation)]
        df = pl.DataFrame(schema=cols)

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

    # get column by name
    assert_series_equal(df.get_column("b"), pl.Series("b", [1.0, 2.0, 3.0]))

    # get column by index
    assert_series_equal(df.to_series(1), pl.Series("b", [1.0, 2.0, 3.0]))
    assert_series_equal(df.to_series(-1), pl.Series("c", ["a", "b", "c"]))

    # select columns by mask
    assert df[:2, :1].rows() == [(1,), (2,)]
    assert df[:2, ["a"]].rows() == [(1,), (2,)]

    # column selection by string(s) in first dimension
    assert df["a"].to_list() == [1, 2, 3]
    assert df["b"].to_list() == [1.0, 2.0, 3.0]
    assert df["c"].to_list() == ["a", "b", "c"]

    # row selection by integers(s) in first dimension
    assert_frame_equal(df[0], pl.DataFrame({"a": [1], "b": [1.0], "c": ["a"]}))
    assert_frame_equal(df[-1], pl.DataFrame({"a": [3], "b": [3.0], "c": ["c"]}))

    # row, column selection when using two dimensions
    assert df[:, "a"].to_list() == [1, 2, 3]
    assert df[:, 1].to_list() == [1.0, 2.0, 3.0]
    assert df[:2, 2].to_list() == ["a", "b"]

    assert_frame_equal(
        df[[1, 2]], pl.DataFrame({"a": [2, 3], "b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert_frame_equal(
        df[[-1, -2]], pl.DataFrame({"a": [3, 2], "b": [3.0, 2.0], "c": ["c", "b"]})
    )

    assert df[["a", "b"]].columns == ["a", "b"]
    assert_frame_equal(
        df[[1, 2], [1, 2]], pl.DataFrame({"b": [2.0, 3.0], "c": ["b", "c"]})
    )
    assert typing.cast(str, df[1, 2]) == "b"
    assert typing.cast(float, df[1, 1]) == 2.0
    assert typing.cast(int, df[2, 0]) == 3

    assert df[[2], ["a", "b"]].rows() == [(3, 3.0)]
    assert df.to_series(0).name == "a"
    assert (df["a"] == df["a"]).sum() == 3
    assert (df["c"] == df["a"].cast(str)).sum() == 0
    assert df[:, "a":"b"].rows() == [(1, 1.0), (2, 2.0), (3, 3.0)]  # type: ignore[misc]
    assert df[:, "a":"c"].columns == ["a", "b", "c"]  # type: ignore[misc]
    expect = pl.DataFrame({"c": ["b"]})
    assert_frame_equal(df[1, [2]], expect)
    expect = pl.DataFrame({"b": [1.0, 3.0]})
    assert_frame_equal(df[[0, 2], [1]], expect)
    assert typing.cast(str, df[0, "c"]) == "a"
    assert typing.cast(str, df[1, "c"]) == "b"
    assert typing.cast(str, df[2, "c"]) == "c"
    assert typing.cast(int, df[0, "a"]) == 1

    # more slicing
    expect = pl.DataFrame({"a": [3, 2, 1], "b": [3.0, 2.0, 1.0], "c": ["c", "b", "a"]})
    assert_frame_equal(df[::-1], expect)
    expect = pl.DataFrame({"a": [1, 2], "b": [1.0, 2.0], "c": ["a", "b"]})
    assert_frame_equal(df[:-1], expect)

    expect = pl.DataFrame({"a": [1, 3], "b": [1.0, 3.0], "c": ["a", "c"]})
    assert_frame_equal(df[::2], expect)

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


def test_from_arrow(monkeypatch: Any) -> None:
    monkeypatch.setenv("POLARS_ACTIVATE_DECIMAL", "1")
    tbl = pa.table(
        {
            "a": pa.array([1, 2], pa.timestamp("s")),
            "b": pa.array([1, 2], pa.timestamp("ms")),
            "c": pa.array([1, 2], pa.timestamp("us")),
            "d": pa.array([1, 2], pa.timestamp("ns")),
            "e": pa.array([1, 2], pa.int32()),
            "decimal1": pa.array([1, 2], pa.decimal128(2, 1)),
        }
    )
    record_batches = tbl.to_batches(max_chunksize=1)
    expected_schema = {
        "a": pl.Datetime("ms"),
        "b": pl.Datetime("ms"),
        "c": pl.Datetime("us"),
        "d": pl.Datetime("ns"),
        "e": pl.Int32,
        "decimal1": pl.Decimal(1, 2),
    }
    expected_data = [
        (
            datetime(1970, 1, 1, 0, 0, 1),
            datetime(1970, 1, 1, 0, 0, 0, 1000),
            datetime(1970, 1, 1, 0, 0, 0, 1),
            datetime(1970, 1, 1, 0, 0),
            1,
            Decimal("1.0"),
        ),
        (
            datetime(1970, 1, 1, 0, 0, 2),
            datetime(1970, 1, 1, 0, 0, 0, 2000),
            datetime(1970, 1, 1, 0, 0, 0, 2),
            datetime(1970, 1, 1, 0, 0),
            2,
            Decimal("2.0"),
        ),
    ]
    for arrow_data in (tbl, record_batches, (rb for rb in record_batches)):
        df = cast(pl.DataFrame, pl.from_arrow(arrow_data))
        assert df.schema == expected_schema
        assert df.rows() == expected_data

    empty_tbl = tbl[:0]  # no rows
    df = cast(pl.DataFrame, pl.from_arrow(empty_tbl))
    assert df.schema == expected_schema
    assert df.rows() == []

    # try a single column dtype override
    for t in (tbl, empty_tbl):
        df = pl.DataFrame(t, schema_overrides={"e": pl.Int8})
        override_schema = expected_schema.copy()
        override_schema["e"] = pl.Int8
        assert df.schema == override_schema
        assert df.rows() == expected_data[: (len(df))]

    # init from record batches with overrides
    df = pl.DataFrame(
        {
            "id": ["a123", "b345", "c567", "d789", "e101"],
            "points": [99, 45, 50, 85, 35],
        }
    )
    tbl = df.to_arrow()
    batches = tbl.to_batches(max_chunksize=3)

    df0: pl.DataFrame = pl.from_arrow(batches)  # type: ignore[assignment]
    df1: pl.DataFrame = pl.from_arrow(  # type: ignore[assignment]
        data=batches,
        schema=["x", "y"],
        schema_overrides={"y": pl.Int32},
    )
    df2: pl.DataFrame = pl.from_arrow(  # type: ignore[assignment]
        data=batches[0],
        schema=["x", "y"],
        schema_overrides={"y": pl.Int32},
    )

    assert df0.rows() == df.rows()
    assert df1.rows() == df.rows()
    assert df2.rows() == df.rows()[:3]

    assert df0.schema == {"id": pl.Utf8, "points": pl.Int64}
    assert df1.schema == {"x": pl.Utf8, "y": pl.Int32}
    assert df2.schema == {"x": pl.Utf8, "y": pl.Int32}

    with pytest.raises(TypeError, match="Cannot convert str"):
        pl.from_arrow(data="xyz")

    with pytest.raises(TypeError, match="Cannot convert int"):
        pl.from_arrow(data=(x for x in (1, 2, 3)))


def test_dataframe_membership_operator() -> None:
    # cf. issue #4032
    df = pl.DataFrame({"name": ["Jane", "John"], "age": [20, 30]})
    assert "name" in df
    assert "phone" not in df
    assert df._ipython_key_completions_() == ["name", "age"]


def test_sort() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    assert_frame_equal(df.sort("a"), pl.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]}))
    assert_frame_equal(
        df.sort(["a", "b"]), pl.DataFrame({"a": [1, 2, 3], "b": [2, 1, 3]})
    )


def test_sort_maintain_order() -> None:
    l1 = (
        pl.LazyFrame({"A": [1] * 4, "B": ["A", "B", "C", "D"]})
        .sort("A", maintain_order=True)
        .slice(0, 3)
        .collect()["B"]
        .to_list()
    )
    l2 = (
        pl.LazyFrame({"A": [1] * 4, "B": ["A", "B", "C", "D"]})
        .sort("A")
        .collect()
        .slice(0, 3)["B"]
        .to_list()
    )
    assert l1 == l2 == ["A", "B", "C"]


def test_replace() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    s = pl.Series("c", [True, False, True])
    with pytest.deprecated_call():
        df.replace("a", s)
    assert_frame_equal(df, pl.DataFrame({"a": [True, False, True], "b": [1, 2, 3]}))


def test_assignment() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [2, 3, 4]})
    df = df.with_columns(pl.col("foo").alias("foo"))
    # make sure that assignment does not change column order
    assert df.columns == ["foo", "bar"]
    df = df.with_columns(
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


def test_take_misc(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars

    # Out of bounds error.
    with pytest.raises(pl.ComputeError):
        (
            df.sort("fruits").select(
                [pl.col("B").reverse().take([1, 2]).implode().over("fruits"), "fruits"]
            )
        )

    # Null indices.
    assert_frame_equal(
        df.select(pl.col("fruits").take(pl.Series([0, None]))),
        pl.DataFrame({"fruits": ["banana", None]}),
    )

    for index in [[0, 1], pl.Series([0, 1]), np.array([0, 1])]:
        out = df.sort("fruits").select(
            [
                pl.col("B")
                .reverse()
                .take(index)  # type: ignore[arg-type]
                .over("fruits", mapping_strategy="join"),
                "fruits",
            ]
        )

        assert out[0, "B"].to_list() == [2, 3]
        assert out[4, "B"].to_list() == [1, 4]

    out = df.sort("fruits").select(
        [pl.col("B").reverse().take(pl.lit(1)).over("fruits"), "fruits"]
    )
    assert out[0, "B"] == 3
    assert out[4, "B"] == 4


def test_pipe() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, None, 8]})

    def _multiply(data: pl.DataFrame, mul: int) -> pl.DataFrame:
        return data * mul

    result = df.pipe(_multiply, mul=3)

    assert_frame_equal(result, df * 3)


def test_explode() -> None:
    df = pl.DataFrame({"letters": ["c", "a"], "nrs": [[1, 2], [1, 3]]})
    out = df.explode("nrs")
    assert out["letters"].to_list() == ["c", "c", "a", "a"]
    assert out["nrs"].to_list() == [1, 2, 1, 3]


@pytest.mark.parametrize(
    ("stack", "exp_shape", "exp_columns"),
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
        assert_frame_equal(df, expected)
    else:
        df_out = df.hstack(df2, in_place=False)
        assert_frame_equal(df_out, expected)


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
    with pytest.raises(pl.ComputeError):
        pl.read_parquet(f)


@pytest.mark.parametrize(
    "read_function", [pl.read_parquet, pl.read_csv, pl.read_ipc, pl.read_avro]
)
def test_read_missing_file(read_function: Callable[[Any], pl.DataFrame]) -> None:
    with pytest.raises(FileNotFoundError, match="fake_file"):
        read_function("fake_file")


def test_shift() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    a = df.shift(1)
    b = pl.DataFrame(
        {"A": [None, "a", "b"], "B": [None, 1, 3]},
    )
    assert_frame_equal(a, b)


def test_custom_group_by() -> None:
    df = pl.DataFrame({"a": [1, 2, 1, 1], "b": ["a", "b", "c", "c"]})
    out = df.group_by("b", maintain_order=True).agg(
        [pl.col("a").map_elements(lambda x: x.sum(), return_dtype=pl.Int64)]
    )
    assert out.rows() == [("a", 1), ("b", 2), ("c", 2)]


def test_multiple_columns_drop() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    # List input
    out = df.drop(["a", "b"])
    assert out.columns == ["c"]
    # Positional input
    out = df.drop("b", "c")
    assert out.columns == ["a"]


def test_concat() -> None:
    df1 = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3], "c": [1, 2, 3]})
    df2 = pl.concat([df1, df1])

    assert df2.shape == (6, 3)
    assert df2.n_chunks() == 1  # the default is to rechunk
    assert df2.rows() == df1.rows() + df1.rows()
    assert pl.concat([df1, df1], rechunk=False).n_chunks() == 2

    # concat from generator of frames
    df3 = pl.concat(items=(df1 for _ in range(2)))
    assert_frame_equal(df2, df3)

    # check that df4 is not modified following concat of itself
    df4 = pl.from_records(((1, 2), (1, 2)))
    _ = pl.concat([df4, df4, df4])

    assert df4.shape == (2, 2)
    assert df4.rows() == [(1, 1), (2, 2)]

    # misc error conditions
    with pytest.raises(ValueError):
        _ = pl.concat([])

    with pytest.raises(ValueError):
        pl.concat([df1, df1], how="rubbish")  # type: ignore[arg-type]


def test_arg_where() -> None:
    s = pl.Series([True, False, True, False])
    assert_series_equal(pl.arg_where(s, eager=True).cast(int), pl.Series([0, 2]))


def test_to_dummies() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5]})
    dummies = df.to_dummies()

    assert dummies["A_a"].to_list() == [1, 0, 0]
    assert dummies["A_b"].to_list() == [0, 1, 0]
    assert dummies["A_c"].to_list() == [0, 0, 1]

    df = pl.DataFrame({"a": [1, 2, 3]})
    res = df.to_dummies()

    expected = pl.DataFrame(
        {"a_1": [1, 0, 0], "a_2": [0, 1, 0], "a_3": [0, 0, 1]}
    ).with_columns(pl.all().cast(pl.UInt8))
    assert_frame_equal(res, expected)

    df = pl.DataFrame(
        {
            "i": [1, 2, 3],
            "category": ["dog", "cat", "cat"],
        },
        schema={"i": pl.Int32, "category": pl.Categorical},
    )
    expected = pl.DataFrame(
        {
            "i": [1, 2, 3],
            "category|cat": [0, 1, 1],
            "category|dog": [1, 0, 0],
        },
        schema={"i": pl.Int32, "category|cat": pl.UInt8, "category|dog": pl.UInt8},
    )
    for _cols in ("category", cs.string()):
        result = df.to_dummies(columns=["category"], separator="|")
        assert_frame_equal(result, expected)

    # test sorted fast path
    assert pl.DataFrame({"x": pl.arange(0, 3, eager=True)}).to_dummies("x").to_dict(
        False
    ) == {"x_0": [1, 0, 0], "x_1": [0, 1, 0], "x_2": [0, 0, 1]}


def test_to_dummies_drop_first() -> None:
    df = pl.DataFrame(
        {
            "foo": [0, 1, 2],
            "bar": [3, 4, 5],
            "baz": ["x", "y", "z"],
        }
    )
    dm = df.to_dummies()
    dd = df.to_dummies(drop_first=True)

    assert dd.columns == ["foo_1", "foo_2", "bar_4", "bar_5", "baz_y", "baz_z"]
    assert set(dm.columns) - set(dd.columns) == {"foo_0", "bar_3", "baz_x"}
    assert dm.select(dd.columns).frame_equal(dd)
    assert dd.rows() == [
        (0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 1, 0),
        (0, 1, 0, 1, 0, 1),
    ]


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
    assert_frame_equal(df, pl.DataFrame(data))


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

    assert_series_equal(
        df.fold(lambda s1, s2: s1 + s2), pl.Series("a", [4.0, 5.0, 9.0])
    )
    assert_series_equal(
        df.fold(lambda s1, s2: s1.zip_with(s1 < s2, s2)),
        pl.Series("a", [1.0, 1.0, 3.0]),
    )

    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.fold(lambda s1, s2: s1 + s2)
    assert_series_equal(out, pl.Series("a", ["foo11.0", "bar22.0", "233.0"]))

    df = pl.DataFrame({"a": [3, 2, 1], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})
    # just check dispatch. values are tested on rust side.
    assert len(df.sum(axis=1)) == 3
    assert len(df.mean(axis=1)) == 3
    assert len(df.min(axis=1)) == 3
    assert len(df.max(axis=1)) == 3

    df_width_one = df[["a"]]
    assert_series_equal(df_width_one.fold(lambda s1, s2: s1), df["a"])


def test_fold_filter() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [0, 1, 2]})

    out = df.filter(
        pl.fold(
            acc=pl.lit(True),
            function=lambda a, b: a & b,
            exprs=[pl.col(c) > 1 for c in df.columns],
        )
    )

    assert out.shape == (1, 2)
    assert out.rows() == [(3, 2)]

    out = df.filter(
        pl.fold(
            acc=pl.lit(True),
            function=lambda a, b: a | b,
            exprs=[pl.col(c) > 1 for c in df.columns],
        )
    )

    assert out.shape == (3, 2)
    assert out.rows() == [(1, 0), (2, 1), (3, 2)]


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


def test_init_series_edge_cases() -> None:
    # confirm that we don't modify the name of the input series in-place
    s1 = pl.Series("X", [1, 2, 3])
    df1 = pl.DataFrame({"A": s1}, schema_overrides={"A": pl.UInt8})
    assert s1.name == "X"
    assert df1["A"].name == "A"

    # init same series object under different names
    df2 = pl.DataFrame({"A": s1, "B": s1})
    assert df2.rows(named=True) == [
        {"A": 1, "B": 1},
        {"A": 2, "B": 2},
        {"A": 3, "B": 3},
    ]

    # empty series names should not be overwritten
    s2 = pl.Series([1, 2, 3])
    s3 = pl.Series([2, 3, 4])
    df3 = pl.DataFrame([s2, s3])
    assert s2.name == s3.name == ""
    assert df3.columns == ["column_0", "column_1"]


def test_head_group_by() -> None:
    commodity_prices = {
        "commodity": [
            "Wheat",
            "Wheat",
            "Wheat",
            "Wheat",
            "Corn",
            "Corn",
            "Corn",
            "Corn",
            "Corn",
        ],
        "location": [
            "StPaul",
            "StPaul",
            "StPaul",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
            "Chicago",
        ],
        "seller": [
            "Bob",
            "Charlie",
            "Susan",
            "Paul",
            "Ed",
            "Mary",
            "Paul",
            "Charlie",
            "Norman",
        ],
        "price": [1.0, 0.7, 0.8, 0.55, 2.0, 3.0, 2.4, 1.8, 2.1],
    }
    df = pl.DataFrame(commodity_prices)

    # this query flexes the wildcard exclusion quite a bit.
    keys = ["commodity", "location"]
    out = (
        df.sort(by="price", descending=True)
        .group_by(keys, maintain_order=True)
        .agg([pl.col("*").exclude(keys).head(2).keep_name()])
        .explode(pl.col("*").exclude(keys))
    )

    assert out.shape == (5, 4)
    assert out.rows() == [
        ("Corn", "Chicago", "Mary", 3.0),
        ("Corn", "Chicago", "Paul", 2.4),
        ("Wheat", "StPaul", "Bob", 1.0),
        ("Wheat", "StPaul", "Susan", 0.8),
        ("Wheat", "Chicago", "Paul", 0.55),
    ]

    df = pl.DataFrame(
        {"letters": ["c", "c", "a", "c", "a", "b"], "nrs": [1, 2, 3, 4, 5, 6]}
    )
    out = df.group_by("letters").tail(2).sort("letters")
    assert_frame_equal(
        out,
        pl.DataFrame({"letters": ["a", "a", "b", "c", "c"], "nrs": [3, 5, 6, 2, 4]}),
    )
    out = df.group_by("letters").head(2).sort("letters")
    assert_frame_equal(
        out,
        pl.DataFrame({"letters": ["a", "a", "b", "c", "c"], "nrs": [3, 5, 6, 1, 2]}),
    )


def test_is_null_is_not_null() -> None:
    df = pl.DataFrame({"nrs": [1, 2, None]})
    assert df.select(pl.col("nrs").is_null())["nrs"].to_list() == [False, False, True]
    assert df.select(pl.col("nrs").is_not_null())["nrs"].to_list() == [
        True,
        True,
        False,
    ]


def test_is_nan_is_not_nan() -> None:
    df = pl.DataFrame({"nrs": np.array([1, 2, np.nan])})
    assert df.select(pl.col("nrs").is_nan())["nrs"].to_list() == [False, False, True]
    assert df.select(pl.col("nrs").is_not_nan())["nrs"].to_list() == [True, True, False]


def test_is_finite_is_infinite() -> None:
    df = pl.DataFrame({"nrs": np.array([1, 2, np.inf])})
    assert df.select(pl.col("nrs").is_infinite())["nrs"].to_list() == [
        False,
        False,
        True,
    ]
    assert df.select(pl.col("nrs").is_finite())["nrs"].to_list() == [True, True, False]


def test_len() -> None:
    df = pl.DataFrame({"nrs": [1, 2, 3]})
    assert cast(int, df.select(pl.col("nrs").len()).item()) == 3
    assert len(pl.DataFrame()) == 0


def test_multiple_column_sort() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [2, 2, 3], "c": [1.0, 2.0, 3.0]})
    out = df.sort([pl.col("b"), pl.col("c").reverse()])
    assert list(out["c"]) == [2.0, 1.0, 3.0]
    assert list(out["b"]) == [2, 2, 3]

    # Explicitly specify numpy dtype because of different defaults on Windows
    df = pl.DataFrame({"a": np.arange(1, 4, dtype=np.int64), "b": ["a", "a", "b"]})

    assert_frame_equal(
        df.sort("a", descending=True),
        pl.DataFrame({"a": [3, 2, 1], "b": ["b", "a", "a"]}),
    )
    assert_frame_equal(
        df.sort("b", descending=True),
        pl.DataFrame({"a": [3, 1, 2], "b": ["b", "a", "a"]}),
    )
    assert_frame_equal(
        df.sort(["b", "a"], descending=[False, True]),
        pl.DataFrame({"a": [2, 1, 3], "b": ["a", "a", "b"]}),
    )


def test_cast_frame() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.5, 3.0],
            "b": [4, 5, None],
            "c": [True, False, True],
            "d": [date(2020, 1, 2), date(2021, 3, 4), date(2022, 5, 6)],
        }
    )

    # cast via col:dtype map
    assert df.cast(
        dtypes={"b": pl.Float32, "c": pl.Utf8, "d": pl.Datetime("ms")}
    ).schema == {
        "a": pl.Float64,
        "b": pl.Float32,
        "c": pl.Utf8,
        "d": pl.Datetime("ms"),
    }

    # cast via selector:dtype map
    assert df.cast(
        {
            cs.numeric(): pl.UInt8,
            cs.temporal(): pl.Utf8,
        }
    ).rows() == [
        (1, 4, True, "2020-01-02"),
        (2, 5, False, "2021-03-04"),
        (3, None, True, "2022-05-06"),
    ]

    # cast all fields to a single type
    assert df.cast(pl.Utf8).to_dict(False) == {
        "a": ["1.0", "2.5", "3.0"],
        "b": ["4", "5", None],
        "c": ["true", "false", "true"],
        "d": ["2020-01-02", "2021-03-04", "2022-05-06"],
    }


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
    df = df.with_columns(pl.col("e").cast(pl.Categorical))
    expected = pl.DataFrame(
        {
            "describe": [
                "count",
                "null_count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ],
            "a": [3.0, 0.0, 2.2666667, 1.101514, 1.0, 1.0, 2.8, 3.0, 3.0],
            "b": [3.0, 1.0, 4.5, 0.7071067811865476, 4.0, 4.0, 5.0, 5.0, 5.0],
            "c": [
                3.0,
                0.0,
                0.6666666666666666,
                0.5773502588272095,
                0.0,
                None,
                None,
                None,
                1.0,
            ],
            "d": ["3", "1", None, None, "b", None, None, None, "c"],
            "e": ["3", "1", None, None, None, None, None, None, None],
            "f": ["3", "0", None, None, "2020-01-01", None, None, None, "2022-01-01"],
        }
    )
    assert_frame_equal(df.describe(), expected)

    # struct
    df = pl.DataFrame(
        {
            "numerical": [1, 2, 1, None],
            "struct": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}, None],
            "list": [[1, 2], [3, 4], [1, 2], None],
        }
    )

    assert df.describe().to_dict(False) == {
        "describe": [
            "count",
            "null_count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ],
        "numerical": [
            4.0,
            1.0,
            1.3333333333333333,
            0.5773502691896257,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
        ],
        "struct": ["4", "1", None, None, None, None, None, None, None],
        "list": ["4", "1", None, None, None, None, None, None, None],
    }

    for pcts in (None, []):  # type:ignore[var-annotated]
        assert df.describe(percentiles=pcts).rows() == [
            ("count", 4.0, "4", "4"),
            ("null_count", 1.0, "1", "1"),
            ("mean", 1.3333333333333333, None, None),
            ("std", 0.5773502691896257, None, None),
            ("min", 1.0, None, None),
            ("max", 2.0, None, None),
        ]

    described = df.describe(percentiles=(0.2, 0.4, 0.5, 0.6, 0.8))
    assert described.schema == {
        "describe": pl.Utf8,
        "numerical": pl.Float64,
        "struct": pl.Utf8,
        "list": pl.Utf8,
    }
    assert described.rows() == [
        ("count", 4.0, "4", "4"),
        ("null_count", 1.0, "1", "1"),
        ("mean", 1.3333333333333333, None, None),
        ("std", 0.5773502691896257, None, None),
        ("min", 1.0, None, None),
        ("20%", 1.0, None, None),
        ("40%", 1.0, None, None),
        ("50%", 1.0, None, None),
        ("60%", 1.0, None, None),
        ("80%", 2.0, None, None),
        ("max", 2.0, None, None),
    ]


def test_duration_arithmetic() -> None:
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


def test_assign() -> None:
    # check if can assign in case of a single column
    df = pl.DataFrame({"a": [1, 2, 3]})
    # test if we can assign in case of single column
    df = df.with_columns(pl.col("a") * 2)
    assert list(df["a"]) == [2, 4, 6]


@pytest.mark.parametrize(
    ("order", "f_contiguous", "c_contiguous"),
    [("fortran", True, False), ("c", False, True)],
)
def test_to_numpy(order: IndexOrder, f_contiguous: bool, c_contiguous: bool) -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    out_array = df.to_numpy(order=order)
    expected_array = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    assert_array_equal(out_array, expected_array)
    assert out_array.flags["F_CONTIGUOUS"] == f_contiguous
    assert out_array.flags["C_CONTIGUOUS"] == c_contiguous

    structured_array = df.to_numpy(structured=True, order=order)
    expected_array = np.array(
        [(1, 1.0), (2, 2.0), (3, 3.0)], dtype=[("a", "<i8"), ("b", "<f8")]
    )
    assert_array_equal(structured_array, expected_array)
    assert structured_array.flags["F_CONTIGUOUS"]

    # check string conversion; if no nulls can optimise as a fixed-width dtype
    df = pl.DataFrame({"s": ["x", "y", None]})
    assert df["s"].has_validity()
    assert_array_equal(
        df.to_numpy(structured=True),
        np.array([("x",), ("y",), (None,)], dtype=[("s", "O")]),
    )
    assert not df["s"][:2].has_validity()
    assert_array_equal(
        df[:2].to_numpy(structured=True),
        np.array([("x",), ("y",)], dtype=[("s", "<U1")]),
    )


def test_to_numpy_structured() -> None:
    # round-trip structured array: validate init/export
    structured_array = np.array(
        [
            ("Google Pixel 7", 521.90, True),
            ("Apple iPhone 14 Pro", 999.00, True),
            ("OnePlus 11", 699.00, True),
            ("Samsung Galaxy S23 Ultra", 1199.99, False),
        ],
        dtype=np.dtype(
            [
                ("product", "U24"),
                ("price_usd", "float64"),
                ("in_stock", "bool"),
            ]
        ),
    )
    df = pl.from_numpy(structured_array)
    assert df.schema == {
        "product": pl.Utf8,
        "price_usd": pl.Float64,
        "in_stock": pl.Boolean,
    }
    exported_array = df.to_numpy(structured=True)
    assert exported_array["product"].dtype == np.dtype("U24")
    assert_array_equal(exported_array, structured_array)

    # none/nan values
    df = pl.DataFrame({"x": ["a", None, "b"], "y": [5.5, None, -5.5]})
    exported_array = df.to_numpy(structured=True)

    assert exported_array.dtype == np.dtype([("x", object), ("y", float)])
    for name in df.columns:
        assert_equal(
            list(exported_array[name]),
            (
                df[name].fill_null(float("nan"))
                if df.schema[name] in FLOAT_DTYPES
                else df[name]
            ).to_list(),
        )


def test__array__() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})

    out_array = np.asarray(df.to_numpy())
    expected_array = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float64)
    assert_array_equal(out_array, expected_array)
    assert out_array.flags["F_CONTIGUOUS"] is True

    out_array = np.asarray(df.to_numpy(), np.uint8)
    expected_array = np.array([[1, 1], [2, 2], [3, 3]], dtype=np.uint8)
    assert_array_equal(out_array, expected_array)
    assert out_array.flags["F_CONTIGUOUS"] is True


def test_arg_sort_by(df: pl.DataFrame) -> None:
    idx_df = df.select(
        pl.arg_sort_by(["int_nulls", "floats"], descending=[False, True]).alias("idx")
    )
    assert (idx_df["idx"] == [1, 0, 2]).all()

    idx_df = df.select(
        pl.arg_sort_by(["int_nulls", "floats"], descending=False).alias("idx")
    )
    assert (idx_df["idx"] == [1, 0, 2]).all()

    df = pl.DataFrame({"x": [0, 0, 0, 1, 1, 2], "y": [9, 9, 8, 7, 6, 6]})
    for expr, expected in (
        (pl.arg_sort_by(["x", "y"]), [2, 0, 1, 4, 3, 5]),
        (pl.arg_sort_by(["x", "y"], descending=[True, True]), [5, 3, 4, 0, 1, 2]),
        (pl.arg_sort_by(["x", "y"], descending=[True, False]), [5, 4, 3, 2, 0, 1]),
        (pl.arg_sort_by(["x", "y"], descending=[False, True]), [0, 1, 2, 3, 4, 5]),
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
        },
        schema_overrides={"a": pl.Float64},
    )
    out = (
        df.lazy()
        .with_columns(pl.Series("e", [2, 1, 3], pl.Int32))
        .with_columns(pl.col("e").cast(pl.Float32))
        .collect()
    )
    expected_schema = {
        "a": pl.Float64,
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
            schema=expected_schema,  # type: ignore[arg-type]
        ),
        out,
        atol=0.00001,
    )


def test_to_html() -> None:
    # check it does not panic/error, and appears to contain
    # a reasonable table with suitably escaped html entities.
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "<bar>": ["a", "b", "c"],
            "<baz": ["a", "b", "c"],
            "spam>": ["a", "b", "c"],
        }
    )
    html = df._repr_html_()
    for match in (
        "<table",
        'class="dataframe"',
        "<th>foo</th>",
        "<th>&lt;bar&gt;</th>",
        "<th>&lt;baz</th>",
        "<th>spam&gt;</th>",
        "<td>1</td>",
        "<td>2</td>",
        "<td>3</td>",
    ):
        assert match in html, f"Expected to find {match!r} in html repr"


def test_rename(df: pl.DataFrame) -> None:
    out = df.rename({"strings": "bars", "int": "foos"})
    # check if we can select these new columns
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


def test_from_generator_or_iterable() -> None:
    # generator function
    def gen(n: int, *, strkey: bool = True) -> Iterator[Any]:
        for i in range(n):
            yield (str(i) if strkey else i), 1 * i, 2**i, 3**i

    # iterable object
    class Rows:
        def __init__(self, n: int, *, strkey: bool = True):
            self._n = n
            self._strkey = strkey

        def __iter__(self) -> Iterator[Any]:
            yield from gen(self._n, strkey=self._strkey)

    # check init from column-oriented generator
    assert_frame_equal(
        pl.DataFrame(data=gen(4, strkey=False), orient="col"),
        pl.DataFrame(
            data=[(0, 0, 1, 1), (1, 1, 2, 3), (2, 2, 4, 9), (3, 3, 8, 27)], orient="col"
        ),
    )
    # check init from row-oriented generators (more common)
    expected = pl.DataFrame(
        data=list(gen(4)), schema=["a", "b", "c", "d"], orient="row"
    )
    for generated_frame in (
        pl.DataFrame(data=gen(4), schema=["a", "b", "c", "d"]),
        pl.DataFrame(data=Rows(4), schema=["a", "b", "c", "d"]),
        pl.DataFrame(data=(x for x in Rows(4)), schema=["a", "b", "c", "d"]),
    ):
        assert_frame_equal(expected, generated_frame)
        assert generated_frame.schema == {
            "a": pl.Utf8,
            "b": pl.Int64,
            "c": pl.Int64,
            "d": pl.Int64,
        }

    # test 'iterable_to_pydf' directly to validate 'chunk_size' behaviour
    cols = ["a", "b", ("c", pl.Int8), "d"]

    expected_data = [("0", 0, 1, 1), ("1", 1, 2, 3), ("2", 2, 4, 9), ("3", 3, 8, 27)]
    expected_schema = [("a", pl.Utf8), ("b", pl.Int64), ("c", pl.Int8), ("d", pl.Int64)]

    for params in (
        {"data": Rows(4)},
        {"data": gen(4), "chunk_size": 2},
        {"data": Rows(4), "chunk_size": 3},
        {"data": gen(4), "infer_schema_length": None},
        {"data": Rows(4), "infer_schema_length": 1},
        {"data": gen(4), "chunk_size": 2},
        {"data": Rows(4), "infer_schema_length": 5},
        {"data": gen(4), "infer_schema_length": 3, "chunk_size": 2},
        {"data": gen(4), "infer_schema_length": None, "chunk_size": 3},
    ):
        d = iterable_to_pydf(schema=cols, **params)  # type: ignore[arg-type]
        assert expected_data == d.row_tuples()
        assert expected_schema == list(zip(d.columns(), d.dtypes()))

    # ref: issue #6489 (initial chunk_size cannot be smaller than 'infer_schema_length')
    df = pl.DataFrame(
        data=iter(([{"col": None}] * 1000) + [{"col": ["a", "b", "c"]}]),
        infer_schema_length=1001,
    )
    assert df.schema == {"col": pl.List(pl.Utf8)}
    assert df[-2:]["col"].to_list() == [None, ["a", "b", "c"]]

    # empty iterator
    assert_frame_equal(
        pl.DataFrame(data=gen(0), schema=["a", "b", "c", "d"]),
        pl.DataFrame(schema=["a", "b", "c", "d"]),
    )

    # dict-related generator-views
    d = {0: "x", 1: "y", 2: "z"}
    df = pl.DataFrame(
        {
            "keys": d.keys(),
            "vals": d.values(),
            "itms": d.items(),
        }
    )
    assert df.to_dict(False) == {
        "keys": [0, 1, 2],
        "vals": ["x", "y", "z"],
        "itms": [(0, "x"), (1, "y"), (2, "z")],
    }
    if sys.version_info >= (3, 11):
        df = pl.DataFrame(
            {
                "rev_keys": reversed(d.keys()),
                "rev_vals": reversed(d.values()),
                "rev_itms": reversed(d.items()),
            }
        )
        assert df.to_dict(False) == {
            "rev_keys": [2, 1, 0],
            "rev_vals": ["z", "y", "x"],
            "rev_itms": [(2, "z"), (1, "y"), (0, "x")],
        }


def test_from_rows() -> None:
    df = pl.from_records([[1, 2, "foo"], [2, 3, "bar"]])
    assert_frame_equal(
        df,
        pl.DataFrame(
            {"column_0": [1, 2], "column_1": [2, 3], "column_2": ["foo", "bar"]}
        ),
    )
    df = pl.from_records(
        [[1, datetime.fromtimestamp(100)], [2, datetime.fromtimestamp(2398754908)]],
        schema_overrides={"column_0": pl.UInt32},
        orient="row",
    )
    assert df.dtypes == [pl.UInt32, pl.Datetime]

    # auto-inference with same num rows/cols
    data = [(1, 2, "foo"), (2, 3, "bar"), (3, 4, "baz")]
    df = pl.from_records(data)
    assert data == df.rows()


def test_from_rows_of_dicts() -> None:
    records = [
        {"id": 1, "value": 100, "_meta": "a"},
        {"id": 2, "value": 101, "_meta": "b"},
    ]
    df_init: Callable[..., Any]
    for df_init in (pl.from_dicts, pl.DataFrame):  # type:ignore[assignment]
        df1 = df_init(records)
        assert df1.rows() == [(1, 100, "a"), (2, 101, "b")]

        overrides = {
            "id": pl.Int16,
            "value": pl.Int32,
        }
        df2 = df_init(records, schema_overrides=overrides)
        assert df2.rows() == [(1, 100, "a"), (2, 101, "b")]
        assert df2.schema == {"id": pl.Int16, "value": pl.Int32, "_meta": pl.Utf8}

        df3 = df_init(records, schema=overrides)
        assert df3.rows() == [(1, 100), (2, 101)]
        assert df3.schema == {"id": pl.Int16, "value": pl.Int32}


def test_repeat_by_unequal_lengths_panic() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "y", "z"],
        }
    )
    with pytest.raises(
        pl.ComputeError,
        match="repeat_by argument and the Series should have equal length, "
        "or at least one of them should have length 1",
    ):
        df.select(pl.col("a").repeat_by(pl.Series([2, 2])))


@pytest.mark.parametrize(
    ("value", "values_expect"),
    [
        (1.2, [[1.2], [1.2, 1.2], [1.2, 1.2, 1.2]]),
        (True, [[True], [True, True], [True, True, True]]),
        ("x", [["x"], ["x", "x"], ["x", "x", "x"]]),
        (b"a", [[b"a"], [b"a", b"a"], [b"a", b"a", b"a"]]),
    ],
)
def test_repeat_by_broadcast_left(
    value: float | bool | str, values_expect: list[list[float | bool | str]]
) -> None:
    df = pl.DataFrame(
        {
            "n": [1, 2, 3],
        }
    )
    expected = pl.DataFrame({"values": values_expect})
    result = df.select(pl.lit(value).repeat_by(pl.col("n")).alias("values"))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("a", "a_expected"),
    [
        ([1.2, 2.2, 3.3], [[1.2, 1.2, 1.2], [2.2, 2.2, 2.2], [3.3, 3.3, 3.3]]),
        ([True, False], [[True, True, True], [False, False, False]]),
        (["x", "y", "z"], [["x", "x", "x"], ["y", "y", "y"], ["z", "z", "z"]]),
        (
            [b"a", b"b", b"c"],
            [[b"a", b"a", b"a"], [b"b", b"b", b"b"], [b"c", b"c", b"c"]],
        ),
    ],
)
def test_repeat_by_broadcast_right(
    a: list[float | bool | str], a_expected: list[list[float | bool | str]]
) -> None:
    df = pl.DataFrame(
        {
            "a": a,
        }
    )
    expected = pl.DataFrame({"a": a_expected})
    result = df.select(pl.col("a").repeat_by(3))
    assert_frame_equal(result, expected)
    result = df.select(pl.col("a").repeat_by(pl.lit(3)))
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("a", "a_expected"),
    [
        (["foo", "bar"], [["foo", "foo"], ["bar", "bar", "bar"]]),
        ([1, 2], [[1, 1], [2, 2, 2]]),
        ([True, False], [[True, True], [False, False, False]]),
        (
            [b"a", b"b"],
            [[b"a", b"a"], [b"b", b"b", b"b"]],
        ),
    ],
)
def test_repeat_by(
    a: list[float | bool | str], a_expected: list[list[float | bool | str]]
) -> None:
    df = pl.DataFrame({"a": a, "n": [2, 3]})
    expected = pl.DataFrame({"a": a_expected})
    result = df.select(pl.col("a").repeat_by("n"))
    assert_frame_equal(result, expected)


def test_join_dates() -> None:
    dts_in = pl.datetime_range(
        datetime(2021, 6, 24),
        datetime(2021, 6, 24, 10, 0, 0),
        interval=timedelta(hours=1),
        closed="left",
        eager=True,
    )
    dts = (
        dts_in.cast(int)
        .map_elements(lambda x: x + np.random.randint(1_000 * 60, 60_000 * 60))
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
    left = pl.DataFrame({"a": [-10, 5, 10], "left_val": ["a", "b", "c"]}).with_columns(
        pl.col("a").set_sorted()
    )
    right = pl.DataFrame(
        {"a": [1, 2, 3, 6, 7], "right_val": [1, 2, 3, 6, 7]}
    ).with_columns(pl.col("a").set_sorted())

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
    out = df.with_columns((pl.lit("Dr. ") + pl.col("name")).alias("graduated_name"))
    assert out["graduated_name"][0] == "Dr. ham"
    assert out["graduated_name"][1] == "Dr. spam"


def test_dot_product() -> None:
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
    import platform

    # m1 hash different random source seed
    if platform.mac_ver()[-1] != "arm64":
        expected = pl.Series(
            "s",
            [13477868900383131459, 6344663067812082469, 16840582678788620208],
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
    # see if we can do a group_by, drop_duplicates on a DataFrame with objects.
    # this requires that the hashing and aggregations are done on python objects

    df = pl.DataFrame({"a": [1, 1, 3, 4], "b": [1, 1, 2, 2]})

    class Foo:
        def __hash__(self) -> int:
            return 0

        def __eq__(self, other: Any) -> bool:
            return True

    df = df.with_columns(pl.col("a").map_elements(lambda x: Foo()).alias("foo"))
    assert df.group_by(["foo"]).first().shape == (1, 3)
    assert df.unique().shape == (3, 3)


def test_unique_unit_rows() -> None:
    df = pl.DataFrame({"a": [1], "b": [None]})

    # 'unique' one-row frame should be equal to the original frame
    assert_frame_equal(df, df.unique(subset="a"))
    for col in df.columns:
        assert df.n_unique(subset=[col]) == 1


def test_panic() -> None:
    # may contain some tests that yielded a panic in polars or pl_arrow
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


def test_group_by_cat_list() -> None:
    grouped = (
        pl.DataFrame(
            [
                pl.Series("str_column", ["a", "b", "b", "a", "b"]),
                pl.Series("int_column", [1, 1, 2, 2, 3]),
            ]
        )
        .with_columns(pl.col("str_column").cast(pl.Categorical).alias("cat_column"))
        .group_by("int_column", maintain_order=True)
        .agg([pl.col("cat_column")])["cat_column"]
    )

    out = grouped.explode()
    assert out.dtype == pl.Categorical
    assert out[0] == "a"


def test_group_by_agg_n_unique_floats() -> None:
    # tests proper dispatch
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    for dtype in [pl.Float32, pl.Float64]:
        out = df.group_by("a", maintain_order=True).agg(
            [pl.col("b").cast(dtype).n_unique()]
        )
        assert out["b"].to_list() == [2, 1]


def test_select_by_dtype(df: pl.DataFrame) -> None:
    out = df.select(pl.col(pl.Utf8))
    assert out.columns == ["strings", "strings_nulls"]
    out = df.select(pl.col([pl.Utf8, pl.Boolean]))
    assert out.columns == ["bools", "bools_nulls", "strings", "strings_nulls"]
    out = df.select(pl.col(INTEGER_DTYPES))
    assert out.columns == ["int", "int_nulls"]

    with pl.Config() as cfg:
        cfg.set_auto_structify(True)
        out = df.select(ints=pl.col(INTEGER_DTYPES))
        assert out.schema == {
            "ints": pl.Struct(
                [pl.Field("int", pl.Int64), pl.Field("int_nulls", pl.Int64)]
            )
        }


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

    out = df.group_by("groups", maintain_order=True).agg(pl.col("a").alias("a"))
    assert sys.getrefcount(foos[0]) == base_count + 2
    s = out["a"].list.explode()
    assert sys.getrefcount(foos[0]) == base_count + 3
    del s
    assert sys.getrefcount(foos[0]) == base_count + 2

    assert out["a"].list.explode().to_list() == foos
    assert sys.getrefcount(foos[0]) == base_count + 2
    del out
    assert sys.getrefcount(foos[0]) == base_count + 1
    del df
    assert sys.getrefcount(foos[0]) == base_count


def test_group_by_order_dispatch() -> None:
    df = pl.DataFrame({"x": list("bab"), "y": range(3)})

    result = df.group_by("x", maintain_order=True).count()
    expected = pl.DataFrame(
        {"x": ["b", "a"], "count": [2, 1]}, schema_overrides={"count": pl.UInt32}
    )
    assert_frame_equal(result, expected)

    result = df.group_by("x", maintain_order=True).all()
    expected = pl.DataFrame({"x": ["b", "a"], "y": [[0, 2], [1]]})
    assert_frame_equal(result, expected)


def test_partitioned_group_by_order() -> None:
    # check if group ordering is maintained.
    # we only have 30 groups, so this triggers a partitioned group by
    df = pl.DataFrame({"x": [chr(v) for v in range(33, 63)], "y": range(30)})
    out = df.group_by("x", maintain_order=True).agg(pl.all().implode())
    assert_series_equal(out["x"], df["x"])


def test_schema() -> None:
    df = pl.DataFrame(
        {"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0], "ham": ["a", "b", "c"]}
    )
    expected = {"foo": pl.Int64, "bar": pl.Float64, "ham": pl.Utf8}
    assert df.schema == expected


def test_schema_equality() -> None:
    lf = pl.LazyFrame({"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0]})
    lf_rev = lf.select("bar", "foo")

    assert lf.schema != lf_rev.schema
    assert lf.collect().schema != lf_rev.collect().schema


def test_df_schema_unique() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(pl.DuplicateError):
        df.columns = ["a", "a"]

    with pytest.raises(pl.DuplicateError):
        df.rename({"b": "a"})


def test_cleared() -> None:
    df = pl.DataFrame(
        {"a": [1, 2], "b": [True, False]}, schema_overrides={"a": pl.UInt32}
    )
    dfc = df.clear()
    assert dfc.schema == df.schema
    assert dfc.rows() == []

    dfc = df.clear(3)
    assert dfc.schema == df.schema
    assert dfc.rows() == [(None, None), (None, None), (None, None)]


def test_empty_projection() -> None:
    empty_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).select([])
    assert empty_df.rows() == []
    assert empty_df.schema == {}
    assert empty_df.shape == (0, 0)


def test_with_column_renamed() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.rename({"b": "c"})
    expected = pl.DataFrame({"a": [1, 2], "c": [3, 4]})
    assert_frame_equal(result, expected)


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
    assert_frame_equal(out, expected)

    # 6195
    ldf = pl.DataFrame(
        {
            "weekday": [
                1,
            ],
            "priority": [
                2,
            ],
            "roundNumber": [
                3,
            ],
            "flag": [
                4,
            ],
        }
    ).lazy()

    # Rename some columns (note: swapping two columns)
    rename_dict = {
        "weekday": "priority",
        "priority": "weekday",
        "roundNumber": "round_number",
    }
    ldf = ldf.rename(rename_dict)

    # Select some columns
    ldf = ldf.select(["priority", "weekday", "round_number"])

    assert ldf.collect().to_dict(False) == {
        "priority": [1],
        "weekday": [2],
        "round_number": [3],
    }


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
    assert_frame_equal(df.fill_null(4), pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert_frame_equal(
        df.fill_null(strategy="max"), pl.DataFrame({"a": [1, 2], "b": [3, 3]})
    )

    # utf8 and list data
    # utf8 goes via binary
    df = pl.DataFrame(
        {
            "c": [
                ["Apple", "Orange"],
                ["Apple", "Orange"],
                None,
                ["Carrot"],
                None,
                None,
            ],
            "b": ["Apple", "Orange", None, "Carrot", None, None],
        }
    )

    assert df.select(
        [
            pl.all().forward_fill().suffix("_forward"),
            pl.all().backward_fill().suffix("_backward"),
        ]
    ).to_dict(False) == {
        "c_forward": [
            ["Apple", "Orange"],
            ["Apple", "Orange"],
            ["Apple", "Orange"],
            ["Carrot"],
            ["Carrot"],
            ["Carrot"],
        ],
        "b_forward": ["Apple", "Orange", "Orange", "Carrot", "Carrot", "Carrot"],
        "c_backward": [
            ["Apple", "Orange"],
            ["Apple", "Orange"],
            ["Carrot"],
            ["Carrot"],
            None,
            None,
        ],
        "b_backward": ["Apple", "Orange", "Carrot", "Carrot", None, None],
    }
    # categoricals
    df = pl.DataFrame(pl.Series("cat", ["a", None], dtype=pl.Categorical))
    s = df.select(pl.col("cat").fill_null(strategy="forward"))["cat"]
    assert s.dtype == pl.Categorical
    assert s.to_list() == ["a", "a"]


def test_fill_nan() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, float("nan")]})
    assert_frame_equal(
        df.fill_nan(4),
        pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0]}),
    )
    assert_frame_equal(
        df.fill_nan(None),
        pl.DataFrame({"a": [1, 2], "b": [3.0, None]}),
    )
    assert df["b"].fill_nan(5.0).to_list() == [3.0, 5.0]
    df = pl.DataFrame(
        {
            "a": [1.0, np.nan, 3.0],
            "b": [datetime(1, 2, 2), datetime(2, 2, 2), datetime(3, 2, 2)],
        }
    )
    assert df.fill_nan(2.0).dtypes == [pl.Float64, pl.Datetime]


def test_backward_fill() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    col_a_backward_fill = df.select([pl.col("a").backward_fill()])["a"]
    assert_series_equal(col_a_backward_fill, pl.Series("a", [1, 3, 3]).cast(pl.Float64))


def test_shift_and_fill() -> None:
    df = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6, 7, 8],
            "ham": ["a", "b", "c"],
        }
    )
    result = df.shift_and_fill(fill_value=0, n=1)
    expected = pl.DataFrame(
        {
            "foo": [0, 1, 2],
            "bar": [0, 6, 7],
            "ham": ["0", "a", "b"],
        }
    )
    assert_frame_equal(result, expected)


def test_is_duplicated() -> None:
    df = pl.DataFrame({"foo": [1, 2, 2], "bar": [6, 7, 7]})
    assert_series_equal(df.is_duplicated(), pl.Series("", [False, True, True]))


def test_is_unique() -> None:
    df = pl.DataFrame({"foo": [1, 2, 2], "bar": [6, 7, 7]})

    assert_series_equal(df.is_unique(), pl.Series("", [True, False, False]))
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


def test_shrink_to_fit() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]})

    assert df.shrink_to_fit(in_place=True) is df
    assert df.shrink_to_fit(in_place=False) is not df
    assert_frame_equal(df.shrink_to_fit(in_place=False), df)


def test_arithmetic() -> None:
    df = pl.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})

    for df_mul in (df * 2, 2 * df):
        expected = pl.DataFrame({"a": [2.0, 4.0], "b": [6.0, 8.0]})
        assert_frame_equal(df_mul, expected)

    for df_plus in (df + 2, 2 + df):
        expected = pl.DataFrame({"a": [3.0, 4.0], "b": [5.0, 6.0]})
        assert_frame_equal(df_plus, expected)

    df_div = df / 2
    expected = pl.DataFrame({"a": [0.5, 1.0], "b": [1.5, 2.0]})
    assert_frame_equal(df_div, expected)

    df_minus = df - 2
    expected = pl.DataFrame({"a": [-1.0, 0.0], "b": [1.0, 2.0]})
    assert_frame_equal(df_minus, expected)

    df_mod = df % 2
    expected = pl.DataFrame({"a": [1.0, 0.0], "b": [1.0, 0.0]})
    assert_frame_equal(df_mod, expected)

    df2 = pl.DataFrame({"c": [10]})

    out = df + df2
    expected = pl.DataFrame({"a": [11.0, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    out = df - df2
    expected = pl.DataFrame({"a": [-9.0, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    out = df / df2
    expected = pl.DataFrame({"a": [0.1, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    out = df * df2
    expected = pl.DataFrame({"a": [10.0, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    out = df % df2
    expected = pl.DataFrame({"a": [1.0, None], "b": [None, None]}).with_columns(
        pl.col("b").cast(pl.Float64)
    )
    assert_frame_equal(out, expected)

    # cannot do arithmetic with a sequence
    with pytest.raises(TypeError, match="operation not supported"):
        _ = df + [1]  # type: ignore[operator]


def test_df_series_division() -> None:
    df = pl.DataFrame(
        {
            "a": [2, 2, 4, 4, 6, 6],
            "b": [2, 2, 10, 5, 6, 6],
        }
    )
    s = pl.Series([2, 2, 2, 2, 2, 2])
    assert (df / s).to_dict(False) == {
        "a": [1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
        "b": [1.0, 1.0, 5.0, 2.5, 3.0, 3.0],
    }
    assert (df // s).to_dict(False) == {
        "a": [1, 1, 2, 2, 3, 3],
        "b": [1, 1, 5, 2, 3, 3],
    }


def test_add_string() -> None:
    df = pl.DataFrame({"a": ["hi", "there"], "b": ["hello", "world"]})
    expected = pl.DataFrame(
        {"a": ["hi hello", "there hello"], "b": ["hello hello", "world hello"]}
    )
    assert_frame_equal((df + " hello"), expected)

    expected = pl.DataFrame(
        {"a": ["hello hi", "hello there"], "b": ["hello hello", "hello world"]}
    )
    assert_frame_equal(("hello " + df), expected)


def test_getitem() -> None:
    """Test all the methods to use [] on a dataframe."""
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [3, 4, 5, 6]})

    # expression
    assert_frame_equal(
        df.select(pl.col("a")), pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    )

    # multiple slices.
    # The first element refers to the rows, the second element to columns
    assert_frame_equal(df[:, :], df)

    # str, always refers to a column name
    assert_series_equal(df["a"], pl.Series("a", [1.0, 2.0, 3.0, 4.0]))

    # int, always refers to a row index (zero-based): index=1 => second row
    assert_frame_equal(df[1], pl.DataFrame({"a": [2.0], "b": [4]}))

    # int, int.
    # The first element refers to the rows, the second element to columns
    assert df[2, 1] == 5
    assert df[2, -2] == 3.0

    with pytest.raises(IndexError):
        # Column index out of bounds
        df[2, 2]

    with pytest.raises(IndexError):
        # Column index out of bounds
        df[2, -3]

    # int, list[int].
    # The first element refers to the rows, the second element to columns
    assert_frame_equal(df[2, [1, 0]], pl.DataFrame({"b": [5], "a": [3.0]}))
    assert_frame_equal(df[2, [-1, -2]], pl.DataFrame({"b": [5], "a": [3.0]}))

    with pytest.raises(IndexError):
        # Column index out of bounds
        df[2, [2, 0]]

    with pytest.raises(IndexError):
        # Column index out of bounds
        df[2, [2, -3]]

    # range, refers to rows
    assert_frame_equal(df[range(1, 3)], pl.DataFrame({"a": [2.0, 3.0], "b": [4, 5]}))

    # slice. Below an example of taking every second row
    assert_frame_equal(df[1::2], pl.DataFrame({"a": [2.0, 4.0], "b": [4, 6]}))

    # slice, empty slice
    assert df[:0].columns == ["a", "b"]
    assert len(df[:0]) == 0

    # make mypy happy
    empty: list[int] = []

    # empty list with column selector drops rows but keeps columns
    assert_frame_equal(df[empty, :], df[:0])

    # empty list without column select return empty frame
    assert_frame_equal(df[empty], pl.DataFrame({}))

    # numpy array: assumed to be row indices if integers, or columns if strings

    # numpy array: positive idxs and empty idx
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
        assert_frame_equal(
            df[np.array([1, 0, 3, 2, 3, 0], dtype=np_dtype)],
            pl.DataFrame(
                {"a": [2.0, 1.0, 4.0, 3.0, 4.0, 1.0], "b": [4, 3, 6, 5, 6, 3]}
            ),
        )
        assert df[np.array([], dtype=np_dtype)].columns == ["a", "b"]

    # numpy array: positive and negative idxs.
    for np_dtype in (np.int8, np.int16, np.int32, np.int64):
        assert_frame_equal(
            df[np.array([-1, 0, -3, -2, 3, -4], dtype=np_dtype)],
            pl.DataFrame(
                {"a": [4.0, 1.0, 2.0, 3.0, 4.0, 1.0], "b": [6, 3, 4, 5, 6, 3]}
            ),
        )

    # note that we cannot use floats (even if they could be casted to integer without
    # loss)
    with pytest.raises(TypeError):
        _ = df[np.array([1.0])]

    # sequences (lists or tuples; tuple only if length != 2)
    # if strings or list of expressions, assumed to be column names
    # if bools, assumed to be a row mask
    # if integers, assumed to be row indices
    assert_frame_equal(df[["a", "b"]], df)
    assert_frame_equal(df.select([pl.col("a"), pl.col("b")]), df)
    assert_frame_equal(
        df[[1, -4, -1, 2, 1]],
        pl.DataFrame({"a": [2.0, 1.0, 4.0, 3.0, 2.0], "b": [4, 3, 6, 5, 4]}),
    )

    # pl.Series: strings for column selections.
    assert_frame_equal(df[pl.Series("", ["a", "b"])], df)

    # pl.Series: positive idxs or empty idxs for row selection.
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
        assert_frame_equal(
            df[pl.Series("", [1, 0, 3, 2, 3, 0], dtype=pl_dtype)],
            pl.DataFrame(
                {"a": [2.0, 1.0, 4.0, 3.0, 4.0, 1.0], "b": [4, 3, 6, 5, 6, 3]}
            ),
        )
        assert df[pl.Series("", [], dtype=pl_dtype)].columns == ["a", "b"]

    # pl.Series: positive and negative idxs for row selection.
    for pl_dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        assert_frame_equal(
            df[pl.Series("", [-1, 0, -3, -2, 3, -4], dtype=pl_dtype)],
            pl.DataFrame(
                {"a": [4.0, 1.0, 2.0, 3.0, 4.0, 1.0], "b": [6, 3, 4, 5, 6, 3]}
            ),
        )

    # Boolean masks not supported
    with pytest.raises(TypeError):
        df[np.array([True, False, True])]
    with pytest.raises(TypeError):
        df[[True, False, True], [False, True]]  # type: ignore[index]
    with pytest.raises(TypeError):
        df[pl.Series([True, False, True]), "b"]

    # 5343
    df = pl.DataFrame(
        {
            f"foo{col}": [n**col for n in range(5)]  # 5 rows
            for col in range(12)  # 12 columns
        }
    )
    assert df[4, 4] == 256
    assert df[4, 5] == 1024
    assert_frame_equal(df[4, [2]], pl.DataFrame({"foo2": [16]}))
    assert_frame_equal(df[4, [5]], pl.DataFrame({"foo5": [1024]}))


@pytest.mark.parametrize(
    ("as_series", "inner_dtype"), [(True, pl.Series), (False, list)]
)
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
    df = pl.DataFrame({"a": [1, 2, 3]}, schema_overrides={"a": pl.UInt8})
    out = df.with_columns(pl.Series("s", [[1, 2]]))
    assert out.shape == (3, 2)
    assert out.schema == {"a": pl.UInt8, "s": pl.List(pl.Int64)}
    assert out.rows() == [(1, [1, 2]), (2, [1, 2]), (3, [1, 2])]


def test_product() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2, 3],
            "flt": [-1.0, 12.0, 9.0],
            "bool_0": [True, False, True],
            "bool_1": [True, True, True],
        },
        schema_overrides={
            "int": pl.UInt16,
            "flt": pl.Float32,
        },
    )
    out = df.product()
    expected = pl.DataFrame({"int": [6], "flt": [-108.0], "bool_0": [0], "bool_1": [1]})
    assert_frame_not_equal(out, expected, check_dtype=True)
    assert_frame_equal(out, expected, check_dtype=False)


def test_first_last_expression(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(pl.first())
    assert out.columns == ["A"]

    out = df.select(pl.last())
    assert out.columns == ["cars"]


def test_is_between(fruits_cars: pl.DataFrame) -> None:
    result = fruits_cars.select(pl.col("A").is_between(2, 4)).to_series()
    assert_series_equal(result, pl.Series("A", [False, True, True, True, False]))

    result = fruits_cars.select(pl.col("A").is_between(2, 4, closed="none")).to_series()
    assert_series_equal(result, pl.Series("A", [False, False, True, False, False]))

    result = fruits_cars.select(pl.col("A").is_between(2, 4, closed="both")).to_series()
    assert_series_equal(result, pl.Series("A", [False, True, True, True, False]))

    result = fruits_cars.select(
        pl.col("A").is_between(2, 4, closed="right")
    ).to_series()
    assert_series_equal(result, pl.Series("A", [False, False, True, True, False]))

    result = fruits_cars.select(pl.col("A").is_between(2, 4, closed="left")).to_series()
    assert_series_equal(result, pl.Series("A", [False, True, True, False, False]))


def test_is_between_data_types() -> None:
    df = pl.DataFrame(
        {
            "flt": [1.4, 1.2, 2.5],
            "int": [2, 3, 4],
            "str": ["xyz", "str", "abc"],
            "date": [date(2020, 1, 1), date(2020, 2, 2), date(2020, 3, 3)],
            "datetime": [
                datetime(2020, 1, 1, 0, 0, 0),
                datetime(2020, 1, 1, 10, 0, 0),
                datetime(2020, 1, 1, 12, 0, 0),
            ],
            "tm": [time(10, 30), time(0, 45), time(15, 15)],
        }
    )

    # on purpose, for float and int, we pass in a mixture of bound data types
    assert_series_equal(
        df.select(pl.col("flt").is_between(1, 2.3))[:, 0],
        pl.Series("flt", [True, True, False]),
    )
    assert_series_equal(
        df.select(pl.col("int").is_between(1.5, 3))[:, 0],
        pl.Series("int", [True, True, False]),
    )
    assert_series_equal(
        df.select(pl.col("date").is_between(date(2019, 1, 1), date(2020, 2, 5)))[:, 0],
        pl.Series("date", [True, True, False]),
    )
    assert_series_equal(
        df.select(
            pl.col("datetime").is_between(
                datetime(2020, 1, 1, 5, 0, 0), datetime(2020, 1, 1, 11, 0, 0)
            )
        )[:, 0],
        pl.Series("datetime", [False, True, False]),
    )
    assert_series_equal(
        df.select(
            pl.col("str").is_between(pl.lit("str"), pl.lit("zzz"), closed="left")
        )[:, 0],
        pl.Series("str", [True, True, False]),
    )
    assert_series_equal(
        df.select(
            pl.col("tm")
            .is_between(time(0, 45), time(10, 30), closed="right")
            .alias("tm_between")
        )[:, 0],
        pl.Series("tm_between", [True, False, False]),
    )


def test_empty_is_in() -> None:
    df_empty_isin = pl.DataFrame({"foo": ["a", "b", "c", "d"]}).filter(
        pl.col("foo").is_in([])
    )
    assert df_empty_isin.shape == (0, 1)
    assert df_empty_isin.rows() == []
    assert df_empty_isin.schema == {"foo": pl.Utf8}


def test_group_by_slice_expression_args() -> None:
    df = pl.DataFrame({"groups": ["a"] * 10 + ["b"] * 20, "vals": range(30)})

    out = (
        df.group_by("groups", maintain_order=True)
        .agg([pl.col("vals").slice(pl.count() * 0.1, (pl.count() // 5))])
        .explode("vals")
    )

    expected = pl.DataFrame(
        {"groups": ["a", "a", "b", "b", "b", "b"], "vals": [1, 2, 12, 13, 14, 15]}
    )
    assert_frame_equal(out, expected)


def test_join_suffixes() -> None:
    df_a = pl.DataFrame({"A": [1], "B": [1]})
    df_b = pl.DataFrame({"A": [1], "B": [1]})

    join_strategies: list[JoinStrategy] = ["left", "inner", "outer", "cross"]
    for how in join_strategies:
        # no need for an assert, we error if wrong
        df_a.join(df_b, on="A", suffix="_y", how=how)["B_y"]

    df_a.join_asof(df_b, on=pl.col("A").set_sorted(), suffix="_y")["B_y"]


def test_explode_empty() -> None:
    df = (
        pl.DataFrame({"x": ["a", "a", "b", "b"], "y": [1, 1, 2, 2]})
        .group_by("x", maintain_order=True)
        .agg(pl.col("y").take([]))
    )
    assert df.explode("y").to_dict(False) == {"x": ["a", "b"], "y": [None, None]}

    df = pl.DataFrame({"x": ["1", "2", "4"], "y": [["a", "b", "c"], ["d"], []]})
    assert_frame_equal(
        df.explode("y"),
        pl.DataFrame({"x": ["1", "1", "1", "2", "4"], "y": ["a", "b", "c", "d", None]}),
    )

    df = pl.DataFrame(
        {
            "letters": ["a"],
            "numbers": [[]],
        }
    )
    assert df.explode("numbers").to_dict(False) == {"letters": ["a"], "numbers": [None]}


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

    result = lhs.join_asof(
        rhs, on=pl.col("a").set_sorted(), by=["by", "by2"], strategy="backward"
    ).select(["a", "by"])
    expected = pl.DataFrame({"a": [-20, -19, 8, 12, 14], "by": [1, 1, 2, 2, 2]})
    assert_frame_equal(result, expected)


def test_partition_by() -> None:
    df = pl.DataFrame(
        {
            "foo": ["A", "A", "B", "B", "C"],
            "N": [1, 2, 2, 4, 2],
            "bar": ["k", "l", "m", "m", "l"],
        }
    )

    expected = [
        {"foo": ["A"], "N": [1], "bar": ["k"]},
        {"foo": ["A"], "N": [2], "bar": ["l"]},
        {"foo": ["B", "B"], "N": [2, 4], "bar": ["m", "m"]},
        {"foo": ["C"], "N": [2], "bar": ["l"]},
    ]
    assert [
        a.to_dict(False) for a in df.partition_by("foo", "bar", maintain_order=True)
    ] == expected
    assert [
        a.to_dict(False) for a in df.partition_by(cs.string(), maintain_order=True)
    ] == expected

    expected = [
        {"N": [1]},
        {"N": [2]},
        {"N": [2, 4]},
        {"N": [2]},
    ]
    assert [
        a.to_dict(False)
        for a in df.partition_by(["foo", "bar"], maintain_order=True, include_key=False)
    ] == expected
    assert [
        a.to_dict(False)
        for a in df.partition_by("foo", "bar", maintain_order=True, include_key=False)
    ] == expected

    assert [a.to_dict(False) for a in df.partition_by("foo", maintain_order=True)] == [
        {"foo": ["A", "A"], "N": [1, 2], "bar": ["k", "l"]},
        {"foo": ["B", "B"], "N": [2, 4], "bar": ["m", "m"]},
        {"foo": ["C"], "N": [2], "bar": ["l"]},
    ]

    df = pl.DataFrame({"a": ["one", "two", "one", "two"], "b": [1, 2, 3, 4]})
    assert df.partition_by(cs.all(), as_dict=True)["one", 1].to_dict(False) == {
        "a": ["one"],
        "b": [1],
    }
    assert df.partition_by(["a"], as_dict=True)["one"].to_dict(False) == {
        "a": ["one", "one"],
        "b": [1, 3],
    }

    # test with both as_dict and include_key=False
    df = pl.DataFrame(
        {
            "a": pl.int_range(0, 100, dtype=pl.UInt8, eager=True),
            "b": pl.int_range(0, 100, dtype=pl.UInt8, eager=True),
            "c": pl.int_range(0, 100, dtype=pl.UInt8, eager=True),
            "d": pl.int_range(0, 100, dtype=pl.UInt8, eager=True),
        }
    ).sample(n=100_000, with_replacement=True, shuffle=True)

    partitions = df.partition_by(["a", "b"], as_dict=True, include_key=False)
    assert all(key == value.row(0) for key, value in partitions.items())


def test_list_of_list_of_struct() -> None:
    expected = [{"list_of_list_of_struct": [[{"a": 1}, {"a": 2}]]}]
    pa_df = pa.Table.from_pylist(expected)

    df = pl.from_arrow(pa_df)
    assert df.rows() == [([[{"a": 1}, {"a": 2}]],)]  # type: ignore[union-attr]
    assert df.to_dicts() == expected  # type: ignore[union-attr]

    df = pl.from_arrow(pa_df[:0])
    assert df.to_dicts() == []  # type: ignore[union-attr]


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


def test_lower_bound_upper_bound(fruits_cars: pl.DataFrame) -> None:
    res_expr = fruits_cars.select(pl.col("A").lower_bound())
    assert res_expr.item() == -9223372036854775808

    res_expr = fruits_cars.select(pl.col("B").upper_bound())
    assert res_expr.item() == 9223372036854775807

    with pytest.raises(pl.ComputeError):
        fruits_cars.select(pl.col("fruits").upper_bound())


def test_selection_misc() -> None:
    df = pl.DataFrame({"x": "abc"}, schema={"x": pl.Utf8})

    # literal values (as scalar/list)
    for zero in (0, [0]):
        assert df.select(zero)["literal"].to_list() == [0]  # type: ignore[arg-type]
    assert df.select(literal=0)["literal"].to_list() == [0]

    # expect string values to be interpreted as cols
    for x in ("x", ["x"], pl.col("x")):
        assert df.select(x).rows() == [("abc",)]  # type: ignore[arg-type]

    # string col + lit
    assert df.with_columns(["x", 0]).to_dicts() == [{"x": "abc", "literal": 0}]


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
    assert test_df.select(pl.exclude("foo") * pl.exclude("foo")).to_dict(False) == {
        "a": [1, 4, 9, 16],
        "b": [25, 36, 49, 64],
        "c": [81, 100, 121, 144],
    }
    assert test_df.select(pl.col("^\\w$") * pl.col("^\\w$")).to_dict(False) == {
        "a": [1, 4, 9, 16],
        "b": [25, 36, 49, 64],
        "c": [81, 100, 121, 144],
    }

    # kwargs
    with pl.Config() as cfg:
        cfg.set_auto_structify(True)

        df = test_df.select(
            pl.col("^\\w$").alias("re"),
            odd=(pl.col(INTEGER_DTYPES) % 2).suffix("_is_odd"),
            maxes=pl.all().max().suffix("_max"),
        ).head(2)
        # ┌───────────┬───────────┬─────────────┐
        # │ re        ┆ odd       ┆ maxes       │
        # │ ---       ┆ ---       ┆ ---         │
        # │ struct[3] ┆ struct[4] ┆ struct[4]   │
        # ╞═══════════╪═══════════╪═════════════╡
        # │ {1,5,9}   ┆ {1,1,1,1} ┆ {4,8,12,16} │
        # │ {2,6,10}  ┆ {0,0,0,0} ┆ {4,8,12,16} │
        # └───────────┴───────────┴─────────────┘
        assert df.rows() == [
            (
                {"a": 1, "b": 5, "c": 9},
                {"a_is_odd": 1, "b_is_odd": 1, "c_is_odd": 1, "foo_is_odd": 1},
                {"a_max": 4, "b_max": 8, "c_max": 12, "foo_max": 16},
            ),
            (
                {"a": 2, "b": 6, "c": 10},
                {"a_is_odd": 0, "b_is_odd": 0, "c_is_odd": 0, "foo_is_odd": 0},
                {"a_max": 4, "b_max": 8, "c_max": 12, "foo_max": 16},
            ),
        ]


def test_unique_on_sorted() -> None:
    df = pl.DataFrame(data={"a": [1, 1, 3], "b": [1, 2, 3]})
    for subset in ("a", cs.starts_with("x", "a")):
        assert df.with_columns([pl.col("a").set_sorted()]).unique(
            subset=subset, keep="last"  # type: ignore[arg-type]
        ).to_dict(False) == {
            "a": [1, 3],
            "b": [2, 3],
        }


def test_len_compute(df: pl.DataFrame) -> None:
    df = df.with_columns(pl.struct(["list_bool", "cat"]).alias("struct"))
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


def test_filter_multiple_predicates() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2],
            "b": [1, 1, 2, 2, 2],
            "c": [1, 1, 2, 3, 4],
        }
    )

    # multiple predicates
    expected = pl.DataFrame({"a": [1, 1, 1], "b": [1, 1, 2], "c": [1, 1, 2]})
    for out in (
        df.filter(pl.col("a") == 1, pl.col("b") <= 2),  # positional/splat
        df.filter([pl.col("a") == 1, pl.col("b") <= 2]),  # as list
    ):
        assert_frame_equal(out, expected)

    # multiple kwargs
    assert_frame_equal(
        df.filter(a=1, b=2),
        pl.DataFrame({"a": [1], "b": [2], "c": [2]}),
    )

    # both positional and keyword args
    assert_frame_equal(
        pl.DataFrame({"a": [2], "b": [2], "c": [3]}),
        df.filter(pl.col("c") < 4, a=2, b=2),
    )

    # boolean mask
    out = df.filter([True, False, False, False, True])
    expected = pl.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 4]})
    assert_frame_equal(out, expected)

    # multiple boolean masks
    out = df.filter(
        np.array([True, True, False, True, False]),
        np.array([True, False, True, True, False]),
    )
    expected = pl.DataFrame({"a": [1, 2], "b": [1, 2], "c": [1, 3]})
    assert_frame_equal(out, expected)


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
        match=r"DataFrame object does not support `Series` assignment by index"
        r"\n\nUse `DataFrame.with_columns`.",
    ):
        df["new"] = np.random.rand(10)

    with pytest.raises(
        TypeError,
        match=r"not allowed to set DataFrame by boolean mask in the row position"
        r"\n\nConsider using `DataFrame.with_columns`.",
    ):
        df[df["ham"] > 0.5, "ham"] = "a"
    with pytest.raises(
        TypeError,
        match=r"not allowed to set DataFrame by boolean mask in the row position"
        r"\n\nConsider using `DataFrame.with_columns`.",
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
    with pytest.raises(TypeError):
        df[:, [1]] = 1  # type: ignore[index]
    with pytest.raises(TypeError):
        df[True, :] = 1  # type: ignore[index]

    # needs to be a 2 element tuple
    with pytest.raises(ValueError):
        df[(1, 2, 3)] = 1

    # we cannot index with any type, such as bool
    with pytest.raises(TypeError):
        df[True] = 1  # type: ignore[index]


def test_series_iter_over_frame() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [2, 3, 4], "c": [3, 4, 5]})

    expected = {
        0: pl.Series("a", [1, 2, 3]),
        1: pl.Series("b", [2, 3, 4]),
        2: pl.Series("c", [3, 4, 5]),
    }
    for idx, s in enumerate(df):
        assert_series_equal(s, expected[idx])

    expected = {
        0: pl.Series("c", [3, 4, 5]),
        1: pl.Series("b", [2, 3, 4]),
        2: pl.Series("a", [1, 2, 3]),
    }
    for idx, s in enumerate(reversed(df)):
        assert_series_equal(s, expected[idx])


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


def test_init_datetimes_with_timezone() -> None:
    tz_us = "America/New_York"
    tz_europe = "Europe/Amsterdam"

    dtm = datetime(2022, 10, 12, 12, 30)
    for time_unit in DTYPE_TEMPORAL_UNITS | frozenset([None]):
        for type_overrides in (
            {
                "schema": [
                    ("d1", pl.Datetime(time_unit, tz_us)),
                    ("d2", pl.Datetime(time_unit, tz_europe)),
                ]
            },
            {
                "schema_overrides": {
                    "d1": pl.Datetime(time_unit, tz_us),
                    "d2": pl.Datetime(time_unit, tz_europe),
                }
            },
        ):
            result = pl.DataFrame(  # type: ignore[arg-type]
                data={"d1": [dtm], "d2": [dtm]},
                **type_overrides,
            )
            expected = pl.DataFrame(
                {"d1": ["2022-10-12 12:30"], "d2": ["2022-10-12 12:30"]}
            ).with_columns(
                pl.col("d1").str.to_datetime(time_unit=time_unit, time_zone=tz_us),
                pl.col("d2").str.to_datetime(time_unit=time_unit, time_zone=tz_europe),
            )
            assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    (
        "tzinfo",
        "offset",
        "dtype_time_zone",
        "expected_time_zone",
        "expected_item",
        "warn",
    ),
    [
        (None, "", None, None, datetime(2020, 1, 1), False),
        (
            timezone(timedelta(hours=-8)),
            "-08:00",
            "UTC",
            "UTC",
            datetime(2020, 1, 1, 8, tzinfo=timezone.utc),
            False,
        ),
        (
            timezone(timedelta(hours=-8)),
            "-08:00",
            None,
            "UTC",
            datetime(2020, 1, 1, 8, tzinfo=timezone.utc),
            True,
        ),
    ],
)
def test_init_vs_strptime_consistency(
    tzinfo: timezone | None,
    offset: str,
    dtype_time_zone: str | None,
    expected_time_zone: str,
    expected_item: datetime,
    warn: bool,
) -> None:
    msg = r"UTC time zone"
    context_manager: contextlib.AbstractContextManager[pytest.WarningsRecorder | None]
    if warn:
        context_manager = pytest.warns(TimeZoneAwareConstructorWarning, match=msg)
    else:
        context_manager = contextlib.nullcontext()
    with context_manager:
        result_init = pl.Series(
            [datetime(2020, 1, 1, tzinfo=tzinfo)],
            dtype=pl.Datetime("us", dtype_time_zone),
        )
    result_strptime = pl.Series([f"2020-01-01 00:00{offset}"]).str.strptime(
        pl.Datetime("us", dtype_time_zone)
    )
    assert result_init.dtype == pl.Datetime("us", expected_time_zone)
    assert result_init.item() == expected_item
    assert_series_equal(result_init, result_strptime)


def test_init_vs_strptime_consistency_raises() -> None:
    msg = "-aware datetimes are converted to UTC"
    with pytest.raises(ValueError, match=msg):
        pl.Series(
            [datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=-8)))],
            dtype=pl.Datetime("us", "US/Pacific"),
        )
    with pytest.raises(ComputeError, match=msg):
        pl.Series(["2020-01-01 00:00-08:00"]).str.strptime(
            pl.Datetime("us", "US/Pacific")
        )


def test_init_physical_with_timezone() -> None:
    tz_uae = "Asia/Dubai"
    tz_asia = "Asia/Tokyo"

    dtm_us = 1665577800000000
    for time_unit in DTYPE_TEMPORAL_UNITS | frozenset([None]):
        dtm = {"ms": dtm_us // 1_000, "ns": dtm_us * 1_000}.get(str(time_unit), dtm_us)
        df = pl.DataFrame(
            data={"d1": [dtm], "d2": [dtm]},
            schema=[
                ("d1", pl.Datetime(time_unit, tz_uae)),
                ("d2", pl.Datetime(time_unit, tz_asia)),
            ],
        )
        assert (df["d1"].to_physical() == df["d2"].to_physical()).all()
        assert df.rows() == [
            (
                datetime(2022, 10, 12, 16, 30, tzinfo=ZoneInfo(tz_uae)),
                datetime(2022, 10, 12, 21, 30, tzinfo=ZoneInfo(tz_asia)),
            )
        ]


@pytest.mark.parametrize("divop", [floordiv, truediv])
def test_floordiv_truediv(divop: Callable[..., Any]) -> None:
    # validate truediv/floordiv dataframe ops against python
    df1 = pl.DataFrame(
        data={
            "x": [0, -1, -2, -3],
            "y": [-0.0, -3.0, 5.0, -7.0],
            "z": [10, 3, -5, 7],
        }
    )

    # scalar
    for n in (3, 3.0, -3, -3.0):
        py_div = [tuple(divop(elem, n) for elem in row) for row in df1.rows()]
        df_div = divop(df1, n).rows()
        assert py_div == df_div

    # series
    xdf, s = df1["x"].to_frame(), pl.Series([2] * 4)
    assert list(divop(xdf, s)["x"]) == [divop(x, 2) for x in list(df1["x"])]

    # frame
    df2 = pl.DataFrame(
        data={
            "x": [2, -2, 2, 3],
            "y": [4, 4, -4, 8],
            "z": [0.5, 2.0, -2.0, -3],
        }
    )
    df_div = divop(df1, df2).rows()
    for i, (row1, row2) in enumerate(zip(df1.rows(), df2.rows())):
        for j, (elem1, elem2) in enumerate(zip(row1, row2)):
            assert divop(elem1, elem2) == df_div[i][j]


def test_glimpse(capsys: Any) -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.8, 3.0],
            "b": [4, 5, None],
            "c": [True, False, True],
            "d": [None, "b", "c"],
            "e": ["usd", "eur", None],
            "f": pl.datetime_range(
                datetime(2023, 1, 1),
                datetime(2023, 1, 3),
                "1d",
                time_unit="us",
                eager=True,
            ),
            "g": pl.datetime_range(
                datetime(2023, 1, 1),
                datetime(2023, 1, 3),
                "1d",
                time_unit="ms",
                eager=True,
            ),
            "h": pl.datetime_range(
                datetime(2023, 1, 1),
                datetime(2023, 1, 3),
                "1d",
                time_unit="ns",
                eager=True,
            ),
            "i": [[5, 6], [3, 4], [9, 8]],
            "j": [[5.0, 6.0], [3.0, 4.0], [9.0, 8.0]],
            "k": [["A", "a"], ["B", "b"], ["C", "c"]],
        }
    )
    result = df.glimpse(return_as_string=True)

    expected = textwrap.dedent(
        """\
        Rows: 3
        Columns: 11
        $ a          <f64> 1.0, 2.8, 3.0
        $ b          <i64> 4, 5, None
        $ c         <bool> True, False, True
        $ d          <str> None, 'b', 'c'
        $ e          <str> 'usd', 'eur', None
        $ f <datetime[μs]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
        $ g <datetime[ms]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
        $ h <datetime[ns]> 2023-01-01 00:00:00, 2023-01-02 00:00:00, 2023-01-03 00:00:00
        $ i    <list[i64]> [5, 6], [3, 4], [9, 8]
        $ j    <list[f64]> [5.0, 6.0], [3.0, 4.0], [9.0, 8.0]
        $ k    <list[str]> ['A', 'a'], ['B', 'b'], ['C', 'c']
        """
    )
    assert result == expected

    # the default is to print to the console
    df.glimpse(return_as_string=False)
    # remove the last newline on the capsys
    assert capsys.readouterr().out[:-1] == expected

    colc = "a" * 96
    df = pl.DataFrame({colc: [11, 22, 33, 44, 55, 66]})
    result = df.glimpse(
        return_as_string=True, max_colname_length=20, max_items_per_column=4
    )
    expected = textwrap.dedent(
        """\
        Rows: 6
        Columns: 1
        $ aaaaaaaaaaaaaaaaaaa… <i64> 11, 22, 33, 44
        """
    )
    assert result == expected


@pytest.mark.parametrize(
    ("subset", "keep", "expected_mask"),
    [
        (None, "first", [True, True, True, False]),
        ("a", "first", [True, True, False, False]),
        (["a", "b"], "first", [True, True, False, False]),
        (("a", "b"), "last", [True, False, False, True]),
        (("a", "b"), "none", [True, False, False, False]),
    ],
)
def test_unique(
    subset: str | Sequence[str], keep: UniqueKeepStrategy, expected_mask: list[bool]
) -> None:
    df = pl.DataFrame({"a": [1, 2, 2, 2], "b": [3, 4, 4, 4], "c": [5, 6, 7, 7]})

    result = df.unique(maintain_order=True, subset=subset, keep=keep)
    expected = df.filter(expected_mask)
    assert_frame_equal(result, expected)


def test_iter_slices() -> None:
    df = pl.DataFrame(
        {
            "a": range(95),
            "b": date(2023, 1, 1),
            "c": "klmnopqrstuvwxyz",
        }
    )
    batches = list(df.iter_slices(n_rows=50))

    assert len(batches[0]) == 50
    assert len(batches[1]) == 45
    assert batches[1].rows() == df[50:].rows()


def test_frame_equal() -> None:
    # Values are checked
    df1 = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": [6.0, 7.0, 8.0],
            "ham": ["a", "b", "c"],
        }
    )
    df2 = pl.DataFrame(
        {
            "foo": [3, 2, 1],
            "bar": [8.0, 7.0, 6.0],
            "ham": ["c", "b", "a"],
        }
    )

    assert df1.frame_equal(df1)
    assert not df1.frame_equal(df2)

    # Column names are checked
    df3 = pl.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [6.0, 7.0, 8.0],
            "c": ["a", "b", "c"],
        }
    )
    assert not df1.frame_equal(df3)

    # Datatypes are NOT checked
    df = pl.DataFrame(
        {
            "foo": [1, 2, None],
            "bar": [6.0, 7.0, None],
            "ham": ["a", "b", None],
        }
    )
    assert df.frame_equal(df.with_columns(pl.col("foo").cast(pl.Int8)))
    assert df.frame_equal(df.with_columns(pl.col("ham").cast(pl.Categorical)))

    # The null_equal parameter determines if None values are considered equal
    assert df.frame_equal(df)
    assert not df.frame_equal(df, null_equal=False)


def test_format_empty_df() -> None:
    df = pl.DataFrame(
        [
            pl.Series("val1", [], dtype=pl.Categorical),
            pl.Series("val2", [], dtype=pl.Categorical),
        ]
    ).select(
        [
            pl.format("{}:{}", pl.col("val1"), pl.col("val2")).alias("cat"),
        ]
    )
    assert df.shape == (0, 1)
    assert df.dtypes == [pl.Utf8]


def test_deadlocks_3409() -> None:
    assert (
        pl.DataFrame({"col1": [[1, 2, 3]]})
        .with_columns(
            pl.col("col1").list.eval(
                pl.element().map_elements(lambda x: x, return_dtype=pl.Int64)
            )
        )
        .to_dict(False)
    ) == {"col1": [[1, 2, 3]]}

    assert (
        pl.DataFrame({"col1": [1, 2, 3]})
        .with_columns(
            [pl.col("col1").cumulative_eval(pl.element().map_batches(lambda x: 0))]
        )
        .to_dict(False)
    ) == {"col1": [0, 0, 0]}


def test_clip() -> None:
    clip_exprs = [
        pl.col("a").clip(pl.col("min"), pl.col("max")).alias("clip"),
        pl.col("a").clip_min(pl.col("min")).alias("clip_min"),
        pl.col("a").clip_max(pl.col("max")).alias("clip_max"),
    ]

    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "min": [0, -1, 4, None, 4],
            "max": [2, 1, 8, 5, None],
        }
    )

    assert df.select(clip_exprs).to_dict(False) == {
        "clip": [1, 1, 4, None, None],
        "clip_min": [1, 2, 4, None, 5],
        "clip_max": [1, 1, 3, 4, None],
    }

    df = pl.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0],
            "min": [0, -1.0, 4.0, None, 4.0],
            "max": [2.0, 1.0, 8.0, 5.0, None],
        }
    )

    assert df.select(clip_exprs).to_dict(False) == {
        "clip": [1.0, 1.0, 4.0, None, None],
        "clip_min": [1.0, 2.0, 4.0, None, 5.0],
        "clip_max": [1.0, 1.0, 3.0, 4.0, None],
    }

    df = pl.DataFrame(
        {
            "a": [
                datetime(1995, 6, 5, 10, 30),
                datetime(1995, 6, 5),
                datetime(2023, 10, 20, 18, 30, 6),
                None,
                datetime(2023, 9, 24),
                datetime(2000, 1, 10),
            ],
            "min": [
                datetime(1995, 6, 5, 10, 29),
                datetime(1996, 6, 5),
                datetime(2020, 9, 24),
                datetime(2020, 1, 1),
                None,
                datetime(2000, 1, 1),
            ],
            "max": [
                datetime(1995, 7, 21, 10, 30),
                datetime(2000, 1, 1),
                datetime(2023, 9, 20, 18, 30, 6),
                datetime(2000, 1, 1),
                datetime(1993, 3, 13),
                None,
            ],
        }
    )

    assert df.select(clip_exprs).to_dict(False) == {
        "clip": [
            datetime(1995, 6, 5, 10, 30),
            datetime(1996, 6, 5),
            datetime(2023, 9, 20, 18, 30, 6),
            None,
            None,
            None,
        ],
        "clip_min": [
            datetime(1995, 6, 5, 10, 30),
            datetime(1996, 6, 5),
            datetime(2023, 10, 20, 18, 30, 6),
            None,
            None,
            datetime(2000, 1, 10),
        ],
        "clip_max": [
            datetime(1995, 6, 5, 10, 30),
            datetime(1995, 6, 5),
            datetime(2023, 9, 20, 18, 30, 6),
            None,
            datetime(1993, 3, 13),
            None,
        ],
    }


def test_cum_agg() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 2]})
    assert_series_equal(
        df.select(pl.col("a").cumsum())["a"], pl.Series("a", [1, 3, 6, 8])
    )
    assert_series_equal(
        df.select(pl.col("a").cummin())["a"], pl.Series("a", [1, 1, 1, 1])
    )
    assert_series_equal(
        df.select(pl.col("a").cummax())["a"], pl.Series("a", [1, 2, 3, 3])
    )
    assert_series_equal(
        df.select(pl.col("a").cumprod())["a"], pl.Series("a", [1, 2, 6, 12])
    )


def test_floor() -> None:
    df = pl.DataFrame({"a": [1.8, 1.2, 3.0]})
    col_a_floor = df.select(pl.col("a").floor())["a"]
    assert_series_equal(col_a_floor, pl.Series("a", [1, 1, 3]).cast(pl.Float64))


def test_floor_divide() -> None:
    x = 10.4
    step = 0.5
    df = pl.DataFrame({"x": [x]})
    assert df.with_columns(pl.col("x") // step)[0, 0] == x // step


def test_round() -> None:
    df = pl.DataFrame({"a": [1.8, 1.2, 3.0]})
    col_a_rounded = df.select(pl.col("a").round(decimals=0))["a"]
    assert_series_equal(col_a_rounded, pl.Series("a", [2, 1, 3]).cast(pl.Float64))


def test_dot() -> None:
    df = pl.DataFrame({"a": [1.8, 1.2, 3.0], "b": [3.2, 1, 2]})
    assert df.select(pl.col("a").dot(pl.col("b"))).item() == 12.96


def test_ufunc() -> None:
    df = pl.DataFrame([pl.Series("a", [1, 2, 3, 4], dtype=pl.UInt8)])
    out = df.select(
        [
            np.power(pl.col("a"), 2).alias("power_uint8"),  # type: ignore[call-overload]
            np.power(pl.col("a"), 2.0).alias("power_float64"),  # type: ignore[call-overload]
            np.power(pl.col("a"), 2, dtype=np.uint16).alias("power_uint16"),  # type: ignore[call-overload]
        ]
    )
    expected = pl.DataFrame(
        [
            pl.Series("power_uint8", [1, 4, 9, 16], dtype=pl.UInt8),
            pl.Series("power_float64", [1.0, 4.0, 9.0, 16.0], dtype=pl.Float64),
            pl.Series("power_uint16", [1, 4, 9, 16], dtype=pl.UInt16),
        ]
    )
    assert_frame_equal(out, expected)
    assert out.dtypes == expected.dtypes


def test_ufunc_expr_not_first() -> None:
    """Check numpy ufunc expressions also work if expression not the first argument."""
    df = pl.DataFrame([pl.Series("a", [1, 2, 3], dtype=pl.Float64)])
    out = df.select(
        [
            np.power(2.0, cast(Any, pl.col("a"))).alias("power"),
            (2.0 / cast(Any, pl.col("a"))).alias("divide_scalar"),
            (np.array([2, 2, 2]) / cast(Any, pl.col("a"))).alias("divide_array"),
        ]
    )
    expected = pl.DataFrame(
        [
            pl.Series("power", [2**1, 2**2, 2**3], dtype=pl.Float64),
            pl.Series("divide_scalar", [2 / 1, 2 / 2, 2 / 3], dtype=pl.Float64),
            pl.Series("divide_array", [2 / 1, 2 / 2, 2 / 3], dtype=pl.Float64),
        ]
    )
    assert_frame_equal(out, expected)


def test_ufunc_multiple_expressions() -> None:
    # example from https://github.com/pola-rs/polars/issues/6770
    df = pl.DataFrame(
        {
            "v": [
                -4.293,
                -2.4659,
                -1.8378,
                -0.2821,
                -4.5649,
                -3.8128,
                -7.4274,
                3.3443,
                3.8604,
                -4.2200,
            ],
            "u": [
                -11.2268,
                6.3478,
                7.1681,
                3.4986,
                2.7320,
                -1.0695,
                -10.1408,
                11.2327,
                6.6623,
                -8.1412,
            ],
        }
    )
    expected = np.arctan2(df.get_column("v"), df.get_column("u"))
    result = df.select(np.arctan2(pl.col("v"), pl.col("u")))[:, 0]  # type: ignore[call-overload]
    assert_series_equal(expected, result)  # type: ignore[arg-type]


def test_unstack() -> None:
    from string import ascii_uppercase

    df = pl.DataFrame(
        {
            "col1": list(ascii_uppercase[0:9]),
            "col2": pl.int_range(0, 9, eager=True),
            "col3": pl.int_range(-9, 0, eager=True),
        }
    )
    assert df.unstack(step=3, how="vertical").to_dict(False) == {
        "col1_0": ["A", "B", "C"],
        "col1_1": ["D", "E", "F"],
        "col1_2": ["G", "H", "I"],
        "col2_0": [0, 1, 2],
        "col2_1": [3, 4, 5],
        "col2_2": [6, 7, 8],
        "col3_0": [-9, -8, -7],
        "col3_1": [-6, -5, -4],
        "col3_2": [-3, -2, -1],
    }

    assert df.unstack(step=3, how="horizontal").to_dict(False) == {
        "col1_0": ["A", "D", "G"],
        "col1_1": ["B", "E", "H"],
        "col1_2": ["C", "F", "I"],
        "col2_0": [0, 3, 6],
        "col2_1": [1, 4, 7],
        "col2_2": [2, 5, 8],
        "col3_0": [-9, -6, -3],
        "col3_1": [-8, -5, -2],
        "col3_2": [-7, -4, -1],
    }

    for column_subset in (("col2", "col3"), cs.integer()):
        assert df.unstack(
            step=3,
            how="horizontal",
            columns=column_subset,  # type: ignore[arg-type]
        ).to_dict(False) == {
            "col2_0": [0, 3, 6],
            "col2_1": [1, 4, 7],
            "col2_2": [2, 5, 8],
            "col3_0": [-9, -6, -3],
            "col3_1": [-8, -5, -2],
            "col3_2": [-7, -4, -1],
        }


def test_window_deadlock() -> None:
    np.random.seed(12)

    df = pl.DataFrame(
        {
            "nrs": [1, 2, 3, None, 5],
            "names": ["foo", "ham", "spam", "egg", None],
            "random": np.random.rand(5),
            "groups": ["A", "A", "B", "C", "B"],
        }
    )

    df = df.select(
        [
            pl.col("*"),  # select all
            pl.col("random").sum().over("groups").alias("sum[random]/groups"),
            pl.col("random").implode().over("names").alias("random/name"),
        ]
    )


def test_sum_empty_column_names() -> None:
    df = pl.DataFrame({"x": [], "y": []}, schema={"x": pl.Boolean, "y": pl.Boolean})
    expected = pl.DataFrame(
        {"x": [0], "y": [0]}, schema={"x": pl.UInt32, "y": pl.UInt32}
    )
    assert_frame_equal(df.sum(), expected)


def test_flags() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.flags == {
        "a": {"SORTED_ASC": False, "SORTED_DESC": False},
        "b": {"SORTED_ASC": False, "SORTED_DESC": False},
    }
    assert df.set_sorted("a").flags == {
        "a": {"SORTED_ASC": True, "SORTED_DESC": False},
        "b": {"SORTED_ASC": False, "SORTED_DESC": False},
    }


def test_interchange() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3.0, 4.0], "c": ["foo", "bar"]})
    dfi = df.__dataframe__()

    # Testing some random properties to make sure conversion happened correctly
    assert dfi.num_rows() == 2
    assert dfi.get_column(0).dtype[1] == 64
    assert dfi.get_column_by_name("c").get_buffers()["data"][0].bufsize == 6
