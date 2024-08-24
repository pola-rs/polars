from __future__ import annotations

import sys
import typing
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from io import BytesIO
from operator import floordiv, truediv
from typing import TYPE_CHECKING, Any, Callable, Iterator, Sequence, cast

import numpy as np
import pyarrow as pa
import pytest

import polars as pl
import polars.selectors as cs
from polars._utils.construction import iterable_to_pydf
from polars.datatypes import DTYPE_TEMPORAL_UNITS
from polars.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    DuplicateError,
    InvalidOperationError,
    OutOfBoundsError,
)
from polars.testing import (
    assert_frame_equal,
    assert_frame_not_equal,
    assert_series_equal,
)
from tests.unit.conftest import INTEGER_DTYPES

if TYPE_CHECKING:
    from zoneinfo import ZoneInfo

    from polars import Expr
    from polars._typing import JoinStrategy, UniqueKeepStrategy
else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


def test_version() -> None:
    pl.__version__


def test_null_count() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": ["a", "b", None]})
    assert df.null_count().shape == (1, 2)
    assert df.null_count().row(0) == (0, 1)
    assert df.null_count().row(np.int64(0)) == (0, 1)  # type: ignore[call-overload]


@pytest.mark.parametrize("input", [None, (), [], {}, pa.Table.from_arrays([])])
def test_init_empty(input: Any) -> None:
    # test various flavours of empty init
    df = pl.DataFrame(input)
    assert df.shape == (0, 0)
    assert df.is_empty()


def test_df_bool_ambiguous() -> None:
    empty_df = pl.DataFrame()
    with pytest.raises(TypeError, match="ambiguous"):
        not empty_df


def test_special_char_colname_init() -> None:
    from string import punctuation

    cols = [(c, pl.Int8) for c in punctuation]
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

    with pytest.raises(ComputeError):
        df > "2"  # noqa: B015

    # Series
    s = pl.Series([3, 1])
    assert_frame_equal(df >= s, pl.DataFrame({"a": [False, True], "b": [True, True]}))

    # DataFrame
    other = pl.DataFrame({"a": [1, 2], "b": [2, 3]})
    assert_frame_equal(
        df == other, pl.DataFrame({"a": [True, True], "b": [False, False]})
    )
    assert_frame_equal(
        df != other, pl.DataFrame({"a": [False, False], "b": [True, True]})
    )
    assert_frame_equal(
        df > other, pl.DataFrame({"a": [False, False], "b": [True, True]})
    )
    assert_frame_equal(
        df < other, pl.DataFrame({"a": [False, False], "b": [False, False]})
    )
    assert_frame_equal(
        df >= other, pl.DataFrame({"a": [True, True], "b": [True, True]})
    )
    assert_frame_equal(
        df <= other, pl.DataFrame({"a": [True, True], "b": [False, False]})
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
    with pytest.raises(ComputeError):
        df == pl.DataFrame({"a": [1, 2], "b": ["x", "y"]})  # noqa: B015


def test_column_selection() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0], "c": ["a", "b", "c"]})

    # get column by name
    b = pl.Series("b", [1.0, 2.0, 3.0])
    assert_series_equal(df["b"], b)
    assert_series_equal(df.get_column("b"), b)

    with pytest.raises(ColumnNotFoundError, match="x"):
        df.get_column("x")

    default_series = pl.Series("x", ["?", "?", "?"])
    assert_series_equal(df.get_column("x", default=default_series), default_series)

    assert df.get_column("x", default=None) is None

    # get column by index
    assert_series_equal(df.to_series(1), pl.Series("b", [1.0, 2.0, 3.0]))
    assert_series_equal(df.to_series(-1), pl.Series("c", ["a", "b", "c"]))


def test_mixed_sequence_selection() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = df.select(["a", pl.col("b"), pl.lit("c")])
    expected = pl.DataFrame({"a": [1, 2], "b": [3, 4], "literal": ["c", "c"]})
    assert_frame_equal(result, expected)


def test_from_arrow(monkeypatch: Any) -> None:
    tbl = pa.table(
        {
            "a": pa.array([1, 2], pa.timestamp("s")),
            "b": pa.array([1, 2], pa.timestamp("ms")),
            "c": pa.array([1, 2], pa.timestamp("us")),
            "d": pa.array([1, 2], pa.timestamp("ns")),
            "e": pa.array([1, 2], pa.int32()),
            "decimal1": pa.array([1, 2], pa.decimal128(2, 1)),
            "struct": pa.array(
                [{"a": 1}, {"a": 2}], pa.struct([pa.field("a", pa.int32())])
            ),
        }
    )
    record_batches = tbl.to_batches(max_chunksize=1)
    expected_schema = {
        "a": pl.Datetime("ms"),
        "b": pl.Datetime("ms"),
        "c": pl.Datetime("us"),
        "d": pl.Datetime("ns"),
        "e": pl.Int32,
        "decimal1": pl.Decimal(2, 1),
        "struct": pl.Struct({"a": pl.Int32()}),
    }
    expected_data = [
        (
            datetime(1970, 1, 1, 0, 0, 1),
            datetime(1970, 1, 1, 0, 0, 0, 1000),
            datetime(1970, 1, 1, 0, 0, 0, 1),
            datetime(1970, 1, 1, 0, 0),
            1,
            Decimal("1.0"),
            {"a": 1},
        ),
        (
            datetime(1970, 1, 1, 0, 0, 2),
            datetime(1970, 1, 1, 0, 0, 0, 2000),
            datetime(1970, 1, 1, 0, 0, 0, 2),
            datetime(1970, 1, 1, 0, 0),
            2,
            Decimal("2.0"),
            {"a": 2},
        ),
    ]
    for arrow_data in (tbl, record_batches, (rb for rb in record_batches)):
        df = cast(pl.DataFrame, pl.from_arrow(arrow_data))
        assert df.schema == expected_schema
        assert df.rows() == expected_data

    # record batches (inc. empty)
    for b, n_expected in (
        (record_batches[0], 1),
        (record_batches[0][:0], 0),
    ):
        df = cast(pl.DataFrame, pl.from_arrow(b))
        assert df.schema == expected_schema
        assert df.rows() == expected_data[:n_expected]

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

    assert df0.schema == {"id": pl.String, "points": pl.Int64}
    assert df1.schema == {"x": pl.String, "y": pl.Int32}
    assert df2.schema == {"x": pl.String, "y": pl.Int32}

    with pytest.raises(TypeError, match="Cannot convert str"):
        pl.from_arrow(data="xyz")

    with pytest.raises(TypeError, match="Cannot convert int"):
        pl.from_arrow(data=(x for x in (1, 2, 3)))


@pytest.mark.parametrize(
    "data",
    [
        pa.Table.from_pydict(
            {
                "struct": pa.array(
                    [{"a": 1}, {"a": 2}], pa.struct([pa.field("a", pa.int32())])
                ),
            }
        ),
        pa.Table.from_pydict(
            {
                "struct": pa.chunked_array(
                    [[{"a": 1}], [{"a": 2}]], pa.struct([pa.field("a", pa.int32())])
                ),
            }
        ),
    ],
)
def test_from_arrow_struct_column(data: pa.Table) -> None:
    df = cast(pl.DataFrame, pl.from_arrow(data=data))
    expected_schema = pl.Schema({"struct": pl.Struct({"a": pl.Int32()})})
    expected_data = [({"a": 1},), ({"a": 2},)]
    assert df.schema == expected_schema
    assert df.rows() == expected_data


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


def test_sort_multi_output_exprs_01() -> None:
    df = pl.DataFrame(
        {
            "dts": [date(2077, 10, 3), date(2077, 10, 2), date(2077, 10, 2)],
            "strs": ["abc", "def", "ghi"],
            "vals": [10.5, 20.3, 15.7],
        }
    )

    expected = pl.DataFrame(
        {
            "dts": [date(2077, 10, 2), date(2077, 10, 2), date(2077, 10, 3)],
            "strs": ["ghi", "def", "abc"],
            "vals": [15.7, 20.3, 10.5],
        }
    )
    assert_frame_equal(expected, df.sort(pl.col("^(d|v).*$")))
    assert_frame_equal(expected, df.sort(cs.temporal() | cs.numeric()))
    assert_frame_equal(expected, df.sort(cs.temporal(), cs.numeric(), cs.binary()))

    expected = pl.DataFrame(
        {
            "dts": [date(2077, 10, 3), date(2077, 10, 2), date(2077, 10, 2)],
            "strs": ["abc", "def", "ghi"],
            "vals": [10.5, 20.3, 15.7],
        }
    )
    assert_frame_equal(
        expected,
        df.sort(pl.col("^(d|v).*$"), descending=[True]),
    )
    assert_frame_equal(
        expected,
        df.sort(cs.temporal() | cs.numeric(), descending=[True]),
    )
    assert_frame_equal(
        expected,
        df.sort(cs.temporal(), cs.numeric(), descending=[True, True]),
    )

    with pytest.raises(
        ValueError,
        match=r"the length of `descending` \(2\) does not match the length of `by` \(1\)",
    ):
        df.sort(by=[cs.temporal()], descending=[True, False])

    with pytest.raises(
        ValueError,
        match=r"the length of `nulls_last` \(3\) does not match the length of `by` \(2\)",
    ):
        df.sort("dts", "strs", nulls_last=[True, False, True])

    with pytest.raises(
        ComputeError,
        match="No columns selected for sorting",
    ):
        df.sort(pl.col("^xxx$"))


@pytest.mark.parametrize(
    ("by_explicit", "desc_explicit", "by_multi", "desc_multi"),
    [
        (
            ["w", "x", "y", "z"],
            [False, False, True, True],
            [cs.integer(), cs.string()],
            [False, True],
        ),
        (
            ["w", "y", "z"],
            [True, True, False],
            [pl.col("^(w|y)$"), pl.col("^z.*$")],
            [True, False],
        ),
        (
            ["z", "w", "x"],
            [True, False, False],
            [pl.col("z"), cs.numeric()],
            [True, False],
        ),
    ],
)
def test_sort_multi_output_exprs_02(
    by_explicit: list[str],
    desc_explicit: list[bool],
    by_multi: list[Expr],
    desc_multi: list[bool],
) -> None:
    df = pl.DataFrame(
        {
            "w": [100, 100, 100, 100, 200, 200, 200, 200],
            "x": [888, 888, 444, 444, 888, 888, 444, 888],
            "y": ["b", "b", "a", "a", "b", "b", "a", "a"],
            "z": ["x", "y", "x", "y", "x", "y", "x", "y"],
        }
    )
    res1 = df.sort(*by_explicit, descending=desc_explicit)
    res2 = df.sort(*by_multi, descending=desc_multi)
    assert_frame_equal(res1, res2)


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


@pytest.mark.parametrize("nulls_last", [False, True], ids=["nulls_first", "nulls_last"])
def test_sort_maintain_order_descending_repeated_nulls(nulls_last: bool) -> None:
    got = (
        pl.LazyFrame({"A": [None, -1, 1, 1, None], "B": [1, 2, 3, 4, 5]})
        .sort("A", descending=True, maintain_order=True, nulls_last=nulls_last)
        .collect()
    )
    if nulls_last:
        expect = pl.DataFrame({"A": [1, 1, -1, None, None], "B": [3, 4, 2, 1, 5]})
    else:
        expect = pl.DataFrame({"A": [None, None, 1, 1, -1], "B": [1, 5, 3, 4, 2]})
    assert_frame_equal(got, expect)


def test_replace() -> None:
    df = pl.DataFrame({"a": [2, 1, 3], "b": [1, 2, 3]})
    s = pl.Series("c", [True, False, True])
    df._replace("a", s)
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


def test_insert_column() -> None:
    df = (
        pl.DataFrame({"z": [3, 4, 5]})
        .insert_column(0, pl.Series("x", [1, 2, 3]))
        .insert_column(-1, pl.Series("y", [2, 3, 4]))
    )
    expected_df = pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [3, 4, 5]})
    assert_frame_equal(expected_df, df)


def test_replace_column() -> None:
    df = (
        pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [3, 4, 5]})
        .replace_column(0, pl.Series("a", [4, 5, 6]))
        .replace_column(-2, pl.Series("b", [5, 6, 7]))
        .replace_column(-1, pl.Series("c", [6, 7, 8]))
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


def test_to_series_bad_inputs() -> None:
    df = pl.DataFrame({"x": [1, 2, 3], "y": [2, 3, 4], "z": [3, 4, 5]})

    with pytest.raises(IndexError, match="index 5 is out of bounds"):
        df.to_series(5)

    with pytest.raises(IndexError, match="index -100 is out of bounds"):
        df.to_series(-100)

    with pytest.raises(
        TypeError, match="'str' object cannot be interpreted as an integer"
    ):
        df.to_series("x")  # type: ignore[arg-type]


def test_gather_every() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, 4], "b": ["w", "x", "y", "z"]})
    expected_df = pl.DataFrame({"a": [1, 3], "b": ["w", "y"]})
    assert_frame_equal(expected_df, df.gather_every(2))

    expected_df = pl.DataFrame({"a": [2, 4], "b": ["x", "z"]})
    assert_frame_equal(expected_df, df.gather_every(2, offset=1))


def test_gather_every_agg() -> None:
    df = pl.DataFrame(
        {
            "g": [1, 1, 1, 2, 2, 2],
            "a": ["a", "b", "c", "d", "e", "f"],
        }
    )
    out = df.group_by(pl.col("g")).agg(pl.col("a").gather_every(2)).sort("g")
    expected = pl.DataFrame(
        {
            "g": [1, 2],
            "a": [["a", "c"], ["d", "f"]],
        }
    )
    assert_frame_equal(out, expected)


def test_take_misc(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars

    # Out of bounds error.
    with pytest.raises(OutOfBoundsError):
        df.sort("fruits").select(
            pl.col("B").reverse().gather([1, 2]).implode().over("fruits"),
            "fruits",
        )

    # Null indices.
    assert_frame_equal(
        df.select(pl.col("fruits").gather(pl.Series([0, None]))),
        pl.DataFrame({"fruits": ["banana", None]}),
    )

    for index in [[0, 1], pl.Series([0, 1]), np.array([0, 1])]:
        out = df.sort("fruits").select(
            pl.col("B")
            .reverse()
            .gather(index)  # type: ignore[arg-type]
            .over("fruits", mapping_strategy="join"),
            "fruits",
        )

        assert out[0, "B"].to_list() == [2, 3]
        assert out[4, "B"].to_list() == [1, 4]

    out = df.sort("fruits").select(
        pl.col("B").reverse().get(pl.lit(1)).over("fruits"),
        "fruits",
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
    with pytest.raises(ComputeError):
        pl.read_parquet(f)


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
    df2 = pl.concat([df1, df1], rechunk=True)

    assert df2.shape == (6, 3)
    assert df2.n_chunks() == 1
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
    result = pl.DataFrame({"x": pl.arange(0, 3, eager=True)}).to_dummies("x")
    expected = pl.DataFrame(
        {"x_0": [1, 0, 0], "x_1": [0, 1, 0], "x_2": [0, 0, 1]}
    ).with_columns(pl.all().cast(pl.UInt8))
    assert_frame_equal(result, expected)


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
    assert_frame_equal(dm.select(dd.columns), dd)
    assert dd.rows() == [
        (0, 0, 0, 0, 0, 0),
        (1, 0, 1, 0, 1, 0),
        (0, 1, 0, 1, 0, 1),
    ]


def test_to_pandas(df: pl.DataFrame) -> None:
    # pyarrow cannot deal with unsigned dictionary integer yet.
    # pyarrow cannot convert a time64 w/ non-zero nanoseconds
    df = df.drop(["cat", "time", "enum"])
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
    assert len(df.sum_horizontal()) == 3
    assert len(df.mean_horizontal()) == 3
    assert len(df.min_horizontal()) == 3
    assert len(df.max_horizontal()) == 3

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
        .agg([pl.col("*").exclude(keys).head(2).name.keep()])
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
        dtypes={"b": pl.Float32, "c": pl.String, "d": pl.Datetime("ms")}
    ).schema == {
        "a": pl.Float64,
        "b": pl.Float32,
        "c": pl.String,
        "d": pl.Datetime("ms"),
    }

    # cast via selector:dtype map
    assert df.cast(
        {
            cs.numeric(): pl.UInt8,
            cs.temporal(): pl.String,
        }
    ).rows() == [
        (1, 4, True, "2020-01-02"),
        (2, 5, False, "2021-03-04"),
        (3, None, True, "2022-05-06"),
    ]

    # cast all fields to a single type
    assert df.cast(pl.String).to_dict(as_series=False) == {
        "a": ["1.0", "2.5", "3.0"],
        "b": ["4", "5", None],
        "c": ["true", "false", "true"],
        "d": ["2020-01-02", "2021-03-04", "2022-05-06"],
    }


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
        "c": pl.String,
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
            orient="row",
        ),
        out,
        atol=0.00001,
    )


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
            "a": pl.String,
            "b": pl.Int64,
            "c": pl.Int64,
            "d": pl.Int64,
        }

    # test 'iterable_to_pydf' directly to validate 'chunk_size' behaviour
    cols = ["a", "b", ("c", pl.Int8), "d"]

    expected_data = [("0", 0, 1, 1), ("1", 1, 2, 3), ("2", 2, 4, 9), ("3", 3, 8, 27)]
    expected_schema = [
        ("a", pl.String),
        ("b", pl.Int64),
        ("c", pl.Int8),
        ("d", pl.Int64),
    ]

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
    assert df.schema == {"col": pl.List(pl.String)}
    assert df[-2:]["col"].to_list() == [None, ["a", "b", "c"]]

    # empty iterator
    assert_frame_equal(
        pl.DataFrame(data=gen(0), schema=["a", "b", "c", "d"]),
        pl.DataFrame(schema=["a", "b", "c", "d"]),
    )


def test_from_rows() -> None:
    df = pl.from_records([[1, 2, "foo"], [2, 3, "bar"]], orient="row")
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
    df = pl.from_records(data, orient="row")
    assert data == df.rows()


def test_from_rows_of_dicts() -> None:
    records = [
        {"id": 1, "value": 100, "_meta": "a"},
        {"id": 2, "value": 101, "_meta": "b"},
    ]
    df_init: Callable[..., Any]
    for df_init in (pl.from_dicts, pl.DataFrame):
        df1 = df_init(records)
        assert df1.rows() == [(1, 100, "a"), (2, 101, "b")]

        overrides = {
            "id": pl.Int16,
            "value": pl.Int32,
        }
        df2 = df_init(records, schema_overrides=overrides)
        assert df2.rows() == [(1, 100, "a"), (2, 101, "b")]
        assert df2.schema == {"id": pl.Int16, "value": pl.Int32, "_meta": pl.String}

        df3 = df_init(records, schema=overrides)
        assert df3.rows() == [(1, 100), (2, 101)]
        assert df3.schema == {"id": pl.Int16, "value": pl.Int32}


def test_from_records_with_schema_overrides_12032() -> None:
    # the 'id' fields contains an int value that exceeds Int64 and doesn't have an exact
    # Float64 representation; confirm that the override is applied *during* inference,
    # not as a post-inference cast, so we maintain the accuracy of the original value.
    rec = [
        {"id": 9187643043065364490, "x": 333, "y": None},
        {"id": 9223671840084328467, "x": 666.5, "y": 1698177261953686},
        {"id": 9187643043065364505, "x": 999, "y": 9223372036854775807},
    ]
    df = pl.from_records(rec, schema_overrides={"x": pl.Float32, "id": pl.UInt64})
    assert df.schema == OrderedDict(
        [
            ("id", pl.UInt64),
            ("x", pl.Float32),
            ("y", pl.Int64),
        ]
    )
    assert rec == df.rows(named=True)


def test_from_large_uint64_misc() -> None:
    uint_data = [[9187643043065364490, 9223671840084328467, 9187643043065364505]]

    df = pl.DataFrame(uint_data, orient="col", schema_overrides={"column_0": pl.UInt64})
    assert df["column_0"].dtype == pl.UInt64
    assert df["column_0"].to_list() == uint_data[0]

    for overrides in ({}, {"column_1": pl.UInt64}):
        df = pl.DataFrame(
            uint_data,
            orient="row",
            schema_overrides=overrides,
        )
        assert df.schema == OrderedDict(
            [
                ("column_0", pl.Int64),
                ("column_1", pl.UInt64),
                ("column_2", pl.Int64),
            ]
        )
        assert df.row(0) == tuple(uint_data[0])


def test_repeat_by_unequal_lengths_panic() -> None:
    df = pl.DataFrame(
        {
            "a": ["x", "y", "z"],
        }
    )
    with pytest.raises(
        ComputeError,
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

    result = pl.Series([1, 2, 3]) @ pl.Series([4, 5, 6])
    assert isinstance(result, int)
    assert result == 32

    result = pl.Series([1, 2, 3]) @ pl.Series([4.0, 5.0, 6.0])
    assert isinstance(result, float)
    assert result == 32.0

    result = pl.Series([1.0, 2.0, 3.0]) @ pl.Series([4.0, 5.0, 6.0])
    assert isinstance(result, float)
    assert result == 32.0

    with pytest.raises(
        InvalidOperationError, match="`dot` operation not supported for dtype `bool`"
    ):
        pl.Series([True, False, False, True]) @ pl.Series([4, 5, 6, 7])

    with pytest.raises(
        InvalidOperationError, match="`dot` operation not supported for dtype `str`"
    ):
        pl.Series([1, 2, 3, 4]) @ pl.Series(["True", "False", "False", "True"])


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
            [8661293245726181094, 9565952849861441858, 2921274555702885622],
            dtype=pl.UInt64,
        )
        result = df.hash_rows(*seeds)
        assert_series_equal(expected, result, check_names=False, check_exact=True)
        result = df["s"].hash(*seeds)
        assert_series_equal(expected, result, check_names=False, check_exact=True)
        result = df.select([pl.col("s").hash(*seeds)])["s"]
        assert_series_equal(expected, result, check_names=False, check_exact=True)


@pytest.mark.slow()
@pytest.mark.parametrize(
    "e",
    [
        pl.int_range(1_000_000),
        # Test code path for null_count > 0
        pl.when(pl.int_range(1_000_000) != 0).then(pl.int_range(1_000_000)),
    ],
)
def test_hash_collision_multiple_columns_equal_values_15390(e: pl.Expr) -> None:
    df = pl.select(e.alias("a"))

    for n_columns in (1, 2, 3, 4):
        s = df.select(pl.col("a").alias(f"x{i}") for i in range(n_columns)).hash_rows()

        vc = s.sort().value_counts(sort=True)
        max_bucket_size = vc["count"][0]

        assert max_bucket_size == 1


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
    df = pl.DataFrame({"a": [1], "b": [None]}, schema={"a": pl.Int64, "b": pl.Float32})

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


def test_horizontal_agg() -> None:
    df = pl.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})

    assert_series_equal(df.sum_horizontal(), pl.Series("sum", [2, 2, 6]))
    assert_series_equal(
        df.sum_horizontal(ignore_nulls=False), pl.Series("sum", [2, None, 6])
    )
    assert_series_equal(
        df.mean_horizontal(ignore_nulls=False), pl.Series("mean", [1.0, None, 3.0])
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


def test_group_by_agg_n_unique_empty_group_idx_path() -> None:
    df = pl.DataFrame(
        {
            "key": [1, 1, 1, 2, 2, 2],
            "value": [1, 2, 3, 4, 5, 6],
            "filt": [True, True, True, False, False, False],
        }
    )
    out = df.group_by("key", maintain_order=True).agg(
        pl.col("value").filter("filt").n_unique().alias("n_unique")
    )
    expected = pl.DataFrame(
        {
            "key": [1, 2],
            "n_unique": pl.Series([3, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(out, expected)


def test_group_by_agg_n_unique_empty_group_slice_path() -> None:
    df = pl.DataFrame(
        {
            "key": [1, 1, 1, 2, 2, 2],
            "value": [1, 2, 3, 4, 5, 6],
            "filt": [False, False, False, False, False, False],
        }
    )
    out = df.group_by("key", maintain_order=True).agg(
        pl.col("value").filter("filt").n_unique().alias("n_unique")
    )
    expected = pl.DataFrame(
        {
            "key": [1, 2],
            "n_unique": pl.Series([0, 0], dtype=pl.UInt32),
        }
    )
    assert_frame_equal(out, expected)


def test_select_by_dtype(df: pl.DataFrame) -> None:
    out = df.select(pl.col(pl.String))
    assert out.columns == ["strings", "strings_nulls"]
    out = df.select(pl.col([pl.String, pl.Boolean]))
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


def test_with_row_index() -> None:
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    out = df.with_row_index()
    assert out["index"].to_list() == [0, 1, 2]

    out = df.lazy().with_row_index().collect()
    assert out["index"].to_list() == [0, 1, 2]


def test_with_row_index_bad_offset() -> None:
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    with pytest.raises(ValueError, match="cannot be negative"):
        df.with_row_index(offset=-1)
    with pytest.raises(
        ValueError, match="cannot be greater than the maximum index value"
    ):
        df.with_row_index(offset=2**32)


def test_with_row_index_bad_offset_lazy() -> None:
    lf = pl.LazyFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    with pytest.raises(ValueError, match="cannot be negative"):
        lf.with_row_index(offset=-1)
    with pytest.raises(
        ValueError, match="cannot be greater than the maximum index value"
    ):
        lf.with_row_index(offset=2**32)


def test_with_row_count_deprecated() -> None:
    df = pl.DataFrame({"a": [1, 1, 3], "b": [1.0, 2.0, 2.0]})

    with pytest.deprecated_call():
        out = df.with_row_count()
    assert out["row_nr"].to_list() == [0, 1, 2]

    with pytest.deprecated_call():
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
    # foos and sys.getrefcount have a reference.
    base_count = 2
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


@pytest.mark.parametrize("name", [None, "n", ""])
def test_group_by_order_dispatch(name: str | None) -> None:
    df = pl.DataFrame({"x": list("bab"), "y": range(3)})
    lf = df.lazy()

    result = df.group_by("x", maintain_order=True).len(name=name)
    lazy_result = lf.group_by("x").len(name=name).sort(by="x", descending=True)

    name = "len" if name is None else name
    expected = pl.DataFrame(
        data={"x": ["b", "a"], name: [2, 1]},
        schema_overrides={name: pl.UInt32},
    )
    assert_frame_equal(result, expected)
    assert_frame_equal(lazy_result.collect(), expected)

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
    expected = {"foo": pl.Int64, "bar": pl.Float64, "ham": pl.String}
    assert df.schema == expected


def test_schema_equality() -> None:
    lf = pl.LazyFrame({"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0]})
    lf_rev = lf.select("bar", "foo")

    assert lf.collect_schema() != lf_rev.collect_schema()
    assert lf.collect().schema != lf_rev.collect().schema


def test_df_schema_unique() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(DuplicateError):
        df.columns = ["a", "a"]

    with pytest.raises(DuplicateError):
        df.rename({"b": "a"})


def test_empty_projection() -> None:
    empty_df = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).select([])
    assert empty_df.rows() == []
    assert empty_df.schema == {}
    assert empty_df.shape == (0, 0)


def test_fill_null() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [3, None]})
    assert_frame_equal(df.fill_null(4), pl.DataFrame({"a": [1, 2], "b": [3, 4]}))
    assert_frame_equal(
        df.fill_null(strategy="max"), pl.DataFrame({"a": [1, 2], "b": [3, 3]})
    )

    # string and list data
    # string goes via binary
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
        pl.all().forward_fill().name.suffix("_forward"),
        pl.all().backward_fill().name.suffix("_backward"),
    ).to_dict(as_series=False) == {
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


def test_forward_fill() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    fill = df.select(pl.col("a").forward_fill())["a"]
    assert_series_equal(fill, pl.Series("a", [1, 1, 3]).cast(pl.Float64))

    df = pl.DataFrame({"a": [None, 1, None]})
    fill = df.select(pl.col("a").forward_fill())["a"]
    assert_series_equal(fill, pl.Series("a", [None, 1, 1]).cast(pl.Int64))


def test_backward_fill() -> None:
    df = pl.DataFrame({"a": [1.0, None, 3.0]})
    fill = df.select(pl.col("a").backward_fill())["a"]
    assert_series_equal(fill, pl.Series("a", [1, 3, 3]).cast(pl.Float64))

    df = pl.DataFrame({"a": [None, 1, None]})
    fill = df.select(pl.col("a").backward_fill())["a"]
    assert_series_equal(fill, pl.Series("a", [1, 1, None]).cast(pl.Int64))


def test_shrink_to_fit() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3], "bar": [6, 7, 8], "ham": ["a", "b", "c"]})

    assert df.shrink_to_fit(in_place=True) is df
    assert df.shrink_to_fit(in_place=False) is not df
    assert_frame_equal(df.shrink_to_fit(in_place=False), df)


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
            "str": ["a", "b", "c"],
        },
        schema_overrides={
            "int": pl.UInt16,
            "flt": pl.Float32,
        },
    )
    out = df.product()
    expected = pl.DataFrame(
        {"int": [6], "flt": [-108.0], "bool_0": [0], "bool_1": [1], "str": [None]}
    )
    assert_frame_not_equal(out, expected, check_dtypes=True)
    assert_frame_equal(out, expected, check_dtypes=False)


def test_first_last_nth_expressions(fruits_cars: pl.DataFrame) -> None:
    df = fruits_cars
    out = df.select(pl.first())
    assert out.columns == ["A"]

    out = df.select(pl.last())
    assert out.columns == ["cars"]

    out = df.select(pl.nth(0))
    assert out.columns == ["A"]

    out = df.select(pl.nth(1))
    assert out.columns == ["fruits"]

    out = df.select(pl.nth(-2))
    assert out.columns == ["B"]


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
    assert df_empty_isin.schema == {"foo": pl.String}


def test_group_by_slice_expression_args() -> None:
    df = pl.DataFrame({"groups": ["a"] * 10 + ["b"] * 20, "vals": range(30)})

    out = (
        df.group_by("groups", maintain_order=True)
        .agg([pl.col("vals").slice((pl.len() * 0.1).cast(int), (pl.len() // 5))])
        .explode("vals")
    )

    expected = pl.DataFrame(
        {"groups": ["a", "a", "b", "b", "b", "b"], "vals": [1, 2, 12, 13, 14, 15]}
    )
    assert_frame_equal(out, expected)


def test_join_suffixes() -> None:
    df_a = pl.DataFrame({"A": [1], "B": [1]})
    df_b = pl.DataFrame({"A": [1], "B": [1]})

    join_strategies: list[JoinStrategy] = ["left", "inner", "full", "cross"]
    for how in join_strategies:
        # no need for an assert, we error if wrong
        df_a.join(df_b, on="A", suffix="_y", how=how)["B_y"]

    df_a.join_asof(df_b, on=pl.col("A").set_sorted(), suffix="_y")["B_y"]


def test_explode_empty() -> None:
    df = (
        pl.DataFrame({"x": ["a", "a", "b", "b"], "y": [1, 1, 2, 2]})
        .group_by("x", maintain_order=True)
        .agg(pl.col("y").gather([]))
    )
    assert df.explode("y").to_dict(as_series=False) == {
        "x": ["a", "b"],
        "y": [None, None],
    }

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
    assert df.explode("numbers").to_dict(as_series=False) == {
        "letters": ["a"],
        "numbers": [None],
    }


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


def test_list_of_list_of_struct() -> None:
    expected = [{"list_of_list_of_struct": [[{"a": 1}, {"a": 2}]]}]
    pa_df = pa.Table.from_pylist(expected)

    df = pl.from_arrow(pa_df)
    assert df.rows() == [([[{"a": 1}, {"a": 2}]],)]  # type: ignore[union-attr]
    assert df.to_dicts() == expected  # type: ignore[union-attr]

    df = pl.from_arrow(pa_df[:0])
    assert df.to_dicts() == []  # type: ignore[union-attr]


def test_concat_to_empty() -> None:
    assert pl.concat([pl.DataFrame([]), pl.DataFrame({"a": [1]})]).to_dict(
        as_series=False
    ) == {"a": [1]}


def test_fill_null_limits() -> None:
    assert pl.DataFrame(
        {
            "a": [1, None, None, None, 5, 6, None, None, None, 10],
            "b": ["a", None, None, None, "b", "c", None, None, None, "d"],
            "c": [True, None, None, None, False, True, None, None, None, False],
        }
    ).select(
        pl.all().fill_null(strategy="forward", limit=2),
        pl.all().fill_null(strategy="backward", limit=2).name.suffix("_backward"),
    ).to_dict(as_series=False) == {
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

    with pytest.raises(ComputeError):
        fruits_cars.select(pl.col("fruits").upper_bound())


def test_selection_misc() -> None:
    df = pl.DataFrame({"x": "abc"}, schema={"x": pl.String})

    # literal values (as scalar/list)
    for zero in (0, [0]):
        assert df.select(zero)["literal"].to_list() == [0]
    assert df.select(literal=0)["literal"].to_list() == [0]

    # expect string values to be interpreted as cols
    for x in ("x", ["x"], pl.col("x")):
        assert df.select(x).rows() == [("abc",)]

    # string col + lit
    assert df.with_columns(["x", 0]).to_dicts() == [{"x": "abc", "literal": 0}]


def test_selection_regex_and_multicol() -> None:
    test_df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4],
            "b": [5, 6, 7, 8],
            "c": [9, 10, 11, 12],
            "foo": [13, 14, 15, 16],
        },
        schema_overrides={"foo": pl.UInt8},
    )

    # Selection only
    test_df.select(
        pl.col(["a", "b", "c"]).name.suffix("_list"),
        pl.all().exclude("foo").name.suffix("_wild"),
        pl.col("^\\w$").name.suffix("_regex"),
    )

    # Multi * Single
    assert test_df.select(pl.col(["a", "b", "c"]) * pl.col("foo")).to_dict(
        as_series=False
    ) == {
        "a": [13, 28, 45, 64],
        "b": [65, 84, 105, 128],
        "c": [117, 140, 165, 192],
    }
    assert test_df.select(pl.all().exclude("foo") * pl.col("foo")).to_dict(
        as_series=False
    ) == {
        "a": [13, 28, 45, 64],
        "b": [65, 84, 105, 128],
        "c": [117, 140, 165, 192],
    }

    assert test_df.select(pl.col("^\\w$") * pl.col("foo")).to_dict(as_series=False) == {
        "a": [13, 28, 45, 64],
        "b": [65, 84, 105, 128],
        "c": [117, 140, 165, 192],
    }

    # Multi * Multi
    result = test_df.select(pl.col(["a", "b", "c"]) * pl.col(["a", "b", "c"]))
    expected = {"a": [1, 4, 9, 16], "b": [25, 36, 49, 64], "c": [81, 100, 121, 144]}

    assert result.to_dict(as_series=False) == expected
    assert test_df.select(pl.exclude("foo") * pl.exclude("foo")).to_dict(
        as_series=False
    ) == {
        "a": [1, 4, 9, 16],
        "b": [25, 36, 49, 64],
        "c": [81, 100, 121, 144],
    }
    assert test_df.select(pl.col("^\\w$") * pl.col("^\\w$")).to_dict(
        as_series=False
    ) == {
        "a": [1, 4, 9, 16],
        "b": [25, 36, 49, 64],
        "c": [81, 100, 121, 144],
    }

    # kwargs
    with pl.Config() as cfg:
        cfg.set_auto_structify(True)

        df = test_df.select(
            pl.col("^\\w$").alias("re"),
            odd=(pl.col(INTEGER_DTYPES) % 2).name.suffix("_is_odd"),
            maxes=pl.all().max().name.suffix("_max"),
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


@pytest.mark.parametrize("subset", ["a", cs.starts_with("x", "a")])
def test_unique_on_sorted(subset: Any) -> None:
    df = pl.DataFrame(data={"a": [1, 1, 3], "b": [1, 2, 3]})

    result = df.with_columns([pl.col("a").set_sorted()]).unique(
        subset=subset,
        keep="last",
    )

    expected = pl.DataFrame({"a": [1, 3], "b": [2, 3]})
    assert_frame_equal(result, expected)


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

    assert df.to_dict(as_series=False) == {
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
    for time_unit in DTYPE_TEMPORAL_UNITS:
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
            result = pl.DataFrame(
                data={
                    "d1": [dtm.replace(tzinfo=ZoneInfo(tz_us))],
                    "d2": [dtm.replace(tzinfo=ZoneInfo(tz_europe))],
                },
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
    ),
    [
        (None, "", None, None, datetime(2020, 1, 1)),
        (
            timezone(timedelta(hours=-8)),
            "-08:00",
            "UTC",
            "UTC",
            datetime(2020, 1, 1, 8, tzinfo=timezone.utc),
        ),
        (
            timezone(timedelta(hours=-8)),
            "-08:00",
            None,
            "UTC",
            datetime(2020, 1, 1, 8, tzinfo=timezone.utc),
        ),
    ],
)
def test_init_vs_strptime_consistency(
    tzinfo: timezone | None,
    offset: str,
    dtype_time_zone: str | None,
    expected_time_zone: str,
    expected_item: datetime,
) -> None:
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


def test_init_vs_strptime_consistency_converts() -> None:
    result = pl.Series(
        [datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=-8)))],
        dtype=pl.Datetime("us", "US/Pacific"),
    ).item()
    assert result == datetime(2020, 1, 1, 0, 0, tzinfo=ZoneInfo(key="US/Pacific"))
    result = (
        pl.Series(["2020-01-01 00:00-08:00"])
        .str.strptime(pl.Datetime("us", "US/Pacific"))
        .item()
    )
    assert result == datetime(2020, 1, 1, 0, 0, tzinfo=ZoneInfo(key="US/Pacific"))


def test_init_physical_with_timezone() -> None:
    tz_uae = "Asia/Dubai"
    tz_asia = "Asia/Tokyo"

    dtm_us = 1665577800000000
    for time_unit in DTYPE_TEMPORAL_UNITS:
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


def test_format_empty_df() -> None:
    df = pl.DataFrame(
        [
            pl.Series("val1", [], dtype=pl.Categorical),
            pl.Series("val2", [], dtype=pl.Categorical),
        ]
    ).select(
        pl.format("{}:{}", pl.col("val1"), pl.col("val2")).alias("cat"),
    )
    assert df.shape == (0, 1)
    assert df.dtypes == [pl.String]


def test_deadlocks_3409() -> None:
    assert (
        pl.DataFrame({"col1": [[1, 2, 3]]})
        .with_columns(
            pl.col("col1").list.eval(
                pl.element().map_elements(lambda x: x, return_dtype=pl.Int64)
            )
        )
        .to_dict(as_series=False)
    ) == {"col1": [[1, 2, 3]]}

    assert (
        pl.DataFrame({"col1": [1, 2, 3]})
        .with_columns(
            pl.col("col1").cumulative_eval(pl.element().map_batches(lambda x: 0))
        )
        .to_dict(as_series=False)
    ) == {"col1": [0, 0, 0]}


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


def test_unstack() -> None:
    from string import ascii_uppercase

    df = pl.DataFrame(
        {
            "col1": list(ascii_uppercase[0:9]),
            "col2": pl.int_range(0, 9, eager=True),
            "col3": pl.int_range(-9, 0, eager=True),
        }
    )
    assert df.unstack(step=3, how="vertical").to_dict(as_series=False) == {
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

    assert df.unstack(step=3, how="horizontal").to_dict(as_series=False) == {
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
            columns=column_subset,
        ).to_dict(as_series=False) == {
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

    _df = df.select(
        pl.col("*"),  # select all
        pl.col("random").sum().over("groups").alias("sum[random]/groups"),
        pl.col("random").implode().over("names").alias("random/name"),
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


def test_from_dicts_undeclared_column_dtype() -> None:
    data = [{"a": 1, "b": 2}]
    result = pl.from_dicts(data, schema=["x"])
    assert result.schema == {"x": pl.Null}


def test_from_dicts_with_override() -> None:
    data = [
        {"a": "1", "b": str(2**64 - 1), "c": "1"},
        {"a": "1", "b": "1", "c": "-5.0"},
    ]
    override = {"a": pl.Int32, "b": pl.UInt64, "c": pl.Float32}
    result = pl.from_dicts(data, schema_overrides=override)
    assert_frame_equal(
        result,
        pl.DataFrame(
            {
                "a": pl.Series([1, 1], dtype=pl.Int32),
                "b": pl.Series([2**64 - 1, 1], dtype=pl.UInt64),
                "c": pl.Series([1.0, -5.0], dtype=pl.Float32),
            }
        ),
    )


def test_from_records_u64_12329() -> None:
    s = pl.from_records([{"a": 9908227375760408577}])
    assert s.dtypes == [pl.UInt64]
    assert s["a"][0] == 9908227375760408577


def test_negative_slice_12642() -> None:
    df = pl.DataFrame({"x": range(5)})
    assert_frame_equal(df.slice(-2, 1), df.tail(2).head(1))


def test_iter_columns() -> None:
    df = pl.DataFrame({"a": [1, 1, 2], "b": [4, 5, 6]})
    iter_columns = df.iter_columns()
    assert_series_equal(next(iter_columns), pl.Series("a", [1, 1, 2]))
    assert_series_equal(next(iter_columns), pl.Series("b", [4, 5, 6]))


def test_get_column_index() -> None:
    df = pl.DataFrame({"actual": [1001], "expected": [1000]})

    assert df.get_column_index("actual") == 0
    assert df.get_column_index("expected") == 1

    with pytest.raises(ColumnNotFoundError, match="missing"):
        df.get_column_index("missing")
