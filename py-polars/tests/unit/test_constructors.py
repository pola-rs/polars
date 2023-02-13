from __future__ import annotations

import typing
from datetime import date, datetime
from decimal import Decimal
from random import shuffle
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_init_dict() -> None:
    # Empty dictionary
    df = pl.DataFrame({})
    assert df.shape == (0, 0)

    # Empty dictionary/values
    df = pl.DataFrame({"a": [], "b": []})
    assert df.shape == (0, 2)
    assert df.schema == {"a": pl.Float32, "b": pl.Float32}

    for df in (
        pl.DataFrame({}, schema={"a": pl.Date, "b": pl.Utf8}),
        pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Date, "b": pl.Utf8}),
    ):
        assert df.shape == (0, 2)
        assert df.schema == {"a": pl.Date, "b": pl.Utf8}

    # List of empty list/tuple
    df = pl.DataFrame({"a": [[]], "b": [()]})
    expected = {"a": pl.List(pl.Float64), "b": pl.List(pl.Float64)}
    assert df.schema == expected
    assert df.rows() == [([], [])]

    # Mixed dtypes
    df = pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]})
    assert df.shape == (3, 2)
    assert df.columns == ["a", "b"]
    assert df.dtypes == [pl.Int64, pl.Float64]

    df = pl.DataFrame(
        data={"a": [1, 2, 3], "b": [1.0, 2.0, 3.0]},
        schema=[("a", pl.Int8), ("b", pl.Float32)],
    )
    assert df.schema == {"a": pl.Int8, "b": pl.Float32}

    # Values contained in tuples
    df = pl.DataFrame({"a": (1, 2, 3), "b": [1.0, 2.0, 3.0]})
    assert df.shape == (3, 2)

    # Datetime/Date types (from both python and integer values)
    py_datetimes = (
        datetime(2022, 12, 31, 23, 59, 59),
        datetime(2022, 12, 31, 23, 59, 59),
    )
    py_dates = (date(2022, 12, 31), date(2022, 12, 31))
    int_datetimes = [1672531199000000, 1672531199000000]
    int_dates = [19357, 19357]

    for dates, datetimes, coldefs in (
        # test inferred and explicit (given both py/polars dtypes)
        (py_dates, py_datetimes, None),
        (py_dates, py_datetimes, [("dt", date), ("dtm", datetime)]),
        (py_dates, py_datetimes, [("dt", pl.Date), ("dtm", pl.Datetime)]),
        (int_dates, int_datetimes, [("dt", date), ("dtm", datetime)]),
        (int_dates, int_datetimes, [("dt", pl.Date), ("dtm", pl.Datetime)]),
    ):
        df = pl.DataFrame(
            data={"dt": dates, "dtm": datetimes},
            schema=coldefs,
        )
        assert df.schema == {"dt": pl.Date, "dtm": pl.Datetime}
        assert df.rows() == list(zip(py_dates, py_datetimes))

    # Overriding dict column names/types
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, schema=["c", "d"])
    assert df.columns == ["c", "d"]

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        schema=["c", ("d", pl.Int8)],
    )  # partial type info (allowed, but mypy doesn't like it ;p)
    assert df.schema == {"c": pl.Int64, "d": pl.Int8}

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, schema=[("c", pl.Int8), ("d", pl.Int16)]
    )
    assert df.schema == {"c": pl.Int8, "d": pl.Int16}

    dfe = df.cleared()
    assert df.schema == dfe.schema
    assert len(dfe) == 0

    # empty nested objects
    for empty_val in [None, "", {}, []]:  # type: ignore[var-annotated]
        test = [{"field": {"sub_field": empty_val, "sub_field_2": 2}}]
        df = pl.DataFrame(test, schema={"field": pl.Object})
        assert df.to_dict(False)["field"][0] == test[0]["field"]


def test_init_dataclasses_and_namedtuple() -> None:
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

            # partial dtypes override
            df = DF(  # type: ignore[operator]
                data=trades,
                schema_overrides={"timestamp": pl.Datetime("ms"), "size": pl.Int32},
            )
            assert df.schema == {
                "timestamp": pl.Datetime("ms"),
                "ticker": pl.Utf8,
                "price": pl.Float64,
                "size": pl.Int32,
            }

        # in conjunction with full 'columns' override (rename/downcast)
        df = pl.DataFrame(
            data=trades,
            schema=[
                ("ts", pl.Datetime("ms")),
                ("tk", pl.Categorical),
                ("pc", pl.Float32),
                ("sz", pl.UInt16),
            ],
        )
        assert df.schema == {
            "ts": pl.Datetime("ms"),
            "tk": pl.Categorical,
            "pc": pl.Float32,
            "sz": pl.UInt16,
        }
        assert df.rows() == raw_data


def test_init_ndarray(monkeypatch: Any) -> None:
    # Empty array
    df = pl.DataFrame(np.array([]))
    assert_frame_equal(df, pl.DataFrame())

    # 1D array
    df = pl.DataFrame(np.array([1, 2, 3], dtype=np.int64), schema=["a"])
    expected = pl.DataFrame({"a": [1, 2, 3]})
    assert_frame_equal(df, expected)

    df = pl.DataFrame(np.array([1, 2, 3]), schema=[("a", pl.Int32)])
    expected = pl.DataFrame({"a": [1, 2, 3]}).with_columns(pl.col("a").cast(pl.Int32))
    assert_frame_equal(df, expected)

    # 2D array (or 2x 1D array) - should default to column orientation
    for data in (
        np.array([[1, 2], [3, 4]], dtype=np.int64),
        [np.array([1, 2], dtype=np.int64), np.array([3, 4], dtype=np.int64)],
    ):
        df = pl.DataFrame(data, orient="col")
        expected = pl.DataFrame({"column_0": [1, 2], "column_1": [3, 4]})
        assert_frame_equal(df, expected)

    df = pl.DataFrame([[1, 2.0, "a"], [None, None, None]], orient="row")
    expected = pl.DataFrame(
        {"column_0": [1, None], "column_1": [2.0, None], "column_2": ["a", None]}
    )
    assert_frame_equal(df, expected)

    df = pl.DataFrame(
        data=[[1, 2.0, "a"], [None, None, None]],
        schema=[("x", pl.Boolean), ("y", pl.Int32), "z"],
        orient="row",
    )
    assert df.rows() == [(True, 2, "a"), (None, None, None)]
    assert df.schema == {"x": pl.Boolean, "y": pl.Int32, "z": pl.Utf8}

    # 2D array - default to column orientation
    df = pl.DataFrame(np.array([[1, 2], [3, 4]], dtype=np.int64))
    expected = pl.DataFrame({"column_0": [1, 3], "column_1": [2, 4]})
    assert_frame_equal(df, expected)

    # no orientation is numpy convention
    df = pl.DataFrame(np.ones((3, 1), dtype=np.int64))
    assert df.shape == (3, 1)

    # 2D array - row orientation inferred
    df = pl.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64), schema=["a", "b", "c"]
    )
    expected = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    assert_frame_equal(df, expected)

    # 2D array - column orientation inferred
    df = pl.DataFrame(
        np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64), schema=["a", "b"]
    )
    expected = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_frame_equal(df, expected)

    # 2D array - orientation conflicts with columns
    with pytest.raises(ValueError):
        pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), schema=["a", "b"], orient="row")
    with pytest.raises(ValueError):
        pl.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6]]),
            schema=[("a", pl.UInt32), ("b", pl.UInt32)],
            orient="row",
        )

    # 3D array
    with pytest.raises(ValueError):
        _ = pl.DataFrame(np.random.randn(2, 2, 2))

    # Wrong orient value
    with pytest.raises(ValueError):
        df = pl.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6]]),
            orient="wrong",  # type: ignore[arg-type]
        )

    # Dimensions mismatch
    with pytest.raises(ValueError):
        _ = pl.DataFrame(np.array([1, 2, 3]), schema=[])
    with pytest.raises(ValueError):
        _ = pl.DataFrame(np.array([[1, 2], [3, 4]]), schema=["a"])

    # NumPy not available
    monkeypatch.setattr(
        pl.internals.dataframe.frame, "_check_for_numpy", lambda x: False
    )
    with pytest.raises(ValueError):
        pl.DataFrame(np.array([1, 2, 3]), schema=["a"])

    # 2D numpy arrays
    df = pl.DataFrame({"a": np.arange(5, dtype=np.int64).reshape(1, -1)})
    assert df.dtypes == [pl.List(pl.Int64)]
    assert df.shape == (1, 1)

    df = pl.DataFrame({"a": np.arange(10, dtype=np.int64).reshape(2, -1)})
    assert df.dtypes == [pl.List(pl.Int64)]
    assert df.shape == (2, 1)
    assert df.rows() == [([0, 1, 2, 3, 4],), ([5, 6, 7, 8, 9],)]

    # numpy arrays containing NaN
    df0 = pl.DataFrame(
        data={"x": [1.0, 2.5, float("nan")], "y": [4.0, float("nan"), 6.5]},
    )
    df1 = pl.DataFrame(
        data={"x": np.array([1.0, 2.5, np.nan]), "y": np.array([4.0, np.nan, 6.5])},
    )
    df2 = pl.DataFrame(
        data={"x": np.array([1.0, 2.5, np.nan]), "y": np.array([4.0, np.nan, 6.5])},
        nan_to_null=True,
    )
    assert_frame_equal(df0, df1, nans_compare_equal=True)
    assert df2.rows() == [(1.0, 4.0), (2.5, None), (None, 6.5)]


def test_init_arrow() -> None:
    # Handle unnamed column
    df = pl.DataFrame(pa.table({"a": [1, 2], None: [3, 4]}))
    expected = pl.DataFrame({"a": [1, 2], "None": [3, 4]})
    assert_frame_equal(df, expected)

    # Rename columns
    df = pl.DataFrame(pa.table({"a": [1, 2], "b": [3, 4]}), schema=["c", "d"])
    expected = pl.DataFrame({"c": [1, 2], "d": [3, 4]})
    assert_frame_equal(df, expected)

    df = pl.DataFrame(
        pa.table({"a": [1, 2], None: [3, 4]}),
        schema=[("c", pl.Int32), ("d", pl.Float32)],
    )
    assert df.schema == {"c": pl.Int32, "d": pl.Float32}
    assert df.rows() == [(1, 3.0), (2, 4.0)]

    # Bad columns argument
    with pytest.raises(ValueError):
        pl.DataFrame(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), schema=["c", "d", "e"])


def test_init_series() -> None:
    # List of Series
    df = pl.DataFrame([pl.Series("a", [1, 2, 3]), pl.Series("b", [4, 5, 6])])
    expected = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert_frame_equal(df, expected)

    # Tuple of Series
    df = pl.DataFrame((pl.Series("a", (1, 2, 3)), pl.Series("b", (4, 5, 6))))
    assert_frame_equal(df, expected)

    df = pl.DataFrame(
        (pl.Series("a", (1, 2, 3)), pl.Series("b", (4, 5, 6))),
        schema=[("x", pl.Float64), ("y", pl.Float64)],
    )
    assert df.schema == {"x": pl.Float64, "y": pl.Float64}
    assert df.rows() == [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]

    # List of unnamed Series
    df = pl.DataFrame([pl.Series([1, 2, 3]), pl.Series([4, 5, 6])])
    expected = pl.DataFrame(
        [pl.Series("column_0", [1, 2, 3]), pl.Series("column_1", [4, 5, 6])]
    )
    assert_frame_equal(df, expected)

    df = pl.DataFrame([pl.Series([0.0]), pl.Series([1.0])])
    assert df.schema == {"column_0": pl.Float64, "column_1": pl.Float64}
    assert df.rows() == [(0.0, 1.0)]

    df = pl.DataFrame(
        [pl.Series([None]), pl.Series([1.0])],
        schema=[("x", pl.Date), ("y", pl.Boolean)],
    )
    assert df.schema == {"x": pl.Date, "y": pl.Boolean}
    assert df.rows() == [(None, True)]

    # Single Series
    df = pl.DataFrame(pl.Series("a", [1, 2, 3]))
    expected = pl.DataFrame({"a": [1, 2, 3]})
    assert df.schema == {"a": pl.Int64}
    assert_frame_equal(df, expected)

    df = pl.DataFrame(pl.Series("a", [1, 2, 3]), schema=[("a", pl.UInt32)])
    assert df.rows() == [(1,), (2,), (3,)]
    assert df.schema == {"a": pl.UInt32}

    # nested list, with/without explicit dtype
    s1 = pl.Series([[[2, 2]]])
    assert s1.dtype == pl.List(pl.List(pl.Int64))

    s2 = pl.Series([[[2, 2]]], dtype=pl.List(pl.List(pl.UInt8)))
    assert s2.dtype == pl.List(pl.List(pl.UInt8))

    s3 = pl.Series(dtype=pl.List(pl.List(pl.UInt8)))
    assert s3.dtype == pl.List(pl.List(pl.UInt8))

    # numpy data containing NaN values
    s0 = pl.Series("n", [1.0, 2.5, float("nan")])
    s1 = pl.Series("n", np.array([1.0, 2.5, float("nan")]))
    s2 = pl.Series("n", np.array([1.0, 2.5, float("nan")]), nan_to_null=True)

    assert_series_equal(s0, s1, nans_compare_equal=True)
    assert s2.to_list() == [1.0, 2.5, None]


def test_init_seq_of_seq() -> None:
    # List of lists
    df = pl.DataFrame([[1, 2, 3], [4, 5, 6]], schema=["a", "b", "c"])
    expected = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    assert_frame_equal(df, expected)

    df = pl.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        schema=[("a", pl.Int8), ("b", pl.Int16), ("c", pl.Int32)],
    )
    assert df.schema == {"a": pl.Int8, "b": pl.Int16, "c": pl.Int32}
    assert df.rows() == [(1, 2, 3), (4, 5, 6)]

    # Tuple of tuples, default to column orientation
    df = pl.DataFrame(((1, 2, 3), (4, 5, 6)))
    expected = pl.DataFrame({"column_0": [1, 2, 3], "column_1": [4, 5, 6]})
    assert_frame_equal(df, expected)

    # Row orientation
    df = pl.DataFrame(((1, 2), (3, 4)), schema=("a", "b"), orient="row")
    expected = pl.DataFrame({"a": [1, 3], "b": [2, 4]})
    assert_frame_equal(df, expected)

    df = pl.DataFrame(
        ((1, 2), (3, 4)), schema=(("a", pl.Float32), ("b", pl.Float32)), orient="row"
    )
    assert df.schema == {"a": pl.Float32, "b": pl.Float32}
    assert df.rows() == [(1.0, 2.0), (3.0, 4.0)]

    # Wrong orient value
    with pytest.raises(ValueError):
        df = pl.DataFrame(((1, 2), (3, 4)), orient="wrong")  # type: ignore[arg-type]


def test_init_1d_sequence() -> None:
    # Empty list
    df = pl.DataFrame([])
    assert_frame_equal(df, pl.DataFrame())

    # List/array of strings
    data = ["a", "b", "c"]
    for a in (data, np.array(data)):
        df = pl.DataFrame(a, schema=["s"])
        expected = pl.DataFrame({"s": data})
        assert_frame_equal(df, expected)

    df = pl.DataFrame([None, True, False], schema=[("xx", pl.Int8)])
    assert df.schema == {"xx": pl.Int8}
    assert df.rows() == [(None,), (1,), (0,)]

    # String sequence
    assert pl.DataFrame("abc", schema=["s"]).to_dict(False) == {"s": ["a", "b", "c"]}


def test_init_pandas(monkeypatch: Any) -> None:
    pandas_df = pd.DataFrame([[1, 2], [3, 4]], columns=[1, 2])

    # integer column names
    df = pl.DataFrame(pandas_df)
    expected = pl.DataFrame({"1": [1, 3], "2": [2, 4]})
    assert_frame_equal(df, expected)
    assert df.schema == {"1": pl.Int64, "2": pl.Int64}

    # override column names, types
    df = pl.DataFrame(pandas_df, schema=[("x", pl.Float64), ("y", pl.Float64)])
    assert df.schema == {"x": pl.Float64, "y": pl.Float64}
    assert df.rows() == [(1.0, 2.0), (3.0, 4.0)]

    # subclassed pandas object, with/without data & overrides
    class XSeries(pd.Series):
        @property
        def _constructor(self) -> type:
            return XSeries

    df = pl.DataFrame(
        data=[
            XSeries(name="x", data=[], dtype=np.dtype("<M8[ns]")),
            XSeries(name="y", data=[], dtype=np.dtype("f8")),
            XSeries(name="z", data=[], dtype=np.dtype("?")),
        ],
    )
    assert df.schema == {"x": pl.Datetime("ns"), "y": pl.Float64, "z": pl.Boolean}
    assert df.rows() == []

    df = pl.DataFrame(
        data=[
            XSeries(
                name="x",
                data=[datetime(2022, 10, 31, 10, 30, 45, 123456)],
                dtype=np.dtype("<M8[ns]"),
            )
        ],
        schema={"colx": pl.Datetime("us")},
    )
    assert df.schema == {"colx": pl.Datetime("us")}
    assert df.rows() == [(datetime(2022, 10, 31, 10, 30, 45, 123456),)]

    # pandas is not available
    monkeypatch.setattr(
        pl.internals.dataframe.frame, "_check_for_pandas", lambda x: False
    )
    with pytest.raises(ValueError):
        pl.DataFrame(pandas_df)


def test_init_errors() -> None:
    # Length mismatch
    with pytest.raises(pl.ShapeError):
        pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0, 4.0]})

    # Columns don't match data dimensions
    with pytest.raises(pl.ShapeError):
        pl.DataFrame([[1, 2], [3, 4]], schema=["a", "b", "c"])

    # Unmatched input
    with pytest.raises(ValueError):
        pl.DataFrame(0)


def test_init_records() -> None:
    dicts = [
        {"a": 1, "b": 2},
        {"b": 1, "a": 2},
        {"a": 1, "b": 2},
    ]
    df = pl.DataFrame(dicts)
    expected = pl.DataFrame({"a": [1, 2, 1], "b": [2, 1, 2]})
    assert_frame_equal(df, expected)
    assert df.to_dicts() == dicts

    df_cd = pl.DataFrame(dicts, schema=["c", "d"])
    expected = pl.DataFrame({"c": [1, 2, 1], "d": [2, 1, 2]})
    assert_frame_equal(df_cd, expected)


def test_init_records_schema_order() -> None:
    cols: list[str] = ["a", "b", "c", "d"]
    data: list[dict[str, int]] = [
        {"c": 3, "b": 2, "a": 1},
        {"b": 2, "d": 4},
        {},
        {"a": 1, "b": 2, "c": 3},
        {"d": 4, "b": 2, "a": 1},
        {"c": 3, "b": 2},
    ]
    lookup = {"a": 1, "b": 2, "c": 3, "d": 4, "e": None}

    # ensure field values are loaded according to the declared schema order
    for _ in range(8):
        shuffle(data)
        shuffle(cols)

        df = pl.from_dicts(dicts=data, schema=cols)
        for col in df.columns:
            assert all(value in (None, lookup[col]) for value in df[col].to_list())

    # have schema override inferred types, omit some columns, add a new one
    schema = {"a": pl.Int8, "c": pl.Int16, "e": pl.Int32}
    df = pl.from_dicts(dicts=data, schema=schema)

    assert df.schema == schema
    for col in df.columns:
        assert all(value in (None, lookup[col]) for value in df[col].to_list())


def test_init_only_columns() -> None:
    df = pl.DataFrame(schema=["a", "b", "c"])
    expected = pl.DataFrame({"a": [], "b": [], "c": []})
    assert_frame_equal(df, expected)

    # Validate construction with various flavours of no/empty data
    no_data: Any
    for no_data in (None, {}, []):
        df = pl.DataFrame(
            data=no_data,
            schema=[
                ("a", pl.Date),
                ("b", pl.UInt64),
                ("c", pl.Int8),
                ("d", pl.List(pl.UInt8)),
            ],
        )
        expected = pl.DataFrame({"a": [], "b": [], "c": []}).with_columns(
            [
                pl.col("a").cast(pl.Date),
                pl.col("b").cast(pl.UInt64),
                pl.col("c").cast(pl.Int8),
            ]
        )
        expected.insert_at_idx(3, pl.Series("d", [], pl.List(pl.UInt8)))

        assert df.shape == (0, 4)
        assert_frame_equal(df, expected)
        assert df.dtypes == [pl.Date, pl.UInt64, pl.Int8, pl.List]
        assert df.schema["d"].inner == pl.UInt8  # type: ignore[union-attr]

        dfe = df.cleared()
        assert len(dfe) == 0
        assert df.schema == dfe.schema
        assert dfe.shape == df.shape


def test_from_dicts_list_without_dtype() -> None:
    assert pl.from_dicts(
        [{"id": 1, "hint": ["some_text_here"]}, {"id": 2, "hint": [None]}]
    ).to_dict(False) == {"id": [1, 2], "hint": [["some_text_here"], [None]]}


def test_from_dicts_list_struct_without_inner_dtype() -> None:
    assert pl.DataFrame(
        {
            "users": [
                [{"category": "A"}, {"category": "B"}],
                [{"category": None}, {"category": None}],
            ],
            "days_of_week": [1, 2],
        }
    ).to_dict(False) == {
        "users": [
            [{"category": "A"}, {"category": "B"}],
            [{"category": None}, {"category": None}],
        ],
        "days_of_week": [1, 2],
    }

    # 5611
    df = pl.from_dicts(
        [
            {"a": []},
            {"a": [{"b": 1}]},
        ]
    )
    assert df.to_dict(False) == {"a": [[], [{"b": 1}]]}


def test_upcast_primitive_and_strings() -> None:
    assert pl.Series([1, 1.0, 1]).dtype == pl.Float64
    assert pl.Series([1, 1, "1.0"]).dtype == pl.Utf8
    assert pl.Series([1, 1.0, "1.0"]).dtype == pl.Utf8
    assert pl.Series([True, 1]).dtype == pl.Int64
    assert pl.Series([True, 1.0]).dtype == pl.Float64
    assert pl.Series([True, "1.0"]).dtype == pl.Utf8
    assert pl.from_dict({"a": [1, 2.1, 3], "b": [4, 5, 6.4]}).dtypes == [
        pl.Float64,
        pl.Float64,
    ]


def test_u64_lit_5031() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3]}).with_columns(pl.col("foo").cast(pl.UInt64))
    assert df.filter(pl.col("foo") < (1 << 64) - 20).shape == (3, 1)
    assert df["foo"].to_list() == [1, 2, 3]


def test_from_dicts_missing_columns() -> None:
    data = [
        {"a": 1},
        {"b": 2},
    ]

    assert pl.from_dicts(data).to_dict(False) == {"a": [1, None], "b": [None, 2]}


@typing.no_type_check
def test_from_rows_dtype() -> None:
    # 50 is the default inference length
    # 5182
    df = pl.DataFrame(
        data=[(None, None)] * 50 + [("1.23", None)],
        schema=[("foo", pl.Utf8), ("bar", pl.Utf8)],
        orient="row",
    )
    assert df.dtypes == [pl.Utf8, pl.Utf8]
    assert df.null_count().row(0) == (50, 51)

    type1 = [{"c1": 206, "c2": "type1", "c3": {"x1": "abcd", "x2": "jkl;"}}]
    type2 = [
        {"c1": 208, "c2": "type2", "c3": {"a1": "abcd", "a2": "jkl;", "a3": "qwerty"}}
    ]

    df = pl.DataFrame(
        data=type1 * 50 + type2,
        schema=[("c1", pl.Int32), ("c2", pl.Object), ("c3", pl.Object)],
    )
    assert df.dtypes == [pl.Int32, pl.Object, pl.Object]

    # 50 is the default inference length
    # 5266
    type1 = [{"c1": 206, "c2": "type1", "c3": {"x1": "abcd", "x2": "jkl;"}}]
    type2 = [
        {"c1": 208, "c2": "type2", "c3": {"a1": "abcd", "a2": "jkl;", "a3": "qwerty"}}
    ]

    df = pl.DataFrame(
        data=type1 * 50 + type2,
        schema=[("c1", pl.Int32), ("c2", pl.Object), ("c3", pl.Object)],
    )
    assert df.dtypes == [pl.Int32, pl.Object, pl.Object]
    assert df.null_count().row(0) == (0, 0, 0)


def test_from_dicts_schema() -> None:
    data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]

    # let polars infer the dtypes, but inform it about a 3rd column.
    for schema, overrides in (
        ({"a": pl.Unknown, "b": pl.Unknown, "c": pl.Int32}, None),
        ({"a": None, "b": None, "c": None}, {"c": pl.Int32}),
        (["a", "b", ("c", pl.Int32)], None),
    ):
        df = pl.from_dicts(
            data,
            schema=schema,  # type: ignore[arg-type]
            schema_overrides=overrides,
        )
        assert df.dtypes == [pl.Int64, pl.Int64, pl.Int32]
        assert df.to_dict(False) == {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [None, None, None],
        }


def test_nested_read_dict_4143() -> None:
    assert pl.from_dicts(
        [
            {
                "id": 1,
                "hint": [
                    {"some_text_here": "text", "list_": [1, 2, 4]},
                    {"some_text_here": "text", "list_": [1, 2, 4]},
                ],
            },
            {
                "id": 2,
                "hint": [
                    {"some_text_here": None, "list_": [1]},
                    {"some_text_here": None, "list_": [2]},
                ],
            },
        ]
    ).to_dict(False) == {
        "hint": [
            [
                {"some_text_here": "text", "list_": [1, 2, 4]},
                {"some_text_here": "text", "list_": [1, 2, 4]},
            ],
            [
                {"some_text_here": None, "list_": [1]},
                {"some_text_here": None, "list_": [2]},
            ],
        ],
        "id": [1, 2],
    }

    out = pl.from_dicts(
        [
            {
                "id": 1,
                "hint": [
                    {"some_text_here": "text", "list_": [1, 2, 4]},
                    {"some_text_here": "text", "list_": [1, 2, 4]},
                ],
            },
            {
                "id": 2,
                "hint": [
                    {"some_text_here": "text", "list_": []},
                    {"some_text_here": "text", "list_": []},
                ],
            },
        ]
    )

    assert out.dtypes == [
        pl.Int64,
        pl.List(pl.Struct({"some_text_here": pl.Utf8, "list_": pl.List(pl.Int64)})),
    ]
    assert out.to_dict(False) == {
        "id": [1, 2],
        "hint": [
            [
                {"some_text_here": "text", "list_": [1, 2, 4]},
                {"some_text_here": "text", "list_": [1, 2, 4]},
            ],
            [
                {"some_text_here": "text", "list_": []},
                {"some_text_here": "text", "list_": []},
            ],
        ],
    }


@typing.no_type_check
def test_from_records_nullable_structs() -> None:
    records = [
        {"id": 1, "items": [{"item_id": 100, "description": None}]},
        {"id": 1, "items": [{"item_id": 100, "description": "hi"}]},
    ]

    schema = [
        ("id", pl.UInt16),
        (
            "items",
            pl.List(
                pl.Struct(
                    [pl.Field("item_id", pl.UInt32), pl.Field("description", pl.Utf8)]
                )
            ),
        ),
    ]

    for s in [schema, None]:
        assert pl.DataFrame(records, schema=s, orient="row").to_dict(False) == {
            "id": [1, 1],
            "items": [
                [{"item_id": 100, "description": None}],
                [{"item_id": 100, "description": "hi"}],
            ],
        }

    # check initialisation without any records
    df = pl.DataFrame(schema=schema)
    dict_schema = dict(schema)

    assert df.to_dict(False) == {"id": [], "items": []}
    assert df.schema == dict_schema

    s = pl.Series("items", dtype=dict_schema["items"])
    assert s.to_frame().to_dict(False) == {"items": []}
    assert s.dtype == dict_schema["items"]
    assert s.to_list() == []


def test_from_categorical_in_struct_defined_by_schema() -> None:
    df = pl.DataFrame(
        {
            "a": [
                {"value": "foo", "counts": 1},
                {"value": "bar", "counts": 2},
            ]
        },
        schema={"a": pl.Struct({"value": pl.Categorical, "counts": pl.UInt32})},
    )
    out = df.unnest("a")
    assert out.schema == {"value": pl.Categorical, "counts": pl.UInt32}
    assert out.to_dict(False) == {"value": ["foo", "bar"], "counts": [1, 2]}
