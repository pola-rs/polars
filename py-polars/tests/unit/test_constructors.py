import typing
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

import polars as pl


def test_init_dict() -> None:
    # Empty dictionary
    df = pl.DataFrame({})
    assert df.shape == (0, 0)

    # Empty dictionary/values
    df = pl.DataFrame({"a": [], "b": []})
    assert df.shape == (0, 2)
    assert df.schema == {"a": pl.Float32, "b": pl.Float32}

    for df in (
        pl.DataFrame({}, columns={"a": pl.Date, "b": pl.Utf8}),
        pl.DataFrame({"a": [], "b": []}, columns={"a": pl.Date, "b": pl.Utf8}),
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
        columns=[("a", pl.Int8), ("b", pl.Float32)],
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
            columns=coldefs,
        )
        assert df.schema == {"dt": pl.Date, "dtm": pl.Datetime}
        assert df.rows() == list(zip(py_dates, py_datetimes))

    # Overriding dict column names/types
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}, columns=["c", "d"])
    assert df.columns == ["c", "d"]

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]},
        columns=["c", ("d", pl.Int8)],
    )  # partial type info (allowed, but mypy doesn't like it ;p)
    assert df.schema == {"c": pl.Int64, "d": pl.Int8}

    df = pl.DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6]}, columns=[("c", pl.Int8), ("d", pl.Int16)]
    )
    assert df.schema == {"c": pl.Int8, "d": pl.Int16}

    dfe = df.cleared()
    assert df.schema == dfe.schema
    assert len(dfe) == 0


def test_init_ndarray(monkeypatch: Any) -> None:
    # Empty array
    df = pl.DataFrame(np.array([]))
    assert df.frame_equal(pl.DataFrame())

    # 1D array
    df = pl.DataFrame(np.array([1, 2, 3]), columns=["a"])
    truth = pl.DataFrame({"a": [1, 2, 3]})
    assert df.frame_equal(truth)

    df = pl.DataFrame(np.array([1, 2, 3]), columns=[("a", pl.Int32)])
    truth = pl.DataFrame({"a": [1, 2, 3]}).with_column(pl.col("a").cast(pl.Int32))
    assert df.frame_equal(truth)

    # 2D array - default to column orientation
    df = pl.DataFrame(np.array([[1, 2], [3, 4]]), orient="col")
    truth = pl.DataFrame({"column_0": [1, 2], "column_1": [3, 4]})
    assert df.frame_equal(truth)

    df = pl.DataFrame([[1, 2.0, "a"], [None, None, None]], orient="row")
    truth = pl.DataFrame(
        {"column_0": [1, None], "column_1": [2.0, None], "column_2": ["a", None]}
    )
    assert df.frame_equal(truth)

    df = pl.DataFrame(
        data=[[1, 2.0, "a"], [None, None, None]],
        columns=[("x", pl.Boolean), ("y", pl.Int32), "z"],
        orient="row",
    )
    assert df.rows() == [(True, 2, "a"), (None, None, None)]
    assert df.schema == {"x": pl.Boolean, "y": pl.Int32, "z": pl.Utf8}

    # 2D array - default to column orientation
    df = pl.DataFrame(np.array([[1, 2], [3, 4]]))
    truth = pl.DataFrame({"column_0": [1, 3], "column_1": [2, 4]})
    assert df.frame_equal(truth)

    # no orientation is numpy convention
    df = pl.DataFrame(np.ones((3, 1)))
    assert df.shape == (3, 1)

    # 2D array - row orientation inferred
    df = pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["a", "b", "c"])
    truth = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    assert df.frame_equal(truth)

    # 2D array - column orientation inferred
    df = pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["a", "b"])
    truth = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.frame_equal(truth)

    # 2D array - orientation conflicts with columns
    with pytest.raises(ValueError):
        pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), columns=["a", "b"], orient="row")
    with pytest.raises(ValueError):
        pl.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6]]),
            columns=[("a", pl.UInt32), ("b", pl.UInt32)],
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
        _ = pl.DataFrame(np.array([1, 2, 3]), columns=[])
    with pytest.raises(ValueError):
        _ = pl.DataFrame(np.array([[1, 2], [3, 4]]), columns=["a"])

    # NumPy not available
    monkeypatch.setattr(
        pl.internals.dataframe.frame, "_check_for_numpy", lambda x: False
    )
    with pytest.raises(ValueError):
        pl.DataFrame(np.array([1, 2, 3]), columns=["a"])

    # 2D numpy arrays
    df = pl.DataFrame({"a": np.arange(5, dtype=np.int64).reshape(1, -1)})
    assert df.dtypes == [pl.List(pl.Int64)]
    assert df.shape == (1, 1)
    df = pl.DataFrame({"a": np.arange(10, dtype=np.int64).reshape(2, -1)})
    assert df.dtypes == [pl.List(pl.Int64)]
    assert df.shape == (2, 1)
    assert df.rows() == [([0, 1, 2, 3, 4],), ([5, 6, 7, 8, 9],)]


def test_init_arrow() -> None:
    # Handle unnamed column
    df = pl.DataFrame(pa.table({"a": [1, 2], None: [3, 4]}))
    truth = pl.DataFrame({"a": [1, 2], "None": [3, 4]})
    assert df.frame_equal(truth)

    # Rename columns
    df = pl.DataFrame(pa.table({"a": [1, 2], "b": [3, 4]}), columns=["c", "d"])
    truth = pl.DataFrame({"c": [1, 2], "d": [3, 4]})
    assert df.frame_equal(truth)

    df = pl.DataFrame(
        pa.table({"a": [1, 2], None: [3, 4]}),
        columns=[("c", pl.Int32), ("d", pl.Float32)],
    )
    assert df.schema == {"c": pl.Int32, "d": pl.Float32}
    assert df.rows() == [(1, 3.0), (2, 4.0)]

    # Bad columns argument
    with pytest.raises(ValueError):
        pl.DataFrame(
            pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}), columns=["c", "d", "e"]
        )


def test_init_series() -> None:
    # List of Series
    df = pl.DataFrame([pl.Series("a", [1, 2, 3]), pl.Series("b", [4, 5, 6])])
    truth = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    assert df.frame_equal(truth)

    # Tuple of Series
    df = pl.DataFrame((pl.Series("a", (1, 2, 3)), pl.Series("b", (4, 5, 6))))
    assert df.frame_equal(truth)

    df = pl.DataFrame(
        (pl.Series("a", (1, 2, 3)), pl.Series("b", (4, 5, 6))),
        columns=[("x", pl.Float64), ("y", pl.Float64)],
    )
    assert df.schema == {"x": pl.Float64, "y": pl.Float64}
    assert df.rows() == [(1.0, 4.0), (2.0, 5.0), (3.0, 6.0)]

    # List of unnamed Series
    df = pl.DataFrame([pl.Series([1, 2, 3]), pl.Series([4, 5, 6])])
    truth = pl.DataFrame(
        [pl.Series("column_0", [1, 2, 3]), pl.Series("column_1", [4, 5, 6])]
    )
    assert df.frame_equal(truth)

    df = pl.DataFrame([pl.Series([0.0]), pl.Series([1.0])])
    assert df.schema == {"column_0": pl.Float64, "column_1": pl.Float64}
    assert df.rows() == [(0.0, 1.0)]

    df = pl.DataFrame(
        [pl.Series([None]), pl.Series([1.0])],
        columns=[("x", pl.Date), ("y", pl.Boolean)],
    )
    assert df.schema == {"x": pl.Date, "y": pl.Boolean}
    assert df.rows() == [(None, True)]

    # Single Series
    df = pl.DataFrame(pl.Series("a", [1, 2, 3]))
    truth = pl.DataFrame({"a": [1, 2, 3]})
    assert df.schema == {"a": pl.Int64}
    assert df.frame_equal(truth)

    df = pl.DataFrame(pl.Series("a", [1, 2, 3]), columns=[("a", pl.UInt32)])
    assert df.rows() == [(1,), (2,), (3,)]
    assert df.schema == {"a": pl.UInt32}

    # nested list
    assert pl.Series([[[2, 2]]]).dtype == pl.List(pl.List(pl.Int64))


def test_init_seq_of_seq() -> None:
    # List of lists
    df = pl.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
    truth = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    assert df.frame_equal(truth)

    df = pl.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        columns=[("a", pl.Int8), ("b", pl.Int16), ("c", pl.Int32)],
    )
    assert df.schema == {"a": pl.Int8, "b": pl.Int16, "c": pl.Int32}
    assert df.rows() == [(1, 2, 3), (4, 5, 6)]

    # Tuple of tuples, default to column orientation
    df = pl.DataFrame(((1, 2, 3), (4, 5, 6)))
    truth = pl.DataFrame({"column_0": [1, 2, 3], "column_1": [4, 5, 6]})
    assert df.frame_equal(truth)

    # Row orientation
    df = pl.DataFrame(((1, 2), (3, 4)), columns=("a", "b"), orient="row")
    truth = pl.DataFrame({"a": [1, 3], "b": [2, 4]})
    assert df.frame_equal(truth)

    df = pl.DataFrame(
        ((1, 2), (3, 4)), columns=(("a", pl.Float32), ("b", pl.Float32)), orient="row"
    )
    assert df.schema == {"a": pl.Float32, "b": pl.Float32}
    assert df.rows() == [(1.0, 2.0), (3.0, 4.0)]

    # Wrong orient value
    with pytest.raises(ValueError):
        df = pl.DataFrame(((1, 2), (3, 4)), orient="wrong")  # type: ignore[arg-type]


def test_init_1d_sequence() -> None:
    # Empty list
    df = pl.DataFrame([])
    assert df.frame_equal(pl.DataFrame())

    # List of strings
    df = pl.DataFrame(["a", "b", "c"], columns=["hi"])
    truth = pl.DataFrame({"hi": ["a", "b", "c"]})
    assert df.frame_equal(truth)

    df = pl.DataFrame([None, True, False], columns=[("xx", pl.Int8)])
    assert df.schema == {"xx": pl.Int8}
    assert df.rows() == [(None,), (1,), (0,)]

    # String sequence
    assert pl.DataFrame("abc", columns=["s"]).to_dict(False) == {"s": ["a", "b", "c"]}


def test_init_pandas(monkeypatch: Any) -> None:
    pandas_df = pd.DataFrame([[1, 2], [3, 4]], columns=[1, 2])

    # integer column names
    df = pl.DataFrame(pandas_df)
    truth = pl.DataFrame({"1": [1, 3], "2": [2, 4]})
    assert df.frame_equal(truth)
    assert df.schema == {"1": pl.Int64, "2": pl.Int64}

    # override column names, types
    df = pl.DataFrame(pandas_df, columns=[("x", pl.Float64), ("y", pl.Float64)])
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
        columns={"colx": pl.Datetime("us")},
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
        pl.DataFrame([[1, 2], [3, 4]], columns=["a", "b", "c"])

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
    assert df.frame_equal(expected)
    assert df.to_dicts() == dicts

    df_cd = pl.DataFrame(dicts, columns=["c", "d"])
    expected = pl.DataFrame({"c": [1, 2, 1], "d": [2, 1, 2]})
    assert df_cd.frame_equal(expected)

    df_xy = pl.DataFrame(dicts, columns=[("x", pl.UInt32), ("y", pl.UInt32)])
    expected = pl.DataFrame({"x": [1, 2, 1], "y": [2, 1, 2]}).with_columns(
        [pl.col("x").cast(pl.UInt32), pl.col("y").cast(pl.UInt32)]
    )
    assert df_xy.frame_equal(expected)
    assert df_xy.schema == {"x": pl.UInt32, "y": pl.UInt32}
    assert df_xy.rows() == [(1, 2), (2, 1), (1, 2)]


def test_init_only_columns() -> None:
    df = pl.DataFrame(columns=["a", "b", "c"])
    truth = pl.DataFrame({"a": [], "b": [], "c": []})
    assert df.shape == (0, 3)
    assert df.frame_equal(truth, null_equal=True)
    assert df.dtypes == [pl.Float32, pl.Float32, pl.Float32]

    # Validate construction with various flavours of no/empty data
    no_data: Any
    for no_data in (None, {}, []):
        df = pl.DataFrame(
            data=no_data,
            columns=[
                ("a", pl.Date),
                ("b", pl.UInt64),
                ("c", pl.Int8),
                ("d", pl.List(pl.UInt8)),
            ],
        )
        truth = pl.DataFrame({"a": [], "b": [], "c": []}).with_columns(
            [
                pl.col("a").cast(pl.Date),
                pl.col("b").cast(pl.UInt64),
                pl.col("c").cast(pl.Int8),
            ]
        )
        truth.insert_at_idx(3, pl.Series("d", [], pl.List(pl.UInt8)))

        assert df.shape == (0, 4)
        assert df.frame_equal(truth, null_equal=True)
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
    df = pl.DataFrame({"foo": [1, 2, 3]}).with_column(pl.col("foo").cast(pl.UInt64))
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
        columns=[("foo", pl.Utf8), ("bar", pl.Utf8)],
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
        columns=[("c1", pl.Int32), ("c2", pl.Object), ("c3", pl.Object)],
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
        columns=[("c1", pl.Int32), ("c2", pl.Object), ("c3", pl.Object)],
    )
    assert df.dtypes == [pl.Int32, pl.Object, pl.Object]
    assert df.null_count().row(0) == (0, 0, 0)


def test_from_dicts_schema() -> None:
    data = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]

    # let polars infer the dtypes
    # but inform about a 3rd column
    df = pl.from_dicts(
        data, schema_overrides={"a": pl.Unknown, "b": pl.Unknown, "c": pl.Int32}
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

    for columns in [schema, None]:
        assert pl.DataFrame(records, orient="row", columns=columns).to_dict(False) == {
            "id": [1, 1],
            "items": [
                [{"item_id": 100, "description": None}],
                [{"item_id": 100, "description": "hi"}],
            ],
        }
