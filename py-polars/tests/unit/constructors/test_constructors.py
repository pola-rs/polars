from __future__ import annotations

from collections import OrderedDict, namedtuple
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from random import shuffle
from typing import TYPE_CHECKING, Any, List, Literal, NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from packaging.version import parse as parse_version
from pydantic import BaseModel, Field, TypeAdapter

import polars as pl
from polars._utils.construction.utils import try_get_type_hints
from polars.datatypes import numpy_char_code_to_dtype
from polars.dependencies import dataclasses, pydantic
from polars.exceptions import ShapeError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from zoneinfo import ZoneInfo

    from polars._typing import PolarsDataType

else:
    from polars._utils.convert import string_to_zoneinfo as ZoneInfo


# -----------------------------------------------------------------------------------
# nested dataclasses, models, namedtuple classes (can't be defined inside test func)
# -----------------------------------------------------------------------------------
@dataclasses.dataclass
class _TestBazDC:
    d: datetime
    e: float
    f: str


@dataclasses.dataclass
class _TestBarDC:
    a: str
    b: int
    c: _TestBazDC


@dataclasses.dataclass
class _TestFooDC:
    x: int
    y: _TestBarDC


class _TestBazPD(pydantic.BaseModel):
    d: datetime
    e: float
    f: str


class _TestBarPD(pydantic.BaseModel):
    a: str
    b: int
    c: _TestBazPD


class _TestFooPD(pydantic.BaseModel):
    x: int
    y: _TestBarPD


class _TestBazNT(NamedTuple):
    d: datetime
    e: float
    f: str


class _TestBarNT(NamedTuple):
    a: str
    b: int
    c: _TestBazNT


class _TestFooNT(NamedTuple):
    x: int
    y: _TestBarNT


# --------------------------------------------------------------------------------


def test_init_dict() -> None:
    # Empty dictionary
    df = pl.DataFrame({})
    assert df.shape == (0, 0)

    # Empty dictionary/values
    df = pl.DataFrame({"a": [], "b": []})
    assert df.shape == (0, 2)
    assert df.schema == {"a": pl.Null, "b": pl.Null}

    for df in (
        pl.DataFrame({}, schema={"a": pl.Date, "b": pl.String}),
        pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Date, "b": pl.String}),
    ):
        assert df.shape == (0, 2)
        assert df.schema == {"a": pl.Date, "b": pl.String}

    # List of empty list
    df = pl.DataFrame({"a": [[]], "b": [[]]})
    expected = {"a": pl.List(pl.Null), "b": pl.List(pl.Null)}
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

    # empty nested objects
    for empty_val in [None, "", {}, []]:  # type: ignore[var-annotated]
        test = [{"field": {"sub_field": empty_val, "sub_field_2": 2}}]
        df = pl.DataFrame(test, schema={"field": pl.Object})
        assert df["field"][0] == test[0]["field"]


def test_error_string_dtypes() -> None:
    with pytest.raises(TypeError, match="cannot parse input"):
        pl.DataFrame(
            data={"x": [1, 2], "y": [3, 4], "z": [5, 6]},
            schema={"x": "i16", "y": "i32", "z": "f32"},  # type: ignore[dict-item]
        )

    with pytest.raises(TypeError, match="cannot parse input"):
        pl.Series("n", [1, 2, 3], dtype="f32")  # type: ignore[arg-type]


def test_init_structured_objects() -> None:
    # validate init from dataclass, namedtuple, and pydantic model objects
    @dataclasses.dataclass
    class TradeDC:
        timestamp: datetime
        ticker: str
        price: Decimal
        size: int | None = None

    class TradePD(pydantic.BaseModel):
        timestamp: datetime
        ticker: str
        price: Decimal
        size: int

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
    columns = ["timestamp", "ticker", "price", "size"]

    for TradeClass in (TradeDC, TradeNT, TradePD):
        trades = [TradeClass(**dict(zip(columns, values))) for values in raw_data]  # type: ignore[arg-type]

        for DF in (pl.DataFrame, pl.from_records):
            df = DF(data=trades)
            assert df.schema == {
                "timestamp": pl.Datetime("us"),
                "ticker": pl.String,
                "price": pl.Decimal(scale=1),
                "size": pl.Int64,
            }
            assert df.rows() == raw_data

            # partial dtypes override
            df = DF(
                data=trades,
                schema_overrides={"timestamp": pl.Datetime("ms"), "size": pl.Int32},
            )
            assert df.schema == {
                "timestamp": pl.Datetime("ms"),
                "ticker": pl.String,
                "price": pl.Decimal(scale=1),
                "size": pl.Int32,
            }

        # in conjunction with full 'columns' override (rename/downcast)
        df = pl.DataFrame(
            data=trades,
            schema=[
                ("ts", pl.Datetime("ms")),
                ("tk", pl.Categorical),
                ("pc", pl.Decimal(scale=1)),
                ("sz", pl.UInt16),
            ],
        )
        assert df.schema == {
            "ts": pl.Datetime("ms"),
            "tk": pl.Categorical,
            "pc": pl.Decimal(scale=1),
            "sz": pl.UInt16,
        }
        assert df.rows() == raw_data

        # cover a miscellaneous edge-case when detecting the annotations
        assert try_get_type_hints(obj=type(None)) == {}


def test_init_pydantic_2x() -> None:
    class PageView(BaseModel):
        user_id: str
        ts: datetime = Field(alias=["ts", "$date"])  # type: ignore[literal-required, arg-type]
        path: str = Field("?", alias=["url", "path"])  # type: ignore[literal-required, arg-type]
        referer: str = Field("?", alias="referer")
        event: Literal["leave", "enter"] = Field("enter")
        time_on_page: int = Field(0, serialization_alias="top")

    data_json = """
    [{
        "user_id": "x",
        "ts": {"$date": "2021-01-01T00:00:00.000Z"},
        "url": "/latest/foobar",
        "referer": "https://google.com",
        "event": "enter",
        "top": 123
    }]
    """
    adapter: TypeAdapter[Any] = TypeAdapter(List[PageView])
    models = adapter.validate_json(data_json)

    result = pl.DataFrame(models)

    expected = pl.DataFrame(
        {
            "user_id": ["x"],
            "ts": [datetime(2021, 1, 1, 0, 0)],
            "path": ["?"],
            "referer": ["https://google.com"],
            "event": ["enter"],
            "time_on_page": [0],
        }
    )
    assert_frame_equal(result, expected)


def test_init_structured_objects_unhashable() -> None:
    # cover an edge-case with namedtuple fields that aren't hashable

    class Test(NamedTuple):
        dt: datetime
        info: dict[str, int]

    test_data = [
        Test(datetime(2017, 1, 1), {"a": 1, "b": 2}),
        Test(datetime(2017, 1, 2), {"a": 2, "b": 2}),
    ]
    df = pl.DataFrame(test_data)
    # shape: (2, 2)
    # ┌─────────────────────┬───────────┐
    # │ dt                  ┆ info      │
    # │ ---                 ┆ ---       │
    # │ datetime[μs]        ┆ struct[2] │
    # ╞═════════════════════╪═══════════╡
    # │ 2017-01-01 00:00:00 ┆ {1,2}     │
    # │ 2017-01-02 00:00:00 ┆ {2,2}     │
    # └─────────────────────┴───────────┘
    assert df.schema == {
        "dt": pl.Datetime(time_unit="us", time_zone=None),
        "info": pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Int64)]),
    }
    assert df.rows() == test_data


@pytest.mark.parametrize(
    ("foo", "bar", "baz"),
    [
        (_TestFooDC, _TestBarDC, _TestBazDC),
        (_TestFooPD, _TestBarPD, _TestBazPD),
        (_TestFooNT, _TestBarNT, _TestBazNT),
    ],
)
def test_init_structured_objects_nested(foo: Any, bar: Any, baz: Any) -> None:
    data = [
        foo(
            x=100,
            y=bar(
                a="hello",
                b=800,
                c=baz(d=datetime(2023, 4, 12, 10, 30), e=-10.5, f="world"),
            ),
        )
    ]
    df = pl.DataFrame(data)
    # shape: (1, 2)
    # ┌─────┬───────────────────────────────────┐
    # │ x   ┆ y                                 │
    # │ --- ┆ ---                               │
    # │ i64 ┆ struct[3]                         │
    # ╞═════╪═══════════════════════════════════╡
    # │ 100 ┆ {"hello",800,{2023-04-12 10:30:0… │
    # └─────┴───────────────────────────────────┘

    assert df.schema == {
        "x": pl.Int64,
        "y": pl.Struct(
            [
                pl.Field("a", pl.String),
                pl.Field("b", pl.Int64),
                pl.Field(
                    "c",
                    pl.Struct(
                        [
                            pl.Field("d", pl.Datetime("us")),
                            pl.Field("e", pl.Float64),
                            pl.Field("f", pl.String),
                        ]
                    ),
                ),
            ]
        ),
    }
    assert df.row(0) == (
        100,
        {
            "a": "hello",
            "b": 800,
            "c": {
                "d": datetime(2023, 4, 12, 10, 30),
                "e": -10.5,
                "f": "world",
            },
        },
    )

    # validate nested schema override
    override_struct_schema: dict[str, PolarsDataType] = {
        "x": pl.Int16,
        "y": pl.Struct(
            [
                pl.Field("a", pl.String),
                pl.Field("b", pl.Int32),
                pl.Field(
                    name="c",
                    dtype=pl.Struct(
                        [
                            pl.Field("d", pl.Datetime("ms")),
                            pl.Field("e", pl.Float32),
                            pl.Field("f", pl.String),
                        ]
                    ),
                ),
            ]
        ),
    }
    for schema, schema_overrides in (
        (None, override_struct_schema),
        (override_struct_schema, None),
    ):
        df = (
            pl.DataFrame(data, schema=schema, schema_overrides=schema_overrides)
            .unnest("y")
            .unnest("c")
        )
        # shape: (1, 6)
        # ┌─────┬───────┬─────┬─────────────────────┬───────┬───────┐
        # │ x   ┆ a     ┆ b   ┆ d                   ┆ e     ┆ f     │
        # │ --- ┆ ---   ┆ --- ┆ ---                 ┆ ---   ┆ ---   │
        # │ i16 ┆ str   ┆ i32 ┆ datetime[ms]        ┆ f32   ┆ str   │
        # ╞═════╪═══════╪═════╪═════════════════════╪═══════╪═══════╡
        # │ 100 ┆ hello ┆ 800 ┆ 2023-04-12 10:30:00 ┆ -10.5 ┆ world │
        # └─────┴───────┴─────┴─────────────────────┴───────┴───────┘
        assert df.schema == {
            "x": pl.Int16,
            "a": pl.String,
            "b": pl.Int32,
            "d": pl.Datetime("ms"),
            "e": pl.Float32,
            "f": pl.String,
        }
        assert df.row(0) == (
            100,
            "hello",
            800,
            datetime(2023, 4, 12, 10, 30),
            -10.5,
            "world",
        )


def test_dataclasses_initvar_typing() -> None:
    @dataclasses.dataclass
    class ABC:
        x: date
        y: float
        z: dataclasses.InitVar[list[str]] = None

    # should be able to parse the initvar typing...
    abc = ABC(x=date(1999, 12, 31), y=100.0)
    df = pl.DataFrame([abc])

    # ...but should not load the initvar field into the DataFrame
    assert dataclasses.asdict(abc) == df.rows(named=True)[0]


def test_collections_namedtuple() -> None:
    TestData = namedtuple("TestData", ["id", "info"])
    nt_data = [TestData(1, "a"), TestData(2, "b"), TestData(3, "c")]

    result = pl.DataFrame(nt_data)
    expected = pl.DataFrame({"id": [1, 2, 3], "info": ["a", "b", "c"]})
    assert_frame_equal(result, expected)

    result = pl.DataFrame({"data": nt_data, "misc": ["x", "y", "z"]})
    expected = pl.DataFrame(
        {
            "data": [
                {"id": 1, "info": "a"},
                {"id": 2, "info": "b"},
                {"id": 3, "info": "c"},
            ],
            "misc": ["x", "y", "z"],
        }
    )
    assert_frame_equal(result, expected)


def test_init_ndarray() -> None:
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

    # 2D array (or 2x 1D array) - should default to column orientation (if C-contiguous)
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
    assert df.schema == {"x": pl.Boolean, "y": pl.Int32, "z": pl.String}

    # 2D array - default to column orientation
    df = pl.DataFrame(np.array([[1, 2], [3, 4]], dtype=np.int64))
    expected = pl.DataFrame({"column_0": [1, 3], "column_1": [2, 4]})
    assert_frame_equal(df, expected)

    # no orientation, numpy convention
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

    # List column from 2D array with single-column schema
    df = pl.DataFrame(np.arange(4).reshape(-1, 1).astype(np.int64), schema=["a"])
    assert_frame_equal(df, pl.DataFrame({"a": [0, 1, 2, 3]}))
    assert np.array_equal(df.to_numpy(), np.arange(4).reshape(-1, 1).astype(np.int64))

    df = pl.DataFrame(np.arange(4).reshape(-1, 2).astype(np.int64), schema=["a"])
    assert_frame_equal(
        df,
        pl.DataFrame(
            {"a": [[0, 1], [2, 3]]}, schema={"a": pl.Array(pl.Int64, shape=2)}
        ),
    )

    # 2D numpy arrays
    df = pl.DataFrame({"a": np.arange(5, dtype=np.int64).reshape(1, -1)})
    assert df.dtypes == [pl.Array(pl.Int64, shape=5)]
    assert df.shape == (1, 1)

    df = pl.DataFrame({"a": np.arange(10, dtype=np.int64).reshape(2, -1)})
    assert df.dtypes == [pl.Array(pl.Int64, shape=5)]
    assert df.shape == (2, 1)
    assert df.rows() == [([0, 1, 2, 3, 4],), ([5, 6, 7, 8, 9],)]

    test_rows = [(1, 2), (3, 4)]
    df = pl.DataFrame([np.array(test_rows[0]), np.array(test_rows[1])], orient="row")
    expected = pl.DataFrame(test_rows, orient="row")
    assert_frame_equal(df, expected)

    # round trip export/init
    for shape in ((4, 4), (4, 8), (8, 4)):
        np_ones = np.ones(shape=shape, dtype=np.float64)
        names = [f"c{i}" for i in range(shape[1])]

        df = pl.DataFrame(np_ones, schema=names)
        assert_frame_equal(df, pl.DataFrame(np.asarray(df), schema=names))


def test_init_ndarray_errors() -> None:
    # 2D array: orientation conflicts with columns
    with pytest.raises(ValueError):
        pl.DataFrame(np.array([[1, 2, 3], [4, 5, 6]]), schema=["a", "b"], orient="row")

    with pytest.raises(ValueError):
        pl.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6]]),
            schema=[("a", pl.UInt32), ("b", pl.UInt32)],
            orient="row",
        )

    # Invalid orient value
    with pytest.raises(ValueError):
        pl.DataFrame(
            np.array([[1, 2, 3], [4, 5, 6]]),
            orient="wrong",  # type: ignore[arg-type]
        )

    # Dimensions mismatch
    with pytest.raises(ValueError):
        _ = pl.DataFrame(np.array([1, 2, 3]), schema=[])

    # Cannot init with 3D array
    with pytest.raises(ValueError):
        _ = pl.DataFrame(np.random.randn(2, 2, 2))


def test_init_ndarray_nan() -> None:
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
    assert_frame_equal(df0, df1)
    assert df2.rows() == [(1.0, 4.0), (2.5, None), (None, 6.5)]

    s0 = pl.Series("n", [1.0, 2.5, float("nan")])
    s1 = pl.Series("n", np.array([1.0, 2.5, float("nan")]))
    s2 = pl.Series("n", np.array([1.0, 2.5, float("nan")]), nan_to_null=True)

    assert_series_equal(s0, s1)
    assert s2.to_list() == [1.0, 2.5, None]


def test_init_ndarray_square() -> None:
    # 2D square array; ensure that we maintain convention
    # (first axis = rows) with/without an explicit schema
    arr = np.arange(4).reshape(2, 2)
    assert (
        [(0, 1), (2, 3)]
        == pl.DataFrame(arr).rows()
        == pl.DataFrame(arr, schema=["a", "b"]).rows()
    )
    # check that we tie-break square arrays using fortran vs c-contiguous row/col major
    df_c = pl.DataFrame(
        data=np.array([[1, 2], [3, 4]], dtype=np.int64, order="C"),
        schema=["x", "y"],
    )
    assert_frame_equal(df_c, pl.DataFrame({"x": [1, 3], "y": [2, 4]}))

    df_f = pl.DataFrame(
        data=np.array([[1, 2], [3, 4]], dtype=np.int64, order="F"),
        schema=["x", "y"],
    )
    assert_frame_equal(df_f, pl.DataFrame({"x": [1, 2], "y": [3, 4]}))


def test_init_numpy_unavailable(monkeypatch: Any) -> None:
    monkeypatch.setattr(pl.dataframe.frame, "_check_for_numpy", lambda x: False)
    with pytest.raises(TypeError):
        pl.DataFrame(np.array([1, 2, 3]), schema=["a"])


def test_init_numpy_scalars() -> None:
    df = pl.DataFrame(
        {
            "bool": [np.bool_(True), np.bool_(False)],
            "i8": [np.int8(16), np.int8(64)],
            "u32": [np.uint32(1234), np.uint32(9876)],
        }
    )
    df_expected = pl.from_records(
        data=[(True, 16, 1234), (False, 64, 9876)],
        schema=OrderedDict([("bool", pl.Boolean), ("i8", pl.Int8), ("u32", pl.UInt32)]),
        orient="row",
    )
    assert_frame_equal(df, df_expected)


def test_null_array_print_format() -> None:
    pa_tbl_null = pa.table({"a": [None, None]})
    df_null = pl.from_arrow(pa_tbl_null)
    assert df_null.shape == (2, 1)
    assert df_null.dtypes == [pl.Null]  # type: ignore[union-attr]
    assert df_null.rows() == [(None,), (None,)]  # type: ignore[union-attr]

    assert (
        str(df_null) == "shape: (2, 1)\n"
        "┌──────┐\n"
        "│ a    │\n"
        "│ ---  │\n"
        "│ null │\n"
        "╞══════╡\n"
        "│ null │\n"
        "│ null │\n"
        "└──────┘"
    )


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


def test_init_from_frame() -> None:
    df1 = pl.DataFrame({"id": [0, 1], "misc": ["a", "b"], "val": [-10, 10]})
    assert_frame_equal(df1, pl.DataFrame(df1))

    df2 = pl.DataFrame(df1, schema=["a", "b", "c"])
    assert_frame_equal(df2, pl.DataFrame(df2))

    df3 = pl.DataFrame(df1, schema=["a", "b", "c"], schema_overrides={"val": pl.Int8})
    assert_frame_equal(df3, pl.DataFrame(df3))

    assert df1.schema == {"id": pl.Int64, "misc": pl.String, "val": pl.Int64}
    assert df2.schema == {"a": pl.Int64, "b": pl.String, "c": pl.Int64}
    assert df3.schema == {"a": pl.Int64, "b": pl.String, "c": pl.Int8}
    assert df1.rows() == df2.rows() == df3.rows()

    s1 = pl.Series("s", df3)
    s2 = pl.Series(df3)

    assert s1.name == "s"
    assert s2.name == ""


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
    col0 = pl.Series("column_0", [1, 2, 3])
    col1 = pl.Series("column_1", [4, 5, 6])
    expected = pl.DataFrame([col0, col1])
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

    nested_dtype = pl.List(pl.List(pl.UInt8))
    s3 = pl.Series("x", dtype=nested_dtype)
    s4 = pl.Series(s3)
    for s in (s3, s4):
        assert s.dtype == nested_dtype
        assert s.to_list() == []
        assert s.name == "x"

    s5 = pl.Series("", df, dtype=pl.Int8)
    assert_series_equal(s5, pl.Series("", [1, 2, 3], dtype=pl.Int8))


@pytest.mark.parametrize(
    ("dtype", "expected_dtype"),
    [
        (int, pl.Int64),
        (bytes, pl.Binary),
        (float, pl.Float64),
        (str, pl.String),
        (date, pl.Date),
        (time, pl.Time),
        (datetime, pl.Datetime("us")),
        (timedelta, pl.Duration("us")),
        (Decimal, pl.Decimal(precision=None, scale=0)),
    ],
)
def test_init_py_dtype(dtype: Any, expected_dtype: PolarsDataType) -> None:
    for s in (
        pl.Series("s", [None], dtype=dtype),
        pl.Series("s", [], dtype=dtype),
    ):
        assert s.dtype == expected_dtype

    for df in (
        pl.DataFrame({"col": [None]}, schema={"col": dtype}),
        pl.DataFrame({"col": []}, schema={"col": dtype}),
    ):
        assert df.schema == {"col": expected_dtype}


def test_init_py_dtype_misc_float() -> None:
    assert pl.Series([100], dtype=float).dtype == pl.Float64  # type: ignore[arg-type]

    df = pl.DataFrame(
        {"x": [100.0], "y": [200], "z": [None]},
        schema={"x": float, "y": float, "z": float},
    )
    assert df.schema == {"x": pl.Float64, "y": pl.Float64, "z": pl.Float64}
    assert df.rows() == [(100.0, 200.0, None)]


def test_init_seq_of_seq() -> None:
    # List of lists
    df = pl.DataFrame([[1, 2, 3], [4, 5, 6]], schema=["a", "b", "c"], orient="row")
    expected = pl.DataFrame({"a": [1, 4], "b": [2, 5], "c": [3, 6]})
    assert_frame_equal(df, expected)

    df = pl.DataFrame(
        [[1, 2, 3], [4, 5, 6]],
        schema=[("a", pl.Int8), ("b", pl.Int16), ("c", pl.Int32)],
        orient="row",
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
    result = pl.DataFrame("abc", schema=["s"])
    expected = pl.DataFrame({"s": ["a", "b", "c"]})
    assert_frame_equal(result, expected)

    # datetimes sequence
    df = pl.DataFrame([datetime(2020, 1, 1)], schema={"ts": pl.Datetime("ms")})
    assert df.schema == {"ts": pl.Datetime("ms")}
    df = pl.DataFrame(
        [datetime(2020, 1, 1, tzinfo=timezone.utc)], schema={"ts": pl.Datetime("ms")}
    )
    assert df.schema == {"ts": pl.Datetime("ms", "UTC")}
    df = pl.DataFrame(
        [datetime(2020, 1, 1, tzinfo=timezone(timedelta(hours=1)))],
        schema={"ts": pl.Datetime("ms")},
    )
    assert df.schema == {"ts": pl.Datetime("ms", "UTC")}
    df = pl.DataFrame(
        [datetime(2020, 1, 1, tzinfo=ZoneInfo("Asia/Kathmandu"))],
        schema={"ts": pl.Datetime("ms")},
    )
    assert df.schema == {"ts": pl.Datetime("ms", "UTC")}


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
    class XSeries(pd.Series):  # type: ignore[type-arg]
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
    monkeypatch.setattr(pl.dataframe.frame, "_check_for_pandas", lambda x: False)

    # pandas 2.2 and higher implement the Arrow PyCapsule Interface, so the constructor
    # will still work even without using pandas APIs
    if parse_version(pd.__version__) >= parse_version("2.2.0"):
        df = pl.DataFrame(pandas_df)
        assert_frame_equal(df, expected)

    else:
        with pytest.raises(TypeError):
            pl.DataFrame(pandas_df)


def test_init_errors() -> None:
    # Length mismatch
    with pytest.raises(ShapeError):
        pl.DataFrame({"a": [1, 2, 3], "b": [1.0, 2.0, 3.0, 4.0]})

    # Columns don't match data dimensions
    with pytest.raises(ShapeError):
        pl.DataFrame([[1, 2], [3, 4]], schema=["a", "b", "c"])

    # Unmatched input
    with pytest.raises(TypeError):
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

    df_cd = pl.DataFrame(dicts, schema=["a", "c", "d"])
    expected_values = {
        "a": [1, 2, 1],
        "c": [None, None, None],
        "d": [None, None, None],
    }
    assert df_cd.to_dict(as_series=False) == expected_values

    data = {"a": 1, "b": 2, "c": 3}

    df1 = pl.from_dicts([data])
    assert df1.columns == ["a", "b", "c"]

    df1.columns = ["x", "y", "z"]
    assert df1.columns == ["x", "y", "z"]

    df2 = pl.from_dicts([data], schema=["c", "b", "a"])
    assert df2.columns == ["c", "b", "a"]

    for colname in ("c", "b", "a"):
        result = pl.from_dicts([data], schema=[colname])
        expected_values = {colname: [data[colname]]}
        assert result.to_dict(as_series=False) == expected_values


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

    for constructor in (pl.from_dicts, pl.DataFrame):
        # ensure field values are loaded according to the declared schema order
        for _ in range(8):
            shuffle(data)
            shuffle(cols)

            df = constructor(data, schema=cols)
            for col in df.columns:
                assert all(value in (None, lookup[col]) for value in df[col].to_list())

        # have schema override inferred types, omit some columns, add a new one
        schema = {"a": pl.Int8, "c": pl.Int16, "e": pl.Int32}
        df = constructor(data, schema=schema)

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
            pl.col("a").cast(pl.Date),
            pl.col("b").cast(pl.UInt64),
            pl.col("c").cast(pl.Int8),
        )
        expected.insert_column(3, pl.Series("d", [], pl.List(pl.UInt8)))

        assert df.shape == (0, 4)
        assert_frame_equal(df, expected)
        assert df.dtypes == [pl.Date, pl.UInt64, pl.Int8, pl.List]
        assert pl.List(pl.UInt8).is_(df.schema["d"])

        dfe = df.clear()
        assert len(dfe) == 0
        assert df.schema == dfe.schema
        assert dfe.shape == df.shape


def test_from_dicts_list_without_dtype() -> None:
    result = pl.from_dicts(
        [{"id": 1, "hint": ["some_text_here"]}, {"id": 2, "hint": [None]}]
    )
    expected = pl.DataFrame({"id": [1, 2], "hint": [["some_text_here"], [None]]})
    assert_frame_equal(result, expected)


def test_from_dicts_list_struct_without_inner_dtype() -> None:
    df = pl.DataFrame(
        {
            "users": [
                [{"category": "A"}, {"category": "B"}],
                [{"category": None}, {"category": None}],
            ],
            "days_of_week": [1, 2],
        }
    )
    expected = {
        "users": [
            [{"category": "A"}, {"category": "B"}],
            [{"category": None}, {"category": None}],
        ],
        "days_of_week": [1, 2],
    }
    assert df.to_dict(as_series=False) == expected


def test_from_dicts_list_struct_without_inner_dtype_5611() -> None:
    result = pl.from_dicts(
        [
            {"a": []},
            {"a": [{"b": 1}]},
        ]
    )
    expected = pl.DataFrame({"a": [[], [{"b": 1}]]})
    assert_frame_equal(result, expected)


def test_from_dict_upcast_primitive() -> None:
    df = pl.from_dict({"a": [1, 2.1, 3], "b": [4, 5, 6.4]}, strict=False)
    assert df.dtypes == [pl.Float64, pl.Float64]


def test_u64_lit_5031() -> None:
    df = pl.DataFrame({"foo": [1, 2, 3]}).with_columns(pl.col("foo").cast(pl.UInt64))
    assert df.filter(pl.col("foo") < (1 << 64) - 20).shape == (3, 1)
    assert df["foo"].to_list() == [1, 2, 3]


def test_from_dicts_missing_columns() -> None:
    # missing columns from some of the data dicts
    data = [{"a": 1}, {"b": 2}]
    result = pl.from_dicts(data)
    expected = pl.DataFrame({"a": [1, None], "b": [None, 2]})
    assert_frame_equal(result, expected)

    # partial schema with some columns missing; only load the declared keys
    data = [{"a": 1, "b": 2}]
    result = pl.from_dicts(data, schema=["a"])
    expected = pl.DataFrame({"a": [1]})
    assert_frame_equal(result, expected)


def test_from_dicts_schema_columns_do_not_match() -> None:
    data = [{"a": 1, "b": 2}]
    result = pl.from_dicts(data, schema=["x"])
    expected = pl.DataFrame({"x": [None]})
    assert_frame_equal(result, expected)


def test_from_rows_dtype() -> None:
    # 50 is the default inference length
    # 5182
    df = pl.DataFrame(
        data=[(None, None)] * 50 + [("1.23", None)],
        schema=[("foo", pl.String), ("bar", pl.String)],
        orient="row",
    )
    assert df.dtypes == [pl.String, pl.String]
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

    dc = _TestBazDC(d=datetime(2020, 2, 22), e=42.0, f="xyz")
    df = pl.DataFrame([[dc]], schema={"d": pl.Object})
    assert df.schema == {"d": pl.Object}
    assert df.item() == dc


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
        assert df.to_dict(as_series=False) == {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
            "c": [None, None, None],
        }

    # provide data that resolves to an empty frame (ref: scalar
    # expansion shortcut), with schema/override hints
    schema = {"colx": pl.String, "coly": pl.Int32}

    for param in ("schema", "schema_overrides"):
        df = pl.DataFrame({"colx": [], "coly": 0}, **{param: schema})  # type: ignore[arg-type]
        assert df.schema == schema


def test_nested_read_dicts_4143() -> None:
    result = pl.from_dicts(
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
    )
    expected = {
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
    assert result.to_dict(as_series=False) == expected


def test_nested_read_dicts_4143_2() -> None:
    result = pl.from_dicts(
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

    assert result.dtypes == [
        pl.Int64,
        pl.List(pl.Struct({"some_text_here": pl.String, "list_": pl.List(pl.Int64)})),
    ]
    expected = {
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
    assert result.to_dict(as_series=False) == expected


def test_from_records_nullable_structs() -> None:
    records = [
        {"id": 1, "items": [{"item_id": 100, "description": None}]},
        {"id": 1, "items": [{"item_id": 100, "description": "hi"}]},
    ]

    schema: list[tuple[str, PolarsDataType]] = [
        ("id", pl.UInt16),
        (
            "items",
            pl.List(
                pl.Struct(
                    [pl.Field("item_id", pl.UInt32), pl.Field("description", pl.String)]
                )
            ),
        ),
    ]

    schema_options: list[list[tuple[str, PolarsDataType]] | None] = [schema, None]
    for s in schema_options:
        result = pl.DataFrame(records, schema=s, orient="row")
        expected = {
            "id": [1, 1],
            "items": [
                [{"item_id": 100, "description": None}],
                [{"item_id": 100, "description": "hi"}],
            ],
        }
        assert result.to_dict(as_series=False) == expected

    # check initialisation without any records
    df = pl.DataFrame(schema=schema)
    dict_schema = dict(schema)
    assert df.to_dict(as_series=False) == {"id": [], "items": []}
    assert df.schema == dict_schema

    dtype: PolarsDataType = dict_schema["items"]
    series = pl.Series("items", dtype=dtype)
    assert series.to_frame().to_dict(as_series=False) == {"items": []}
    assert series.dtype == dict_schema["items"]
    assert series.to_list() == []


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

    result = df.unnest("a")

    expected = pl.DataFrame(
        {"value": ["foo", "bar"], "counts": [1, 2]},
        schema={"value": pl.Categorical, "counts": pl.UInt32},
    )
    assert_frame_equal(result, expected, categorical_as_str=True)


def test_nested_schema_construction() -> None:
    schema = {
        "node_groups": pl.List(
            pl.Struct(
                [
                    pl.Field("parent_node_group_id", pl.UInt8),
                    pl.Field(
                        "nodes",
                        pl.List(
                            pl.Struct(
                                [
                                    pl.Field("name", pl.String),
                                    pl.Field(
                                        "sub_nodes",
                                        pl.List(
                                            pl.Struct(
                                                [
                                                    pl.Field("internal_id", pl.UInt64),
                                                    pl.Field("value", pl.UInt32),
                                                ]
                                            )
                                        ),
                                    ),
                                ]
                            )
                        ),
                    ),
                ]
            )
        )
    }
    df = pl.DataFrame(
        {
            "node_groups": [
                [{"nodes": []}, {"nodes": [{"name": "", "sub_nodes": []}]}],
            ]
        },
        schema=schema,
    )

    assert df.schema == schema
    assert df.to_dict(as_series=False) == {
        "node_groups": [
            [
                {"parent_node_group_id": None, "nodes": []},
                {
                    "parent_node_group_id": None,
                    "nodes": [{"name": "", "sub_nodes": []}],
                },
            ]
        ]
    }


def test_nested_schema_construction2() -> None:
    schema = {
        "node_groups": pl.List(
            pl.Struct(
                [
                    pl.Field(
                        "nodes",
                        pl.List(
                            pl.Struct(
                                [
                                    pl.Field("name", pl.String),
                                    pl.Field("time", pl.UInt32),
                                ]
                            )
                        ),
                    )
                ]
            )
        )
    }
    df = pl.DataFrame(
        [
            {"node_groups": [{"nodes": [{"name": "a", "time": 0}]}]},
            {"node_groups": [{"nodes": []}]},
        ],
        schema=schema,
    )
    assert df.schema == schema
    assert df.to_dict(as_series=False) == {
        "node_groups": [[{"nodes": [{"name": "a", "time": 0}]}], [{"nodes": []}]]
    }


def test_arrow_to_pyseries_with_one_chunk_does_not_copy_data() -> None:
    from polars._utils.construction import arrow_to_pyseries

    original_array = pa.chunked_array([[1, 2, 3]], type=pa.int64())
    pyseries = arrow_to_pyseries("", original_array)
    assert (
        pyseries.get_chunks()[0]._get_buffer_info()[0]
        == original_array.chunks[0].buffers()[1].address
    )


def test_init_with_explicit_binary_schema() -> None:
    df = pl.DataFrame({"a": [b"hello", b"world"]}, schema={"a": pl.Binary})
    assert df.schema == {"a": pl.Binary}
    assert df["a"].to_list() == [b"hello", b"world"]

    s = pl.Series("a", [b"hello", b"world"], dtype=pl.Binary)
    assert s.dtype == pl.Binary
    assert s.to_list() == [b"hello", b"world"]


def test_nested_categorical() -> None:
    s = pl.Series([["a"]], dtype=pl.List(pl.Categorical))
    assert s.to_list() == [["a"]]
    assert s.dtype == pl.List(pl.Categorical)


def test_datetime_date_subclasses() -> None:
    class FakeDate(date): ...

    class FakeDateChild(FakeDate): ...

    class FakeDatetime(FakeDate, datetime): ...

    result = pl.Series([FakeDate(2020, 1, 1)])
    expected = pl.Series([date(2020, 1, 1)])
    assert_series_equal(result, expected)

    result = pl.Series([FakeDateChild(2020, 1, 1)])
    expected = pl.Series([date(2020, 1, 1)])
    assert_series_equal(result, expected)

    result = pl.Series([FakeDatetime(2020, 1, 1, 3)])
    expected = pl.Series([datetime(2020, 1, 1, 3)])
    assert_series_equal(result, expected)


def test_list_null_constructor() -> None:
    s = pl.Series("a", [[None], [None]], dtype=pl.List(pl.Null))
    assert s.dtype == pl.List(pl.Null)
    assert s.to_list() == [[None], [None]]

    # nested
    dtype = pl.List(pl.List(pl.Int8))
    values = [
        [],
        [[], []],
        [[33, 112]],
    ]
    s = pl.Series(
        name="colx",
        values=values,
        dtype=dtype,
    )
    assert s.dtype == dtype
    assert s.to_list() == values

    # nested
    # small order change has influence
    dtype = pl.List(pl.List(pl.Int8))
    values = [
        [[], []],
        [],
        [[33, 112]],
    ]
    s = pl.Series(
        name="colx",
        values=values,
        dtype=dtype,
    )
    assert s.dtype == dtype
    assert s.to_list() == values


def test_numpy_float_construction_av() -> None:
    np_dict = {"a": np.float64(1)}
    assert_frame_equal(pl.DataFrame(np_dict), pl.DataFrame({"a": 1.0}))


def test_df_init_dict_raise_on_expression_input() -> None:
    with pytest.raises(
        TypeError,
        match="passing Expr objects to the DataFrame constructor is not supported",
    ):
        pl.DataFrame({"a": pl.int_range(0, 3)})
    with pytest.raises(TypeError):
        pl.DataFrame({"a": pl.int_range(0, 3), "b": [3, 4, 5]})

    # Passing a list of expressions is allowed
    df = pl.DataFrame({"a": [pl.int_range(0, 3)]})
    assert df.get_column("a").dtype == pl.Object


def test_df_schema_sequences() -> None:
    schema = [
        ["address", pl.String],
        ["key", pl.Int64],
        ["value", pl.Float32],
    ]
    df = pl.DataFrame(schema=schema)  # type: ignore[arg-type]
    assert df.schema == {"address": pl.String, "key": pl.Int64, "value": pl.Float32}


def test_df_schema_sequences_incorrect_length() -> None:
    schema = [
        ["address", pl.String, pl.Int8],
        ["key", pl.Int64],
        ["value", pl.Float32],
    ]
    with pytest.raises(ValueError):
        pl.DataFrame(schema=schema)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("input", "infer_func", "expected_dtype"),
    [
        ("f8", numpy_char_code_to_dtype, pl.Float64),
        ("f4", numpy_char_code_to_dtype, pl.Float32),
        ("i4", numpy_char_code_to_dtype, pl.Int32),
        ("u1", numpy_char_code_to_dtype, pl.UInt8),
        ("?", numpy_char_code_to_dtype, pl.Boolean),
        ("m8", numpy_char_code_to_dtype, pl.Duration("us")),
        ("M8", numpy_char_code_to_dtype, pl.Datetime("us")),
    ],
)
def test_numpy_inference(
    input: Any,
    infer_func: Callable[[Any], PolarsDataType],
    expected_dtype: PolarsDataType,
) -> None:
    result = infer_func(input)
    assert result == expected_dtype


def test_array_construction() -> None:
    payload = [[1, 2, 3], None, [4, 2, 3]]

    dtype = pl.Array(pl.Int64, 3)
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # inner type
    dtype = pl.Array(pl.UInt8, 2)
    payload = [[1, 2], None, [3, 4]]
    s = pl.Series(payload, dtype=dtype)
    assert s.dtype == dtype
    assert s.to_list() == payload

    # create using schema
    df = pl.DataFrame(
        schema={
            "a": pl.Array(pl.Float32, 3),
            "b": pl.Array(pl.Datetime("ms"), 5),
        }
    )
    assert df.dtypes == [
        pl.Array(pl.Float32, 3),
        pl.Array(pl.Datetime("ms"), 5),
    ]
    assert df.rows() == []

    # from dicts
    rows = [
        {"row_id": "a", "data": [1, 2, 3]},
        {"row_id": "b", "data": [2, 3, 4]},
    ]
    schema = {"row_id": pl.String(), "data": pl.Array(inner=pl.Int64, shape=3)}
    df = pl.from_dicts(rows, schema=schema)
    assert df.schema == schema
    assert df.rows() == [("a", [1, 2, 3]), ("b", [2, 3, 4])]


class PyCapsuleStreamHolder:
    """
    Hold the Arrow C Stream pycapsule.

    A class that exposes _only_ the Arrow C Stream interface via Arrow PyCapsules. This
    ensures that the consumer is seeing _only_ the `__arrow_c_stream__` dunder, and that
    nothing else (e.g. the dataframe or array interface) is actually being used.
    """

    arrow_obj: Any

    def __init__(self, arrow_obj: object) -> None:
        self.arrow_obj = arrow_obj

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self.arrow_obj.__arrow_c_stream__(requested_schema)


class PyCapsuleArrayHolder:
    """
    Hold the Arrow C Array pycapsule.

    A class that exposes _only_ the Arrow C Array interface via Arrow PyCapsules. This
    ensures that the consumer is seeing _only_ the `__arrow_c_array__` dunder, and that
    nothing else (e.g. the dataframe or array interface) is actually being used.
    """

    arrow_obj: Any

    def __init__(self, arrow_obj: object) -> None:
        self.arrow_obj = arrow_obj

    def __arrow_c_array__(self, requested_schema: object = None) -> object:
        return self.arrow_obj.__arrow_c_array__(requested_schema)


def test_pycapsule_interface(df: pl.DataFrame) -> None:
    df = df.rechunk()
    pyarrow_table = df.to_arrow()

    # Array via C data interface
    pyarrow_array = pyarrow_table["bools"].chunk(0)
    round_trip_series = pl.Series(PyCapsuleArrayHolder(pyarrow_array))
    assert df["bools"].equals(round_trip_series, check_dtypes=True, check_names=False)

    # empty Array via C data interface
    empty_pyarrow_array = pa.array([], type=pyarrow_array.type)
    round_trip_series = pl.Series(PyCapsuleArrayHolder(empty_pyarrow_array))
    assert df["bools"].dtype == round_trip_series.dtype

    # RecordBatch via C array interface
    pyarrow_record_batch = pyarrow_table.to_batches()[0]
    round_trip_df = pl.DataFrame(PyCapsuleArrayHolder(pyarrow_record_batch))
    assert df.equals(round_trip_df)

    # ChunkedArray via C stream interface
    pyarrow_chunked_array = pyarrow_table["bools"]
    round_trip_series = pl.Series(PyCapsuleStreamHolder(pyarrow_chunked_array))
    assert df["bools"].equals(round_trip_series, check_dtypes=True, check_names=False)

    # empty ChunkedArray via C stream interface
    empty_chunked_array = pa.chunked_array([], type=pyarrow_chunked_array.type)
    round_trip_series = pl.Series(PyCapsuleStreamHolder(empty_chunked_array))
    assert df["bools"].dtype == round_trip_series.dtype

    # Table via C stream interface
    round_trip_df = pl.DataFrame(PyCapsuleStreamHolder(pyarrow_table))
    assert df.equals(round_trip_df)

    # empty Table via C stream interface
    empty_df = df[:0].to_arrow()
    round_trip_df = pl.DataFrame(PyCapsuleStreamHolder(empty_df))
    orig_schema = df.schema
    round_trip_schema = round_trip_df.schema

    # The "enum" schema is not preserved because categories are lost via C data
    # interface
    orig_schema.pop("enum")
    round_trip_schema.pop("enum")

    assert orig_schema == round_trip_schema

    # RecordBatchReader via C stream interface
    pyarrow_reader = pa.RecordBatchReader.from_batches(
        pyarrow_table.schema, pyarrow_table.to_batches()
    )
    round_trip_df = pl.DataFrame(PyCapsuleStreamHolder(pyarrow_reader))
    assert df.equals(round_trip_df)


@pytest.mark.parametrize(
    "tz",
    [
        None,
        ZoneInfo("Asia/Tokyo"),
        ZoneInfo("Europe/Amsterdam"),
        ZoneInfo("UTC"),
        timezone.utc,
    ],
)
def test_init_list_of_dicts_with_timezone(tz: Any) -> None:
    dt = datetime(2023, 1, 1, 0, 0, 0, 0, tzinfo=tz)

    df = pl.DataFrame([{"dt": dt}, {"dt": dt}])
    expected = pl.DataFrame({"dt": [dt, dt]})
    assert_frame_equal(df, expected)

    assert df.schema == {"dt": pl.Datetime("us", time_zone=tz and "UTC")}
