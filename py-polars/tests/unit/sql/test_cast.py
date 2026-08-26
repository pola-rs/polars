from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import (
    ComputeError,
    InvalidOperationError,
    SQLInterfaceError,
)
from polars.testing import assert_frame_equal


def test_cast() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5],
            "c": ["a", "b", "c", "d", "e"],
            "d": [True, False, True, False, True],
            "e": [-1, 0, None, 1, 2],
        }
    )

    # test various dtype casts, using standard ("CAST <col> AS <dtype>")
    # and postgres-specific ("<col>::<dtype>") cast syntax
    with pl.SQLContext(df=df, eager=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              -- float
              CAST(a AS DOUBLE PRECISION) AS a_f64,
              a::real AS a_f32,
              b::float(24) AS b_f32,
              b::float(25) AS b_f64,
              e::float8 AS e_f64,
              e::float4 AS e_f32,

              -- integer
              CAST(b AS TINYINT) AS b_i8,
              CAST(b AS SMALLINT) AS b_i16,
              b::bigint AS b_i64,
              d::tinyint AS d_i8,
              d::hugeint AS d_i128,
              d::uhugeint as d_u128,
              a::int1 AS a_i8,
              a::int2 AS a_i16,
              a::int4 AS a_i32,
              a::int8 AS a_i64,

              -- unsigned integer
              CAST(a AS TINYINT UNSIGNED) AS a_u8,
              d::uint1 AS d_u8,
              a::uint2 AS a_u16,
              b::uint4 AS b_u32,
              b::uint8 AS b_u64,
              CAST(a AS BIGINT UNSIGNED) AS a_u64,
              b::utinyint AS b_u8,
              b::usmallint AS b_u16,
              a::uinteger AS a_u32,
              d::ubigint AS d_u64,

              -- string/binary
              CAST(a AS CHAR) AS a_char,
              CAST(b AS VARCHAR) AS b_varchar,
              c::blob AS c_blob,
              c::bytes AS c_bytes,
              c::VARBINARY AS c_varbinary,
              CAST(d AS CHARACTER VARYING) AS d_charvar,

              -- boolean
              e::bool AS e_bool,
              e::boolean AS e_boolean
            FROM df
            """
        )
    assert res.schema == {
        "a_f64": pl.Float64,
        "a_f32": pl.Float32,
        "b_f32": pl.Float32,
        "b_f64": pl.Float64,
        "e_f64": pl.Float64,
        "e_f32": pl.Float32,
        "b_i8": pl.Int8,
        "b_i16": pl.Int16,
        "b_i64": pl.Int64,
        "d_i8": pl.Int8,
        "d_i128": pl.Int128,
        "d_u128": pl.UInt128,
        "a_i8": pl.Int8,
        "a_i16": pl.Int16,
        "a_i32": pl.Int32,
        "a_i64": pl.Int64,
        "a_u8": pl.UInt8,
        "d_u8": pl.UInt8,
        "a_u16": pl.UInt16,
        "b_u32": pl.UInt32,
        "b_u64": pl.UInt64,
        "a_u64": pl.UInt64,
        "b_u8": pl.UInt8,
        "b_u16": pl.UInt16,
        "a_u32": pl.UInt32,
        "d_u64": pl.UInt64,
        "a_char": pl.String,
        "b_varchar": pl.String,
        "c_blob": pl.Binary,
        "c_bytes": pl.Binary,
        "c_varbinary": pl.Binary,
        "d_charvar": pl.String,
        "e_bool": pl.Boolean,
        "e_boolean": pl.Boolean,
    }
    assert res.select(cs.by_dtype(pl.Float32)).rows() == pytest.approx(
        [
            (1.0, 1.100000023841858, -1.0),
            (2.0, 2.200000047683716, 0.0),
            (3.0, 3.299999952316284, None),
            (4.0, 4.400000095367432, 1.0),
            (5.0, 5.5, 2.0),
        ]
    )
    assert res.select(cs.by_dtype(pl.Float64)).rows() == [
        (1.0, 1.1, -1.0),
        (2.0, 2.2, 0.0),
        (3.0, 3.3, None),
        (4.0, 4.4, 1.0),
        (5.0, 5.5, 2.0),
    ]
    assert res.select(cs.integer()).rows() == [
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (2, 2, 2, 0, 0, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 0),
        (3, 3, 3, 1, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1),
        (4, 4, 4, 0, 0, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4, 4, 4, 4, 0),
        (5, 5, 5, 1, 1, 1, 5, 5, 5, 5, 5, 1, 5, 5, 5, 5, 5, 5, 5, 1),
    ]
    assert res.select(cs.string()).rows() == [
        ("1", "1.1", "true"),
        ("2", "2.2", "false"),
        ("3", "3.3", "true"),
        ("4", "4.4", "false"),
        ("5", "5.5", "true"),
    ]
    assert res.select(cs.binary()).rows() == [
        (b"a", b"a", b"a"),
        (b"b", b"b", b"b"),
        (b"c", b"c", b"c"),
        (b"d", b"d", b"d"),
        (b"e", b"e", b"e"),
    ]
    assert res.select(cs.boolean()).rows() == [
        (True, True),
        (False, False),
        (None, None),
        (True, True),
        (True, True),
    ]

    with pytest.raises(
        SQLInterfaceError,
        match="use of FORMAT is not currently supported in CAST",
    ):
        pl.SQLContext(df=df, eager=True).execute(
            "SELECT CAST(a AS STRING FORMAT 'HEX') FROM df"
        )


@pytest.mark.parametrize(
    ("values", "cast_op", "exc", "error"),
    [
        (
            [1.0, -1.0],
            "values::uint8",
            InvalidOperationError,
            "conversion from `f64` to `u64` failed",
        ),
        (
            [10, 0, -1],
            "values::uint4",
            InvalidOperationError,
            "conversion from `i64` to `u32` failed",
        ),
        (
            [int(1e8)],
            "values::int1",
            InvalidOperationError,
            "conversion from `i64` to `i8` failed",
        ),
        (
            ["a", "b"],
            "values::time",
            ComputeError,
            "could not find an appropriate format to parse times",
        ),
        (
            ["a", "b"],
            "values::int4",
            InvalidOperationError,
            "conversion from `str` to `i32` failed",
        ),
    ],
)
def test_cast_errors(
    values: Any, cast_op: str, exc: type[Exception], error: str
) -> None:
    df = pl.DataFrame({"values": values})

    # invalid CAST should raise an error...
    with pytest.raises(exc, match=error):
        df.sql(f"SELECT {cast_op} FROM self")

    # ... or return `null` values if using TRY_CAST
    target_type = cast_op.split("::")[1]
    res = df.sql(f"SELECT TRY_CAST(values AS {target_type}) AS cast_values FROM self")
    assert None in res.to_series()


@pytest.mark.parametrize(
    ("sql_type", "dtype", "value", "expected"),
    [
        ("date", pl.Date, "2000-02-01", date(2000, 2, 1)),
        (
            "timestamp",
            pl.Datetime("us"),
            "2000-02-01 12:30:00",
            datetime(2000, 2, 1, 12, 30),
        ),
        (
            "datetime",
            pl.Datetime("us"),
            "2000-02-01 12:30:00",
            datetime(2000, 2, 1, 12, 30),
        ),
        ("time", pl.Time, "12:30:00", time(12, 30)),
    ],
)
def test_cast_string_to_temporal(
    sql_type: str, dtype: pl.DataType, value: str, expected: Any
) -> None:
    df = pl.DataFrame({"s": [value, None]})

    for cast_op in (
        f"CAST(s AS {sql_type})",
        f"TRY_CAST(s AS {sql_type})",
        f"s::{sql_type}",
    ):
        res = df.sql(f"SELECT {cast_op} AS x FROM self")
        assert_frame_equal(
            res, pl.DataFrame({"x": [expected, None]}, schema={"x": dtype})
        )

    for cast_op in (
        f"CAST('{value}' AS {sql_type})",
        f"TRY_CAST('{value}' AS {sql_type})",
        f"'{value}'::{sql_type}",
    ):
        res = df.sql(f"SELECT {cast_op} AS x FROM self")
        assert_frame_equal(
            res, pl.DataFrame({"x": [expected, expected]}, schema={"x": dtype})
        )


@pytest.mark.parametrize(
    ("sql_type", "dtype"),
    [
        ("date", pl.Date),
        ("timestamp", pl.Datetime("us")),
        ("time", pl.Time),
    ],
)
def test_try_cast_string_to_temporal_nulls(sql_type: str, dtype: pl.DataType) -> None:
    df = pl.DataFrame({"s": ["not a temporal value"]})

    for operand in ("s", "'not a temporal value'"):
        res = df.sql(f"SELECT TRY_CAST({operand} AS {sql_type}) AS x FROM self")
        assert_frame_equal(res, pl.DataFrame({"x": [None]}, schema={"x": dtype}))

        with pytest.raises(ComputeError, match="could not find an appropriate format"):
            df.sql(f"SELECT CAST({operand} AS {sql_type}) AS x FROM self")


def test_cast_temporal_to_temporal_is_not_parsed() -> None:
    df = pl.DataFrame(
        {"dtm": [datetime(2000, 2, 1, 12, 30)]},
        schema={"dtm": pl.Datetime("us")},
    )
    res = df.sql(
        """
        SELECT
          CAST(dtm AS date) AS d,
          CAST(dtm AS time) AS t
        FROM self
        """
    )
    assert_frame_equal(
        res,
        pl.DataFrame({"d": [date(2000, 2, 1)], "t": [time(12, 30)]}),
    )


def test_cast_string_to_date_in_between() -> None:
    df = pl.DataFrame(
        {"d": [date(1999, 1, 1), date(1999, 3, 1), date(2000, 1, 1)]},
    )
    res = df.sql(
        "SELECT * FROM self WHERE d BETWEEN CAST('1999-02-22' AS date) AND CAST('1999-03-24' AS date)"
    )
    assert_frame_equal(res, pl.DataFrame({"d": [date(1999, 3, 1)]}))


def test_temporal_in_string_list() -> None:
    df = pl.DataFrame(
        {"d": [date(1999, 1, 1), date(1999, 3, 1), date(2000, 1, 1)]},
    )
    res = df.sql("SELECT * FROM self WHERE d IN ('1999-03-01', '2000-01-01')")
    assert_frame_equal(res, pl.DataFrame({"d": [date(1999, 3, 1), date(2000, 1, 1)]}))


def test_try_cast_string_to_temporal_partial() -> None:
    df = pl.DataFrame({"s": ["2000-02-01", "nope"]})

    res = df.sql("SELECT TRY_CAST(s AS date) AS x FROM self")
    assert_frame_equal(
        res, pl.DataFrame({"x": [date(2000, 2, 1), None]}, schema={"x": pl.Date})
    )

    with pytest.raises(InvalidOperationError, match=r"conversion .* failed"):
        df.sql("SELECT CAST(s AS date) AS x FROM self")


@pytest.mark.may_fail_cloud  # reason: eager construct to_struct
@pytest.mark.xfail  # this is a construct we cannot deal with anymore
def test_cast_json() -> None:
    df = pl.DataFrame({"txt": ['{"a":[1,2,3],"b":["x","y","z"],"c":5.0}']})

    with pl.SQLContext(df=df, eager=True) as ctx:
        for json_cast in ("txt::json", "CAST(txt AS JSON)"):
            res = ctx.execute(f"SELECT {json_cast} AS j FROM df")

            assert res.schema == {
                "j": pl.Struct(
                    {
                        "a": pl.List(pl.Int64),
                        "b": pl.List(pl.String),
                        "c": pl.Float64,
                    },
                )
            }
            assert_frame_equal(
                res.unnest("j"),
                pl.DataFrame(
                    {
                        "a": [[1, 2, 3]],
                        "b": [["x", "y", "z"]],
                        "c": [5.0],
                    }
                ),
            )
