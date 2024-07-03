from __future__ import annotations

from typing import Any

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import InvalidOperationError, SQLInterfaceError
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
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (2, 2, 2, 0, 2, 2, 2, 2, 2, 0, 2, 2, 2, 2),
        (3, 3, 3, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3),
        (4, 4, 4, 0, 4, 4, 4, 4, 4, 0, 4, 4, 4, 4),
        (5, 5, 5, 1, 5, 5, 5, 5, 5, 1, 5, 5, 5, 5),
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
    ("values", "cast_op", "error"),
    [
        ([1.0, -1.0], "values::uint8", "conversion from `f64` to `u64` failed"),
        ([10, 0, -1], "values::uint4", "conversion from `i64` to `u32` failed"),
        ([int(1e8)], "values::int1", "conversion from `i64` to `i8` failed"),
        (["a", "b"], "values::date", "conversion from `str` to `date` failed"),
        (["a", "b"], "values::time", "conversion from `str` to `time` failed"),
        (["a", "b"], "values::int4", "conversion from `str` to `i32` failed"),
    ],
)
def test_cast_errors(values: Any, cast_op: str, error: str) -> None:
    df = pl.DataFrame({"values": values})

    # invalid CAST should raise an error...
    with pytest.raises(InvalidOperationError, match=error):
        df.sql(f"SELECT {cast_op} FROM self")

    # ... or return `null` values if using TRY_CAST
    target_type = cast_op.split("::")[1]
    res = df.sql(f"SELECT TRY_CAST(values AS {target_type}) AS cast_values FROM self")
    assert None in res.to_series()


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
