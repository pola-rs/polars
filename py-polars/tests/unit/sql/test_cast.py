from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import ComputeError


def test_cast() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5],
            "c": ["a", "b", "c", "d", "e"],
            "d": [True, False, True, False, True],
        }
    )
    # test various dtype casts, using standard ("CAST <col> AS <dtype>")
    # and postgres-specific ("<col>::<dtype>") cast syntax
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              -- float
              CAST(a AS DOUBLE PRECISION) AS a_f64,
              a::real AS a_f32,
              -- integer
              CAST(b AS TINYINT) AS b_i8,
              CAST(b AS SMALLINT) AS b_i16,
              b::bigint AS b_i64,
              d::tinyint AS d_i8,
              -- string/binary
              CAST(a AS CHAR) AS a_char,
              CAST(b AS VARCHAR) AS b_varchar,
              c::blob AS c_blob,
              c::bytes AS c_bytes,
              c::VARBINARY AS c_varbinary,
              CAST(d AS CHARACTER VARYING) AS d_charvar,
            FROM df
            """
        )
    assert res.schema == {
        "a_f64": pl.Float64,
        "a_f32": pl.Float32,
        "b_i8": pl.Int8,
        "b_i16": pl.Int16,
        "b_i64": pl.Int64,
        "d_i8": pl.Int8,
        "a_char": pl.String,
        "b_varchar": pl.String,
        "c_blob": pl.Binary,
        "c_bytes": pl.Binary,
        "c_varbinary": pl.Binary,
        "d_charvar": pl.String,
    }
    assert res.rows() == [
        (1.0, 1.0, 1, 1, 1, 1, "1", "1.1", b"a", b"a", b"a", "true"),
        (2.0, 2.0, 2, 2, 2, 0, "2", "2.2", b"b", b"b", b"b", "false"),
        (3.0, 3.0, 3, 3, 3, 1, "3", "3.3", b"c", b"c", b"c", "true"),
        (4.0, 4.0, 4, 4, 4, 0, "4", "4.4", b"d", b"d", b"d", "false"),
        (5.0, 5.0, 5, 5, 5, 1, "5", "5.5", b"e", b"e", b"e", "true"),
    ]

    with pytest.raises(ComputeError, match="unsupported use of FORMAT in CAST"):
        pl.SQLContext(df=df, eager_execution=True).execute(
            "SELECT CAST(a AS STRING FORMAT 'HEX') FROM df"
        )
