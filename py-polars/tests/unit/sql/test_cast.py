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
    SQLSyntaxError,
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

    # a column operand keeps the frame's null; a literal broadcasts to both rows
    for operand, rows in (("s", [expected, None]), (f"'{value}'", [expected] * 2)):
        for cast_op in (
            f"CAST({operand} AS {sql_type})",
            f"TRY_CAST({operand} AS {sql_type})",
            f"{operand}::{sql_type}",
        ):
            res = df.sql(f"SELECT {cast_op} AS x FROM self")
            assert_frame_equal(res, pl.DataFrame({"x": rows}, schema={"x": dtype}))


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


@pytest.mark.parametrize(
    "sql_type",
    ["STRUCT<y VARCHAR, x BIGINT>", "STRUCT<x BIGINT, y VARCHAR, z DOUBLE>"],
)
def test_cast_struct_fields_strict(sql_type: str) -> None:
    # CAST, like a strict `Series.cast`, rejects reordered or added struct fields
    df = pl.DataFrame({"s": [{"x": 1, "y": "a"}]})
    with pytest.raises(InvalidOperationError, match="failed in column 's'"):
        df.sql(f"SELECT CAST(s AS {sql_type}) AS s FROM self")


@pytest.mark.parametrize(
    ("sql_type", "dtype"),
    [
        # reorder and add a field
        (
            "STRUCT<y VARCHAR, x BIGINT, z DOUBLE>",
            pl.Struct({"y": pl.String, "x": pl.Int64, "z": pl.Float64}),
        ),
        # drop a field, narrow another and add one
        ("STRUCT<x INT, z DOUBLE>", pl.Struct({"x": pl.Int32, "z": pl.Float64})),
    ],
)
def test_try_cast_struct(sql_type: str, dtype: pl.DataType) -> None:
    # TRY_CAST matches struct fields by name; an added field is null
    df = pl.DataFrame({"s": [{"x": 1, "y": "a"}, {"x": 2, "y": None}]})
    res = df.sql(f"SELECT TRY_CAST(s AS {sql_type}) AS s FROM self")
    assert_frame_equal(res, df.select(pl.col("s").cast(dtype, strict=False)))
    assert res["s"].struct.field("z").null_count() == 2


@pytest.mark.parametrize(
    ("sql_type", "dtype"),
    [
        ("STRUCT<a INT, b VARCHAR>", pl.Struct({"a": pl.Int32, "b": pl.String})),
        (
            'STRUCT<a INT[], "My Field" STRUCT<c DATE>>',
            pl.Struct({"a": pl.List(pl.Int32), "My Field": pl.Struct({"c": pl.Date})}),
        ),
        ("MAP(VARCHAR, INT)", pl.Map(pl.String, pl.Int32)),
        (
            "MAP(VARCHAR, STRUCT<x DOUBLE>)",
            pl.Map(pl.String, pl.Struct({"x": pl.Float64})),
        ),
    ],
)
def test_struct_and_map_type_names(sql_type: str, dtype: pl.DataType) -> None:
    res = pl.sql(f"SELECT CAST(NULL AS {sql_type}) AS c", eager=True)
    assert res.schema == {"c": dtype}

    with pl.SQLContext() as ctx:
        ctx.execute(f"CREATE TABLE tbl (c {sql_type})")
        assert ctx.execute("SELECT * FROM tbl", eager=True).schema == {"c": dtype}


@pytest.mark.parametrize(
    ("sql", "exc", "error"),
    [
        (
            "SELECT CAST(NULL AS STRUCT) AS c",
            SQLSyntaxError,
            "STRUCT type requires at least one field",
        ),
        (
            "SELECT CAST(NULL AS STRUCT<INT>) AS c",
            SQLInterfaceError,
            "STRUCT fields must be named",
        ),
        (
            "SELECT CAST(NULL AS STRUCT<a INT, a VARCHAR>) AS c",
            SQLSyntaxError,
            "STRUCT field 'a' is given more than once",
        ),
        (
            "SELECT CAST(NULL AS STRUCT<a NOT_A_TYPE>) AS c",
            SQLInterfaceError,
            'datatype "NOT_A_TYPE" is not currently supported',
        ),
        # only a STRUCT (MAP) value or NULL casts to STRUCT (MAP)
        (
            "SELECT CAST('1' AS STRUCT<x INT>) AS c",
            SQLInterfaceError,
            "CAST from str to STRUCT is not supported",
        ),
        (
            "SELECT TRY_CAST('{}' AS MAP(VARCHAR, INT)) AS c",
            SQLInterfaceError,
            "TRY_CAST from str to MAP is not supported",
        ),
        # a literal is named by the type it becomes
        (
            "SELECT CAST(1 AS STRUCT<x INT>) AS c",
            SQLInterfaceError,
            "CAST from i32 to STRUCT is not supported",
        ),
        (
            "SELECT CAST(1 AS STRUCT<x INT>[]) AS c",
            SQLInterfaceError,
            r"CAST from i32 to STRUCT\[\] is not supported; only STRUCT\[\] values",
        ),
        # a nested mismatch says where it is
        (
            "SELECT CAST([1] AS STRUCT<x INT>[]) AS c",
            SQLInterfaceError,
            r"CAST from list\[i64\] to STRUCT\[\] is not supported: list element is i64; "
            r"only STRUCT values",
        ),
        (
            "SELECT CAST(CAST(NULL AS STRUCT<x INT>) AS STRUCT<x STRUCT<y INT>>) AS c",
            SQLInterfaceError,
            r"CAST from struct\[1\] to STRUCT is not supported: field 'x' is i32; "
            r"only STRUCT values",
        ),
        (
            "SELECT CAST(CAST(NULL AS MAP(VARCHAR, INT)) "
            "AS STRUCT<key STRUCT<x VARCHAR>, value INT>[]) AS c",
            SQLInterfaceError,
            r"CAST from map\[str, i32\] to STRUCT\[\] is not supported: "
            r"field 'key' of map entry is str; only STRUCT values",
        ),
    ],
)
def test_struct_and_map_type_name_errors(
    sql: str, exc: type[Exception], error: str
) -> None:
    with pytest.raises(exc, match=error):
        pl.sql(sql, eager=True)


@pytest.mark.parametrize(
    "cast_op",
    [
        # string-like sources
        "CAST(s AS STRUCT<x INT>)",
        "CAST(c AS STRUCT<x INT>)",
        "CAST(e AS MAP(VARCHAR, INT))",
        # other sources, including into a 1-field struct
        "CAST(1 AS STRUCT<x INT>)",
        "CAST(i AS STRUCT<x INT>)",
        "TRY_CAST(i AS STRUCT<x INT, y INT>)",
        "CAST(1.5 AS STRUCT<x INT>)",
        "CAST(d AS STRUCT<x DATE>)",
        "CAST(l AS STRUCT<x INT>)",
        "CAST(i AS MAP(VARCHAR, INT))",
        "CAST(CAST(NULL AS INT) AS STRUCT<x INT>)",
        # a struct is not a map, and the other way round
        "CAST(st AS MAP(VARCHAR, INT))",
        "CAST(m AS STRUCT<x INT>)",
        # at any depth: a struct field, a list element, a map value or key
        "CAST(st AS STRUCT<x STRUCT<y INT>>)",
        "TRY_CAST(ls AS STRUCT<y INT>[])",
        "CAST(m AS MAP(VARCHAR, STRUCT<y INT>))",
        "CAST(m AS STRUCT<key STRUCT<x VARCHAR>, value INT>[])",
        # a list target needs a list or map source
        "TRY_CAST(i AS STRUCT<x INT>[])",
        "CAST(CAST(NULL AS INT) AS STRUCT<x INT>[])",
        "CAST(st AS STRUCT<x INT>[])",
        "CAST(i AS MAP(VARCHAR, INT)[])",
        "CAST(st AS STRUCT<x STRUCT<y INT>[]>)",
        # windows
        "CAST(FIRST_VALUE(i) OVER () AS STRUCT<x INT>)",
        "CAST(SUM(i) OVER () + 1 AS STRUCT<x INT>)",
        # array literals and scalar subqueries
        "ARRAY[CAST('1' AS STRUCT<x INT>)]",
        "ARRAY[CAST(1 AS STRUCT<x INT>), '2']",
        "CAST((SELECT '1') AS STRUCT<x INT>)",
    ],
)
def test_cast_to_nested_errors(cast_op: str) -> None:
    df = pl.DataFrame(
        {
            "s": ["1"],
            "c": pl.Series(["1"], dtype=pl.Categorical),
            "e": pl.Series(["1"], dtype=pl.Enum(["1"])),
            "i": [1],
            "d": [date(2024, 1, 1)],
            "l": [[1]],
            "ls": [["1"]],
            "st": [{"x": 1}],
            "m": pl.Series(
                [[{"key": "a", "value": 1}]], dtype=pl.Map(pl.String, pl.Int32)
            ),
        }
    )
    with pytest.raises(
        SQLInterfaceError,
        match=r"(TRY_)?CAST from .+ is not supported(: .+)?; only (STRUCT|MAP)(\[\])* values",
    ):
        df.sql(f"SELECT {cast_op} AS r FROM self")


def test_cast_map_to_map() -> None:
    df = pl.DataFrame(
        {
            "m": pl.Series(
                [[{"key": "a", "value": 1}]], dtype=pl.Map(pl.String, pl.Int32)
            )
        }
    )
    res = df.sql("SELECT CAST(m AS MAP(VARCHAR, BIGINT)) AS m FROM self")
    assert_frame_equal(res, df.select(pl.col("m").cast(pl.Map(pl.String, pl.Int64))))


@pytest.mark.parametrize(
    ("cast_op", "column", "dtype"),
    [
        (
            "CAST(nst AS STRUCT<x STRUCT<y BIGINT>>)",
            "nst",
            pl.Struct({"x": pl.Struct({"y": pl.Int64})}),
        ),
        ("CAST(lst AS STRUCT<y BIGINT>[])", "lst", pl.List(pl.Struct({"y": pl.Int64}))),
        # a map is a list of key/value structs
        (
            "CAST(m AS STRUCT<key VARCHAR, value BIGINT>[])",
            "m",
            pl.List(pl.Struct({"key": pl.String, "value": pl.Int64})),
        ),
    ],
)
def test_cast_nested_structs(cast_op: str, column: str, dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        {
            "nst": [{"x": {"y": 1}}],
            "lst": [[{"y": 1}]],
            "m": pl.Series(
                [[{"key": "a", "value": 1}]], dtype=pl.Map(pl.String, pl.Int32)
            ),
        }
    )
    res = df.sql(f"SELECT {cast_op} AS r FROM self")
    assert_frame_equal(res, df.select(pl.col(column).cast(dtype).alias("r")))


def test_sql_expr_cast_string_to_struct_errors() -> None:
    # without a schema, an explicit string cast still identifies the source
    with pytest.raises(
        SQLInterfaceError, match="CAST from str to STRUCT is not supported"
    ):
        pl.sql_expr("CAST(CAST(s AS VARCHAR) AS STRUCT<x INT>)")
