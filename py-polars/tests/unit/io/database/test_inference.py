from __future__ import annotations

import sqlite3
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.io.database._inference import _infer_dtype_from_database_typename

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import PolarsDataType


@pytest.mark.parametrize(
    ("value", "expected_dtype"),
    [
        # string types
        ("UTF16", pl.String),
        ("char(8)", pl.String),
        ("BPCHAR", pl.String),
        ("nchar[128]", pl.String),
        ("varchar", pl.String),
        ("CHARACTER VARYING(64)", pl.String),
        ("nvarchar(32)", pl.String),
        ("TEXT", pl.String),
        # array types
        ("float32[]", pl.List(pl.Float32)),
        ("double array", pl.List(pl.Float64)),
        ("array[bool]", pl.List(pl.Boolean)),
        ("array of nchar(8)", pl.List(pl.String)),
        ("array[array[int8]]", pl.List(pl.List(pl.Int64))),
        # numeric types
        ("numeric[10,5]", pl.Decimal(10, 5)),
        ("bigdecimal", pl.Decimal),
        ("decimal128(10,5)", pl.Decimal(10, 5)),
        ("double precision", pl.Float64),
        ("floating point", pl.Float64),
        ("numeric", pl.Float64),
        ("real", pl.Float64),
        ("boolean", pl.Boolean),
        ("tinyint", pl.Int8),
        ("smallint", pl.Int16),
        ("int", pl.Int64),
        ("int4", pl.Int32),
        ("int2", pl.Int16),
        ("int(16)", pl.Int16),
        ("ROWID", pl.UInt64),
        ("mediumint", pl.Int32),
        ("unsigned mediumint", pl.UInt32),
        ("cardinal_number", pl.UInt64),
        ("smallserial", pl.Int16),
        ("serial", pl.Int32),
        ("bigserial", pl.Int64),
        # temporal types
        ("timestamp(3)", pl.Datetime("ms")),
        ("timestamp(5)", pl.Datetime("us")),
        ("timestamp(7)", pl.Datetime("ns")),
        ("datetime without tz", pl.Datetime("us")),
        ("duration(2)", pl.Duration("ms")),
        ("interval", pl.Duration("us")),
        ("date", pl.Date),
        ("time", pl.Time),
        ("date32", pl.Date),
        ("time64", pl.Time),
        # binary types
        ("BYTEA", pl.Binary),
        ("BLOB", pl.Binary),
        # miscellaneous
        ("NULL", pl.Null),
    ],
)
def test_dtype_inference_from_string(
    value: str,
    expected_dtype: PolarsDataType,
) -> None:
    inferred_dtype = _infer_dtype_from_database_typename(value)
    assert inferred_dtype == expected_dtype  # type: ignore[operator]


@pytest.mark.parametrize(
    "value",
    [
        "FooType",
        "Unknown",
        "MISSING",
        "XML",  # note: we deliberately exclude "number" as it is ambiguous.
        "Number",  # (could refer to any size of int, float, or decimal dtype)
    ],
)
def test_dtype_inference_from_invalid_string(value: str) -> None:
    with pytest.raises(ValueError, match="cannot infer dtype"):
        _infer_dtype_from_database_typename(value)

    inferred_dtype = _infer_dtype_from_database_typename(
        value=value,
        raise_unmatched=False,
    )
    assert inferred_dtype is None


def test_infer_schema_length(tmp_sqlite_inference_db: Path) -> None:
    # note: first row of this test database contains only NULL values
    conn = sqlite3.connect(tmp_sqlite_inference_db)
    for infer_len in (2, 100, None):
        df = pl.read_database(
            connection=conn,
            query="SELECT * FROM test_data",
            infer_schema_length=infer_len,
        )
        assert df.schema == {"name": pl.String, "value": pl.Float64}

    with pytest.raises(
        ComputeError,
        match='could not append value: "foo" of type: str.*`infer_schema_length`',
    ):
        pl.read_database(
            connection=conn,
            query="SELECT * FROM test_data",
            infer_schema_length=1,
        )
