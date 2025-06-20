from __future__ import annotations
import pytest

import polars as pl
from polars._typing import PolarsDataType
from polars.testing import assert_frame_equal

c = pl.col


@pytest.mark.parametrize(
    ("e", "equiv"),
    [
        (
            c.a.map_batches(lambda x: str(x), pl.dtype_of("b")),
            c.a.map_batches(lambda x: str(x), pl.String),
        ),
        (
            c.a.replace_strict([1, 2, 3, 4, 5], "X", return_dtype=pl.dtype_of("b")),
            pl.repeat("X", pl.len()).alias("a"),
        ),
        (
            pl.int_range(1, 5, 1, dtype=pl.dtype_of("a")),
            pl.int_range(1, 5, 1, dtype=pl.Int64),
        ),
    ],
)
def test_expressions(e: pl.Expr, equiv: pl.Expr) -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["x", "y", "z", "w", "u"],
        }
    )

    assert_frame_equal(
        df.select(e),
        df.select(equiv),
    )


DTYPES = [
    (pl.Null(), "null"),
    (pl.Boolean(), "boolean"),
    (pl.String(), "string"),
    (pl.Binary(), "binary"),
    (pl.Int8(), "int"),
    (pl.Int16(), "int"),
    (pl.Int32(), "int"),
    (pl.Int64(), "int"),
    (pl.Int128(), "int"),
    (pl.UInt8(), "uint"),
    (pl.UInt16(), "uint"),
    (pl.UInt32(), "uint"),
    (pl.UInt64(), "uint"),
    (pl.Float32(), "float"),
    (pl.Float64(), "float"),
    (pl.Decimal(scale=4), "decimal"),
    (pl.Decimal(scale=12), "decimal"),
    (pl.Categorical(), "categorical"),
    (pl.Enum([]), "enum"),
    (pl.Enum(["a", "b"]), "enum"),
    (pl.List(pl.Null()), "list"),
    (pl.List(pl.Array(pl.Int8, 5)), "list"),
    (pl.List(pl.Array(pl.Struct({"x": pl.Int8, "y": pl.String}), 5)), "list"),
    (pl.Array(pl.Null(), 2), "array"),
    (pl.Array(pl.List(pl.Int8), 2), "array"),
    (
        pl.Array(pl.List(pl.Struct({"x": pl.Int8, "y": pl.String})), 2),
        "array",
    ),
    (pl.Struct({"x": pl.Null()}), "struct"),
    (pl.Struct({"x": pl.List(pl.Int8)}), "struct"),
]


@pytest.mark.parametrize(
    ("fn", "fn_tags"),
    [
        ("is_numeric", ["uint", "int", "float", "decimal"]),
        ("is_integer", ["uint", "int"]),
        ("is_float", ["float"]),
        ("is_decimal", ["decimal"]),
        ("is_categorical", ["categorical"]),
        ("is_enum", ["enum"]),
        ("is_nested", ["list", "array", "struct"]),
        ("is_list", ["list"]),
        ("is_array", ["array"]),
        ("is_struct", ["struct"]),
        ("is_temporal", ["date", "datetime", "duration", "time"]),
        ("is_datetime", ["datetime"]),
        ("is_duration", ["duration"]),
        ("is_object", ["object"]),
    ],
)
def test_classification(fn: str, fn_tags: list[str]) -> None:
    f = getattr(pl.DataTypeExpr, fn)
    for dtype, dtype_tag in DTYPES:
        dtype_expr = dtype.to_dtype_expr()
        expr = f(dtype_expr)
        expected = dtype_tag in fn_tags
        assert pl.select(expr).to_series().item() == expected


@pytest.mark.parametrize(
    ("fn", "fn_tag"),
    [
        ("is_signed", "int"),
        ("is_unsigned", "uint"),
    ],
)
def test_int_signed_classification(fn: str, fn_tag: str) -> None:
    for dtype, dtype_tag in DTYPES:
        dtype_expr = dtype.to_dtype_expr()
        expr = getattr(dtype_expr.int, fn)()
        expected = dtype_tag == fn_tag
        assert pl.select(expr).to_series().item() == expected
