from __future__ import annotations

import pytest

import polars as pl
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
    (pl.Null(), "null", "null", 0),
    (pl.Boolean(), "boolean", "bool", 1),
    (pl.String(), "string", "str", None),
    (pl.Binary(), "binary", "binary", None),
    (pl.Int8(), "int", "i8", 8),
    (pl.Int16(), "int", "i16", 16),
    (pl.Int32(), "int", "i32", 32),
    (pl.Int64(), "int", "i64", 64),
    (pl.Int128(), "int", "i128", 128),
    (pl.UInt8(), "uint", "u8", 8),
    (pl.UInt16(), "uint", "u16", 16),
    (pl.UInt32(), "uint", "u32", 32),
    (pl.UInt64(), "uint", "u64", 64),
    (pl.Float32(), "float", "f32", 32),
    (pl.Float64(), "float", "f64", 64),
    (pl.Decimal(scale=4), "decimal", "decimal[*,4]", 128),
    (pl.Decimal(scale=12), "decimal", "decimal[*,12]", 128),
    (pl.Categorical(), "categorical", "cat", 32),
    (pl.Enum([]), "enum", "enum", 32),
    (pl.Enum(["a", "b"]), "enum", "enum", 32),
    (pl.List(pl.Null()), "list", "list[null]", None),
    (pl.List(pl.Array(pl.Int8, 5)), "list", "list[array[i8, 5]]", None),
    (
        pl.List(pl.Array(pl.Struct({"x": pl.Int8, "y": pl.String}), 5)),
        "list",
        "list[array[struct[2], 5]]",
        None,
    ),
    (pl.Array(pl.Null(), 2), "array", "array[null, 2]", 0),
    (pl.Array(pl.List(pl.Int8), 2), "array", "array[list[i8], 2]", None),
    (
        pl.Array(pl.List(pl.Struct({"x": pl.Int8, "y": pl.String})), 2),
        "array",
        "array[list[struct[2]], 2]",
        None,
    ),
    (pl.Struct({"x": pl.Null()}), "struct", "struct[1]", 0),
    (pl.Struct({"x": pl.List(pl.Int8)}), "struct", "struct[1]", None),
    (pl.Date, "date", "date", 32),
    (pl.Datetime, "datetime", "datetime[μs]", 64),
    (pl.Datetime("ns"), "datetime", "datetime[ns]", 64),
    (pl.Datetime("ms"), "datetime", "datetime[ms]", 64),
    (pl.Duration, "duration", "duration[μs]", 64),
    (pl.Duration("ns"), "duration", "duration[ns]", 64),
    (pl.Duration("ms"), "duration", "duration[ms]", 64),
    (pl.Time(), "time", "time", 64),
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
    for dtype, dtype_tag, _, _ in DTYPES:
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
    for dtype, dtype_tag, _, _ in DTYPES:
        dtype_expr = dtype.to_dtype_expr()
        expr = getattr(dtype_expr.int, fn)()
        expected = dtype_tag == fn_tag
        assert pl.select(expr).to_series().item() == expected


def test_inner_dtype() -> None:
    with pytest.raises(pl.exceptions.InvalidOperationError):
        pl.Struct({"x": pl.Int8}).to_dtype_expr().inner_dtype().collect_dtype({})

    arr_dtype = pl.Array(pl.String, 2)
    assert (
        pl.select(arr_dtype.to_dtype_expr().inner_dtype() == pl.String)
        .to_series()
        .item()
    )
    assert (
        pl.select(arr_dtype.to_dtype_expr().arr.inner_dtype() == pl.String)
        .to_series()
        .item()
    )
    with pytest.raises(pl.exceptions.SchemaError):
        arr_dtype.to_dtype_expr().list.inner_dtype().collect_dtype({})

    list_dtype = pl.List(pl.String).to_dtype_expr()
    assert pl.select(list_dtype.inner_dtype() == pl.String).to_series().item()
    assert pl.select(list_dtype.list.inner_dtype() == pl.String).to_series().item()
    with pytest.raises(pl.exceptions.SchemaError):
        list_dtype.arr.inner_dtype().collect_dtype({})


def test_to_string() -> None:
    for dtype, _, dtype_str, _ in DTYPES:
        assert (
            pl.select(dtype.to_dtype_expr().to_string()).to_series().item() == dtype_str
        )


def test_element_bitsize() -> None:
    for dtype, _, _, bitsize in DTYPES:
        if bitsize is None:
            with pytest.raises(pl.exceptions.InvalidOperationError):
                pl.select(dtype.to_dtype_expr().element_bitsize())
        else:
            assert (
                pl.select(dtype.to_dtype_expr().element_bitsize()).to_series().item()
                == bitsize
            )
