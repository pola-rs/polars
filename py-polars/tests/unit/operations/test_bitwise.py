from __future__ import annotations

import typing
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.conftest import INTEGER_DTYPES

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch


@pytest.mark.parametrize("op", ["and_", "or_"])
def test_bitwise_integral_schema(op: str) -> None:
    df = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
    q = df.select(getattr(pl.col("a"), op)(pl.col("b")))
    assert q.collect_schema()["a"] == df.collect_schema()["a"]


@pytest.mark.parametrize("op", ["and_", "or_", "xor"])
def test_bitwise_single_null_value_schema(op: str) -> None:
    df = pl.DataFrame({"a": [True, True]})
    q = df.select(getattr(pl.col("a"), op)(None))
    result_schema = q.collect_schema()
    assert result_schema.len() == 1
    assert "a" in result_schema


def leading_zeros(v: int | None, nb: int) -> int | None:
    if v is None:
        return None

    b = bin(v)[2:]
    blen = len(b) - len(b.lstrip("0"))
    if blen == len(b):
        return nb
    else:
        return nb - len(b) + blen


def leading_ones(v: int | None, nb: int) -> int | None:
    if v is None:
        return None

    b = bin(v)[2:]
    if len(b) < nb:
        return 0
    else:
        return len(b) - len(b.lstrip("1"))


def trailing_zeros(v: int | None, nb: int) -> int | None:
    if v is None:
        return None

    b = bin(v)[2:]
    blen = len(b) - len(b.rstrip("0"))
    if blen == len(b):
        return nb
    else:
        return blen


def trailing_ones(v: int | None) -> int | None:
    if v is None:
        return None

    b = bin(v)[2:]
    return len(b) - len(b.rstrip("1"))


@pytest.mark.parametrize(
    "value",
    [
        0x00,
        0x01,
        0xFCEF_0123,
        0xFFFF_FFFF,
        0xFFF0_FFE1_ABCD_EF01,
        0xAAAA_AAAA_AAAA_AAAA,
        None,
    ],
)
@pytest.mark.parametrize("dtype", [*INTEGER_DTYPES, pl.Boolean])
@typing.no_type_check
def test_bit_counts(value: int, dtype: pl.DataType) -> None:
    bitsize = 8
    if "Boolean" in str(dtype):
        bitsize = 1
    if "16" in str(dtype):
        bitsize = 16
    elif "32" in str(dtype):
        bitsize = 32
    elif "64" in str(dtype):
        bitsize = 64
    elif "128" in str(dtype):
        bitsize = 128

    if bitsize == 1 and value is not None:
        value = value & 1 != 0

        co = 1 if value else 0
        cz = 0 if value else 1
    elif value is not None:
        value = value & ((1 << bitsize) - 1)

        if dtype.is_signed_integer() and value >> (bitsize - 1) > 0:
            value = value - pow(2, bitsize - 1)

        co = value.bit_count()
        cz = bitsize - co
    else:
        co = None
        cz = None

    assert_series_equal(
        pl.Series("a", [value], dtype).bitwise_count_ones(),
        pl.Series("a", [co], pl.UInt32),
    )
    assert_series_equal(
        pl.Series("a", [value], dtype).bitwise_count_zeros(),
        pl.Series("a", [cz], pl.UInt32),
    )
    assert_series_equal(
        pl.Series("a", [value], dtype).bitwise_leading_ones(),
        pl.Series("a", [leading_ones(value, bitsize)], pl.UInt32),
    )
    assert_series_equal(
        pl.Series("a", [value], dtype).bitwise_leading_zeros(),
        pl.Series("a", [leading_zeros(value, bitsize)], pl.UInt32),
    )
    assert_series_equal(
        pl.Series("a", [value], dtype).bitwise_trailing_ones(),
        pl.Series("a", [trailing_ones(value)], pl.UInt32),
    )
    assert_series_equal(
        pl.Series("a", [value], dtype).bitwise_trailing_zeros(),
        pl.Series("a", [trailing_zeros(value, bitsize)], pl.UInt32),
    )


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_bit_aggregations(dtype: pl.DataType) -> None:
    s = pl.Series("a", [0x74, 0x1C, 0x05], dtype)

    df = s.to_frame().select(
        AND=pl.col.a.bitwise_and(),
        OR=pl.col.a.bitwise_or(),
        XOR=pl.col.a.bitwise_xor(),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            [
                pl.Series("AND", [0x04], dtype),
                pl.Series("OR", [0x7D], dtype),
                pl.Series("XOR", [0x6D], dtype),
            ]
        ),
    )


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_bit_aggregations_lazy_no_nulls(dtype: pl.DataType) -> None:
    s = pl.Series("a", [0x74, 0x1C, 0x05], dtype)

    lf = s.to_frame().lazy()

    out = lf.select(
        AND=pl.col.a.bitwise_and(),
        OR=pl.col.a.bitwise_or(),
        XOR=pl.col.a.bitwise_xor(),
    ).collect()

    assert_frame_equal(
        out,
        pl.DataFrame(
            [
                pl.Series("AND", [0x04], dtype),
                pl.Series("OR", [0x7D], dtype),
                pl.Series("XOR", [0x6D], dtype),
            ]
        ),
    )


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_bit_aggregations_lazy_some_nulls(dtype: pl.DataType) -> None:
    s = pl.Series("a", [0x74, None, 0x1C, None, 0x05], dtype)
    out = (
        s.to_frame()
        .lazy()
        .select(
            AND=pl.col.a.bitwise_and(),
            OR=pl.col.a.bitwise_or(),
            XOR=pl.col.a.bitwise_xor(),
        )
        .collect()
    )

    assert_frame_equal(
        out,
        pl.DataFrame(
            [
                pl.Series("AND", [0x04], dtype),
                pl.Series("OR", [0x7D], dtype),
                pl.Series("XOR", [0x6D], dtype),
            ]
        ),
    )


@pytest.mark.parametrize(
    "expr",
    [pl.col("a").bitwise_and(), pl.col("a").bitwise_or(), pl.col("a").bitwise_xor()],
)
def test_bit_aggregations_lazy_all_nulls(expr: pl.Expr) -> None:
    dtype = pl.Int64
    s = pl.Series("a", [None, None, None], dtype)
    out = s.to_frame().lazy().select(OUT=expr).collect()

    assert_frame_equal(
        out,
        pl.DataFrame([pl.Series("OUT", [None], dtype)]),
    )


@pytest.mark.parametrize("dtype", INTEGER_DTYPES)
def test_bit_group_by(dtype: pl.DataType) -> None:
    df = pl.DataFrame(
        [
            pl.Series("g", [4, 1, 1, 2, 3, 2, 4, 4], pl.Int8),
            pl.Series("a", [0x03, 0x74, 0x1C, 0x05, None, 0x70, 0x01, None], dtype),
        ]
    )

    df = df.group_by("g").agg(
        AND=pl.col.a.bitwise_and(),
        OR=pl.col.a.bitwise_or(),
        XOR=pl.col.a.bitwise_xor(),
    )

    assert_frame_equal(
        df,
        pl.DataFrame(
            [
                pl.Series("g", [1, 2, 3, 4], pl.Int8),
                pl.Series("AND", [0x74 & 0x1C, 0x05 & 0x70, None, 0x01], dtype),
                pl.Series("OR", [0x74 | 0x1C, 0x05 | 0x70, None, 0x03], dtype),
                pl.Series("XOR", [0x74 ^ 0x1C, 0x05 ^ 0x70, None, 0x02], dtype),
            ]
        ),
        check_row_order=False,
    )


def test_scalar_bitwise_xor() -> None:
    df = pl.select(
        pl.repeat(pl.lit(0x80, pl.UInt8), i).bitwise_xor().alias(f"l{i}")
        for i in range(5)
    ).transpose()

    assert_series_equal(
        df.to_series(),
        pl.Series("x", [None, 0x80, 0x00, 0x80, 0x00], pl.UInt8),
        check_names=False,
    )


@pytest.mark.parametrize(
    ("expr", "result"),
    [
        (pl.all().bitwise_and(), [True, False, False, True, False, None]),
        (pl.all().bitwise_or(), [True, True, False, True, False, None]),
        (pl.all().bitwise_xor(), [False, True, False, True, False, None]),
    ],
)
def test_bool_bitwise_with_nulls_23314(expr: pl.Expr, result: list[bool]) -> None:
    df = pl.DataFrame(
        {
            "a": [True, True, None],
            "b": [True, False, None],
            "c": [False, False, None],
            "d": [True, None, None],
            "e": [False, None, None],
            "f": [None, None, None],
        },
        schema_overrides={"f": pl.Boolean},
    )
    columns = ["a", "b", "c", "d", "e", "f"]
    out = df.select(expr)
    expected = pl.DataFrame(
        [result], orient="row", schema=columns, schema_overrides={"f": pl.Boolean}
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("expr", "result"),
    [
        (pl.all().bitwise_and(), [True, False, False, False, False, None]),
        (pl.all().bitwise_or(), [True, True, True, False, True, None]),
        (pl.all().bitwise_xor(), [True, False, True, False, True, None]),
    ],
)
def test_bitwise_boolean(expr: pl.Expr, result: list[bool]) -> None:
    lf = pl.LazyFrame(
        {
            "a": [True, True, True],
            "b": [True, False, True],
            "c": [False, True, False],
            "d": [False, False, False],
            "x": [True, False, None],
            "z": [None, None, None],
        },
        schema_overrides={"z": pl.Boolean},
    )

    columns = ["a", "b", "c", "d", "x", "z"]
    expected = pl.DataFrame(
        [result], orient="row", schema=columns, schema_overrides={"z": pl.Boolean}
    )
    out = lf.select(expr).collect()
    assert_frame_equal(out, expected)


# Although there is no way to deterministically trigger the `evict` path
# in the code, the below test will do so with high likelihood
# POLARS_MAX_THREADS is only honored when tested in isolation, see issue #22070
def test_bitwise_boolean_evict_path(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_MAX_THREADS", "1")
    plmonkeypatch.setenv("POLARS_HOT_TABLE_SIZE", "2")
    n_groups = 100
    group_size_pairs = 10
    group_size = group_size_pairs * 2

    col_a = list(range(group_size)) * n_groups
    col_b = [True, False] * group_size_pairs * n_groups
    df = pl.DataFrame({"a": pl.Series(col_a), "b": pl.Series(col_b)}).sort("a")

    out = (
        df.lazy()
        .group_by("a")
        .agg(
            [
                pl.col("b").bitwise_and().alias("bitwise_and"),
                pl.col("b").bitwise_or().alias("bitwise_or"),
                pl.col("b").bitwise_xor().alias("bitwise_xor"),
            ]
        )
        .sort("a")
        .collect()
    )
    expected = pl.DataFrame(
        {
            "a": list(range(group_size)),
            "bitwise_and": [True, False] * group_size_pairs,
            "bitwise_or": [True, False] * group_size_pairs,
            "bitwise_xor": [n_groups % 2 == 1, False] * group_size_pairs,
        }
    )
    assert_frame_equal(out, expected)


def test_bitwise_in_group_by() -> None:
    df = pl.DataFrame(
        {
            "a": [
                111,
                222,
                111,
                222,
                333,
                333,
                999,
                888,
                999,
            ],
        }
    )

    assert_frame_equal(
        df.group_by(pl.lit(1))
        .agg(
            bwand=pl.col.a.bitwise_and(),
            bwor=pl.col.a.bitwise_or(),
            bwxor=pl.col.a.bitwise_xor(),
        )
        .drop("literal"),
        df.select(
            bwand=pl.col.a.bitwise_and(),
            bwor=pl.col.a.bitwise_or(),
            bwxor=pl.col.a.bitwise_xor(),
        ),
    )
