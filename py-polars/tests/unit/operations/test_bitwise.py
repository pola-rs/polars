from __future__ import annotations

import sys
import typing

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


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
@pytest.mark.parametrize(
    "dtype",
    [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Boolean,
    ],
)
@pytest.mark.skipif(sys.version_info < (3, 10), reason="bit_count introduced in 3.10")
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


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64],
)
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


@pytest.mark.parametrize(
    "dtype",
    [pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64],
)
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
