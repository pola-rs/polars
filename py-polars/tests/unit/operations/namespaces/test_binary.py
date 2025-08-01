from __future__ import annotations

import random
import struct
from datetime import date, datetime, time, timedelta
from typing import TYPE_CHECKING

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from polars._typing import PolarsDataType, SizeUnit, TransferEncoding


def test_binary_conversions() -> None:
    df = pl.DataFrame({"blob": [b"abc", None, b"cde"]}).with_columns(
        pl.col("blob").cast(pl.String).alias("decoded_blob")
    )

    assert df.to_dict(as_series=False) == {
        "blob": [b"abc", None, b"cde"],
        "decoded_blob": ["abc", None, "cde"],
    }
    assert df[0, 0] == b"abc"
    assert df[1, 0] is None
    assert df.dtypes == [pl.Binary, pl.String]


def test_contains() -> None:
    df = pl.DataFrame(
        data=[
            (1, b"some * * text"),
            (2, b"(with) special\n * chars"),
            (3, b"**etc...?$"),
            (4, None),
        ],
        schema=["idx", "bin"],
        orient="row",
    )
    for pattern, expected in (
        (b"e * ", [True, False, False, None]),
        (b"text", [True, False, False, None]),
        (b"special", [False, True, False, None]),
        (b"", [True, True, True, None]),
        (b"qwe", [False, False, False, None]),
    ):
        # series
        assert expected == df["bin"].bin.contains(pattern).to_list()
        # frame select
        assert (
            expected == df.select(pl.col("bin").bin.contains(pattern))["bin"].to_list()
        )
        # frame filter
        assert sum(e for e in expected if e is True) == len(
            df.filter(pl.col("bin").bin.contains(pattern))
        )


def test_contains_with_expr() -> None:
    df = pl.DataFrame(
        {
            "bin": [b"some * * text", b"(with) special\n * chars", b"**etc...?$", None],
            "lit1": [b"e * ", b"", b"qwe", b"None"],
            "lit2": [None, b"special\n", b"?!", None],
        }
    )

    assert df.select(
        pl.col("bin").bin.contains(pl.col("lit1")).alias("contains_1"),
        pl.col("bin").bin.contains(pl.col("lit2")).alias("contains_2"),
        pl.col("bin").bin.contains(pl.lit(None)).alias("contains_3"),
    ).to_dict(as_series=False) == {
        "contains_1": [True, True, False, None],
        "contains_2": [None, True, False, None],
        "contains_3": [None, None, None, None],
    }


def test_starts_ends_with() -> None:
    assert pl.DataFrame(
        {
            "a": [b"hamburger", b"nuts", b"lollypop", None],
            "end": [b"ger", b"tg", None, b"anything"],
            "start": [b"ha", b"nga", None, b"anything"],
        }
    ).select(
        pl.col("a").bin.ends_with(b"pop").alias("end_lit"),
        pl.col("a").bin.ends_with(pl.lit(None)).alias("end_none"),
        pl.col("a").bin.ends_with(pl.col("end")).alias("end_expr"),
        pl.col("a").bin.starts_with(b"ham").alias("start_lit"),
        pl.col("a").bin.ends_with(pl.lit(None)).alias("start_none"),
        pl.col("a").bin.starts_with(pl.col("start")).alias("start_expr"),
    ).to_dict(as_series=False) == {
        "end_lit": [False, False, True, None],
        "end_none": [None, None, None, None],
        "end_expr": [True, False, None, None],
        "start_lit": [True, False, False, None],
        "start_none": [None, None, None, None],
        "start_expr": [True, False, None, None],
    }


def test_base64_encode() -> None:
    df = pl.DataFrame({"data": [b"asd", b"qwe"]})

    assert df["data"].bin.encode("base64").to_list() == ["YXNk", "cXdl"]


def test_base64_decode() -> None:
    df = pl.DataFrame({"data": [b"YXNk", b"cXdl"]})

    assert df["data"].bin.decode("base64").to_list() == [b"asd", b"qwe"]


def test_hex_encode() -> None:
    df = pl.DataFrame({"data": [b"asd", b"qwe"]})

    assert df["data"].bin.encode("hex").to_list() == ["617364", "717765"]


def test_hex_decode() -> None:
    df = pl.DataFrame({"data": [b"617364", b"717765"]})

    assert df["data"].bin.decode("hex").to_list() == [b"asd", b"qwe"]


@pytest.mark.parametrize(
    "encoding",
    ["hex", "base64"],
)
def test_compare_encode_between_lazy_and_eager_6814(encoding: TransferEncoding) -> None:
    df = pl.DataFrame({"x": [b"aa", b"bb", b"cc"]})
    expr = pl.col("x").bin.encode(encoding)

    result_eager = df.select(expr)
    dtype = result_eager["x"].dtype

    result_lazy = df.lazy().select(expr).select(pl.col(dtype)).collect()
    assert_frame_equal(result_eager, result_lazy)


@pytest.mark.parametrize(
    "encoding",
    ["hex", "base64"],
)
def test_compare_decode_between_lazy_and_eager_6814(encoding: TransferEncoding) -> None:
    df = pl.DataFrame({"x": [b"d3d3", b"abcd", b"1234"]})
    expr = pl.col("x").bin.decode(encoding)

    result_eager = df.select(expr)
    dtype = result_eager["x"].dtype

    result_lazy = df.lazy().select(expr).select(pl.col(dtype)).collect()
    assert_frame_equal(result_eager, result_lazy)


@pytest.mark.parametrize(
    ("sz", "unit", "expected"),
    [(128, "b", 128), (512, "kb", 0.5), (131072, "mb", 0.125)],
)
def test_binary_size(sz: int, unit: SizeUnit, expected: int | float) -> None:
    df = pl.DataFrame({"data": [b"\x00" * sz]}, schema={"data": pl.Binary})
    for sz in (
        df.select(sz=pl.col("data").bin.size(unit)).item(),  # expr
        df["data"].bin.size(unit).item(),  # series
    ):
        assert sz == expected


@pytest.mark.parametrize(
    ("dtype", "type_size", "struct_type"),
    [
        (pl.Int8, 1, "b"),
        (pl.UInt8, 1, "B"),
        (pl.Int16, 2, "h"),
        (pl.UInt16, 2, "H"),
        (pl.Int32, 4, "i"),
        (pl.UInt32, 4, "I"),
        (pl.Int64, 8, "q"),
        (pl.UInt64, 8, "Q"),
        (pl.Float32, 4, "f"),
        (pl.Float64, 8, "d"),
    ],
)
def test_reinterpret(
    dtype: pl.DataType,
    type_size: int,
    struct_type: str,
) -> None:
    # Make test reproducible
    random.seed(42)

    byte_arr = [random.randbytes(type_size) for _ in range(3)]
    df = pl.DataFrame({"x": byte_arr})

    for endianness in ["little", "big"]:
        # So that mypy doesn't complain
        struct_endianness = "<" if endianness == "little" else ">"
        expected = [
            struct.unpack_from(f"{struct_endianness}{struct_type}", elem_bytes)[0]
            for elem_bytes in byte_arr
        ]
        expected_df = pl.DataFrame({"x": expected}, schema={"x": dtype})

        result = df.select(
            pl.col("x").bin.reinterpret(dtype=dtype, endianness=endianness)  # type: ignore[arg-type]
        )

        assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    ("dtype", "inner_type_size", "struct_type"),
    [
        (pl.Array(pl.Int8, 3), 1, "b"),
        (pl.Array(pl.UInt8, 3), 1, "B"),
        (pl.Array(pl.Int16, 3), 2, "h"),
        (pl.Array(pl.UInt16, 3), 2, "H"),
        (pl.Array(pl.Int32, 3), 4, "i"),
        (pl.Array(pl.UInt32, 3), 4, "I"),
        (pl.Array(pl.Int64, 3), 8, "q"),
        (pl.Array(pl.UInt64, 3), 8, "Q"),
        (pl.Array(pl.Float32, 3), 4, "f"),
        (pl.Array(pl.Float64, 3), 8, "d"),
    ],
)
def test_reinterpret_to_array_numeric_types(
    dtype: pl.Array,
    inner_type_size: int,
    struct_type: str,
) -> None:
    # Make test reproducible
    random.seed(42)

    type_size = inner_type_size
    shape = dtype.shape
    if isinstance(shape, int):
        shape = (shape,)
    for dim_size in dtype.shape:
        type_size *= dim_size

    byte_arr = [random.randbytes(type_size) for _ in range(3)]
    df = pl.DataFrame({"x": byte_arr}, orient="row")

    for endianness in ["little", "big"]:
        result = df.select(
            pl.col("x").bin.reinterpret(dtype=dtype, endianness=endianness)  # type: ignore[arg-type]
        )

        # So that mypy doesn't complain
        struct_endianness = "<" if endianness == "little" else ">"
        expected = []
        for elem_bytes in byte_arr:
            vals = [
                struct.unpack_from(
                    f"{struct_endianness}{struct_type}",
                    elem_bytes[idx : idx + inner_type_size],
                )[0]
                for idx in range(0, type_size, inner_type_size)
            ]
            if len(shape) > 1:
                vals = np.reshape(vals, shape).tolist()
            expected.append(vals)
        expected_df = pl.DataFrame({"x": expected}, schema={"x": dtype})

        assert_frame_equal(result, expected_df)


@pytest.mark.parametrize(
    ("dtype", "binary_value", "expected_values"),
    [
        (pl.Date(), b"\x06\x00\x00\x00", [date(1970, 1, 7)]),
        (
            pl.Datetime(),
            b"\x40\xb6\xfd\xe3\x7c\x00\x00\x00",
            [datetime(1970, 1, 7, 5, 0, 1)],
        ),
        (
            pl.Duration(),
            b"\x03\x00\x00\x00\x00\x00\x00\x00",
            [timedelta(microseconds=3)],
        ),
        (
            pl.Time(),
            b"\x58\x1b\x00\x00\x00\x00\x00\x00",
            [time(microsecond=7)],
        ),
        (
            pl.Int128(),
            b"\x06\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00",
            [6],
        ),
    ],
)
def test_reinterpret_to_additional_types(
    dtype: PolarsDataType, binary_value: bytes, expected_values: list[object]
) -> None:
    series = pl.Series([binary_value])

    # Direct conversion:
    result = series.bin.reinterpret(dtype=dtype, endianness="little")
    assert_series_equal(result, pl.Series(expected_values, dtype=dtype))

    # Array conversion:
    dtype = pl.Array(dtype, 1)
    result = series.bin.reinterpret(dtype=dtype, endianness="little")
    assert_series_equal(result, pl.Series([expected_values], dtype=dtype))


def test_reinterpret_to_array_resulting_in_nulls() -> None:
    series = pl.Series([None, b"short", b"justrite", None, b"waytoolong"])
    as_bin = series.bin.reinterpret(dtype=pl.Array(pl.UInt32(), 2), endianness="little")
    assert as_bin.to_list() == [None, None, [0x7473756A, 0x65746972], None, None]
    as_bin = series.bin.reinterpret(dtype=pl.Array(pl.UInt32(), 2), endianness="big")
    assert as_bin.to_list() == [None, None, [0x6A757374, 0x72697465], None, None]


def test_reinterpret_to_n_dimensional_array() -> None:
    series = pl.Series([b"abcd"])
    for endianness in ["big", "little"]:
        with pytest.raises(
            InvalidOperationError,
            match="reinterpret to a linear Array, and then use reshape",
        ):
            series.bin.reinterpret(
                dtype=pl.Array(pl.UInt32(), (2, 2)),
                endianness=endianness,  # type: ignore[arg-type]
            )


def test_reinterpret_to_zero_length_array() -> None:
    arr_dtype = pl.Array(pl.UInt8, 0)
    result = pl.Series([b"", b""]).bin.reinterpret(dtype=arr_dtype)
    assert_series_equal(result, pl.Series([[], []], dtype=arr_dtype))


@given(
    value1=st.integers(0, 2**63),
    value2=st.binary(min_size=0, max_size=7),
    value3=st.integers(0, 2**63),
)
def test_reinterpret_to_array_different_alignment(
    value1: int, value2: bytes, value3: int
) -> None:
    series = pl.Series([struct.pack("<Q", value1), value2, struct.pack("<Q", value3)])
    arr_dtype = pl.Array(pl.UInt64, 1)
    as_uint64 = series.bin.reinterpret(dtype=arr_dtype, endianness="little")
    assert_series_equal(
        pl.Series([[value1], None, [value3]], dtype=arr_dtype), as_uint64
    )


@pytest.mark.parametrize(
    "bad_dtype",
    [
        pl.Array(pl.Array(pl.UInt8, 1), 1),
        pl.String(),
        pl.Array(pl.List(pl.UInt8()), 1),
        pl.Array(pl.Null(), 1),
        pl.Array(pl.Boolean(), 1),
    ],
)
def test_reinterpret_unsupported(bad_dtype: pl.DataType) -> None:
    series = pl.Series([b"12345678"])
    lazy_df = pl.DataFrame({"s": series}).lazy()
    expected = "cannot reinterpret binary to dtype.*Only numeric or temporal dtype.*"
    for endianness in ["little", "big"]:
        with pytest.raises(InvalidOperationError, match=expected):
            series.bin.reinterpret(dtype=bad_dtype, endianness=endianness)  # type: ignore[arg-type]
        with pytest.raises(InvalidOperationError, match=expected):
            lazy_df.select(
                pl.col("s").bin.reinterpret(dtype=bad_dtype, endianness=endianness)  # type: ignore[arg-type]
            ).collect_schema()


@pytest.mark.parametrize(
    ("dtype", "type_size"),
    [
        (pl.Int128, 16),
    ],
)
def test_reinterpret_int(
    dtype: pl.DataType,
    type_size: int,
) -> None:
    # Function used for testing integers that `struct` or `numpy`
    # doesn't support parsing from bytes.
    # Rather than creating bytes directly, create integer and view it as bytes
    is_signed = dtype.is_signed_integer()

    if is_signed:
        min_val = -(2 ** (type_size - 1))
        max_val = 2 ** (type_size - 1) - 1
    else:
        min_val = 0
        max_val = 2**type_size - 1

    # Make test reproducible
    random.seed(42)

    expected = [random.randint(min_val, max_val) for _ in range(3)]
    expected_df = pl.DataFrame({"x": expected}, schema={"x": dtype})

    for endianness in ["little", "big"]:
        byte_arr = [
            val.to_bytes(type_size, byteorder=endianness, signed=is_signed)  # type: ignore[arg-type]
            for val in expected
        ]
        df = pl.DataFrame({"x": byte_arr})

        result = df.select(
            pl.col("x").bin.reinterpret(dtype=dtype, endianness=endianness)  # type: ignore[arg-type]
        )

        assert_frame_equal(result, expected_df)


def test_reinterpret_invalid() -> None:
    # Fails because buffer has more than 4 bytes
    df = pl.DataFrame({"x": [b"d3d3a"]})
    print(struct.unpack_from("<i", b"d3d3a"))
    assert_frame_equal(
        df.select(pl.col("x").bin.reinterpret(dtype=pl.Int32)),
        pl.DataFrame({"x": [None]}, schema={"x": pl.Int32}),
    )

    # Fails because buffer has less than 4 bytes
    df = pl.DataFrame({"x": [b"d3"]})
    print(df.select(pl.col("x").bin.reinterpret(dtype=pl.Int32)))
    assert_frame_equal(
        df.select(pl.col("x").bin.reinterpret(dtype=pl.Int32)),
        pl.DataFrame({"x": [None]}, schema={"x": pl.Int32}),
    )

    # Fails because dtype is invalid
    with pytest.raises(pl.exceptions.InvalidOperationError):
        df.select(pl.col("x").bin.reinterpret(dtype=pl.String))


@pytest.mark.parametrize("func", ["contains", "starts_with", "ends_with"])
def test_bin_contains_unequal_lengths_22018(func: str) -> None:
    s = pl.Series("a", [b"a", b"xyz"], pl.Binary).bin
    f = getattr(s, func)
    with pytest.raises(pl.exceptions.ShapeError):
        f(pl.Series([b"x", b"y", b"z"]))
