from __future__ import annotations

import random
import struct
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars._typing import SizeUnit, TransferEncoding


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
