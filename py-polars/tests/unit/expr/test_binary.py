from __future__ import annotations

import polars as pl
from polars.testing import assert_frame_equal


def test_binary_slice_basic() -> None:
    """Test basic binary slicing with positive offset and length."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd\xfc\xfb",
                b"\x10\x20\x30",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.slice(1, 3).alias("sliced"))
    expected = pl.DataFrame(
        {
            "sliced": [
                b"\x01\x02\x03",
                b"\xfe\xfd\xfc",
                b"\x20\x30",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_slice_negative_offset() -> None:
    """Test binary slicing with negative offset."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd\xfc\xfb",
                b"\x10\x20\x30",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.slice(-3, 2).alias("sliced"))
    expected = pl.DataFrame(
        {
            "sliced": [
                b"\x02\x03",
                b"\xfd\xfc",
                b"\x10\x20",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_slice_to_end() -> None:
    """Test binary slicing to end (no length specified)."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd\xfc\xfb",
                b"\x10\x20\x30",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.slice(2).alias("sliced"))
    expected = pl.DataFrame(
        {
            "sliced": [
                b"\x02\x03\x04",
                b"\xfd\xfc\xfb",
                b"\x30",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_slice_with_expression() -> None:
    """Test binary slicing with offset as expression."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd\xfc\xfb",
                b"\x10\x20\x30",
                None,
            ],
            "offset": [0, 1, 2, 0],
        }
    )

    result = df.select(pl.col("data").bin.slice(pl.col("offset"), 2).alias("sliced"))
    expected = pl.DataFrame(
        {
            "sliced": [
                b"\x00\x01",
                b"\xfe\xfd",
                b"\x30",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_slice_zero_length() -> None:
    """Test binary slicing with zero length."""
    df = pl.DataFrame({"data": [b"\x00\x01\x02\x03\x04"]})

    result = df.select(pl.col("data").bin.slice(1, 0).alias("sliced"))
    expected = pl.DataFrame({"sliced": [b""]})
    assert_frame_equal(result, expected)


def test_binary_slice_out_of_bounds() -> None:
    """Test binary slicing with out of bounds indices."""
    df = pl.DataFrame({"data": [b"\x00\x01\x02"]})

    # Offset beyond length
    result = df.select(pl.col("data").bin.slice(10, 2).alias("sliced"))
    expected = pl.DataFrame({"sliced": [b""]})
    assert_frame_equal(result, expected)

    # Length beyond available data
    result = df.select(pl.col("data").bin.slice(1, 100).alias("sliced"))
    expected = pl.DataFrame({"sliced": [b"\x01\x02"]})
    assert_frame_equal(result, expected)


def test_binary_head_basic() -> None:
    """Test basic binary head with positive n."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.head(3).alias("head"))
    expected = pl.DataFrame(
        {
            "head": [
                b"\x00\x01\x02",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_head_larger_than_data() -> None:
    """Test binary head with n larger than data length."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.head(10).alias("head"))
    expected = pl.DataFrame(
        {
            "head": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_head_negative() -> None:
    """Test binary head with negative n (all but last n)."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.head(-2).alias("head"))
    expected = pl.DataFrame(
        {
            "head": [
                b"\x00\x01\x02",
                b"\xff",
                b"",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_head_zero() -> None:
    """Test binary head with n=0."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.head(0).alias("head"))
    expected = pl.DataFrame({"head": [b"", b"", b"", None]})
    assert_frame_equal(result, expected)


def test_binary_head_with_expression() -> None:
    """Test binary head with n as expression."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ],
            "n": [2, 1, 1, 0],
        }
    )

    result = df.select(pl.col("data").bin.head(pl.col("n")).alias("head"))
    expected = pl.DataFrame(
        {
            "head": [
                b"\x00\x01",
                b"\xff",
                b"\x10",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_head_default() -> None:
    """Test binary head with default n=5."""
    df = pl.DataFrame({"data": [b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"]})

    result = df.select(pl.col("data").bin.head().alias("head"))
    expected = pl.DataFrame({"head": [b"\x00\x01\x02\x03\x04"]})
    assert_frame_equal(result, expected)


def test_binary_tail_basic() -> None:
    """Test basic binary tail with positive n."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.tail(3).alias("tail"))
    expected = pl.DataFrame(
        {
            "tail": [
                b"\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_tail_larger_than_data() -> None:
    """Test binary tail with n larger than data length."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.tail(10).alias("tail"))
    expected = pl.DataFrame(
        {
            "tail": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_tail_negative() -> None:
    """Test binary tail with negative n (all but first n)."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.tail(-2).alias("tail"))
    expected = pl.DataFrame(
        {
            "tail": [
                b"\x02\x03\x04",
                b"\xfd",
                b"",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_tail_zero() -> None:
    """Test binary tail with n=0."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ]
        }
    )

    result = df.select(pl.col("data").bin.tail(0).alias("tail"))
    expected = pl.DataFrame({"tail": [b"", b"", b"", None]})
    assert_frame_equal(result, expected)


def test_binary_tail_with_expression() -> None:
    """Test binary tail with n as expression."""
    df = pl.DataFrame(
        {
            "data": [
                b"\x00\x01\x02\x03\x04",
                b"\xff\xfe\xfd",
                b"\x10",
                None,
            ],
            "n": [2, 1, 1, 0],
        }
    )

    result = df.select(pl.col("data").bin.tail(pl.col("n")).alias("tail"))
    expected = pl.DataFrame(
        {
            "tail": [
                b"\x03\x04",
                b"\xfd",
                b"\x10",
                None,
            ]
        }
    )
    assert_frame_equal(result, expected)


def test_binary_tail_default() -> None:
    """Test binary tail with default n=5."""
    df = pl.DataFrame({"data": [b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"]})

    result = df.select(pl.col("data").bin.tail().alias("tail"))
    expected = pl.DataFrame({"tail": [b"\x05\x06\x07\x08\x09"]})
    assert_frame_equal(result, expected)


def test_binary_head_then_tail() -> None:
    """Test chaining head and tail operations."""
    df = pl.DataFrame({"data": [b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"]})

    result = df.select(pl.col("data").bin.head(8).bin.tail(6).alias("middle"))
    expected = pl.DataFrame({"middle": [b"\x02\x03\x04\x05\x06\x07"]})
    assert_frame_equal(result, expected)


def test_binary_slice_then_head() -> None:
    """Test chaining slice and head operations."""
    df = pl.DataFrame({"data": [b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"]})

    result = df.select(pl.col("data").bin.slice(2, 6).bin.head(3).alias("combo"))
    expected = pl.DataFrame({"combo": [b"\x02\x03\x04"]})
    assert_frame_equal(result, expected)


def test_binary_tail_then_slice() -> None:
    """Test chaining tail and slice operations."""
    df = pl.DataFrame({"data": [b"\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09"]})

    result = df.select(pl.col("data").bin.tail(7).bin.slice(1, 4).alias("combo"))
    expected = pl.DataFrame({"combo": [b"\x04\x05\x06\x07"]})
    assert_frame_equal(result, expected)


def test_binary_empty() -> None:
    """Test operations on empty binary data."""
    df = pl.DataFrame({"data": [b""]})

    assert_frame_equal(
        df.select(pl.col("data").bin.slice(0, 5)), pl.DataFrame({"data": [b""]})
    )
    assert_frame_equal(
        df.select(pl.col("data").bin.head(5)), pl.DataFrame({"data": [b""]})
    )
    assert_frame_equal(
        df.select(pl.col("data").bin.tail(5)), pl.DataFrame({"data": [b""]})
    )


def test_binary_all_nulls() -> None:
    """Test operations on all-null column."""
    df = pl.DataFrame({"data": [None, None, None]}, schema={"data": pl.Binary})

    assert_frame_equal(df.select(pl.col("data").bin.slice(0, 2)), df)
    assert_frame_equal(df.select(pl.col("data").bin.head(2)), df)
    assert_frame_equal(df.select(pl.col("data").bin.tail(2)), df)


def test_binary_single_byte() -> None:
    """Test operations on single-byte binary data."""
    df = pl.DataFrame({"data": [b"\xff"]})

    assert_frame_equal(df.select(pl.col("data").bin.slice(0, 1)), df)
    assert_frame_equal(df.select(pl.col("data").bin.head(1)), df)
    assert_frame_equal(df.select(pl.col("data").bin.tail(1)), df)
    assert_frame_equal(
        df.select(pl.col("data").bin.slice(0, 0)), pl.DataFrame({"data": [b""]})
    )
