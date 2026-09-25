from __future__ import annotations

from datetime import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

# Test files are from the apache/parquet-testing repository, and were written by
# the Arrow C++ encryption tests (see cpp/src/parquet/encryption/test_encryption_util.cc
# in the apache/arrow repository for details).
FOOTER_KEY = b"0123456789012345"
COLUMN_KEYS = {
    "double_field": b"1234567890123450",
    "float_field": b"1234567890123451",
}
NUM_ROWS = 50

# TODO: Polars misreads bare repeated primitive fields such as `int64_field`
# (a `repeated int64` without a LIST annotation), independent of encryption,
# so this column is excluded when checking data.
UNSUPPORTED_COLUMNS = ["int64_field"]

JULIAN_DAY_OF_EPOCH = 2_440_588
MICROS_PER_DAY = 86_400 * 1_000_000


def expected_data() -> pl.DataFrame:
    n = NUM_ROWS
    return pl.DataFrame(
        {
            "boolean_field": pl.Series([i % 2 == 0 for i in range(n)]),
            "int32_field": pl.Series([time(microsecond=i * 1000) for i in range(n)]),
            "int64_field": pl.Series(
                [[2 * i * 10**12, (2 * i + 1) * 10**12] for i in range(n)],
                dtype=pl.List(pl.Int64),
            ),
            # INT96 values are [nanos low bits, nanos high bits, Julian day] =
            # [i, i + 1, i + 2]. These overflow when read as nanoseconds, so
            # are read as microseconds.
            "int96_field": pl.Series(
                [
                    (i + 2 - JULIAN_DAY_OF_EPOCH) * MICROS_PER_DAY
                    + (((i + 1) << 32) + i) // 1000
                    for i in range(n)
                ],
                dtype=pl.Int64,
            ).cast(pl.Datetime("us")),
            "float_field": pl.Series(np.arange(n, dtype=np.float32) * np.float32(1.1)),
            "double_field": pl.Series([i * 1.1111111 for i in range(n)]),
            "ba_field": pl.Series(
                [f"parquet{i:03}".encode() if i % 2 == 0 else None for i in range(n)],
                dtype=pl.Binary,
            ),
            "flba_field": pl.Series([bytes([i]) * 10 for i in range(n)]),
        }
    )


def test_read_uniform_encryption(io_files_path: Path) -> None:
    path = io_files_path / "parquet-encryption" / "uniform_encryption.parquet.encrypted"
    expected = expected_data()

    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    df = pl.read_parquet(
        path, schema=expected.schema, decryption_properties=decryption_properties
    )

    assert_frame_equal(df.drop(UNSUPPORTED_COLUMNS), expected.drop(UNSUPPORTED_COLUMNS))


def test_read_plaintext_footer_with_column_keys(io_files_path: Path) -> None:
    path = (
        io_files_path
        / "parquet-encryption"
        / "encrypt_columns_plaintext_footer.parquet.encrypted"
    )
    expected = expected_data()

    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, column_keys=COLUMN_KEYS
    )
    df = pl.read_parquet(
        path, schema=expected.schema, decryption_properties=decryption_properties
    )

    assert_frame_equal(df.drop(UNSUPPORTED_COLUMNS), expected.drop(UNSUPPORTED_COLUMNS))


def test_read_plaintext_footer_without_decryption_properties(
    io_files_path: Path,
) -> None:
    path = (
        io_files_path
        / "parquet-encryption"
        / "encrypt_columns_plaintext_footer.parquet.encrypted"
    )
    plaintext_columns = ["boolean_field", "int32_field", "ba_field"]

    # Columns that aren't encrypted can be read without decryption properties
    df = pl.read_parquet(path, columns=plaintext_columns)
    assert_frame_equal(df, expected_data().select(plaintext_columns))

    with pytest.raises(
        pl.exceptions.ComputeError,
        match="Column 'double_field' is encrypted but decryption properties were not provided",
    ):
        pl.read_parquet(path, columns=["double_field"])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"footer_key": "0123456789012345"}, "footer_key must be bytes, got 'str'"),
        (
            {"footer_key": FOOTER_KEY, "column_keys": {"x": "1234567890123450"}},
            "key for column 'x' must be bytes, got 'str'",
        ),
        (
            {"footer_key": FOOTER_KEY, "aad_prefix": "prefix"},
            "aad_prefix must be bytes, got 'str'",
        ),
    ],
)
def test_decryption_properties_keys_must_be_bytes(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(TypeError, match=match):
        pl.ParquetDecryptionProperties(**kwargs)


def test_decryption_properties_with_pyarrow(io_files_path: Path) -> None:
    path = io_files_path / "parquet-encryption" / "uniform_encryption.parquet.encrypted"
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    with pytest.raises(
        ValueError,
        match="Parquet decryption properties cannot be used when use_pyarrow is True",
    ):
        pl.read_parquet(
            path, use_pyarrow=True, decryption_properties=decryption_properties
        )


def uniform_encryption_path(io_files_path: Path) -> Path:
    return io_files_path / "parquet-encryption" / "uniform_encryption.parquet.encrypted"


def test_scan_encrypted_footer_metadata(io_files_path: Path) -> None:
    # Only requires reading the footer, not column data
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(
        uniform_encryption_path(io_files_path),
        decryption_properties=decryption_properties,
    )
    assert lf.collect_schema().names() == expected_data().columns
    assert lf.select(pl.len()).collect().item() == NUM_ROWS


def test_scan_encrypted_footer_without_decryption_properties(
    io_files_path: Path,
) -> None:
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="encrypted footer but decryption properties were not provided",
    ):
        pl.scan_parquet(uniform_encryption_path(io_files_path)).collect_schema()


def test_scan_encrypted_footer_with_wrong_key(io_files_path: Path) -> None:
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=b"1234567890123450"
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="unable to decrypt parquet footer"
    ):
        pl.scan_parquet(
            uniform_encryption_path(io_files_path),
            decryption_properties=decryption_properties,
        ).collect_schema()


def test_serialize_with_decryption_properties(io_files_path: Path) -> None:
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(
        uniform_encryption_path(io_files_path),
        decryption_properties=decryption_properties,
    )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="cannot serialize parquet decryption properties",
    ):
        lf.serialize()
