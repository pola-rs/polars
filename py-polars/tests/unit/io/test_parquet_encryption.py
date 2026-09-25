from __future__ import annotations

from datetime import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.io.conftest import format_file_uri

if TYPE_CHECKING:
    from collections.abc import Callable
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


def local_path(path: Path) -> Path:
    return path


# Tests are run with local paths, and with file:// URIs, which are read in the same
# way as files from cloud storage.
parametrize_source = pytest.mark.parametrize(
    "to_source", [local_path, format_file_uri], ids=["local", "file_uri"]
)

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


@parametrize_source
def test_read_uniform_encryption(
    io_files_path: Path, to_source: Callable[[Path], Any]
) -> None:
    path = to_source(uniform_encryption_path(io_files_path))
    expected = expected_data()

    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    df = pl.read_parquet(
        path, schema=expected.schema, decryption_properties=decryption_properties
    )

    assert_frame_equal(df.drop(UNSUPPORTED_COLUMNS), expected.drop(UNSUPPORTED_COLUMNS))


@pytest.mark.parametrize(
    ("file_name", "aad_prefix"),
    [
        ("encrypt_columns_plaintext_footer", None),
        ("encrypt_columns_and_footer", None),
        # The AAD prefix is stored in the file
        ("encrypt_columns_and_footer_aad", None),
        # The AAD prefix isn't stored in the file, so must be provided
        ("encrypt_columns_and_footer_disable_aad_storage", b"tester"),
    ],
)
@parametrize_source
def test_read_with_column_keys(
    io_files_path: Path,
    file_name: str,
    aad_prefix: bytes | None,
    to_source: Callable[[Path], Any],
) -> None:
    path = to_source(
        io_files_path / "parquet-encryption" / f"{file_name}.parquet.encrypted"
    )
    expected = expected_data()

    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, column_keys=COLUMN_KEYS, aad_prefix=aad_prefix
    )
    df = pl.read_parquet(
        path, schema=expected.schema, decryption_properties=decryption_properties
    )

    assert_frame_equal(df.drop(UNSUPPORTED_COLUMNS), expected.drop(UNSUPPORTED_COLUMNS))


def test_read_with_bloom_filters(io_files_path: Path) -> None:
    # This file has a different schema and data to the other test files
    path = (
        io_files_path
        / "parquet-encryption"
        / "encrypt_columns_and_footer_bloom_filter.parquet.encrypted"
    )
    n = 2000
    expected = pl.DataFrame(
        {
            "double_field": pl.Series(np.arange(n) + 0.5),
            "float_field": pl.Series(np.arange(n, dtype=np.float32) + np.float32(0.25)),
            "int32_field": pl.Series(np.arange(n, dtype=np.int32)),
            "name": pl.Series([f"name_{i}" for i in range(n)]),
        }
    )

    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, column_keys=COLUMN_KEYS
    )
    df = pl.read_parquet(path, decryption_properties=decryption_properties)

    assert_frame_equal(df, expected)


@parametrize_source
def test_read_encrypted_footer_without_column_keys(
    io_files_path: Path, to_source: Callable[[Path], Any]
) -> None:
    path = to_source(
        io_files_path
        / "parquet-encryption"
        / "encrypt_columns_and_footer.parquet.encrypted"
    )
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    expected = expected_data()
    # Columns that aren't encrypted with a column key
    footer_key_columns = [
        c
        for c in expected.columns
        if c not in COLUMN_KEYS and c not in UNSUPPORTED_COLUMNS
    ]

    # Columns encrypted with the footer key can be read without the column keys
    df = pl.read_parquet(
        path,
        columns=footer_key_columns,
        schema=expected.schema,
        decryption_properties=decryption_properties,
    )
    assert_frame_equal(df, expected.select(footer_key_columns))

    # But columns encrypted with a column key can't be read
    match = (
        "Metadata for column '{}' is encrypted and could not be decrypted: "
        "No column decryption key set for encrypted column '{}'"
    )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=match.format("double_field", "double_field"),
    ):
        pl.read_parquet(
            path, columns=["double_field"], decryption_properties=decryption_properties
        )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=match.format("double_field", "double_field"),
    ):
        pl.scan_parquet(path, decryption_properties=decryption_properties).filter(
            pl.col("double_field") > 1.0
        ).select("int32_field").collect()
    with pytest.raises(
        pl.exceptions.ComputeError,
        match=r"Metadata for column '.*_field' is encrypted and could not be decrypted",
    ):
        pl.read_parquet(path, decryption_properties=decryption_properties)


@parametrize_source
def test_read_encrypted_footer_with_some_column_keys(
    io_files_path: Path, to_source: Callable[[Path], Any]
) -> None:
    path = to_source(
        io_files_path
        / "parquet-encryption"
        / "encrypt_columns_and_footer.parquet.encrypted"
    )
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, column_keys={"double_field": COLUMN_KEYS["double_field"]}
    )
    columns = ["boolean_field", "int32_field", "double_field"]

    df = pl.read_parquet(
        path, columns=columns, decryption_properties=decryption_properties
    )
    assert_frame_equal(df, expected_data().select(columns))

    with pytest.raises(
        pl.exceptions.ComputeError,
        match="Metadata for column 'float_field' is encrypted and could not be decrypted",
    ):
        pl.read_parquet(
            path, columns=["float_field"], decryption_properties=decryption_properties
        )


@parametrize_source
def test_read_plaintext_footer_without_decryption_properties(
    io_files_path: Path, to_source: Callable[[Path], Any]
) -> None:
    path = to_source(
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


@parametrize_source
def test_scan_encrypted_footer_metadata(
    io_files_path: Path, to_source: Callable[[Path], Any]
) -> None:
    # Only requires reading the footer, not column data
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(
        to_source(uniform_encryption_path(io_files_path)),
        decryption_properties=decryption_properties,
    )
    assert lf.collect_schema().names() == expected_data().columns
    assert lf.select(pl.len()).collect().item() == NUM_ROWS


@parametrize_source
def test_scan_encrypted_footer_without_decryption_properties(
    io_files_path: Path, to_source: Callable[[Path], Any]
) -> None:
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="encrypted footer but decryption properties were not provided",
    ):
        pl.scan_parquet(
            to_source(uniform_encryption_path(io_files_path))
        ).collect_schema()


@parametrize_source
def test_scan_multiple_encrypted_files(
    io_files_path: Path, to_source: Callable[[Path], Any]
) -> None:
    source = to_source(uniform_encryption_path(io_files_path))
    # No schema is provided, so the schema and row counts come from the footers.
    # INT96 values overflow when read with the inferred nanosecond precision.
    columns = [
        c
        for c in expected_data().columns
        if c not in [*UNSUPPORTED_COLUMNS, "int96_field"]
    ]
    expected = expected_data().select(columns)

    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet([source, source], decryption_properties=decryption_properties)

    assert lf.select(pl.len()).collect().item() == 2 * NUM_ROWS
    assert_frame_equal(lf.select(columns).collect(), pl.concat([expected, expected]))


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
