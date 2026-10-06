from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.io.conftest import format_file_uri

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

# Test files are written with PyArrow, which only supports uniform encryption
# (where the footer and all columns are encrypted with the footer key) when
# using keys directly rather than a KMS.
FOOTER_KEY = b"0123456789012345"
AAD_PREFIX = b"tester"
NUM_ROWS = 1000
ROW_GROUP_SIZE = 250
PAGE_SIZE = 50


def local_path(path: Path) -> Path:
    return path


# Tests are run with local paths, and with file:// URIs, which are read in the same
# way as files from cloud storage.
parametrize_source = pytest.mark.parametrize(
    "to_source", [local_path, format_file_uri], ids=["local", "file_uri"]
)


def expected_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "s": [f"value_{i % 10}" for i in range(NUM_ROWS)],
            "i": pl.Series(range(NUM_ROWS), dtype=pl.Int64),
        }
    )


def write_encrypted(path: Path, **encryption_kwargs: Any) -> Path:
    """
    Write the expected data to an encrypted Parquet file with PyArrow.

    The file has multiple row groups with multiple data pages per column chunk, to
    exercise the row group and page ordinals used in the AAD. The string column is
    dictionary encoded and the integer column is not.
    """
    pytest.importorskip("pyarrow", minversion="25.0.0")
    import pyarrow.parquet as pq
    import pyarrow.parquet.encryption as pe

    encryption_properties = pe.create_encryption_properties(
        FOOTER_KEY, **encryption_kwargs
    )
    pq.write_table(
        expected_data().to_arrow(),
        path,
        encryption_properties=encryption_properties,
        row_group_size=ROW_GROUP_SIZE,
        use_dictionary=["s"],
        # Start a new data page after every write batch
        data_page_size=1,
        write_batch_size=PAGE_SIZE,
        compression="none",
    )

    decryption_properties = pe.create_decryption_properties(
        FOOTER_KEY, aad_prefix=encryption_kwargs.get("aad_prefix")
    )
    metadata = pq.ParquetFile(
        path, decryption_properties=decryption_properties
    ).metadata
    assert metadata.num_row_groups == NUM_ROWS // ROW_GROUP_SIZE
    for i in range(metadata.num_row_groups):
        row_group = metadata.row_group(i)
        assert row_group.column(0).has_dictionary_page
        assert not row_group.column(1).has_dictionary_page

    return path


@pytest.fixture
def encrypted_file_path(tmp_path: Path) -> Path:
    return write_encrypted(tmp_path / "uniform_encryption.parquet")


@parametrize_source
def test_read_uniform_encryption(
    encrypted_file_path: Path, to_source: Callable[[Path], Any]
) -> None:
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    df = pl.read_parquet(
        to_source(encrypted_file_path), decryption_properties=decryption_properties
    )

    assert_frame_equal(df, expected_data())


def test_read_ctr_encryption_unsupported(tmp_path: Path) -> None:
    path = write_encrypted(
        tmp_path / "ctr.parquet", encryption_algorithm="AES_GCM_CTR_V1"
    )
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="The AES_GCM_CTR_V1 encryption algorithm is not yet supported",
    ):
        pl.read_parquet(path, decryption_properties=decryption_properties)


@parametrize_source
def test_read_with_stored_aad_prefix(
    tmp_path: Path, to_source: Callable[[Path], Any]
) -> None:
    path = write_encrypted(tmp_path / "aad.parquet", aad_prefix=AAD_PREFIX)
    # The AAD prefix is stored in the file, so doesn't need to be provided
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    df = pl.read_parquet(to_source(path), decryption_properties=decryption_properties)

    assert_frame_equal(df, expected_data())


@parametrize_source
def test_read_with_unstored_aad_prefix(
    tmp_path: Path, to_source: Callable[[Path], Any]
) -> None:
    path = write_encrypted(
        tmp_path / "aad_not_stored.parquet",
        aad_prefix=AAD_PREFIX,
        store_aad_prefix=False,
    )
    source = to_source(path)

    # The AAD prefix isn't stored in the file, so must be provided
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    with pytest.raises(pl.exceptions.ComputeError):
        pl.read_parquet(source, decryption_properties=decryption_properties)

    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, aad_prefix=AAD_PREFIX
    )
    df = pl.read_parquet(source, decryption_properties=decryption_properties)

    assert_frame_equal(df, expected_data())


@parametrize_source
def test_read_plaintext_footer(
    tmp_path: Path, to_source: Callable[[Path], Any]
) -> None:
    path = write_encrypted(tmp_path / "plaintext_footer.parquet", plaintext_footer=True)
    source = to_source(path)
    expected = expected_data()

    # The footer can be read without decryption properties
    lf = pl.scan_parquet(source)
    assert lf.collect_schema() == expected.schema
    assert lf.select(pl.len()).collect().item() == NUM_ROWS

    # Column data can't be read
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="Column 's' is encrypted but decryption properties were not provided",
    ):
        pl.read_parquet(source, columns=["s"])

    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    df = pl.read_parquet(source, decryption_properties=decryption_properties)
    assert_frame_equal(df, expected)


def test_read_tampered_plaintext_footer(tmp_path: Path) -> None:
    path = write_encrypted(tmp_path / "plaintext_footer.parquet", plaintext_footer=True)
    # Modify the created_by string in the footer, which keeps the footer valid
    # Thrift but invalidates the footer signature.
    data = path.read_bytes()
    original = b"parquet-cpp-arrow"
    assert data.count(original) == 1
    tampered = data.replace(original, b"parquet-cpp-arr0w")

    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    with pytest.raises(
        pl.exceptions.ComputeError, match="Footer signature verification failed"
    ):
        pl.read_parquet(tampered, decryption_properties=decryption_properties)

    # The file can still be read with signature verification disabled
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, verify_footer_signature=False
    )
    df = pl.read_parquet(tampered, decryption_properties=decryption_properties)
    assert_frame_equal(df, expected_data())


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


def test_decryption_properties_with_pyarrow(encrypted_file_path: Path) -> None:
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    with pytest.raises(
        ValueError,
        match="Parquet decryption properties cannot be used when use_pyarrow is True",
    ):
        pl.read_parquet(
            encrypted_file_path,
            use_pyarrow=True,
            decryption_properties=decryption_properties,
        )


@parametrize_source
def test_scan_encrypted_footer_metadata(
    encrypted_file_path: Path, to_source: Callable[[Path], Any]
) -> None:
    # Only requires reading the footer, not column data
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(
        to_source(encrypted_file_path), decryption_properties=decryption_properties
    )
    assert lf.collect_schema() == expected_data().schema
    assert lf.select(pl.len()).collect().item() == NUM_ROWS


@parametrize_source
def test_scan_encrypted_footer_without_decryption_properties(
    encrypted_file_path: Path, to_source: Callable[[Path], Any]
) -> None:
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="encrypted footer but decryption properties were not provided",
    ):
        pl.scan_parquet(to_source(encrypted_file_path)).collect_schema()


@parametrize_source
def test_scan_multiple_encrypted_files(
    encrypted_file_path: Path, to_source: Callable[[Path], Any]
) -> None:
    source = to_source(encrypted_file_path)
    expected = expected_data()

    # No schema is provided, so the schema and row counts come from the footers.
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet([source, source], decryption_properties=decryption_properties)

    assert lf.select(pl.len()).collect().item() == 2 * NUM_ROWS
    assert_frame_equal(lf.collect(), pl.concat([expected, expected]))


def test_scan_encrypted_footer_with_wrong_key(encrypted_file_path: Path) -> None:
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=b"1234567890123450"
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="unable to decrypt parquet footer"
    ):
        pl.scan_parquet(
            encrypted_file_path, decryption_properties=decryption_properties
        ).collect_schema()


def test_serialize_with_decryption_properties(encrypted_file_path: Path) -> None:
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(encrypted_file_path, decryption_properties=decryption_properties)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="cannot serialize parquet decryption properties",
    ):
        lf.serialize()
