from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal
from tests.unit.io.conftest import format_file_uri

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from polars._typing import EngineType
    from tests.conftest import PlMonkeyPatch

FOOTER_KEY = b"0123456789012345"
COLUMN_KEY = b"1234567890123450"
COLUMN_KEY_2 = b"1234567890123451"
AES_256_KEY = b"01234567890123456789012345678901"
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
        to_source(encrypted_file_path), decryption=decryption_properties
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
        pl.read_parquet(path, decryption=decryption_properties)


@parametrize_source
def test_read_with_stored_aad_prefix(
    tmp_path: Path, to_source: Callable[[Path], Any]
) -> None:
    path = write_encrypted(tmp_path / "aad.parquet", aad_prefix=AAD_PREFIX)
    # The AAD prefix is stored in the file, so doesn't need to be provided
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    df = pl.read_parquet(to_source(path), decryption=decryption_properties)

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
        pl.read_parquet(source, decryption=decryption_properties)

    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, aad_prefix=AAD_PREFIX
    )
    df = pl.read_parquet(source, decryption=decryption_properties)

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
    df = pl.read_parquet(source, decryption=decryption_properties)
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
        pl.read_parquet(tampered, decryption=decryption_properties)

    # The file can still be read with signature verification disabled
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, verify_footer_signature=False
    )
    df = pl.read_parquet(tampered, decryption=decryption_properties)
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
            decryption=decryption_properties,
        )


@parametrize_source
def test_scan_encrypted_footer_metadata(
    encrypted_file_path: Path, to_source: Callable[[Path], Any]
) -> None:
    # Only requires reading the footer, not column data
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(
        to_source(encrypted_file_path), decryption=decryption_properties
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
    lf = pl.scan_parquet([source, source], decryption=decryption_properties)

    assert lf.select(pl.len()).collect().item() == 2 * NUM_ROWS
    assert_frame_equal(lf.collect(), pl.concat([expected, expected]))


@pytest.mark.parametrize("mode", ["row_counts", "full"])
@parametrize_source
def test_scan_resolve_metadata_level(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    mode: str,
    to_source: Callable[[Path], Any],
) -> None:
    # Source 0's full footer is always read, but in row_counts mode only the row
    # counts are read from the other footers, so put the encrypted files last.
    encrypted = write_encrypted(tmp_path / "encrypted.parquet")
    plaintext_footer = write_encrypted(
        tmp_path / "plaintext_footer.parquet", plaintext_footer=True
    )
    unencrypted = tmp_path / "unencrypted.parquet"
    expected_data().write_parquet(unencrypted)
    sources = [to_source(p) for p in [unencrypted, encrypted, plaintext_footer]]

    plmonkeypatch.setenv("POLARS_RESOLVE_METADATA_LEVEL", mode)

    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(sources, decryption=decryption_properties)
    assert f"ESTIMATED ROWS: {3 * NUM_ROWS}" in lf.explain(optimized=True)
    assert lf.select(pl.len()).collect().item() == 3 * NUM_ROWS


def test_scan_encrypted_footer_with_wrong_key(encrypted_file_path: Path) -> None:
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=b"1234567890123450"
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="unable to decrypt parquet footer"
    ):
        pl.scan_parquet(
            encrypted_file_path, decryption=decryption_properties
        ).collect_schema()


def test_serialize_with_decryption_properties(encrypted_file_path: Path) -> None:
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(encrypted_file_path, decryption=decryption_properties)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="cannot serialize parquet decryption properties",
    ):
        lf.serialize()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"footer_key": FOOTER_KEY},
        {
            "footer_key": FOOTER_KEY,
            "column_keys": {"x": b"1234567890123450", "y.z": b"1234567890123451"},
        },
        {"footer_key": FOOTER_KEY, "plaintext_footer": True},
        {"footer_key": FOOTER_KEY, "aad_prefix": AAD_PREFIX},
        {"footer_key": FOOTER_KEY, "aad_prefix": AAD_PREFIX, "store_aad_prefix": True},
    ],
)
def test_encryption_properties(kwargs: dict[str, Any]) -> None:
    pl.ParquetEncryptionProperties(**kwargs)


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
def test_encryption_properties_keys_must_be_bytes(
    kwargs: dict[str, Any], match: str
) -> None:
    with pytest.raises(TypeError, match=match):
        pl.ParquetEncryptionProperties(**kwargs)


def test_encryption_properties_store_aad_prefix_requires_aad_prefix() -> None:
    with pytest.raises(ValueError, match="store_aad_prefix requires aad_prefix"):
        pl.ParquetEncryptionProperties(footer_key=FOOTER_KEY, store_aad_prefix=True)


WRITE_OPTIONS: dict[str, Any] = {
    # Uncompressed so that plaintext values can be found in the written bytes.
    "compression": "uncompressed",
    # Write multiple row groups with multiple data pages per column chunk, to
    # exercise the row group and page ordinals used in the AAD.
    "row_group_size": ROW_GROUP_SIZE,
    "data_page_size": 1024,
}


def basic_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series(range(NUM_ROWS), dtype=pl.Int64),
            "secret": [f"secret_{i}" for i in range(NUM_ROWS)],
            "public": [f"public_{i}" for i in range(NUM_ROWS)],
            "part": pl.Series([i % 3 for i in range(NUM_ROWS)], dtype=pl.Int64),
        }
    )


def basic_encryption_properties() -> pl.ParquetEncryptionProperties:
    """Encrypt the footer and the `secret` column, leaving other columns unencrypted."""
    return pl.ParquetEncryptionProperties(
        footer_key=FOOTER_KEY, column_keys={"secret": COLUMN_KEY}
    )


def basic_decryption_properties() -> pl.ParquetDecryptionProperties:
    return pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, column_keys={"secret": COLUMN_KEY}
    )


def parquet_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    assert files
    return files


def assert_basic_encryption(path: Path, expected: pl.DataFrame | None = None) -> None:
    """
    Check a file, or directory of files, written with the basic encryption properties.

    Checks that the files are encrypted, and can be read back with the correct
    decryption properties.
    """
    if expected is None:
        expected = basic_data()

    for file in parquet_files(path):
        data = file.read_bytes()
        assert b"public_" in data
        assert b"secret_" not in data

    with pytest.raises(
        pl.exceptions.ComputeError,
        match="encrypted footer but decryption properties were not provided",
    ):
        pl.scan_parquet(path).collect_schema()

    # Unencrypted columns can be read with only the footer key
    footer_key_only = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    lf = pl.scan_parquet(path, decryption=footer_key_only)
    assert_frame_equal(
        lf.select("id", "public").collect(),
        expected.select("id", "public"),
        check_row_order=False,
    )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="Metadata for column 'secret' is encrypted and could not be decrypted",
    ):
        lf.select("secret").collect()

    df = pl.scan_parquet(path, decryption=basic_decryption_properties()).collect()
    assert_frame_equal(df, expected, check_row_order=False, check_column_order=False)


def test_write_parquet_encrypted(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(
        path, encryption=basic_encryption_properties(), **WRITE_OPTIONS
    )
    assert_basic_encryption(path)


def test_write_parquet_encrypted_partition_by(tmp_path: Path) -> None:
    path = tmp_path / "out"
    basic_data().write_parquet(
        path,
        partition_by="part",
        encryption=basic_encryption_properties(),
        **WRITE_OPTIONS,
    )
    assert len(parquet_files(path)) == 3
    assert_basic_encryption(path)


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_sink_parquet_encrypted(tmp_path: Path, engine: EngineType) -> None:
    path = tmp_path / "out.parquet"
    basic_data().lazy().sink_parquet(
        path,
        encryption=basic_encryption_properties(),
        engine=engine,
        **WRITE_OPTIONS,
    )
    assert_basic_encryption(path)


def test_sink_parquet_encrypted_file_uri(tmp_path: Path) -> None:
    # Written in the same way as files in cloud storage
    path = tmp_path / "out.parquet"
    basic_data().lazy().sink_parquet(
        format_file_uri(path),
        encryption=basic_encryption_properties(),
        **WRITE_OPTIONS,
    )
    assert_basic_encryption(path)


def test_sink_parquet_encrypted_partition_by_key(tmp_path: Path) -> None:
    path = tmp_path / "out"
    basic_data().lazy().sink_parquet(
        pl.PartitionBy(path, key="part"),
        encryption=basic_encryption_properties(),
        mkdir=True,
        **WRITE_OPTIONS,
    )
    assert len(parquet_files(path)) == 3
    assert_basic_encryption(path)


def test_sink_parquet_encrypted_partition_by_max_rows(tmp_path: Path) -> None:
    path = tmp_path / "out"
    basic_data().lazy().sink_parquet(
        pl.PartitionBy(path, max_rows_per_file=400),
        encryption=basic_encryption_properties(),
        mkdir=True,
        **WRITE_OPTIONS,
    )
    assert len(parquet_files(path)) == 3
    assert_basic_encryption(path)


def test_sink_parquet_reencrypt(tmp_path: Path) -> None:
    # Stream an encrypted file into a new file encrypted with different keys
    source = tmp_path / "source.parquet"
    basic_data().write_parquet(
        source,
        encryption=pl.ParquetEncryptionProperties(footer_key=AES_256_KEY),
        **WRITE_OPTIONS,
    )
    path = tmp_path / "out.parquet"
    pl.scan_parquet(
        source,
        decryption=pl.ParquetDecryptionProperties(footer_key=AES_256_KEY),
    ).sink_parquet(path, encryption=basic_encryption_properties(), **WRITE_OPTIONS)
    assert_basic_encryption(path)


def roundtrip_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series(range(NUM_ROWS), dtype=pl.Int64),
            "a": [f"aaa_{i}" for i in range(NUM_ROWS)],
            "b": [f"bbb_{i}" for i in range(NUM_ROWS)],
            # Written with dictionary pages
            "cat": pl.Series(
                [f"cat_{i % 10}" for i in range(NUM_ROWS)], dtype=pl.Categorical
            ),
            "s": [{"x": f"sss_{i}", "y": i} for i in range(NUM_ROWS)],
            "l": [[f"lll_{i}", f"lll_{i + 1}"] for i in range(NUM_ROWS)],
        }
    )


# Values in each column of the round trip data start with a marker, which can be
# searched for in the file bytes to determine whether the column is encrypted.
ROUNDTRIP_MARKERS = {
    "a": b"aaa_",
    "b": b"bbb_",
    "cat": b"cat_",
    "s.x": b"sss_",
    "l.list.element": b"lll_",
}
ROUNDTRIP_LEAF_COLUMNS = ["id", "a", "b", "cat", "s.x", "s.y", "l.list.element"]
ALL_COLUMNS = set(ROUNDTRIP_MARKERS)


@pytest.mark.parametrize(
    ("encryption_kwargs", "decryption_kwargs", "encrypted_columns"),
    [
        pytest.param({}, {}, ALL_COLUMNS, id="uniform"),
        pytest.param(
            {"footer_key": AES_256_KEY},
            {"footer_key": AES_256_KEY},
            ALL_COLUMNS,
            id="uniform_aes_256",
        ),
        pytest.param(
            {"plaintext_footer": True}, {}, ALL_COLUMNS, id="uniform_plaintext_footer"
        ),
        pytest.param(
            {"column_keys": {"a": COLUMN_KEY}},
            {"column_keys": {"a": COLUMN_KEY}},
            {"a"},
            id="column_key",
        ),
        pytest.param(
            {"column_keys": {"a": COLUMN_KEY, "b": COLUMN_KEY_2}},
            {"column_keys": {"a": COLUMN_KEY, "b": COLUMN_KEY_2}},
            {"a", "b"},
            id="column_keys_different_keys",
        ),
        pytest.param(
            {"column_keys": dict.fromkeys(ROUNDTRIP_LEAF_COLUMNS, COLUMN_KEY)},
            {"column_keys": dict.fromkeys(ROUNDTRIP_LEAF_COLUMNS, COLUMN_KEY)},
            ALL_COLUMNS,
            id="column_keys_all_columns",
        ),
        pytest.param(
            {"column_keys": {"a": AES_256_KEY}},
            {"column_keys": {"a": AES_256_KEY}},
            {"a"},
            id="column_key_aes_256",
        ),
        pytest.param(
            {"column_keys": {"s.x": COLUMN_KEY}},
            {"column_keys": {"s.x": COLUMN_KEY}},
            {"s.x"},
            id="column_key_struct_field",
        ),
        pytest.param(
            {"column_keys": {"l.list.element": COLUMN_KEY}},
            {"column_keys": {"l.list.element": COLUMN_KEY}},
            {"l.list.element"},
            id="column_key_list",
        ),
        pytest.param(
            {"plaintext_footer": True, "column_keys": {"a": COLUMN_KEY}},
            {"column_keys": {"a": COLUMN_KEY}},
            {"a"},
            id="column_key_plaintext_footer",
        ),
        pytest.param(
            {"aad_prefix": AAD_PREFIX, "store_aad_prefix": True},
            {},
            ALL_COLUMNS,
            id="aad_prefix_stored",
        ),
        pytest.param(
            {"aad_prefix": AAD_PREFIX},
            {"aad_prefix": AAD_PREFIX},
            ALL_COLUMNS,
            id="aad_prefix_not_stored",
        ),
    ],
)
def test_write_roundtrip(
    tmp_path: Path,
    encryption_kwargs: dict[str, Any],
    decryption_kwargs: dict[str, Any],
    encrypted_columns: set[str],
) -> None:
    df = roundtrip_data()
    path = tmp_path / "out.parquet"
    df.write_parquet(
        path,
        encryption=pl.ParquetEncryptionProperties(
            **{"footer_key": FOOTER_KEY, **encryption_kwargs}
        ),
        **WRITE_OPTIONS,
    )

    data = path.read_bytes()
    for column, marker in ROUNDTRIP_MARKERS.items():
        is_encrypted = marker not in data
        assert is_encrypted == (column in encrypted_columns), column

    decryption_properties = pl.ParquetDecryptionProperties(
        **{"footer_key": FOOTER_KEY, **decryption_kwargs}
    )
    assert_frame_equal(pl.read_parquet(path, decryption=decryption_properties), df)


def test_write_empty_frame(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    df = basic_data().clear()
    df.write_parquet(path, encryption=basic_encryption_properties())
    result = pl.read_parquet(path, decryption=basic_decryption_properties())
    assert_frame_equal(result, df)


@pytest.mark.parametrize("column_keys", [{}, {"a": COLUMN_KEY}])
def test_write_plaintext_footer_readable_without_keys(
    tmp_path: Path, column_keys: dict[str, bytes]
) -> None:
    df = roundtrip_data()
    path = tmp_path / "out.parquet"
    df.write_parquet(
        path,
        encryption=pl.ParquetEncryptionProperties(
            footer_key=FOOTER_KEY, plaintext_footer=True, column_keys=column_keys
        ),
    )

    lf = pl.scan_parquet(path)
    assert lf.collect_schema() == df.schema
    assert lf.select(pl.len()).collect().item() == NUM_ROWS

    with pytest.raises(
        pl.exceptions.ComputeError,
        match="Column 'a' is encrypted but decryption properties were not provided",
    ):
        lf.select("a").collect()

    if column_keys:
        # Only column `a` is encrypted
        assert_frame_equal(lf.select("id", "b").collect(), df.select("id", "b"))


def test_write_plaintext_footer_signature(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(
        path,
        encryption=pl.ParquetEncryptionProperties(
            footer_key=FOOTER_KEY, plaintext_footer=True
        ),
        metadata={"note": "original_value"},
    )
    # Modify the key-value metadata in the footer, which keeps the footer valid
    # Thrift but invalidates the footer signature.
    data = path.read_bytes()
    assert data.count(b"original_value") == 1
    tampered = data.replace(b"original_value", b"0riginal_value")

    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    with pytest.raises(
        pl.exceptions.ComputeError, match="Footer signature verification failed"
    ):
        pl.read_parquet(tampered, decryption=decryption_properties)

    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, verify_footer_signature=False
    )
    df = pl.read_parquet(tampered, decryption=decryption_properties)
    assert_frame_equal(df, basic_data())


def test_write_aad_prefix_not_stored_must_be_provided(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(
        path,
        encryption=pl.ParquetEncryptionProperties(
            footer_key=FOOTER_KEY, aad_prefix=AAD_PREFIX
        ),
    )
    decryption_properties = pl.ParquetDecryptionProperties(footer_key=FOOTER_KEY)
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="encrypted with an AAD prefix that is not stored in the file",
    ):
        pl.read_parquet(path, decryption=decryption_properties)


@pytest.mark.parametrize("store_aad_prefix", [True, False])
def test_write_aad_prefix_mismatch(tmp_path: Path, store_aad_prefix: bool) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(
        path,
        encryption=pl.ParquetEncryptionProperties(
            footer_key=FOOTER_KEY,
            aad_prefix=AAD_PREFIX,
            store_aad_prefix=store_aad_prefix,
        ),
    )
    # A provided AAD prefix is used instead of any stored prefix, and must match
    # the prefix the file was written with.
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, aad_prefix=b"wrong_prefix"
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="unable to decrypt parquet footer"
    ):
        pl.read_parquet(path, decryption=decryption_properties)


def test_write_read_with_wrong_footer_key(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(path, encryption=basic_encryption_properties())
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=COLUMN_KEY_2, column_keys={"secret": COLUMN_KEY}
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="unable to decrypt parquet footer"
    ):
        pl.read_parquet(path, decryption=decryption_properties)


def test_write_read_with_wrong_column_key(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(path, encryption=basic_encryption_properties())
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=FOOTER_KEY, column_keys={"secret": COLUMN_KEY_2}
    )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="Unable to decrypt metadata for column 'secret', the column key may be wrong",
    ):
        pl.read_parquet(path, decryption=decryption_properties)


def test_write_column_key_not_in_schema(tmp_path: Path) -> None:
    encryption_properties = pl.ParquetEncryptionProperties(
        footer_key=FOOTER_KEY,
        column_keys={"secret": COLUMN_KEY, "missing": COLUMN_KEY_2},
    )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="columns with encryption keys specified were not found in the schema: missing",
    ):
        basic_data().write_parquet(
            tmp_path / "out.parquet", encryption=encryption_properties
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"footer_key": b"short_key"},
        {"footer_key": FOOTER_KEY, "column_keys": {"secret": b"short_key"}},
    ],
)
def test_write_invalid_key_length(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    with pytest.raises(pl.exceptions.ComputeError, match="unsupported key length: 9"):
        basic_data().write_parquet(
            tmp_path / "out.parquet",
            encryption=pl.ParquetEncryptionProperties(**kwargs),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"plaintext_footer": True},
        {"aad_prefix": AAD_PREFIX, "store_aad_prefix": True},
        {"aad_prefix": AAD_PREFIX},
    ],
)
def test_write_read_with_pyarrow(tmp_path: Path, kwargs: dict[str, Any]) -> None:
    # PyArrow only supports reading files with uniform encryption when using keys
    # directly rather than a KMS.
    pytest.importorskip("pyarrow", minversion="25.0.0")
    import pyarrow.parquet as pq
    import pyarrow.parquet.encryption as pe

    df = basic_data()
    path = tmp_path / "out.parquet"
    df.write_parquet(
        path,
        encryption=pl.ParquetEncryptionProperties(footer_key=FOOTER_KEY, **kwargs),
        **WRITE_OPTIONS,
    )

    aad_prefix = None if kwargs.get("store_aad_prefix") else kwargs.get("aad_prefix")
    decryption_properties = pe.create_decryption_properties(
        FOOTER_KEY, aad_prefix=aad_prefix
    )
    parquet_file = pq.ParquetFile(path, decryption_properties=decryption_properties)
    assert parquet_file.metadata.num_row_groups == NUM_ROWS // ROW_GROUP_SIZE
    result = pl.from_arrow(parquet_file.read())
    assert isinstance(result, pl.DataFrame)
    assert_frame_equal(result, df)


def test_write_encrypted_predicate_pushdown(tmp_path: Path) -> None:
    # Statistics for encrypted columns are stored in encrypted column metadata
    path = tmp_path / "out.parquet"
    df = basic_data()
    df.write_parquet(path, encryption=basic_encryption_properties(), **WRITE_OPTIONS)
    lf = pl.scan_parquet(path, decryption=basic_decryption_properties())
    for predicate in [
        pl.col("secret") == "secret_500",
        pl.col("id").is_between(10, 20),
    ]:
        assert_frame_equal(lf.filter(predicate).collect(), df.filter(predicate))


def test_write_encrypted_with_pyarrow_raises(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": [1, 2, 3]})
    encryption_properties = pl.ParquetEncryptionProperties(footer_key=FOOTER_KEY)
    with pytest.raises(ValueError, match="cannot be combined with `encryption`"):
        df.write_parquet(
            tmp_path / "out.parquet",
            use_pyarrow=True,
            encryption=encryption_properties,
        )
