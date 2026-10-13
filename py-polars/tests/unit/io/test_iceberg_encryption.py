from __future__ import annotations

import base64
import gzip
import json
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from typing import TYPE_CHECKING, Any, ClassVar

import pyarrow.parquet as pq
import pyarrow.parquet.encryption as pe
import pytest
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (
    clear_manifest_cache,
)
from pyiceberg.schema import Schema
from pyiceberg.table import StaticTable
from pyiceberg.table.snapshots import Operation
from pyiceberg.types import LongType, NestedField

import polars as pl
from polars import _plr
from polars.io.iceberg._dataset import _NativeIcebergScanData
from polars.io.iceberg._encryption import (
    _BytesInputFile,
    _EncryptedInputFile,
    _load_kms_config,
    decode_key_metadata,
)
from polars.testing import assert_frame_equal
from tests.unit.io.iceberg_encryption_utils import (
    encrypt_block,
    encrypt_stream,
    key_metadata,
    write_encrypted_snapshot_metadata,
)
from tests.unit.io.iceberg_utils import (
    iceberg_path_fixture,  # noqa: F401
    new_iceberg_scan_resolver,
    new_iceberg_table,
    sqlcatalog_uses_null_pool_connections,  # noqa: F401
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from polars._typing import EngineType


class _TestKmsClient:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []
        self.properties: dict[str, str] = {}

    def initialize(self, properties: dict[str, str]) -> None:
        self.properties = properties

    def unwrap_key(self, wrapped_key: bytes, wrapping_key_id: str) -> bytes:
        self.calls.append((wrapped_key, wrapping_key_id))
        msg = "unencrypted table should not request a key"
        raise AssertionError(msg)


class _IdentityKms:
    def initialize(self, properties: dict[str, str]) -> None:
        pass

    def unwrap_key(self, wrapped_key: bytes, wrapping_key_id: str) -> bytes:
        assert wrapping_key_id == "test-master"
        return wrapped_key


class _KmsWithoutInitialize:
    def unwrap_key(self, wrapped_key: bytes, wrapping_key_id: str) -> bytes:
        return wrapped_key


class _KmsWithoutUnwrap:
    def initialize(self, properties: dict[str, str]) -> None:
        pass


class _KmsWithRequiredArgument:
    def __init__(self, value: str) -> None:
        pass


@pytest.fixture
def iceberg_kms_endpoint() -> Iterator[
    tuple[str, list[tuple[str | None, dict[str, Any]]], dict[str, str]]
]:
    calls: list[tuple[str | None, dict[str, Any]]] = []
    response = {
        "Plaintext": base64.b64encode(
            bytes.fromhex("fa501fc8fcba0566490e925879392260")
        ).decode(),
        "KeyId": "master-1",
        "EncryptionAlgorithm": "SYMMETRIC_DEFAULT",
    }

    class KmsHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            calls.append((self.headers.get("X-Amz-Target"), json.loads(body)))
            self.send_response(400 if "__type" in response else 200)
            self.send_header("Content-Type", "application/x-amz-json-1.1")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

    with ThreadingHTTPServer(("127.0.0.1", 0), KmsHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_port}", calls, response
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
@pytest.mark.parametrize("uri", [False, True])
def test_different_file_keys_and_encrypted_position_deletes(
    tmp_path: Path, engine: EngineType, uri: bool
) -> None:
    paths = []
    keys = {}
    for i in range(2):
        path = tmp_path / f"data-{i}.parquet"
        key, aad = bytes([i + 1]) * 16, bytes([i + 2]) * 16
        pq.write_table(
            pl.DataFrame({"x": list(range(i * 3, i * 3 + 3))}).to_arrow(),
            path,
            encryption_properties=pe.create_encryption_properties(
                key, aad_prefix=aad, store_aad_prefix=False
            ),
        )
        source = path.as_uri() if uri else str(path)
        paths.append(source)
        keys[source] = (key, aad)
    delete_sources = {}
    for i, source in enumerate(paths):
        delete_path = tmp_path / f"deletes-{i}.parquet"
        delete_source = delete_path.as_uri() if uri else str(delete_path)
        key, aad = bytes([i + 10]) * 16, b"delete"
        pq.write_table(
            pl.DataFrame({"file_path": [source], "pos": [1 - i]}).to_arrow(),
            delete_path,
            encryption_properties=pe.create_encryption_properties(
                key, aad_prefix=aad, store_aad_prefix=False
            ),
        )
        keys[delete_source] = (key, aad)
        delete_sources[i] = [delete_source]
    q = pl.scan_parquet(
        paths,
        decryption=pl.ParquetDecryptionProperties._from_iceberg_keys(keys),
        _deletion_files=("iceberg", (delete_sources, {})),
    )
    expected = pl.DataFrame({"x": [0, 2, 4, 5]})
    assert_frame_equal(q.collect(engine=engine), expected)
    assert_frame_equal(q.tail(2).collect(engine=engine), expected.tail(2))
    assert_frame_equal(
        q.filter(pl.col("x") >= 2).collect(engine=engine),
        expected.filter(pl.col("x") >= 2),
    )
    assert q.select(pl.len()).collect(engine=engine).item() == 4
    keys[paths[1]] = keys[paths[0]]
    with pytest.raises(pl.exceptions.ComputeError):
        pl.scan_parquet(
            paths, decryption=pl.ParquetDecryptionProperties._from_iceberg_keys(keys)
        ).collect(engine=engine)


@pytest.mark.parametrize("key_size", [16, 24, 32])
def test_metadata_authentication_and_multiple_blocks(key_size: int) -> None:
    plaintext = b"avro" * (1024 * 256 + 1)
    key, aad = b"k" * key_size, b"manifest"
    ciphertext = encrypt_stream(plaintext, key, aad)
    assert _plr._iceberg_decrypt(ciphertext, key, aad, True) == plaintext
    for invalid, invalid_aad in [(ciphertext[:-1], aad), (ciphertext, b"wrong")]:
        with pytest.raises(pl.exceptions.ComputeError, match="decryption failed"):
            _plr._iceberg_decrypt(invalid, key, invalid_aad, True)
    assert _plr._iceberg_decrypt(encrypt_stream(b"", key, aad), key, aad, True) == b""
    encoded = key_metadata(key, aad, len(plaintext))
    assert decode_key_metadata(encoded) == (key, aad, len(plaintext))
    with pytest.raises(pl.exceptions.ComputeError):
        decode_key_metadata(encoded[:-1])


def test_metadata_length_and_key_serialization(tmp_path: Path) -> None:
    key, aad = b"k" * 16, b"metadata"
    plaintext = b"a" * (1024 * 1024 + 1)
    ciphertext = encrypt_stream(plaintext, key, aad)
    path = tmp_path / "manifest.avro"
    path.write_bytes(ciphertext)
    encrypted = _EncryptedInputFile(
        PyArrowFileIO().new_input(str(path)),
        key_metadata(key, aad, len(ciphertext)),
    )
    assert len(encrypted) == len(plaintext)
    with encrypted.open() as stream:
        assert stream.read() == plaintext
    for value in (
        encrypted,
        _BytesInputFile(str(path), plaintext),
        pl.ParquetDecryptionProperties._from_iceberg_keys({str(path): (key, aad)}),
    ):
        with pytest.raises(TypeError):
            pickle.dumps(value)
    path.write_bytes(ciphertext[: 8 + 1024 * 1024 + 12 + 16])
    with pytest.raises(pl.exceptions.ComputeError, match="file length"):
        encrypted.open()


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
@pytest.mark.parametrize("compressed", [False, True])
def test_encrypted_table_uses_native_scan(
    tmp_path: Path,
    engine: EngineType,
    compressed: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table, _ = new_iceberg_table(
        tmp_path, schema=Schema(NestedField(1, "x", LongType()))
    )
    table.append(pl.DataFrame({"x": [1, 2, 3]}).to_arrow())
    table.append(pl.DataFrame({"x": [4, 5, 6]}).to_arrow())
    snapshot = table.current_snapshot()
    assert snapshot is not None
    metadata_path = write_encrypted_snapshot_metadata(table, tmp_path, [None, "a"])
    if compressed:
        compressed_path = tmp_path / "v1.gz.metadata.json"
        compressed_path.write_bytes(gzip.compress(metadata_path.read_bytes()))
        metadata_path = compressed_path
    clear_manifest_cache()
    storage_options = {"py-kms-impl": f"{__name__}._RotatingKms"}
    q = pl.scan_iceberg(str(metadata_path), storage_options=storage_options)
    expected = pl.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
    assert "Parquet SCAN" in q.explain()
    assert "ICEBERG-RUST" not in q.explain()
    assert_frame_equal(q.sort("x").collect(engine=engine), expected)
    assert_frame_equal(
        q.filter(pl.col("x") > 3).sort("x").collect(engine=engine), expected.tail(3)
    )
    assert q.select(pl.len()).collect(engine=engine).item() == 6
    assert_frame_equal(
        pl.scan_iceberg(
            str(metadata_path),
            storage_options=storage_options,
            snapshot_id=snapshot.parent_snapshot_id,
        ).collect(engine=engine),
        expected.head(3),
    )
    loaded_table = StaticTable.from_metadata(str(metadata_path))
    monkeypatch.setattr(
        loaded_table.catalog, "supports_server_side_planning", lambda: True
    )
    assert_frame_equal(
        pl.scan_iceberg(loaded_table, storage_options=storage_options)
        .sort("x")
        .collect(engine=engine),
        expected,
    )


class _RotatingKms:
    calls: ClassVar[list[str]] = []
    denied: ClassVar[set[str]] = set()

    def initialize(self, properties: dict[str, str]) -> None:
        pass

    def unwrap_key(self, wrapped_key: bytes, wrapping_key_id: str) -> bytes:
        self.calls.append(wrapping_key_id)
        assert wrapped_key == wrapping_key_id.encode()
        if wrapping_key_id in self.denied:
            msg = "access denied"
            raise PermissionError(msg)
        return wrapping_key_id.encode() * 16


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
def test_encrypted_incremental_append_key_rotation(
    tmp_path: Path, engine: EngineType, monkeypatch: pytest.MonkeyPatch
) -> None:
    table, _ = new_iceberg_table(
        tmp_path, schema=Schema(NestedField(1, "x", LongType()))
    )
    for i in range(5):
        table.append(pl.DataFrame({"x": [i * 10, i * 10 + 1]}).to_arrow())
    ids = [snapshot.snapshot_id for snapshot in table.snapshots()]
    path = write_encrypted_snapshot_metadata(
        table, tmp_path, [None, "a", "b", "b", "c"]
    )
    monkeypatch.setattr(_RotatingKms, "calls", [])
    monkeypatch.setattr(_RotatingKms, "denied", {"c"})
    options = {"py-kms-impl": f"{__name__}._RotatingKms"}
    q = pl.scan_iceberg(
        str(path),
        storage_options=options,
        from_snapshot_id_exclusive=ids[0],
        to_snapshot_id_inclusive=ids[3],
    )
    expected = pl.DataFrame({"x": [10, 11, 20, 21, 30, 31]})
    assert_frame_equal(q.sort("x").collect(engine=engine), expected)
    assert sorted(_RotatingKms.calls) == ["a", "b"]
    assert_frame_equal(
        q.filter(pl.col("x") > 20).sort("x").collect(engine=engine), expected.tail(3)
    )
    assert_frame_equal(q.sort("x").tail(2).collect(engine=engine), expected.tail(2))
    assert q.select(pl.len()).collect(engine=engine).item() == 6

    _RotatingKms.calls.clear()
    monkeypatch.setattr(_RotatingKms, "denied", {"a", "c"})
    assert_frame_equal(
        pl.scan_iceberg(
            str(path),
            storage_options=options,
            from_snapshot_id_exclusive=ids[1],
            to_snapshot_id_inclusive=ids[3],
        )
        .sort("x")
        .collect(engine=engine),
        expected.tail(4),
    )
    assert _RotatingKms.calls == ["b"]
    with pytest.raises(pl.exceptions.ComputeError, match="KMS unwrap_key failed"):
        pl.scan_iceberg(
            str(path),
            storage_options=options,
            from_snapshot_id_exclusive=ids[0],
            to_snapshot_id_inclusive=ids[3],
        ).collect(engine=engine)

    monkeypatch.setattr(_RotatingKms, "denied", {"c"})
    assert_frame_equal(
        pl.scan_iceberg(
            str(path),
            storage_options=options,
            to_snapshot_id_inclusive=ids[3],
        )
        .sort("x")
        .collect(engine=engine),
        pl.DataFrame({"x": [0, 1, 10, 11, 20, 21, 30, 31]}),
    )
    with pytest.raises(pl.exceptions.ComputeError, match="KMS unwrap_key failed"):
        pl.scan_iceberg(
            str(path),
            storage_options=options,
            from_snapshot_id_exclusive=ids[0],
        ).collect(engine=engine)

    loaded_table = StaticTable.from_metadata(str(path))
    loaded_table.metadata = loaded_table.metadata.model_copy(
        update={"current_snapshot_id": ids[2]}
    )
    growing = pl.scan_iceberg(
        loaded_table, storage_options=options, from_snapshot_id_exclusive=ids[0]
    )
    fixed = pl.scan_iceberg(
        loaded_table,
        storage_options=options,
        from_snapshot_id_exclusive=ids[0],
        to_snapshot_id_inclusive=ids[2],
    )
    assert_frame_equal(growing.sort("x").collect(engine=engine), expected.head(4))
    assert_frame_equal(fixed.sort("x").collect(engine=engine), expected.head(4))
    loaded_table.metadata = loaded_table.metadata.model_copy(
        update={"current_snapshot_id": ids[3]}
    )
    assert_frame_equal(growing.sort("x").collect(engine=engine), expected)
    assert_frame_equal(fixed.sort("x").collect(engine=engine), expected.head(4))

    _RotatingKms.calls.clear()
    with pytest.raises(ValueError, match="not a parent ancestor"):
        pl.scan_iceberg(
            str(path),
            storage_options=options,
            from_snapshot_id_exclusive=ids[3],
            to_snapshot_id_inclusive=ids[1],
        ).collect(engine=engine)
    assert _RotatingKms.calls == []


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
@pytest.mark.parametrize(
    ("delete_filter", "operation"),
    [("x == 1", Operation.OVERWRITE), ("x >= 1", Operation.DELETE)],
)
def test_encrypted_incremental_append_skips_non_append(
    tmp_path: Path,
    engine: EngineType,
    monkeypatch: pytest.MonkeyPatch,
    delete_filter: str,
    operation: Operation,
) -> None:
    table, _ = new_iceberg_table(
        tmp_path, schema=Schema(NestedField(1, "x", LongType()))
    )
    table.append(pl.DataFrame({"x": [1, 2]}).to_arrow())
    first = table.current_snapshot()
    assert first is not None
    table.delete(delete_filter)
    current = table.current_snapshot()
    assert current is not None
    assert current.summary is not None
    assert current.summary.operation == operation
    path = write_encrypted_snapshot_metadata(table, tmp_path, ["a", "b"])
    monkeypatch.setattr(_RotatingKms, "calls", [])
    monkeypatch.setattr(_RotatingKms, "denied", {"a", "b"})
    result = pl.scan_iceberg(
        str(path),
        storage_options={"py-kms-impl": f"{__name__}._RotatingKms"},
        from_snapshot_id_exclusive=first.snapshot_id,
    ).collect(engine=engine)
    assert_frame_equal(result, pl.DataFrame(schema={"x": pl.Int64}))
    assert _RotatingKms.calls == []


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
@pytest.mark.filterwarnings("ignore:Call to to_vector, deprecated:DeprecationWarning")
@pytest.mark.parametrize("encrypt_data", [False, True])
def test_puffin_deletion_vector_with_encryption(
    tmp_path: Path, engine: EngineType, encrypt_data: bool
) -> None:
    import zlib

    from pyroaring import BitMap  # type: ignore[import-not-found]

    data_path = tmp_path / "data.parquet"
    pl.DataFrame({"x": [10, 11, 12, 13]}).write_parquet(data_path)
    vector = (1).to_bytes(8, "little") + bytes(4) + BitMap([1, 3]).serialize()
    body = bytes.fromhex("d1d33964") + vector
    blob = len(body).to_bytes(4, "big") + body + zlib.crc32(body).to_bytes(4, "big")
    footer = json.dumps(
        {
            "blobs": [
                {
                    "type": "deletion-vector-v1",
                    "fields": [2147483546],
                    "snapshot-id": 1,
                    "sequence-number": 1,
                    "offset": 4,
                    "length": len(blob),
                    "properties": {"referenced-data-file": str(data_path)},
                }
            ]
        }
    ).encode()
    puffin = (
        b"PFA1"
        + blob
        + b"PFA1"
        + footer
        + len(footer).to_bytes(4, "little")
        + bytes(4)
        + b"PFA1"
    )
    path = tmp_path / "deletes.puffin"
    key, aad = b"k" * 24, b"puffin"
    if encrypt_data:
        key = b"d" * 16
        pq.write_table(
            pl.DataFrame({"x": [10, 11, 12, 13]}).to_arrow(),
            data_path,
            encryption_properties=pe.create_encryption_properties(key),
        )
        path.write_bytes(puffin)
        decryption = pl.ParquetDecryptionProperties(footer_key=key)
    else:
        path.write_bytes(encrypt_stream(puffin, key, aad))
        decryption = pl.ParquetDecryptionProperties._from_iceberg_keys(
            {str(path): (key, aad)}
        )
    q = pl.scan_parquet(
        data_path,
        decryption=decryption,
        _deletion_files=("iceberg", ({}, {0: str(path)})),
    )
    expected = pl.DataFrame({"x": [10, 12]})
    assert_frame_equal(q.collect(engine=engine), expected)
    assert_frame_equal(q.tail(1).collect(engine=engine), expected.tail(1))
    assert q.select(pl.len()).collect(engine=engine).item() == 2


@pytest.mark.slow
@pytest.mark.write_disk
@pytest.mark.filterwarnings(
    "ignore:No preferred file implementation for scheme*:UserWarning"
)
@pytest.mark.ci_only
class TestIcebergKms:
    """KMS configuration and encrypted scan integration."""

    def test_scan_iceberg_kms_implementation(self, iceberg_path: str) -> None:
        table = StaticTable.from_metadata(iceberg_path)
        table.catalog.properties.update(
            {
                "py-kms-impl": f"{__name__}._TestKmsClient",
                "property-precedence": "catalog",
            }
        )
        table.config = {
            "config-property": "config",
            "property-precedence": "config",
        }
        table.metadata.properties["property-precedence"] = "table"

        q = pl.scan_iceberg(table)
        assert q.select("id").head(2).collect().shape == (2, 1)
        assert q.filter(pl.col("id") > 1).select("str").sort(
            "str"
        ).collect().rows() == [
            ("2",),
            ("3",),
        ]

        scan_data = new_iceberg_scan_resolver(table)
        resolved = scan_data._to_dataset_scan_impl(projection=["str"])

        assert isinstance(resolved, _NativeIcebergScanData)
        assert resolved.projected_iceberg_schema.column_names == ["str"]
        config = _load_kms_config(table)
        assert config is not None
        assert isinstance(config.client, _TestKmsClient)
        assert config.client.properties["config-property"] == "config"
        assert config.client.properties["property-precedence"] == "table"

    @pytest.mark.parametrize(
        ("kms_impl", "error", "match"),
        [
            ("InvalidKms", ValueError, "fully qualified class name"),
            ("invalid_module.InvalidKms", ImportError, "failed to import"),
            (
                f"{__name__}.InvalidKms",
                ImportError,
                "KMS class 'InvalidKms' not found",
            ),
            (123, TypeError, "must be a string"),
            (
                f"{__name__}._KmsWithRequiredArgument",
                TypeError,
                "failed to construct",
            ),
            (
                f"{__name__}._KmsWithoutInitialize",
                TypeError,
                "has no initialize",
            ),
            (
                f"{__name__}._KmsWithoutUnwrap",
                TypeError,
                "has no unwrap_key",
            ),
        ],
    )
    def test_load_invalid_kms_implementation(
        self,
        iceberg_path: str,
        kms_impl: Any,
        error: type[Exception],
        match: str,
    ) -> None:
        table = StaticTable.from_metadata(iceberg_path)
        table.config = {"py-kms-impl": kms_impl}

        with pytest.raises(error, match=match):
            _load_kms_config(table)

    def test_kms_implementation_is_not_loaded_from_table_metadata(
        self, iceberg_path: str
    ) -> None:
        table = StaticTable.from_metadata(iceberg_path)
        table.metadata.properties["py-kms-impl"] = f"{__name__}._TestKmsClient"

        assert _load_kms_config(table) is None

    @pytest.mark.parametrize("source", ["catalog", "config", "storage_options"])
    def test_native_kms_configuration(self, iceberg_path: str, source: str) -> None:
        table = StaticTable.from_metadata(iceberg_path)
        properties = {"encryption.kms-type": "aws", "region_name": "us-east-1"}
        storage_options = None
        if source == "catalog":
            table.catalog.properties.update(properties)
        elif source == "config":
            table.config = properties
        else:
            storage_options = properties

        assert _load_kms_config(table, storage_options) is None
        table.metadata.properties.update(
            {
                "encryption.key-id": "master-1",
                "encryption.kms-type": "untrusted",
                "kms.endpoint": "https://untrusted.invalid",
                "region_name": "untrusted",
            }
        )
        config = _load_kms_config(table, storage_options)
        assert config is not None
        assert config.client is None
        assert config.properties["encryption.kms-type"] == "aws"
        assert config.properties["region_name"] == "us-east-1"
        assert "kms.endpoint" not in config.properties

    @pytest.mark.parametrize("kms_type", ["azure", "gcp", "unknown"])
    def test_unsupported_native_kms(self, iceberg_path: str, kms_type: str) -> None:
        table = StaticTable.from_metadata(iceberg_path)
        table.metadata.properties["encryption.key-id"] = "master-1"
        table.config = {"encryption.kms-type": kms_type}

        with pytest.raises(NotImplementedError, match="unsupported Iceberg KMS type"):
            pl.scan_iceberg(table).collect()

    def test_conflicting_kms_configuration(self, iceberg_path: str) -> None:
        table = StaticTable.from_metadata(iceberg_path)
        table.config = {
            "encryption.kms-type": "aws",
            "py-kms-impl": f"{__name__}._TestKmsClient",
        }
        with pytest.raises(ValueError, match="configure only one"):
            pl.scan_iceberg(table).collect()

    def test_native_kms_configuration_precedence(self, iceberg_path: str) -> None:
        table = StaticTable.from_metadata(iceberg_path)
        table.metadata.properties["encryption.key-id"] = "master-1"
        table.catalog.properties.update(
            {"encryption.kms-type": "aws", "region_name": "catalog"}
        )
        table.config = {"region_name": "config"}
        config = _load_kms_config(table)
        assert config is not None
        assert config.properties["region_name"] == "config"
        config = _load_kms_config(table, {"region_name": "explicit"})
        assert config is not None
        assert config.properties["region_name"] == "explicit"

    def test_native_kms_incremental_scan_validates_range(
        self, iceberg_path: str
    ) -> None:
        table = StaticTable.from_metadata(iceberg_path)
        table.metadata.properties["encryption.key-id"] = "master-1"
        table.config = {
            "encryption.kms-type": "aws",
            "region_name": "us-east-1",
            "aws_access_key_id": "test",
            "aws_secret_access_key": "test",
        }
        with pytest.raises(ValueError, match="not a parent ancestor"):
            pl.scan_iceberg(table, from_snapshot_id_exclusive=1).collect()

    @pytest.mark.parametrize("kms_type", ["python", "aws", "aws_env", "aws_denied"])
    @pytest.mark.parametrize("engine", ["streaming", "in-memory"])
    @pytest.mark.parametrize("deletes", ["positions", "vectors"])
    @pytest.mark.filterwarnings(
        "ignore:Call to to_vector, deprecated:DeprecationWarning"
    )
    def test_scan_iceberg_upstream_encrypted_deletes(
        self,
        io_files_path: Path,
        kms_type: str,
        engine: EngineType,
        deletes: str,
        monkeypatch: pytest.MonkeyPatch,
        iceberg_kms_endpoint: tuple[
            str, list[tuple[str | None, dict[str, Any]]], dict[str, str]
        ],
    ) -> None:
        fixture_path = io_files_path / "iceberg-encrypted"
        monkeypatch.chdir(fixture_path)
        metadata_path = fixture_path / f"{deletes}.metadata.json"
        metadata = json.loads(metadata_path.read_text())
        wrapping_key = next(
            key
            for key in metadata["encryption-keys"]
            if key["encrypted-by-id"] == "test-master"
        )
        endpoint, calls, response = iceberg_kms_endpoint
        response["Plaintext"] = wrapping_key["encrypted-key-metadata"]
        response["KeyId"] = "test-master"
        storage_options = (
            {"py-kms-impl": f"{__name__}._IdentityKms"}
            if kms_type == "python"
            else {
                "encryption.kms-type": "aws",
                "kms.endpoint": endpoint,
                "region_name": "us-east-1",
                "aws_access_key_id": "test",
                "aws_secret_access_key": "test",
            }
        )
        if kms_type == "aws_env":
            monkeypatch.setenv(
                "AWS_ACCESS_KEY_ID", storage_options.pop("aws_access_key_id")
            )
            monkeypatch.setenv(
                "AWS_SECRET_ACCESS_KEY", storage_options.pop("aws_secret_access_key")
            )
            monkeypatch.setenv("AWS_REGION", storage_options.pop("region_name"))
        q = pl.scan_iceberg(str(metadata_path), storage_options=storage_options)
        if kms_type == "aws_denied":
            response.clear()
            response.update({"__type": "AccessDeniedException", "message": "denied"})
            with pytest.raises(
                pl.exceptions.ComputeError, match="AWS KMS Decrypt failed"
            ):
                q.collect(engine=engine)
            assert len(calls) == 1
            return
        expected = pl.DataFrame({"x": [0, 2]})
        assert_frame_equal(q.collect(engine=engine), expected)
        if kms_type in {"aws", "aws_env"}:
            assert calls == [
                (
                    "TrentService.Decrypt",
                    {
                        "CiphertextBlob": wrapping_key["encrypted-key-metadata"],
                        "KeyId": "test-master",
                        "EncryptionAlgorithm": "SYMMETRIC_DEFAULT",
                    },
                )
            ]
        else:
            assert calls == []
        assert_frame_equal(
            q.filter(pl.col("x") > 0).collect(engine=engine), expected.tail(1)
        )
        assert_frame_equal(q.tail(1).collect(engine=engine), expected.tail(1))
        assert q.select(pl.len()).collect(engine=engine).item() == 2


@pytest.mark.parametrize("include_length", [False, True])
def test_scan_iceberg_rejects_whole_block_truncation(
    tmp_path: Path, include_length: bool
) -> None:
    import copy

    from pyiceberg.manifest import write_manifest_list

    from polars.io.iceberg._encryption import EncryptedFileIO

    table, _ = new_iceberg_table(
        tmp_path, schema=Schema(NestedField(1, "x", LongType()))
    )
    table.append(pl.DataFrame({"x": [42]}).to_arrow())
    table.append(pl.DataFrame({"x": [43]}).to_arrow())
    metadata_path = write_encrypted_snapshot_metadata(table, tmp_path, [None, "a"])
    loaded = StaticTable.from_metadata(str(metadata_path))
    snapshot = loaded.current_snapshot()
    assert snapshot is not None
    manifests = [
        copy.deepcopy(manifest)
        for manifest in snapshot.manifests(
            EncryptedFileIO(loaded, loaded.io, _RotatingKms())
        )
    ]
    list_path = tmp_path / "aligned-list.avro"
    padding = 0
    # Align Avro and AGS1 blocks so truncation leaves valid Avro.
    for _ in range(4):
        writer = write_manifest_list(
            2,
            table.io.new_output(str(list_path)),
            snapshot.snapshot_id,
            snapshot.parent_snapshot_id,
            snapshot.sequence_number,
            "null",
        )
        writer._meta["padding"] = "x" * padding
        with writer:
            writer.add_manifests(manifests[:1])
            boundary = writer._writer.tell()
            writer.add_manifests(manifests[1:])
        if boundary == 1024 * 1024:
            break
        padding += 1024 * 1024 - boundary
    assert boundary == 1024 * 1024
    key, aad = b"k" * 16, b"manifest-list"
    ciphertext = encrypt_stream(list_path.read_bytes(), key, aad)
    list_path.write_bytes(ciphertext)
    metadata = json.loads(metadata_path.read_text())
    item = metadata["snapshots"][-1]
    item["manifest-list"] = str(list_path)
    entry = next(
        key for key in metadata["encryption-keys"] if key["key-id"] == item["key-id"]
    )
    entry["encrypted-key-metadata"] = base64.b64encode(
        encrypt_block(key_metadata(key, aad, len(ciphertext)), b"a" * 16, b"123")
    ).decode()
    metadata_path.write_text(json.dumps(metadata))
    options = {"py-kms-impl": f"{__name__}._RotatingKms"}
    assert_frame_equal(
        pl.scan_iceberg(str(metadata_path), storage_options=options)
        .sort("x")
        .collect(),
        pl.DataFrame({"x": [42, 43]}),
    )
    if not include_length:
        entry["encrypted-key-metadata"] = base64.b64encode(
            encrypt_block(key_metadata(key, aad), b"a" * 16, b"123")
        ).decode()
        metadata_path.write_text(json.dumps(metadata))
    list_path.write_bytes(ciphertext[: 8 + 1024 * 1024 + 12 + 16])
    clear_manifest_cache()
    with pytest.raises(pl.exceptions.ComputeError, match="file length"):
        pl.scan_iceberg(str(metadata_path), storage_options=options).collect()


@pytest.mark.parametrize("config_native", [False, True])
@pytest.mark.parametrize("explicit", [None, "iceberg", "native"])
def test_encrypted_scan_storage_credential_precedence(
    tmp_path: Path, config_native: bool, explicit: str | None
) -> None:
    table, _ = new_iceberg_table(
        tmp_path, schema=Schema(NestedField(1, "x", LongType()))
    )
    table.append(pl.DataFrame({"x": [42]}).to_arrow())
    metadata_path = write_encrypted_snapshot_metadata(table, tmp_path, ["a"])
    table = StaticTable.from_metadata(str(metadata_path))
    iceberg_keys = ["s3.access-key-id", "s3.secret-access-key", "s3.session-token"]
    native_keys = ["aws_access_key_id", "aws_secret_access_key", "aws_session_token"]
    table.config = {
        "py-kms-impl": f"{__name__}._RotatingKms",
        **dict.fromkeys(native_keys if config_native else iceberg_keys, "stale"),
    }
    table.io.properties.update(dict.fromkeys(iceberg_keys, "vended"))
    resolver = new_iceberg_scan_resolver(table)
    if explicit:
        resolver.table.iceberg_storage_properties = dict.fromkeys(
            native_keys if explicit == "native" else iceberg_keys, "explicit"
        )
    resolved = resolver._to_dataset_scan_impl()
    assert isinstance(resolved, _NativeIcebergScanData)
    assert resolved.storage_options == dict.fromkeys(
        native_keys, "explicit" if explicit else "vended"
    )
    assert_frame_equal(resolved.to_lazyframe().collect(), pl.DataFrame({"x": [42]}))


@pytest.mark.parametrize("reverse", [False, True])
def test_iceberg_storage_aliases_match_file_io_precedence(reverse: bool) -> None:
    from polars.io.iceberg._dataset import (
        _convert_iceberg_to_object_store_storage_options,
    )

    items = [("s3.access-key-id", "vended"), ("aws_access_key_id", "catalog")]
    if reverse:
        items.reverse()
    assert _convert_iceberg_to_object_store_storage_options(dict(items)) == {
        "aws_access_key_id": "vended"
    }
