from __future__ import annotations

import base64
import copy
import gzip
import io
import json
import os
import pickle
from typing import TYPE_CHECKING, ClassVar

import pyarrow.parquet as pq
import pyarrow.parquet.encryption as pe
import pytest
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pyiceberg.avro.encoder import BinaryEncoder
from pyiceberg.io.pyarrow import PyArrowFileIO
from pyiceberg.manifest import (
    clear_manifest_cache,
    write_manifest,
    write_manifest_list,
)
from pyiceberg.schema import Schema
from pyiceberg.table import StaticTable
from pyiceberg.table.snapshots import Operation
from pyiceberg.types import LongType, NestedField

import polars as pl
from polars import _plr
from polars.io.iceberg._encryption import (
    _BytesInputFile,
    _EncryptedInputFile,
    decode_key_metadata,
)
from polars.testing import assert_frame_equal
from tests.unit.io.test_iceberg import (
    new_iceberg_table,
    sqlcatalog_uses_null_pool_connections,  # noqa: F401
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyiceberg.manifest import ManifestFile
    from pyiceberg.table import Table

    from polars._typing import EngineType


KEK = b"0123456789abcdef"


class _TestKms:
    def initialize(self, properties: dict[str, str]) -> None:
        pass

    def unwrap_key(self, wrapped_key: bytes, wrapping_key_id: str) -> bytes:
        assert wrapped_key == b"wrapped-kek"
        assert wrapping_key_id == "master"
        return KEK


def key_metadata(key: bytes, aad: bytes, length: int | None = None) -> bytes:
    output = io.BytesIO(b"\x01")
    output.seek(1)
    encoder = BinaryEncoder(output)
    encoder.write_bytes(key)
    encoder.write_int(1)
    encoder.write_bytes(aad)
    encoder.write_int(int(length is not None))
    if length is not None:
        encoder.write_int(length)
    return output.getvalue()


def encrypt_block(plaintext: bytes, key: bytes, aad: bytes) -> bytes:
    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, aad)


def encrypt_stream(plaintext: bytes, key: bytes, aad: bytes) -> bytes:
    block_size = 1024 * 1024
    return (
        b"AGS1"
        + block_size.to_bytes(4, "little")
        + b"".join(
            encrypt_block(
                plaintext[start : start + block_size],
                key,
                aad + i.to_bytes(4, "little"),
            )
            for i, start in enumerate(range(0, max(len(plaintext), 1), block_size))
        )
    )


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
    manifests = snapshot.manifests(table.io)
    for i, manifest in enumerate(manifests):
        entries = manifest.fetch_manifest_entry(table.io)
        for j, entry in enumerate(entries):
            with table.io.new_input(entry.data_file.file_path).open() as stream:
                original = pq.read_table(stream)
            path = tmp_path / f"data-{i}-{j}.parquet"
            key, aad = os.urandom(16), os.urandom(16)
            pq.write_table(
                original,
                path,
                encryption_properties=pe.create_encryption_properties(
                    key, aad_prefix=aad, store_aad_prefix=False
                ),
            )
            entry.data_file[1] = str(path)
            entry.data_file[5] = path.stat().st_size
            entry.data_file[12] = key_metadata(key, aad)
        manifest_path = tmp_path / f"manifest-{i}.avro"
        with write_manifest(
            2,
            table.spec(),
            table.schema(),
            table.io.new_output(str(manifest_path)),
            snapshot.snapshot_id,
            "null",
        ) as writer:
            for entry in entries:
                writer.add_entry(entry)
        content = manifest_path.read_bytes()
        key, aad = os.urandom(24), os.urandom(16)
        manifest_path.write_bytes(encrypt_stream(content, key, aad))
        manifest[0] = str(manifest_path)
        manifest[1] = manifest_path.stat().st_size
        manifest[14] = key_metadata(key, aad, manifest_path.stat().st_size)
    list_path = tmp_path / "manifest-list.avro"
    with write_manifest_list(
        2,
        table.io.new_output(str(list_path)),
        snapshot.snapshot_id,
        snapshot.parent_snapshot_id,
        snapshot.sequence_number,
        "null",
    ) as writer:
        writer.add_manifests(manifests)
    content = list_path.read_bytes()
    key, aad = os.urandom(32), os.urandom(16)
    list_path.write_bytes(encrypt_stream(content, key, aad))
    metadata = table.metadata.model_dump(mode="json")
    for item in metadata["snapshots"]:
        if item["snapshot-id"] == snapshot.snapshot_id:
            item["manifest-list"] = str(list_path)
            item["key-id"] = "list-key"
    metadata["properties"]["encryption.key-id"] = "master"
    metadata["encryption-keys"] = [
        {
            "key-id": "kek",
            "encrypted-by-id": "master",
            "encrypted-key-metadata": base64.b64encode(b"wrapped-kek").decode(),
            "properties": {"KEY_TIMESTAMP": "123"},
        },
        {
            "key-id": "list-key",
            "encrypted-by-id": "kek",
            "encrypted-key-metadata": base64.b64encode(
                encrypt_block(
                    key_metadata(key, aad, list_path.stat().st_size), KEK, b"123"
                )
            ).decode(),
        },
    ]
    metadata_path = tmp_path / (
        "v1.gz.metadata.json" if compressed else "v1.metadata.json"
    )
    payload = json.dumps(metadata).encode()
    metadata_path.write_bytes(gzip.compress(payload) if compressed else payload)
    clear_manifest_cache()
    storage_options = {"py-kms-impl": f"{__name__}._TestKms"}
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


def write_encrypted_snapshot_metadata(
    table: Table, tmp_path: Path, key_ids: list[str | None]
) -> Path:
    metadata = table.metadata.model_dump(mode="json")
    metadata["properties"]["encryption.key-id"] = "current-master"
    keys = [
        {
            "key-id": f"kek-{key_id}",
            "encrypted-by-id": key_id,
            "encrypted-key-metadata": base64.b64encode(key_id.encode()).decode(),
            "properties": {"KEY_TIMESTAMP": "123"},
        }
        for key_id in sorted({key_id for key_id in key_ids if key_id is not None})
    ]
    rewritten: dict[str, ManifestFile] = {}
    for snapshot, item, key_id in zip(
        table.snapshots(), metadata["snapshots"], key_ids, strict=True
    ):
        manifests = []
        for original_manifest in snapshot.manifests(table.io):
            if original_manifest.manifest_path not in rewritten:
                manifest = copy.deepcopy(original_manifest)
                if key_id is not None:
                    entries = manifest.fetch_manifest_entry(
                        table.io, discard_deleted=False
                    )
                    for j, entry in enumerate(entries):
                        with table.io.new_input(
                            entry.data_file.file_path
                        ).open() as stream:
                            original = pq.read_table(stream)
                        data_path = tmp_path / f"data-{len(rewritten)}-{j}.parquet"
                        key, aad = os.urandom(16), os.urandom(16)
                        pq.write_table(
                            original,
                            data_path,
                            encryption_properties=pe.create_encryption_properties(
                                key, aad_prefix=aad, store_aad_prefix=False
                            ),
                        )
                        entry.data_file[1] = str(data_path)
                        entry.data_file[5] = data_path.stat().st_size
                        entry.data_file[12] = key_metadata(key, aad)
                    manifest_path = tmp_path / f"manifest-{len(rewritten)}.avro"
                    with write_manifest(
                        2,
                        table.spec(),
                        table.schema(),
                        table.io.new_output(str(manifest_path)),
                        snapshot.snapshot_id,
                        "null",
                    ) as writer:
                        for entry in entries:
                            writer.add_entry(entry)
                    content = manifest_path.read_bytes()
                    key, aad = os.urandom(24), os.urandom(16)
                    manifest_path.write_bytes(encrypt_stream(content, key, aad))
                    manifest[0] = str(manifest_path)
                    manifest[1] = manifest_path.stat().st_size
                    manifest[14] = key_metadata(key, aad, manifest_path.stat().st_size)
                rewritten[original_manifest.manifest_path] = manifest
            manifests.append(copy.deepcopy(rewritten[original_manifest.manifest_path]))
        list_path = tmp_path / f"list-{snapshot.snapshot_id}.avro"
        with write_manifest_list(
            2,
            table.io.new_output(str(list_path)),
            snapshot.snapshot_id,
            snapshot.parent_snapshot_id,
            snapshot.sequence_number,
            "null",
        ) as writer:
            writer.add_manifests(manifests)
        if key_id is not None:
            key, aad = os.urandom(32), os.urandom(16)
            list_path.write_bytes(encrypt_stream(list_path.read_bytes(), key, aad))
            item["key-id"] = f"list-{snapshot.snapshot_id}"
            keys.append(
                {
                    "key-id": item["key-id"],
                    "encrypted-by-id": f"kek-{key_id}",
                    "encrypted-key-metadata": base64.b64encode(
                        encrypt_block(
                            key_metadata(key, aad, list_path.stat().st_size),
                            key_id.encode() * 16,
                            b"123",
                        )
                    ).decode(),
                }
            )
        item["manifest-list"] = str(list_path)
    metadata["encryption-keys"] = keys
    path = tmp_path / "encrypted.metadata.json"
    path.write_text(json.dumps(metadata))
    return path


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
