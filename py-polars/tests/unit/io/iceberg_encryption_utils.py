from __future__ import annotations

import base64
import copy
import io
import json
import os
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq
import pyarrow.parquet.encryption as pe
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from pyiceberg.avro.encoder import BinaryEncoder
from pyiceberg.manifest import (
    DATA_FILE_TYPE,
    DEFAULT_READ_VERSION,
    MANIFEST_LIST_FILE_STRUCTS,
    write_manifest,
    write_manifest_list,
)

if TYPE_CHECKING:
    from pathlib import Path

    from pyiceberg.manifest import ManifestFile
    from pyiceberg.table import Table
    from pyiceberg.typedef import Record
    from pyiceberg.types import StructType


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


def _update_fields(record: Record, schema: StructType, **values: Any) -> None:
    for index, field in enumerate(schema.fields):
        if field.name in values:
            record[index] = values[field.name]


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
                        _update_fields(
                            entry.data_file,
                            DATA_FILE_TYPE[DEFAULT_READ_VERSION],
                            file_path=str(data_path),
                            file_size_in_bytes=data_path.stat().st_size,
                            key_metadata=key_metadata(key, aad),
                        )
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
                    _update_fields(
                        manifest,
                        MANIFEST_LIST_FILE_STRUCTS[DEFAULT_READ_VERSION],
                        manifest_path=str(manifest_path),
                        manifest_length=manifest_path.stat().st_size,
                        key_metadata=key_metadata(
                            key, aad, manifest_path.stat().st_size
                        ),
                    )
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
