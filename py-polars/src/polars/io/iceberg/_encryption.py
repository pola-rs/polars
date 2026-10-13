from __future__ import annotations

import base64
import io
import json
from typing import TYPE_CHECKING, Any

from pyiceberg.avro.decoder import new_decoder
from pyiceberg.io import FileIO, InputFile
from pyiceberg.manifest import read_manifest_list
from pyiceberg.serializers import Compressor

from polars import _plr
from polars.exceptions import ComputeError

if TYPE_CHECKING:
    from pyiceberg.io import InputStream, OutputFile
    from pyiceberg.table import Table


def decode_key_metadata(value: bytes) -> tuple[bytes, bytes | None, int | None]:
    if not value or value[0] != 1:
        msg = "Unsupported Iceberg key metadata version"
        raise ComputeError(msg)
    try:
        decoder = new_decoder(value[1:])
        key = decoder.read_bytes()
        aad_tag = decoder.read_int()
        if aad_tag not in (0, 1):
            raise ValueError  # noqa: TRY301
        aad = decoder.read_bytes() if aad_tag else None
        length_tag = decoder.read_int()
        if length_tag not in (0, 1):
            raise ValueError  # noqa: TRY301
        length = decoder.read_int() if length_tag else None
        if (
            len(key) not in (16, 24, 32)
            or (length is not None and length < 0)
            or decoder.tell() != len(value) - 1
        ):
            raise ValueError  # noqa: TRY301
    except Exception:
        msg = "Invalid Iceberg key metadata"
        raise ComputeError(msg) from None
    return key, aad, length


class _BytesInputFile(InputFile):
    def __init__(self, location: str, content: bytes) -> None:
        super().__init__(location)
        self._content = content

    def __len__(self) -> int:
        return len(self._content)

    def exists(self) -> bool:
        return True

    def open(self, seekable: bool = True) -> InputStream:  # noqa: ARG002, FBT001
        return io.BytesIO(self._content)

    def __getstate__(self) -> Any:
        msg = "Cannot serialize decrypted Iceberg metadata"
        raise TypeError(msg)


class _EncryptedInputFile(InputFile):
    def __init__(self, inner: InputFile, metadata: bytes) -> None:
        super().__init__(inner.location)
        self._inner = inner
        self._key, self._aad, self._length = decode_key_metadata(metadata)

    def __len__(self) -> int:
        with self.open() as stream:
            return len(stream.read())

    def exists(self) -> bool:
        return self._inner.exists()

    def open(self, seekable: bool = True) -> InputStream:  # noqa: ARG002, FBT001
        with self._inner.open() as stream:
            ciphertext = stream.read()
        if self._length is not None and len(ciphertext) != self._length:
            msg = "Iceberg encrypted file length does not match key metadata"
            raise ComputeError(msg)
        plaintext = _plr._iceberg_decrypt(ciphertext, self._key, self._aad or b"", True)
        return io.BytesIO(plaintext)

    def __getstate__(self) -> Any:
        msg = "Cannot serialize Iceberg decryption keys"
        raise TypeError(msg)


class EncryptedFileIO(FileIO):
    def __init__(
        self, table: Table, inner: FileIO, snapshot_id: int | None, kms_client: Any
    ) -> None:
        super().__init__(inner.properties)
        self._inner = inner
        self._manifest_lists: dict[str, bytes] = {}
        self._keys: dict[str, bytes] = {}
        with inner.new_input(table.metadata_location).open() as stream:
            compression = Compressor.get_compressor(table.metadata_location)
            with compression.stream_decompressor(stream) as decompressed:
                metadata = json.load(decompressed)
        encryption_keys = {
            entry["key-id"]: entry for entry in metadata.get("encryption-keys", [])
        }
        selected_id = (
            snapshot_id
            if snapshot_id is not None
            else table.metadata.current_snapshot_id
        )
        for snapshot in metadata.get("snapshots", []):
            if snapshot["snapshot-id"] != selected_id:
                continue
            location = snapshot["manifest-list"]
            source = inner.new_input(location)
            if (key_id := snapshot.get("key-id")) is not None:
                try:
                    entry = encryption_keys[key_id]
                    kek = encryption_keys[entry["encrypted-by-id"]]
                    # The timestamp is authenticated when the file key is wrapped.
                    aad = kek["properties"]["KEY_TIMESTAMP"].encode()
                    wrapped_kek = base64.b64decode(
                        kek["encrypted-key-metadata"], validate=True
                    )
                    wrapping_key_id = kek["encrypted-by-id"]
                    wrapped_metadata = base64.b64decode(
                        entry["encrypted-key-metadata"], validate=True
                    )
                except (KeyError, ValueError, TypeError):
                    msg = "Invalid Iceberg encryption key hierarchy"
                    raise ComputeError(msg) from None
                try:
                    kek_bytes = kms_client.unwrap_key(wrapped_kek, wrapping_key_id)
                except Exception:
                    msg = "Iceberg KMS unwrap_key failed (AWS KMS Decrypt failed for native AWS)"
                    raise ComputeError(msg) from None
                key_metadata = _plr._iceberg_decrypt(
                    wrapped_metadata, kek_bytes, aad, False
                )
                source = _EncryptedInputFile(source, key_metadata)
            with source.open() as stream:
                content = stream.read()
            self._manifest_lists[location] = content
            for manifest in read_manifest_list(_BytesInputFile(location, content)):
                if manifest.key_metadata:
                    self._keys[manifest.manifest_path] = manifest.key_metadata

    def new_input(self, location: str) -> InputFile:
        if (content := self._manifest_lists.get(location)) is not None:
            return _BytesInputFile(location, content)
        source = self._inner.new_input(location)
        if (metadata := self._keys.get(location)) is not None:
            return _EncryptedInputFile(source, metadata)
        return source

    def new_output(self, location: str) -> OutputFile:
        msg = "Encrypted Iceberg FileIO only supports reading"
        raise NotImplementedError(msg)

    def delete(self, location: str | InputFile | OutputFile) -> None:
        msg = "Encrypted Iceberg FileIO only supports reading"
        raise NotImplementedError(msg)

    def __getstate__(self) -> Any:
        msg = "Cannot serialize decrypted Iceberg metadata"
        raise TypeError(msg)
