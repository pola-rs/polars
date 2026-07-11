from __future__ import annotations

import base64
import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from os import PathLike

_KEY_MATERIAL_TYPE = "PKMT1"
_KEY_MATERIAL_FILE_PREFIX = "_KEY_MATERIAL_FOR_"
_KEY_MATERIAL_FILE_SUFFIX = ".json"


class KmsClient:
    """Interface implemented by a user-provided key management client."""

    def wrap_key(self, key_bytes: bytes, master_key_identifier: str) -> str:
        """Encrypt a data key with the identified master key."""
        raise NotImplementedError

    def unwrap_key(self, wrapped_key: str, master_key_identifier: str) -> bytes:
        """Recover a data key with the identified master key."""
        raise NotImplementedError


@dataclass
class KmsConnectionConfig:
    """Connection information passed unchanged to the KMS client factory."""

    kms_instance_id: str = ""
    kms_instance_url: str = ""
    key_access_token: str = "DEFAULT"
    custom_kms_conf: dict[str, str] = field(default_factory=dict)

    def refresh_key_access_token(self, value: str) -> None:
        """Replace the access token used to partition CryptoFactory caches."""
        self.key_access_token = value


class EncryptionConfiguration:
    """High-level envelope-encryption settings for one Parquet file."""

    def __init__(
        self,
        footer_key: str,
        column_keys: Mapping[str, Sequence[str]] | None = None,
        *,
        uniform_encryption: bool = False,
        encryption_algorithm: str = "AES_GCM_V1",
        plaintext_footer: bool = False,
        double_wrapping: bool = True,
        cache_lifetime: timedelta = timedelta(minutes=10),
        internal_key_material: bool = True,
        data_key_length_bits: int = 128,
    ) -> None:
        if not footer_key:
            msg = "footer_key must not be empty"
            raise ValueError(msg)
        if encryption_algorithm not in {"AES_GCM_V1", "AES_GCM_CTR_V1"}:
            msg = f"unsupported Parquet encryption algorithm: {encryption_algorithm}"
            raise ValueError(msg)
        if data_key_length_bits not in {128, 192, 256}:
            msg = "data_key_length_bits must be 128, 192, or 256"
            raise ValueError(msg)
        normalized_column_keys = {
            str(master_key): [str(column) for column in columns]
            for master_key, columns in (column_keys or {}).items()
        }
        if uniform_encryption == bool(normalized_column_keys):
            msg = "exactly one of uniform_encryption or column_keys must be configured"
            raise ValueError(msg)
        if plaintext_footer and uniform_encryption:
            msg = "plaintext_footer cannot be used with uniform_encryption"
            raise ValueError(msg)

        self.footer_key = footer_key
        self.column_keys = normalized_column_keys
        self.uniform_encryption = uniform_encryption
        self.encryption_algorithm = encryption_algorithm
        self.plaintext_footer = plaintext_footer
        self.double_wrapping = double_wrapping
        self.cache_lifetime = cache_lifetime
        self.internal_key_material = internal_key_material
        self.data_key_length_bits = data_key_length_bits


class DecryptionConfiguration:
    """High-level settings used while recovering Parquet data keys."""

    def __init__(
        self,
        cache_lifetime: timedelta = timedelta(minutes=10),
    ) -> None:
        self.cache_lifetime = cache_lifetime


class FileEncryptionProperties(dict[str, Any]):
    """Opaque native writer properties produced by :class:`CryptoFactory`."""


class FileDecryptionProperties(dict[str, Any]):
    """Opaque native reader properties produced by :class:`CryptoFactory`."""


def _cache_seconds(value: timedelta) -> float:
    seconds = value.total_seconds()
    if seconds < 0:
        msg = "cache_lifetime must not be negative"
        raise ValueError(msg)
    return seconds


def _key_material_path(parquet_file_path: str | PathLike[str]) -> str:
    path = Path(parquet_file_path)
    return str(
        path.with_name(
            f"{_KEY_MATERIAL_FILE_PREFIX}{path.name}{_KEY_MATERIAL_FILE_SUFFIX}"
        )
    )


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode()


def _read_external_material(path: str, filesystem: Any | None) -> dict[str, str]:
    if filesystem is None:
        return json.loads(Path(path).read_text(encoding="utf8"))
    source = filesystem.open_input_file(path)
    try:
        data = source.read()
    finally:
        source.close()
    return json.loads(bytes(data))


def _write_external_material(
    path: str,
    material: Mapping[str, str],
    filesystem: Any | None,
) -> None:
    data = json.dumps(material, separators=(",", ":"), ensure_ascii=False).encode()
    if filesystem is None:
        Path(path).write_bytes(data)
        return
    sink = filesystem.open_output_stream(path)
    try:
        sink.write(data)
    finally:
        sink.close()


def _plr() -> Any:
    import polars._plr as plr

    return plr


def _kms_cache_key(
    config: KmsConnectionConfig,
) -> tuple[str, str, str, tuple[tuple[str, str], ...]]:
    return (
        config.kms_instance_id,
        config.kms_instance_url,
        config.key_access_token,
        tuple(sorted(config.custom_kms_conf.items())),
    )


class _KeyRetriever:
    def __init__(
        self,
        factory: CryptoFactory,
        kms_connection_config: KmsConnectionConfig,
        cache_lifetime: timedelta,
        parquet_file_path: str | PathLike[str] | None,
        filesystem: Any | None,
    ) -> None:
        self._factory = factory
        self._kms_connection_config = kms_connection_config
        self._cache_lifetime_seconds = _cache_seconds(cache_lifetime)
        self._filesystem = filesystem
        self._external_material_path = (
            _key_material_path(parquet_file_path)
            if parquet_file_path is not None
            else None
        )
        self._external_material: dict[str, str] | None = None

    def __call__(self, key_metadata: bytes) -> bytes:
        metadata = json.loads(key_metadata.decode("utf8"))
        if metadata.get("keyMaterialType") != _KEY_MATERIAL_TYPE:
            msg = "unsupported Parquet key material type"
            raise ValueError(msg)
        if metadata.get("internalStorage") is False:
            if self._external_material_path is None:
                msg = "parquet_file_path is required for external key material"
                raise ValueError(msg)
            if self._external_material is None:
                self._external_material = _read_external_material(
                    self._external_material_path, self._filesystem
                )
            reference = metadata["keyReference"]
            try:
                metadata = json.loads(self._external_material[reference])
            except KeyError as exc:
                msg = f"external key material does not contain {reference!r}"
                raise ValueError(msg) from exc
        return self._unwrap_material(metadata)

    def _unwrap_material(self, material: Mapping[str, Any]) -> bytes:
        connection_config = self._kms_connection_config
        if not connection_config.kms_instance_id and material.get("kmsInstanceID"):
            connection_config = KmsConnectionConfig(
                kms_instance_id=material["kmsInstanceID"],
                kms_instance_url=material.get("kmsInstanceURL", ""),
                key_access_token=connection_config.key_access_token,
                custom_kms_conf=dict(connection_config.custom_kms_conf),
            )
        client = self._factory._get_kms_client(
            connection_config, self._cache_lifetime_seconds
        )
        master_key_id = material["masterKeyID"]
        if not material["doubleWrapping"]:
            key = client.unwrap_key(material["wrappedDEK"], master_key_id)
        else:
            encoded_kek_id = material["keyEncryptionKeyID"]
            kek = self._factory._get_or_unwrap_kek(
                connection_config,
                client,
                encoded_kek_id,
                material["wrappedKEK"],
                master_key_id,
                self._cache_lifetime_seconds,
            )
            encrypted_dek = base64.b64decode(material["wrappedDEK"], validate=True)
            aad = base64.b64decode(encoded_kek_id, validate=True)
            key = bytes(_plr()._parquet_decrypt_key_locally(encrypted_dek, kek, aad))
        if len(key) not in {16, 24, 32}:
            msg = f"KMS returned an invalid data key length: {len(key)} bytes"
            raise ValueError(msg)
        return bytes(key)


class CryptoFactory:
    """Creates native Parquet encryption properties from a KMS client factory."""

    def __init__(
        self,
        kms_client_factory: Callable[[KmsConnectionConfig], KmsClient],
    ) -> None:
        if not callable(kms_client_factory):
            msg = "kms_client_factory must be callable"
            raise TypeError(msg)
        self._kms_client_factory = kms_client_factory
        self._lock = threading.RLock()
        self._kms_client_cache: dict[
            tuple[str, str, str, tuple[tuple[str, str], ...]], tuple[float, KmsClient]
        ] = {}
        self._write_kek_cache: dict[
            tuple[tuple[str, str, str, tuple[tuple[str, str], ...]], str],
            tuple[float, bytes, bytes, str],
        ] = {}
        self._read_kek_cache: dict[
            tuple[tuple[str, str, str, tuple[tuple[str, str], ...]], str],
            tuple[float, bytes],
        ] = {}

    def file_encryption_properties(
        self,
        kms_connection_config: KmsConnectionConfig,
        encryption_config: EncryptionConfiguration,
        parquet_file_path: str | PathLike[str] | None = None,
        filesystem: Any | None = None,
    ) -> FileEncryptionProperties:
        """Create native writer properties and fresh per-file data keys."""
        if not encryption_config.internal_key_material and parquet_file_path is None:
            msg = "parquet_file_path is required for external key material"
            raise ValueError(msg)
        cache_lifetime_seconds = _cache_seconds(encryption_config.cache_lifetime)
        client = self._get_kms_client(kms_connection_config, cache_lifetime_seconds)
        key_length = encryption_config.data_key_length_bits // 8
        external_material: dict[str, str] = {}
        column_counter = 0

        footer_key = secrets.token_bytes(key_length)
        footer_metadata, material = self._wrap_key(
            footer_key,
            encryption_config.footer_key,
            encryption_config,
            kms_connection_config,
            client,
            "footerKey",
            cache_lifetime_seconds,
            is_footer_key=True,
        )
        if material is not None:
            external_material["footerKey"] = material

        column_keys: dict[str, bytes] = {}
        column_key_metadata: dict[str, bytes] = {}
        for master_key_id, columns in encryption_config.column_keys.items():
            for column in columns:
                if not column:
                    msg = f"empty column name configured for key {master_key_id!r}"
                    raise ValueError(msg)
                if column in column_keys:
                    msg = f"multiple master keys configured for column {column!r}"
                    raise ValueError(msg)
                column_key = secrets.token_bytes(key_length)
                reference = f"columnKey{column_counter}"
                column_counter += 1
                metadata, material = self._wrap_key(
                    column_key,
                    master_key_id,
                    encryption_config,
                    kms_connection_config,
                    client,
                    reference,
                    cache_lifetime_seconds,
                    is_footer_key=False,
                )
                column_keys[column] = column_key
                column_key_metadata[column] = metadata
                if material is not None:
                    external_material[reference] = material

        if not encryption_config.internal_key_material:
            assert parquet_file_path is not None
            _write_external_material(
                _key_material_path(parquet_file_path), external_material, filesystem
            )

        properties = FileEncryptionProperties(
            footer_key=footer_key,
            footer_key_metadata=footer_metadata,
            plaintext_footer=encryption_config.plaintext_footer,
            encryption_algorithm=encryption_config.encryption_algorithm,
        )
        if column_keys:
            properties["column_keys"] = column_keys
            properties["column_key_metadata"] = column_key_metadata
        return properties

    def file_decryption_properties(
        self,
        kms_connection_config: KmsConnectionConfig,
        decryption_config: DecryptionConfiguration | None = None,
        parquet_file_path: str | PathLike[str] | None = None,
        filesystem: Any | None = None,
    ) -> FileDecryptionProperties:
        """Create native reader properties backed by the configured KMS."""
        config = decryption_config or DecryptionConfiguration()
        return FileDecryptionProperties(
            key_retriever=_KeyRetriever(
                self,
                kms_connection_config,
                config.cache_lifetime,
                parquet_file_path,
                filesystem,
            )
        )

    def _wrap_key(
        self,
        data_key: bytes,
        master_key_id: str,
        encryption_config: EncryptionConfiguration,
        connection_config: KmsConnectionConfig,
        client: KmsClient,
        reference: str,
        cache_lifetime_seconds: float,
        *,
        is_footer_key: bool,
    ) -> tuple[bytes, str | None]:
        material: dict[str, Any] = {
            "keyMaterialType": _KEY_MATERIAL_TYPE,
            "isFooterKey": is_footer_key,
            "masterKeyID": master_key_id,
            "doubleWrapping": encryption_config.double_wrapping,
        }
        if is_footer_key:
            material["kmsInstanceID"] = connection_config.kms_instance_id
            material["kmsInstanceURL"] = connection_config.kms_instance_url
        if encryption_config.double_wrapping:
            kek, kek_id, wrapped_kek = self._get_or_create_kek(
                connection_config,
                client,
                master_key_id,
                cache_lifetime_seconds,
            )
            locally_wrapped = _plr()._parquet_encrypt_key_locally(data_key, kek, kek_id)
            material["wrappedDEK"] = base64.b64encode(locally_wrapped).decode("ascii")
            material["keyEncryptionKeyID"] = base64.b64encode(kek_id).decode("ascii")
            material["wrappedKEK"] = wrapped_kek
        else:
            material["wrappedDEK"] = client.wrap_key(data_key, master_key_id)

        if encryption_config.internal_key_material:
            material["internalStorage"] = True
            return _json_bytes(material), None
        metadata = {
            "keyMaterialType": _KEY_MATERIAL_TYPE,
            "internalStorage": False,
            "keyReference": reference,
        }
        return _json_bytes(metadata), json.dumps(
            material, separators=(",", ":"), ensure_ascii=False
        )

    def _get_kms_client(
        self,
        config: KmsConnectionConfig,
        cache_lifetime_seconds: float,
    ) -> KmsClient:
        cache_key = _kms_cache_key(config)
        now = time.monotonic()
        with self._lock:
            cached = self._kms_client_cache.get(cache_key)
            if cached is not None and cached[0] >= now:
                return cached[1]
            client = self._kms_client_factory(config)
            if not hasattr(client, "wrap_key") or not hasattr(client, "unwrap_key"):
                msg = "kms_client_factory must return a KmsClient-compatible object"
                raise TypeError(msg)
            self._kms_client_cache[cache_key] = (
                now + cache_lifetime_seconds,
                client,
            )
            return client

    def _get_or_create_kek(
        self,
        config: KmsConnectionConfig,
        client: KmsClient,
        master_key_id: str,
        cache_lifetime_seconds: float,
    ) -> tuple[bytes, bytes, str]:
        cache_key = (_kms_cache_key(config), master_key_id)
        now = time.monotonic()
        with self._lock:
            cached = self._write_kek_cache.get(cache_key)
            if cached is not None and cached[0] >= now:
                return cached[1], cached[2], cached[3]
            kek = secrets.token_bytes(16)
            kek_id = secrets.token_bytes(16)
            wrapped_kek = client.wrap_key(kek, master_key_id)
            self._write_kek_cache[cache_key] = (
                now + cache_lifetime_seconds,
                kek,
                kek_id,
                wrapped_kek,
            )
            return kek, kek_id, wrapped_kek

    def _get_or_unwrap_kek(
        self,
        config: KmsConnectionConfig,
        client: KmsClient,
        encoded_kek_id: str,
        wrapped_kek: str,
        master_key_id: str,
        cache_lifetime_seconds: float,
    ) -> bytes:
        cache_key = (_kms_cache_key(config), encoded_kek_id)
        now = time.monotonic()
        with self._lock:
            cached = self._read_kek_cache.get(cache_key)
            if cached is not None and cached[0] >= now:
                return cached[1]
            kek = bytes(client.unwrap_key(wrapped_kek, master_key_id))
            if len(kek) != 16:
                msg = f"KMS returned an invalid KEK length: {len(kek)} bytes"
                raise ValueError(msg)
            self._read_kek_cache[cache_key] = (
                now + cache_lifetime_seconds,
                kek,
            )
            return kek

    def remove_cache_entries_for_token(self, access_token: str) -> None:
        """Remove cached clients and wrapping keys for one access token."""
        with self._lock:
            self._kms_client_cache = {
                key: value
                for key, value in self._kms_client_cache.items()
                if key[2] != access_token
            }
            self._write_kek_cache = {
                key: value
                for key, value in self._write_kek_cache.items()
                if key[0][2] != access_token
            }
            self._read_kek_cache = {
                key: value
                for key, value in self._read_kek_cache.items()
                if key[0][2] != access_token
            }

    def remove_cache_entries_for_all_tokens(self) -> None:
        """Remove every cached client and wrapping key."""
        with self._lock:
            self._kms_client_cache.clear()
            self._write_kek_cache.clear()
            self._read_kek_cache.clear()

    def rotate_master_keys(
        self,
        kms_connection_config: KmsConnectionConfig,
        parquet_file_path: str | PathLike[str],
        filesystem: Any | None = None,
        *,
        double_wrapping: bool = True,
        cache_lifetime_seconds: float = 600,
    ) -> None:
        """Rewrap every DEK in an external key material file."""
        if cache_lifetime_seconds < 0:
            msg = "cache_lifetime_seconds must not be negative"
            raise ValueError(msg)
        material_path = _key_material_path(parquet_file_path)
        stored_material = _read_external_material(material_path, filesystem)
        retriever = _KeyRetriever(
            self,
            kms_connection_config,
            timedelta(seconds=cache_lifetime_seconds),
            parquet_file_path,
            filesystem,
        )
        client = self._get_kms_client(kms_connection_config, cache_lifetime_seconds)
        rotated: dict[str, str] = {}
        for reference, serialized_material in stored_material.items():
            material = json.loads(serialized_material)
            data_key = retriever._unwrap_material(material)
            rotation_config = EncryptionConfiguration(
                footer_key=material["masterKeyID"],
                uniform_encryption=True,
                double_wrapping=double_wrapping,
                internal_key_material=False,
                data_key_length_bits=len(data_key) * 8,
            )
            _, rotated_material = self._wrap_key(
                data_key,
                material["masterKeyID"],
                rotation_config,
                kms_connection_config,
                client,
                reference,
                cache_lifetime_seconds,
                is_footer_key=bool(material["isFooterKey"]),
            )
            assert rotated_material is not None
            rotated[reference] = rotated_material

        if filesystem is None:
            destination = Path(material_path)
            temporary_path = destination.with_name(f"_TMP{destination.name}")
            _write_external_material(str(temporary_path), rotated, None)
            temporary_path.replace(destination)
        else:
            path = Path(material_path)
            temporary_path_str = str(path.with_name(f"_TMP{path.name}"))
            _write_external_material(temporary_path_str, rotated, filesystem)
            filesystem.move(temporary_path_str, material_path)


__all__ = [
    "CryptoFactory",
    "DecryptionConfiguration",
    "EncryptionConfiguration",
    "FileDecryptionProperties",
    "FileEncryptionProperties",
    "KmsClient",
    "KmsConnectionConfig",
]
