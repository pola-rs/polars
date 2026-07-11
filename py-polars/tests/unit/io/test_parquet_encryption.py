from __future__ import annotations

import base64
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.parquet.encryption as arrow_encryption
import pytest

import polars as pl
from polars.io.parquet import encryption
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

MASTER_KEYS = {
    "footer-master": b"footer-master-key",
    "column-master": b"column-master-key",
}


class InMemoryKmsClient(arrow_encryption.KmsClient):  # type: ignore[misc]
    """Deterministic reversible KMS used only for interoperability tests."""

    def __init__(self) -> None:
        super().__init__()

    def wrap_key(self, key_bytes: bytes, master_key_identifier: str) -> str:
        master_key = MASTER_KEYS[master_key_identifier]
        wrapped = bytes(
            value ^ master_key[index % len(master_key)]
            for index, value in enumerate(key_bytes)
        )
        return base64.b64encode(wrapped).decode("ascii")

    def unwrap_key(self, wrapped_key: str, master_key_identifier: str) -> bytes:
        master_key = MASTER_KEYS[master_key_identifier]
        wrapped = base64.b64decode(wrapped_key, validate=True)
        return bytes(
            value ^ master_key[index % len(master_key)]
            for index, value in enumerate(wrapped)
        )


def kms_config_snapshot(
    config: encryption.KmsConnectionConfig,
) -> tuple[str, str, str, tuple[tuple[str, str], ...]]:
    return (
        config.kms_instance_id,
        config.kms_instance_url,
        config.key_access_token,
        tuple(sorted(config.custom_kms_conf.items())),
    )


def polars_kms_config() -> encryption.KmsConnectionConfig:
    return encryption.KmsConnectionConfig(
        kms_instance_id="test-instance",
        kms_instance_url="memory://test",
        key_access_token="test-token",
    )


def arrow_kms_config() -> arrow_encryption.KmsConnectionConfig:
    return arrow_encryption.KmsConnectionConfig(
        kms_instance_id="test-instance",
        kms_instance_url="memory://test",
        key_access_token="test-token",
    )


def expected_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "public": [1, 2, 3, 4],
            "secret": ["alpha", "beta", "gamma", "delta"],
        }
    )


def from_arrow_table(table: pa.Table) -> pl.DataFrame:
    frame = pl.from_arrow(table)
    assert isinstance(frame, pl.DataFrame)
    return frame


@pytest.mark.write_disk
@pytest.mark.parametrize(
    (
        "algorithm",
        "double_wrapping",
        "internal_material",
        "key_length",
        "plaintext_footer",
    ),
    [
        ("AES_GCM_V1", False, True, 128, True),
        ("AES_GCM_V1", True, False, 192, False),
        ("AES_GCM_CTR_V1", True, True, 256, False),
        ("AES_GCM_CTR_V1", False, False, 128, False),
    ],
)
def test_native_parquet_kms_pyarrow_interoperability(
    tmp_path: Path,
    algorithm: str,
    double_wrapping: bool,
    internal_material: bool,
    key_length: int,
    plaintext_footer: bool,
) -> None:
    frame = expected_frame()
    polars_factory = encryption.CryptoFactory(lambda config: InMemoryKmsClient())
    arrow_factory = arrow_encryption.CryptoFactory(lambda config: InMemoryKmsClient())

    polars_path = tmp_path / f"polars-{algorithm}.parquet"
    polars_encryption_config = encryption.EncryptionConfiguration(
        footer_key="footer-master",
        column_keys={"column-master": ["secret"]},
        encryption_algorithm=algorithm,
        double_wrapping=double_wrapping,
        internal_key_material=internal_material,
        data_key_length_bits=key_length,
        plaintext_footer=plaintext_footer,
    )
    polars_properties = polars_factory.file_encryption_properties(
        polars_kms_config(),
        polars_encryption_config,
        parquet_file_path=polars_path,
    )
    frame.write_parquet(
        polars_path,
        compression="uncompressed",
        encryption_properties=polars_properties,
    )

    arrow_decryption = arrow_factory.file_decryption_properties(
        arrow_kms_config(), parquet_file_path=polars_path
    )
    arrow_result = pq.read_table(polars_path, decryption_properties=arrow_decryption)
    assert_frame_equal(from_arrow_table(arrow_result), frame)
    if plaintext_footer:
        assert_frame_equal(
            pl.read_parquet(polars_path, columns=["public"]), frame.select("public")
        )

    arrow_path = tmp_path / f"arrow-{algorithm}.parquet"
    arrow_encryption_config = arrow_encryption.EncryptionConfiguration(
        footer_key="footer-master",
        column_keys={"column-master": ["secret"]},
        encryption_algorithm=algorithm,
        double_wrapping=double_wrapping,
        internal_key_material=internal_material,
        data_key_length_bits=key_length,
        plaintext_footer=plaintext_footer,
    )
    arrow_properties = arrow_factory.file_encryption_properties(
        arrow_kms_config(),
        arrow_encryption_config,
        parquet_file_path=arrow_path,
    )
    pq.write_table(
        pa.table(frame.to_dict(as_series=False)),
        arrow_path,
        compression="NONE",
        write_statistics=True,
        write_page_index=True,
        encryption_properties=arrow_properties,
    )

    polars_decryption = polars_factory.file_decryption_properties(
        polars_kms_config(), parquet_file_path=arrow_path
    )
    assert_frame_equal(
        pl.read_parquet(arrow_path, decryption_properties=polars_decryption), frame
    )
    assert_frame_equal(
        pl.scan_parquet(arrow_path, decryption_properties=polars_decryption).collect(),
        frame,
    )
    if plaintext_footer:
        assert_frame_equal(
            pl.read_parquet(arrow_path, columns=["public"]), frame.select("public")
        )

    polars_material_path = tmp_path / f"_KEY_MATERIAL_FOR_{polars_path.name}.json"
    arrow_material_path = tmp_path / f"_KEY_MATERIAL_FOR_{arrow_path.name}.json"
    assert polars_material_path.exists() is not internal_material
    assert arrow_material_path.exists() is not internal_material


@pytest.mark.write_disk
def test_native_parquet_kms_uniform_encryption(tmp_path: Path) -> None:
    path = tmp_path / "uniform.parquet"
    frame = expected_frame()
    factory = encryption.CryptoFactory(lambda config: InMemoryKmsClient())
    properties = factory.file_encryption_properties(
        polars_kms_config(),
        encryption.EncryptionConfiguration(
            footer_key="footer-master",
            uniform_encryption=True,
        ),
    )
    frame.write_parquet(path, encryption_properties=properties)

    decryption = factory.file_decryption_properties(polars_kms_config())
    assert_frame_equal(pl.read_parquet(path, decryption_properties=decryption), frame)


@pytest.mark.write_disk
def test_native_parquet_external_key_rotation(tmp_path: Path) -> None:
    path = tmp_path / "rotate.parquet"
    material_path = tmp_path / f"_KEY_MATERIAL_FOR_{path.name}.json"
    frame = expected_frame()
    factory = encryption.CryptoFactory(lambda config: InMemoryKmsClient())
    properties = factory.file_encryption_properties(
        polars_kms_config(),
        encryption.EncryptionConfiguration(
            footer_key="footer-master",
            column_keys={"column-master": ["secret"]},
            internal_key_material=False,
        ),
        parquet_file_path=path,
    )
    frame.write_parquet(path, encryption_properties=properties)
    material_before = material_path.read_bytes()

    factory.remove_cache_entries_for_all_tokens()
    factory.rotate_master_keys(polars_kms_config(), path)
    material_after = material_path.read_bytes()

    assert material_after != material_before
    decryption = factory.file_decryption_properties(
        polars_kms_config(), parquet_file_path=path
    )
    assert_frame_equal(pl.read_parquet(path, decryption_properties=decryption), frame)


def test_native_parquet_kms_cache_is_scoped_by_full_connection_config() -> None:
    seen_configs: list[tuple[str, str, str, tuple[tuple[str, str], ...]]] = []

    def factory(config: encryption.KmsConnectionConfig) -> InMemoryKmsClient:
        seen_configs.append(kms_config_snapshot(config))
        return InMemoryKmsClient()

    crypto_factory = encryption.CryptoFactory(factory)
    encryption_config = encryption.EncryptionConfiguration(
        footer_key="footer-master",
        uniform_encryption=True,
        double_wrapping=True,
    )
    crypto_factory.file_encryption_properties(
        encryption.KmsConnectionConfig(
            kms_instance_id="first",
            kms_instance_url="memory://first",
            key_access_token="shared-token",
            custom_kms_conf={"tenant": "a"},
        ),
        encryption_config,
    )
    crypto_factory.file_encryption_properties(
        encryption.KmsConnectionConfig(
            kms_instance_id="second",
            kms_instance_url="memory://second",
            key_access_token="shared-token",
            custom_kms_conf={"tenant": "b"},
        ),
        encryption_config,
    )

    assert seen_configs == [
        ("first", "memory://first", "shared-token", (("tenant", "a"),)),
        ("second", "memory://second", "shared-token", (("tenant", "b"),)),
    ]


@pytest.mark.write_disk
def test_native_parquet_default_kms_config_uses_material_instance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "material_instance.parquet"
    frame = expected_frame()
    arrow_factory = arrow_encryption.CryptoFactory(lambda config: InMemoryKmsClient())
    arrow_properties = arrow_factory.file_encryption_properties(
        arrow_encryption.KmsConnectionConfig(
            kms_instance_id="material-instance",
            kms_instance_url="memory://material",
            key_access_token="DEFAULT",
        ),
        arrow_encryption.EncryptionConfiguration(
            footer_key="footer-master",
            uniform_encryption=True,
        ),
    )
    pq.write_table(
        pa.table(frame.to_dict(as_series=False)),
        path,
        encryption_properties=arrow_properties,
    )

    seen_configs: list[tuple[str, str, str, tuple[tuple[str, str], ...]]] = []

    def factory(config: encryption.KmsConnectionConfig) -> InMemoryKmsClient:
        seen_configs.append(kms_config_snapshot(config))
        return InMemoryKmsClient()

    polars_factory = encryption.CryptoFactory(factory)
    decryption = polars_factory.file_decryption_properties(
        encryption.KmsConnectionConfig()
    )

    assert_frame_equal(
        pl.read_parquet(path, decryption_properties=decryption),
        frame,
    )
    assert (
        "material-instance",
        "memory://material",
        "DEFAULT",
        (),
    ) in seen_configs


@pytest.mark.write_disk
def test_native_parquet_nested_root_column_key_interoperability(
    tmp_path: Path,
) -> None:
    path = tmp_path / "nested.parquet"
    frame = pl.DataFrame(
        {
            "id": [1, 2],
            "nested": [
                {"public": 10, "private": "alpha"},
                {"public": 20, "private": "beta"},
            ],
        }
    )
    polars_factory = encryption.CryptoFactory(lambda config: InMemoryKmsClient())
    properties = polars_factory.file_encryption_properties(
        polars_kms_config(),
        encryption.EncryptionConfiguration(
            footer_key="footer-master",
            column_keys={"column-master": ["nested"]},
        ),
    )
    frame.write_parquet(path, encryption_properties=properties)

    arrow_factory = arrow_encryption.CryptoFactory(lambda config: InMemoryKmsClient())
    arrow_decryption = arrow_factory.file_decryption_properties(arrow_kms_config())
    assert_frame_equal(
        from_arrow_table(pq.read_table(path, decryption_properties=arrow_decryption)),
        frame,
    )
    polars_decryption = polars_factory.file_decryption_properties(polars_kms_config())
    assert_frame_equal(
        pl.read_parquet(path, decryption_properties=polars_decryption), frame
    )
