from __future__ import annotations

import base64
import re
from datetime import timedelta
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.io.parquet.encryption import (
    CryptoFactory,
    DecryptionConfiguration,
    EncryptionConfiguration,
    KmsClient,
    KmsConnectionConfig,
    ParquetDecryptionConfig,
    ParquetEncryptionConfig,
)
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import EngineType

MASTER_KEYS = {
    "footer_key": b"0123456789012345",
    "column_key": b"1234567890123450",
    "column_key_2": b"1234567890123451",
}
KMS_URL = "https://kms.example.com"
NUM_ROWS = 1000
WRITE_OPTIONS: dict[str, Any] = {
    # Uncompressed so that plaintext values can be found in the written bytes.
    "compression": "uncompressed",
    "row_group_size": 250,
    "data_page_size": 1024,
}


class InMemoryKmsClient(KmsClient):
    """
    A KMS client with master keys held in memory.

    Keys are "wrapped" by XOR-ing them with the master key and prefixing the master
    key ID, which is reversible and lets unwrapping check the master key ID. This is
    not secure, and is only for testing.
    """

    def __init__(self, kms: InMemoryKms) -> None:
        self.kms = kms

    def wrap_key(self, key_bytes: bytes, master_key_identifier: str) -> str:
        self.kms.calls.append(("wrap_key", master_key_identifier))
        master_key = self.kms.master_keys[master_key_identifier]
        wrapped = base64.b64encode(_xor(key_bytes, master_key)).decode()
        return f"{master_key_identifier}:{wrapped}"

    def unwrap_key(self, wrapped_key: str, master_key_identifier: str) -> bytes:
        self.kms.calls.append(("unwrap_key", master_key_identifier))
        if self.kms.unwrap_error is not None:
            raise self.kms.unwrap_error
        key_id, wrapped = wrapped_key.split(":", 1)
        if key_id != master_key_identifier:
            msg = f"key was wrapped with {key_id!r}, not {master_key_identifier!r}"
            raise ValueError(msg)
        master_key = self.kms.master_keys[master_key_identifier]
        return _xor(base64.b64decode(wrapped), master_key)


def _xor(data: bytes, key: bytes) -> bytes:
    return bytes(d ^ key[i % len(key)] for i, d in enumerate(data))


class InMemoryKms:
    """Creates in-memory KMS clients, and records how they are used."""

    def __init__(self, master_keys: dict[str, bytes] | None = None) -> None:
        self.master_keys = MASTER_KEYS if master_keys is None else master_keys
        # The connection configs that clients were created with.
        self.configs: list[KmsConnectionConfig] = []
        # (method, master key ID) for each call to a client.
        self.calls: list[tuple[str, str]] = []
        self.unwrap_error: Exception | None = None

    def create_client(self, config: KmsConnectionConfig) -> InMemoryKmsClient:
        self.configs.append(config)
        return InMemoryKmsClient(self)

    def crypto_factory(self) -> CryptoFactory:
        return CryptoFactory(self.create_client)

    def count_calls(self, method: str) -> int:
        return sum(1 for m, _ in self.calls if m == method)


def encryption_config(
    kms: InMemoryKms,
    connection_config: KmsConnectionConfig | None = None,
    **kwargs: Any,
) -> ParquetEncryptionConfig:
    """Encrypt the `secret` column with `column_key`, unless kwargs say otherwise."""
    if "uniform_encryption" not in kwargs:
        kwargs.setdefault("column_keys", {"column_key": ["secret"]})
    return ParquetEncryptionConfig(
        kms.crypto_factory(),
        connection_config or KmsConnectionConfig(),
        EncryptionConfiguration("footer_key", **kwargs),
    )


def decryption_config(
    kms: InMemoryKms,
    connection_config: KmsConnectionConfig | None = None,
    **kwargs: Any,
) -> ParquetDecryptionConfig:
    return ParquetDecryptionConfig(
        kms.crypto_factory(),
        connection_config or KmsConnectionConfig(),
        DecryptionConfiguration(**kwargs),
    )


def basic_data() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": pl.Series(range(NUM_ROWS), dtype=pl.Int64),
            "secret": [f"secret_{i}" for i in range(NUM_ROWS)],
            "public": [f"public_{i}" for i in range(NUM_ROWS)],
            "part": pl.Series([i % 3 for i in range(NUM_ROWS)], dtype=pl.Int64),
        }
    )


def parquet_files(path: Path) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.rglob("*.parquet"))
    assert files
    return files


def footer_key_material(path: Path) -> dict[str, Any]:
    """
    Get the key material for the footer key, which is stored in the file.

    The key material is JSON in the footer key metadata, which is stored in
    plaintext even when the footer is encrypted.
    """
    import json

    data = path.read_bytes()
    matches = re.findall(rb'\{"keyMaterialType":[^{}]*"isFooterKey":true[^{}]*\}', data)
    assert len(matches) == 1, matches
    material: dict[str, Any] = json.loads(matches[0])
    return material


ROUNDTRIP_PARAMS = [
    pytest.param({"uniform_encryption": True}, {"secret", "public"}, id="uniform"),
    pytest.param({}, {"secret"}, id="column_key"),
    pytest.param(
        {"column_keys": {"column_key": ["secret"], "column_key_2": ["public"]}},
        {"secret", "public"},
        id="column_keys_different_keys",
    ),
    pytest.param(
        {"uniform_encryption": True, "plaintext_footer": True},
        {"secret", "public"},
        id="uniform_plaintext_footer",
    ),
    pytest.param({"plaintext_footer": True}, {"secret"}, id="plaintext_footer"),
    pytest.param({"double_wrapping": False}, {"secret"}, id="single_wrapping"),
]


def assert_encrypted(
    path: Path, encrypted_columns: set[str], *, plaintext_footer: bool
) -> None:
    for file in parquet_files(path):
        data = file.read_bytes()
        magic = b"PAR1" if plaintext_footer else b"PARE"
        assert data[:4] == magic
        assert data[-4:] == magic
        for column in ["secret", "public"]:
            is_encrypted = f"{column}_".encode() not in data
            assert is_encrypted == (column in encrypted_columns), column


@pytest.mark.parametrize(("kwargs", "encrypted_columns"), ROUNDTRIP_PARAMS)
def test_write_parquet_roundtrip(
    tmp_path: Path, kwargs: dict[str, Any], encrypted_columns: set[str]
) -> None:
    kms = InMemoryKms()
    df = basic_data()
    path = tmp_path / "out.parquet"
    df.write_parquet(path, encryption=encryption_config(kms, **kwargs), **WRITE_OPTIONS)
    assert_encrypted(
        path, encrypted_columns, plaintext_footer=kwargs.get("plaintext_footer", False)
    )

    result = pl.read_parquet(path, decryption=decryption_config(kms))
    assert_frame_equal(result, df)


@pytest.mark.parametrize("engine", ["streaming", "in-memory"])
@pytest.mark.parametrize(("kwargs", "encrypted_columns"), ROUNDTRIP_PARAMS)
def test_sink_parquet_roundtrip(
    tmp_path: Path,
    kwargs: dict[str, Any],
    encrypted_columns: set[str],
    engine: EngineType,
) -> None:
    kms = InMemoryKms()
    df = basic_data()
    path = tmp_path / "out.parquet"
    df.lazy().sink_parquet(
        path,
        encryption=encryption_config(kms, **kwargs),
        engine=engine,
        **WRITE_OPTIONS,
    )
    assert_encrypted(
        path, encrypted_columns, plaintext_footer=kwargs.get("plaintext_footer", False)
    )

    lf = pl.scan_parquet(path, decryption=decryption_config(kms))
    assert lf.collect_schema() == df.schema
    assert_frame_equal(lf.collect(engine=engine), df)


def test_kms_wrap_calls(tmp_path: Path) -> None:
    df = basic_data()

    # With single wrapping, the KMS wraps every data encryption key
    kms = InMemoryKms()
    df.write_parquet(
        tmp_path / "single.parquet",
        encryption=encryption_config(kms, double_wrapping=False),
    )
    assert sorted(kms.calls) == [
        ("wrap_key", "column_key"),
        ("wrap_key", "footer_key"),
    ]
    material = footer_key_material(tmp_path / "single.parquet")
    assert material["doubleWrapping"] is False
    assert material["masterKeyID"] == "footer_key"

    # With double wrapping, the KMS wraps key encryption keys, which are cached and
    # reused for later files.
    kms = InMemoryKms()
    encryption = encryption_config(kms)
    df.write_parquet(tmp_path / "double_1.parquet", encryption=encryption)
    assert sorted(kms.calls) == [
        ("wrap_key", "column_key"),
        ("wrap_key", "footer_key"),
    ]
    df.write_parquet(tmp_path / "double_2.parquet", encryption=encryption)
    assert kms.count_calls("wrap_key") == 2
    assert footer_key_material(tmp_path / "double_1.parquet")["doubleWrapping"] is True


def test_read_without_decryption(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(path, encryption=encryption_config(InMemoryKms()))
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="encrypted footer but decryption properties were not provided",
    ):
        pl.read_parquet(path)


def test_read_plaintext_footer_without_decryption(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    df = basic_data()
    df.write_parquet(
        path, encryption=encryption_config(InMemoryKms(), plaintext_footer=True)
    )
    assert_frame_equal(
        pl.read_parquet(path, columns=["id", "public"]), df.drop("secret", "part")
    )
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="Column 'secret' is encrypted but decryption properties were not provided",
    ):
        pl.read_parquet(path, columns=["secret"])


def test_read_with_wrong_keys(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(path, encryption=encryption_config(InMemoryKms()))

    # The data keys are random, so the master keys can't be used directly
    decryption_properties = pl.ParquetDecryptionProperties(
        footer_key=MASTER_KEYS["footer_key"],
        column_keys={"secret": MASTER_KEYS["column_key"]},
    )
    with pytest.raises(
        pl.exceptions.ComputeError, match="unable to decrypt parquet footer"
    ):
        pl.read_parquet(path, decryption=decryption_properties)

    # A KMS with different master keys unwraps the wrong keys
    wrong_kms = InMemoryKms({key_id: key[::-1] for key_id, key in MASTER_KEYS.items()})
    with pytest.raises(pl.exceptions.ComputeError):
        pl.read_parquet(path, decryption=decryption_config(wrong_kms))


def test_scan_partitioned_files(tmp_path: Path) -> None:
    kms = InMemoryKms()
    df = basic_data()
    path = tmp_path / "out"
    df.lazy().sink_parquet(
        pl.PartitionBy(path, key="part"),
        encryption=encryption_config(kms, double_wrapping=False),
        mkdir=True,
        **WRITE_OPTIONS,
    )
    files = parquet_files(path)
    assert len(files) == 3

    # Each file has its own data keys
    assert kms.count_calls("wrap_key") == 2 * len(files)
    wrapped_keys = {footer_key_material(f)["wrappedDEK"] for f in files}
    assert len(wrapped_keys) == len(files)

    result = pl.scan_parquet(path, decryption=decryption_config(kms)).collect()
    assert_frame_equal(result, df, check_row_order=False, check_column_order=False)


def test_connection_config_reaches_client_factory(tmp_path: Path) -> None:
    kms = InMemoryKms()
    connection_config = KmsConnectionConfig(
        "instance_id",
        kms_instance_url=KMS_URL,
        key_access_token="token_1",
        custom_kms_conf={"region": "somewhere"},
    )
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(
        path, encryption=encryption_config(kms, connection_config)
    )
    assert kms.configs
    for config in kms.configs:
        assert type(config) is KmsConnectionConfig
        assert config.kms_instance_id == "instance_id"
        assert config.kms_instance_url == KMS_URL
        assert config.key_access_token == "token_1"
        assert config.custom_kms_conf == {"region": "somewhere"}

    # The instance ID and URL are stored in the file
    material = footer_key_material(path)
    assert material["kmsInstanceID"] == "instance_id"
    assert material["kmsInstanceURL"] == KMS_URL


def test_refresh_key_access_token(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(path, encryption=encryption_config(InMemoryKms()))

    kms = InMemoryKms()
    connection_config = KmsConnectionConfig(key_access_token="token_1")
    decryption = decryption_config(kms, connection_config)
    pl.read_parquet(path, decryption=decryption)
    assert kms.configs
    assert {c.key_access_token for c in kms.configs} == {"token_1"}

    # Later queries create new clients with the new token
    num_configs = len(kms.configs)
    connection_config.refresh_key_access_token("token_2")
    assert connection_config.key_access_token == "token_2"
    pl.read_parquet(path, decryption=decryption)
    assert len(kms.configs) > num_configs
    assert {c.key_access_token for c in kms.configs[num_configs:]} == {"token_2"}


def test_client_factory_config_is_read_only(tmp_path: Path) -> None:
    kms = InMemoryKms()
    basic_data().write_parquet(
        tmp_path / "out.parquet", encryption=encryption_config(kms)
    )
    config = kms.configs[0]
    with pytest.raises(ValueError, match="read-only"):
        config.refresh_key_access_token("token")
    with pytest.raises(ValueError, match="can't be used to read or write files"):
        decryption_config(kms, config)


@pytest.mark.parametrize("read_kms_url", [False, True])
def test_read_kms_url(tmp_path: Path, read_kms_url: bool) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(
        path,
        encryption=encryption_config(
            InMemoryKms(), KmsConnectionConfig(kms_instance_url=KMS_URL)
        ),
    )

    kms = InMemoryKms()
    pl.read_parquet(path, decryption=decryption_config(kms, read_kms_url=read_kms_url))
    expected_url = KMS_URL if read_kms_url else "DEFAULT"
    assert kms.configs
    assert {c.kms_instance_url for c in kms.configs} == {expected_url}

    # A URL in the connection config takes precedence
    kms = InMemoryKms()
    connection_config = KmsConnectionConfig(
        kms_instance_url="https://other.example.com"
    )
    pl.read_parquet(
        path,
        decryption=decryption_config(kms, connection_config, read_kms_url=read_kms_url),
    )
    assert {c.kms_instance_url for c in kms.configs} == {"https://other.example.com"}


def test_pyarrow_reads_polars_file(tmp_path: Path) -> None:
    pq, pe = import_pyarrow_encryption()
    kms = InMemoryKms()
    df = basic_data()
    path = tmp_path / "out.parquet"
    df.write_parquet(
        path,
        encryption=encryption_config(
            kms,
            column_keys={"column_key": ["secret"], "column_key_2": ["public"]},
        ),
        **WRITE_OPTIONS,
    )

    crypto_factory = pe.CryptoFactory(pyarrow_client_factory(kms))
    decryption_properties = crypto_factory.file_decryption_properties(
        pe.KmsConnectionConfig(), pe.DecryptionConfiguration()
    )
    table = pq.read_table(path, decryption_properties=decryption_properties)
    assert_frame_equal(pl.DataFrame(table), df)


def test_polars_reads_pyarrow_file(tmp_path: Path) -> None:
    pq, pe = import_pyarrow_encryption()
    kms = InMemoryKms()
    df = basic_data()
    path = tmp_path / "out.parquet"

    crypto_factory = pe.CryptoFactory(pyarrow_client_factory(kms))
    encryption_properties = crypto_factory.file_encryption_properties(
        pe.KmsConnectionConfig(),
        pe.EncryptionConfiguration(
            footer_key="footer_key",
            column_keys={"column_key": ["secret"], "column_key_2": ["public"]},
        ),
    )
    pq.write_table(df.to_arrow(), path, encryption_properties=encryption_properties)

    # A PyArrow KMS client can be used with Polars
    decryption = ParquetDecryptionConfig(
        CryptoFactory(pyarrow_client_factory(kms)),
        KmsConnectionConfig(),
        DecryptionConfiguration(),
    )
    assert_frame_equal(pl.read_parquet(path, decryption=decryption), df)


def import_pyarrow_encryption() -> tuple[Any, Any]:
    pytest.importorskip("pyarrow", minversion="25.0.0")
    import pyarrow.parquet as pq
    import pyarrow.parquet.encryption as pe

    return pq, pe


def pyarrow_client_factory(kms: InMemoryKms) -> Any:
    """Create clients that subclass PyArrow's KmsClient, as PyArrow requires."""
    import pyarrow.parquet.encryption as pe

    class PyArrowKmsClient(pe.KmsClient):  # type: ignore[misc]
        def __init__(self) -> None:
            pe.KmsClient.__init__(self)
            self.client = InMemoryKmsClient(kms)

        def wrap_key(self, key_bytes: bytes, master_key_identifier: str) -> str:
            return self.client.wrap_key(key_bytes, master_key_identifier)

        def unwrap_key(self, wrapped_key: str, master_key_identifier: str) -> bytes:
            return self.client.unwrap_key(wrapped_key, master_key_identifier)

    def create_client(config: Any) -> PyArrowKmsClient:
        kms.configs.append(config)
        return PyArrowKmsClient()

    return create_client


def test_wrap_key_error(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    encryption = encryption_config(
        InMemoryKms(), column_keys={"missing_key": ["secret"]}
    )
    with pytest.raises(pl.exceptions.ComputeError, match=r"KeyError.*missing_key"):
        basic_data().write_parquet(path, encryption=encryption)


def test_unwrap_key_error(tmp_path: Path) -> None:
    kms = InMemoryKms()
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(path, encryption=encryption_config(kms))

    kms.unwrap_error = RuntimeError("the KMS is unavailable")
    with pytest.raises(pl.exceptions.ComputeError, match="the KMS is unavailable"):
        pl.read_parquet(path, decryption=decryption_config(kms))


def test_unwrap_with_unknown_master_key(tmp_path: Path) -> None:
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(path, encryption=encryption_config(InMemoryKms()))

    kms = InMemoryKms({"footer_key": MASTER_KEYS["footer_key"]})
    lf = pl.scan_parquet(path, decryption=decryption_config(kms))
    assert_frame_equal(lf.select("id").collect(), basic_data().select("id"))
    with pytest.raises(pl.exceptions.ComputeError, match=r"KeyError.*column_key"):
        lf.select("secret").collect()


@pytest.mark.parametrize("partitioned", [False, True])
def test_write_column_not_in_schema(tmp_path: Path, partitioned: bool) -> None:
    kms = InMemoryKms()
    encryption = encryption_config(
        kms, column_keys={"column_key": ["secret", "missing"]}
    )
    path = tmp_path / "out"
    target: Any = pl.PartitionBy(path, key="part") if partitioned else path
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="columns with encryption keys specified were not found in the schema: missing",
    ):
        basic_data().lazy().sink_parquet(target, encryption=encryption, mkdir=True)
    # The columns are checked before any file is opened or key is generated
    assert not path.exists()
    assert kms.calls == []


@pytest.mark.parametrize("include_key", [True, False])
def test_write_encrypted_partition_key(tmp_path: Path, include_key: bool) -> None:
    kms = InMemoryKms()
    encryption = encryption_config(kms, column_keys={"column_key": ["part"]})
    path = tmp_path / "out"
    with pytest.raises(
        pl.exceptions.InvalidOperationError,
        match="cannot encrypt column 'part' as it is a partition key",
    ):
        basic_data().lazy().sink_parquet(
            pl.PartitionBy(path, key="part", include_key=include_key),
            encryption=encryption,
            mkdir=True,
        )
    assert not path.exists()
    assert kms.calls == []


def test_write_nested_column(tmp_path: Path) -> None:
    kms = InMemoryKms()
    df = pl.DataFrame({"s": [{"x": f"secret_{i}", "y": i} for i in range(10)]})
    path = tmp_path / "out.parquet"
    df.write_parquet(
        path,
        encryption=encryption_config(kms, column_keys={"column_key": ["s.x"]}),
        compression="uncompressed",
    )
    assert b"secret_" not in path.read_bytes()
    assert_frame_equal(pl.read_parquet(path, decryption=decryption_config(kms)), df)


def test_use_pyarrow_raises(tmp_path: Path) -> None:
    kms = InMemoryKms()
    path = tmp_path / "out.parquet"
    with pytest.raises(ValueError, match="cannot be combined with `encryption`"):
        basic_data().write_parquet(
            path, use_pyarrow=True, encryption=encryption_config(kms)
        )

    basic_data().write_parquet(path, encryption=encryption_config(kms))
    with pytest.raises(
        ValueError,
        match="Parquet decryption properties cannot be used when use_pyarrow is True",
    ):
        pl.read_parquet(path, use_pyarrow=True, decryption=decryption_config(kms))


def test_serialize_raises(tmp_path: Path) -> None:
    kms = InMemoryKms()
    path = tmp_path / "out.parquet"
    basic_data().write_parquet(path, encryption=encryption_config(kms))
    lf = pl.scan_parquet(path, decryption=decryption_config(kms))
    with pytest.raises(
        pl.exceptions.ComputeError,
        match="cannot serialize parquet decryption properties",
    ):
        lf.serialize()


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        ({}, ValueError, "either `column_keys` or `uniform_encryption` must be set"),
        (
            {"column_keys": {"column_key": []}},
            ValueError,
            "either `column_keys` or `uniform_encryption` must be set",
        ),
        (
            {"column_keys": {"column_key": ["a"]}, "uniform_encryption": True},
            ValueError,
            "cannot set both `column_keys` and `uniform_encryption`",
        ),
        (
            {"column_keys": {"column_key": ["a"], "column_key_2": ["a"]}},
            ValueError,
            "multiple master key ids",
        ),
        (
            {"column_keys": {"column_key": "a"}},
            TypeError,
            "the columns for master key 'column_key' must be a sequence of column names",
        ),
        (
            {"column_keys": {"column_key": [1]}},
            TypeError,
            "the columns for master key 'column_key' must be a sequence of column names",
        ),
        (
            {"column_keys": ["a"]},
            TypeError,
            "`column_keys` must be a mapping from master key ID to column names",
        ),
        (
            {"uniform_encryption": True, "cache_lifetime": 600},
            TypeError,
            "`cache_lifetime` must be a `timedelta`, got 'int'",
        ),
        (
            {"uniform_encryption": True, "cache_lifetime": timedelta(seconds=-1)},
            ValueError,
            "`cache_lifetime` must not be negative",
        ),
    ],
)
def test_encryption_configuration_invalid(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=re.escape(match)):
        EncryptionConfiguration("footer_key", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            {"cache_lifetime": None},
            TypeError,
            "`cache_lifetime` must be a `timedelta`, got 'NoneType'",
        ),
        (
            {"cache_lifetime": timedelta(seconds=-1)},
            ValueError,
            "`cache_lifetime` must not be negative",
        ),
    ],
)
def test_decryption_configuration_invalid(
    kwargs: dict[str, Any], error: type[Exception], match: str
) -> None:
    with pytest.raises(error, match=re.escape(match)):
        DecryptionConfiguration(**kwargs)


def test_configuration_properties() -> None:
    config = EncryptionConfiguration(
        "footer_key",
        column_keys={
            "column_key": ("a", "b"),
            "column_key_2": [],
            "column_key_3": ["c"],
        },
        plaintext_footer=True,
        double_wrapping=False,
        cache_lifetime=timedelta(minutes=1),
    )
    assert config.footer_key == "footer_key"
    assert config.column_keys == {"column_key": ["a", "b"], "column_key_3": ["c"]}
    assert config.uniform_encryption is False
    assert config.plaintext_footer is True
    assert config.double_wrapping is False
    assert config.cache_lifetime == timedelta(minutes=1)

    config = EncryptionConfiguration("footer_key", uniform_encryption=True)
    assert config.column_keys == {}
    assert config.uniform_encryption is True
    assert config.plaintext_footer is False
    assert config.double_wrapping is True
    assert config.cache_lifetime == timedelta(minutes=10)

    decryption = DecryptionConfiguration()
    assert decryption.cache_lifetime == timedelta(minutes=10)
    assert decryption.read_kms_url is False
    decryption = DecryptionConfiguration(cache_lifetime=timedelta(0), read_kms_url=True)
    assert decryption.cache_lifetime == timedelta(0)
    assert decryption.read_kms_url is True

    connection = KmsConnectionConfig()
    assert connection.kms_instance_id == ""
    assert connection.kms_instance_url == ""
    assert connection.key_access_token == "DEFAULT"
    assert connection.custom_kms_conf == {}


def test_crypto_factory_requires_callable() -> None:
    with pytest.raises(
        TypeError, match="`kms_client_factory` must be callable, got 'object'"
    ):
        CryptoFactory(object())  # type: ignore[arg-type]


def test_config_argument_types() -> None:
    crypto_factory = InMemoryKms().crypto_factory()
    connection = KmsConnectionConfig()
    encryption = EncryptionConfiguration("footer_key", uniform_encryption=True)
    decryption = DecryptionConfiguration()

    with pytest.raises(
        TypeError, match="`crypto_factory` must be a `CryptoFactory`, got 'object'"
    ):
        ParquetEncryptionConfig(object(), connection, encryption)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError,
        match="`kms_connection_config` must be a `KmsConnectionConfig`, got 'dict'",
    ):
        ParquetDecryptionConfig(crypto_factory, {}, decryption)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError,
        match="`encryption_config` must be a `EncryptionConfiguration`, "
        "got 'DecryptionConfiguration'",
    ):
        ParquetEncryptionConfig(crypto_factory, connection, decryption)  # type: ignore[arg-type]
    with pytest.raises(
        TypeError,
        match="`decryption_config` must be a `DecryptionConfiguration`, "
        "got 'EncryptionConfiguration'",
    ):
        ParquetDecryptionConfig(crypto_factory, connection, encryption)  # type: ignore[arg-type]


def test_config_wrong_direction(tmp_path: Path) -> None:
    kms = InMemoryKms()
    path = tmp_path / "out.parquet"
    with pytest.raises(
        TypeError,
        match=re.escape(
            "`encryption` must be a `ParquetEncryptionProperties` or "
            "`ParquetEncryptionConfig`, got 'ParquetDecryptionConfig'"
        ),
    ):
        basic_data().write_parquet(path, encryption=decryption_config(kms))  # type: ignore[arg-type]

    basic_data().write_parquet(path)
    with pytest.raises(
        TypeError,
        match=re.escape(
            "`decryption` must be a `ParquetDecryptionProperties` or "
            "`ParquetDecryptionConfig`, got 'ParquetEncryptionConfig'"
        ),
    ):
        pl.scan_parquet(path, decryption=encryption_config(kms))  # type: ignore[arg-type]
