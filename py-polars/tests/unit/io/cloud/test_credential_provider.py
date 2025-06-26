import io
import pickle
from pathlib import Path
from typing import Any

import pytest

import polars as pl
import polars.io.cloud.credential_provider


@pytest.mark.parametrize(
    "io_func",
    [
        *[pl.scan_parquet, pl.read_parquet],
        pl.scan_csv,
        *[pl.scan_ndjson, pl.read_ndjson],
        pl.scan_ipc,
    ],
)
def test_credential_provider_scan(
    io_func: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    err_magic = "err_magic_3"

    def raises(*_: None, **__: None) -> None:
        raise AssertionError(err_magic)

    from polars.io.cloud.credential_provider._builder import CredentialProviderBuilder

    monkeypatch.setattr(CredentialProviderBuilder, "__init__", raises)

    with pytest.raises(AssertionError, match=err_magic):
        io_func("s3://bucket/path", credential_provider="auto")

    with pytest.raises(AssertionError, match=err_magic):
        io_func(
            "s3://bucket/path",
            credential_provider="auto",
            storage_options={"aws_region": "eu-west-1"},
        )

    # We can't test these with the `read_` functions as they end up executing
    # the query
    if io_func.__name__.startswith("scan_"):
        # Passing `None` should disable the automatic instantiation of
        # `CredentialProviderAWS`
        io_func("s3://bucket/path", credential_provider=None)

    err_magic = "err_magic_7"

    def raises_2() -> pl.CredentialProviderFunctionReturn:
        raise AssertionError(err_magic)

    with pytest.raises(AssertionError, match=err_magic):
        io_func("s3://bucket/path", credential_provider=raises_2).collect()


@pytest.mark.parametrize(
    ("provider_class", "path"),
    [
        (polars.io.cloud.credential_provider.CredentialProviderAWS, "s3://.../..."),
        (polars.io.cloud.credential_provider.CredentialProviderGCP, "gs://.../..."),
        (polars.io.cloud.credential_provider.CredentialProviderAzure, "az://.../..."),
    ],
)
def test_credential_provider_serialization_auto_init(
    provider_class: polars.io.cloud.credential_provider.CredentialProvider,
    path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raises_1(*a: Any, **kw: Any) -> None:
        msg = "err_magic_1"
        raise AssertionError(msg)

    monkeypatch.setattr(provider_class, "__init__", raises_1)

    # If this is not set we will get an error before hitting the credential
    # provider logic when polars attempts to retrieve the region from AWS.
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    # Credential provider should not be initialized during query plan construction.
    q = pl.scan_parquet(path)

    # Check baseline - query plan is configured to auto-initialize the credential
    # provider.
    with pytest.raises(AssertionError, match="err_magic_1"):
        q.collect()

    q = pickle.loads(pickle.dumps(q))

    def raises_2(*a: Any, **kw: Any) -> None:
        msg = "err_magic_2"
        raise AssertionError(msg)

    monkeypatch.setattr(provider_class, "__init__", raises_2)

    # Check that auto-initialization happens upon executing the deserialized
    # query.
    with pytest.raises(AssertionError, match="err_magic_2"):
        q.collect()


def test_credential_provider_serialization_custom_provider() -> None:
    err_magic = "err_magic_3"

    class ErrCredentialProvider(pl.CredentialProvider):
        def __call__(self) -> pl.CredentialProviderFunctionReturn:
            raise AssertionError(err_magic)

    lf = pl.scan_parquet(
        "s3://bucket/path", credential_provider=ErrCredentialProvider()
    )

    serialized = lf.serialize()

    lf = pl.LazyFrame.deserialize(io.BytesIO(serialized))

    with pytest.raises(AssertionError, match=err_magic):
        lf.collect()


def test_credential_provider_gcp_skips_config_autoload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_PATH", "__non_existent")

    with pytest.raises(OSError, match="__non_existent"):
        pl.scan_parquet("gs://.../...", credential_provider=None).collect()

    err_magic = "err_magic_3"

    def raises() -> pl.CredentialProviderFunctionReturn:
        raise AssertionError(err_magic)

    with pytest.raises(AssertionError, match=err_magic):
        pl.scan_parquet("gs://.../...", credential_provider=raises).collect()


def test_credential_provider_aws_import_error_with_requested_profile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _session(self: Any) -> None:
        msg = "err_magic_3"
        raise ImportError(msg)

    monkeypatch.setattr(pl.CredentialProviderAWS, "_session", _session)
    monkeypatch.setenv("AWS_REGION", "eu-west-1")

    q = pl.scan_parquet(
        "s3://.../...",
        credential_provider=pl.CredentialProviderAWS(profile_name="test_profile"),
    )

    with pytest.raises(
        pl.exceptions.ComputeError,
        match=(
            "cannot load requested aws_profile 'test_profile': ImportError: err_magic_3"
        ),
    ):
        q.collect()

    q = pl.scan_parquet(
        "s3://.../...",
        storage_options={"aws_profile": "test_profile"},
    )

    with pytest.raises(
        pl.exceptions.ComputeError,
        match=(
            "cannot load requested aws_profile 'test_profile': ImportError: err_magic_3"
        ),
    ):
        q.collect()


@pytest.mark.slow
@pytest.mark.write_disk
def test_credential_provider_aws_endpoint_url_scan_no_parameters(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    tmp_path.mkdir(exist_ok=True)

    _set_default_credentials(tmp_path, monkeypatch)
    cfg_file_path = tmp_path / "config"

    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:333
""")

    # Scan with no parameters should load via CredentialProviderAWS
    q = pl.scan_parquet("s3://.../...")

    capfd.readouterr()

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:333"):
        q.collect()

    capture = capfd.readouterr().err
    lines = capture.splitlines()

    assert "[CredentialProviderAWS]: Loaded endpoint_url: http://localhost:333" in lines


@pytest.mark.slow
@pytest.mark.write_disk
def test_credential_provider_aws_endpoint_url_serde(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    tmp_path.mkdir(exist_ok=True)

    _set_default_credentials(tmp_path, monkeypatch)
    cfg_file_path = tmp_path / "config"

    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:333
""")

    q = pl.scan_parquet("s3://.../...")
    q = pickle.loads(pickle.dumps(q))

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:777
""")

    capfd.readouterr()

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:777"):
        q.collect()


@pytest.mark.slow
@pytest.mark.write_disk
def test_credential_provider_aws_endpoint_url_with_storage_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    tmp_path.mkdir(exist_ok=True)

    _set_default_credentials(tmp_path, monkeypatch)
    cfg_file_path = tmp_path / "config"

    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))
    monkeypatch.setenv("POLARS_VERBOSE", "1")

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:333
""")

    # Previously we would not initialize a credential provider at all if secrets
    # were given under `storage_options`. Now we always initialize so that we
    # can load the `endpoint_url`, but we decide at the very last second whether
    # to also retrieve secrets using the credential provider.
    q = pl.scan_parquet(
        "s3://.../...",
        storage_options={
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
        },
    )

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:333"):
        q.collect()

    capture = capfd.readouterr().err
    lines = capture.splitlines()

    assert (
        "[CredentialProviderAWS]: Will not be used as a provider: unhandled key "
        "in storage_options: 'aws_secret_access_key'"
    ) in lines
    assert "[CredentialProviderAWS]: Loaded endpoint_url: http://localhost:333" in lines


@pytest.mark.parametrize(
    "storage_options",
    [
        {"aws_endpoint_url": "http://localhost:777"},
        {
            "aws_access_key_id": "...",
            "aws_secret_access_key": "...",
            "aws_endpoint_url": "http://localhost:777",
        },
    ],
)
@pytest.mark.slow
@pytest.mark.write_disk
def test_credential_provider_aws_endpoint_url_passed_in_storage_options(
    storage_options: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path.mkdir(exist_ok=True)

    _set_default_credentials(tmp_path, monkeypatch)
    cfg_file_path = tmp_path / "config"
    monkeypatch.setenv("AWS_CONFIG_FILE", str(cfg_file_path))

    cfg_file_path.write_text("""\
[default]
endpoint_url = http://localhost:333
""")

    q = pl.scan_parquet("s3://.../...")

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:333"):
        q.collect()

    # An endpoint_url passed in `storage_options` should take precedence.
    q = pl.scan_parquet(
        "s3://.../...",
        storage_options=storage_options,
    )

    with pytest.raises(IOError, match=r"Error performing HEAD http://localhost:777"):
        q.collect()


def _set_default_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    creds_file_path = tmp_path / "credentials"
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(creds_file_path))

    creds_file_path.write_text("""\
[default]
aws_access_key_id=Z
aws_secret_access_key=Z
""")
