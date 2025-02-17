import io
import pickle
from typing import Any

import pytest

import polars as pl
import polars.io.cloud.credential_provider
from polars.exceptions import ComputeError


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
        # Passing `storage_options` should disable the automatic instantiation of
        # `CredentialProviderAWS`
        io_func(
            "s3://bucket/path",
            credential_provider="auto",
            storage_options={"aws_access_key_id": "polars"},
        )

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
    with pytest.raises(pl.exceptions.ComputeError, match="err_magic_1"):
        q.collect()

    q = pickle.loads(pickle.dumps(q))

    def raises_2(*a: Any, **kw: Any) -> None:
        msg = "err_magic_2"
        raise AssertionError(msg)

    monkeypatch.setattr(provider_class, "__init__", raises_2)

    # Check that auto-initialization happens upon executing the deserialized
    # query.
    with pytest.raises(pl.exceptions.ComputeError, match="err_magic_2"):
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

    with pytest.raises(ComputeError, match=err_magic):
        lf.collect()


def test_credential_provider_skips_google_config_autoload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GOOGLE_SERVICE_ACCOUNT_PATH", "__non_existent")

    with pytest.raises(ComputeError, match="__non_existent"):
        pl.scan_parquet("gs://.../...", credential_provider=None).collect()

    err_magic = "err_magic_3"

    def raises() -> pl.CredentialProviderFunctionReturn:
        raise AssertionError(err_magic)

    # We should get a different error raised by our `raises()` function.
    with pytest.raises(ComputeError, match=err_magic):
        pl.scan_parquet("gs://.../...", credential_provider=raises).collect()
