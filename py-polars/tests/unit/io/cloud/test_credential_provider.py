import io
from typing import Any

import pytest

import polars as pl
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
def test_scan_credential_provider(
    io_func: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    err_magic = "err_magic_3"

    def raises(*_: None, **__: None) -> None:
        raise AssertionError(err_magic)

    monkeypatch.setattr(pl.CredentialProviderAWS, "__init__", raises)

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

    # Note to reader: It is converted to a ComputeError as it is being called
    # from Rust.
    with pytest.raises(ComputeError, match=err_magic):
        io_func("s3://bucket/path", credential_provider=raises_2).collect()


def test_scan_credential_provider_serialization() -> None:
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


def test_credential_provider_skips_config_autoload(
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
