import io
import sys
from typing import Any

import pytest

import polars as pl
from polars.exceptions import ComputeError


@pytest.mark.slow
@pytest.mark.parametrize("format", ["parquet", "csv", "ndjson", "ipc"])
def test_scan_nonexistent_cloud_path_17444(format: str) -> None:
    # https://github.com/pola-rs/polars/issues/17444

    path_str = f"s3://my-nonexistent-bucket/data.{format}"
    scan_function = getattr(pl, f"scan_{format}")

    # Just calling the scan function should not raise any errors
    if format == "ndjson":
        # NDJSON does not have a `retries` parameter yet - so use the default
        result = scan_function(path_str)
    else:
        result = scan_function(path_str, retries=0)
    assert isinstance(result, pl.LazyFrame)

    # Upon collection, it should fail
    with pytest.raises(ComputeError):
        result.collect()


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

    # We can't test these with the `read_` functions as they end up executing
    # the query
    if io_func.__name__.startswith("scan_"):
        # Passing `None` should disable the automatic instantiation of
        # `CredentialProviderAWS`
        io_func("s3://bucket/path", credential_provider=None)
        # Passing `storage_options` should disable the automatic instantiation of
        # `CredentialProviderAWS`
        io_func("s3://bucket/path", credential_provider="auto", storage_options={})

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


def test_scan_credential_provider_serialization_pyversion() -> None:
    lf = pl.scan_parquet(
        "s3://bucket/path", credential_provider=pl.CredentialProviderAWS()
    )

    serialized = lf.serialize()
    serialized = bytearray(serialized)

    # We can't monkeypatch sys.python_version so we just mutate the output
    # instead.

    v = b"PLPYFN"
    i = serialized.index(v) + len(v)
    a, b = serialized[i:][:2]
    serialized_pyver = (a, b)
    assert serialized_pyver == (sys.version_info.minor, sys.version_info.micro)
    # Note: These are loaded as u8's
    serialized[i] = 255
    serialized[i + 1] = 254

    with pytest.raises(ComputeError, match=r"python version.*(3, 255, 254).*differs.*"):
        lf = pl.LazyFrame.deserialize(io.BytesIO(serialized))
