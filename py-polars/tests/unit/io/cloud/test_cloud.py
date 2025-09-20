from __future__ import annotations

import contextlib
from functools import partial

import pytest

import polars as pl
from polars.io.cloud._utils import _is_aws_cloud


@pytest.mark.slow
@pytest.mark.parametrize("format", ["parquet", "csv", "ndjson", "ipc"])
def test_scan_nonexistent_cloud_path_17444(format: str) -> None:
    # https://github.com/pola-rs/polars/issues/17444

    path_str = f"s3://my-nonexistent-bucket/data.{format}"
    scan_function = getattr(pl, f"scan_{format}")
    # Prevent automatic credential provideder instantiation, otherwise CI may fail with
    # * pytest.PytestUnraisableExceptionWarning:
    #   * Exception ignored:
    #     * ResourceWarning: unclosed socket
    scan_function = partial(scan_function, credential_provider=None)

    # Just calling the scan function should not raise any errors
    if format == "ndjson":
        # NDJSON does not have a `retries` parameter yet - so use the default
        result = scan_function(path_str)
    else:
        result = scan_function(path_str, retries=0)
    assert isinstance(result, pl.LazyFrame)

    # Upon collection, it should fail
    with pytest.raises(IOError):
        result.collect()


def test_scan_err_rebuild_store_19933() -> None:
    call_count = 0

    def f() -> None:
        nonlocal call_count
        call_count += 1
        raise AssertionError

    q = pl.scan_parquet(
        "s3://.../...",
        storage_options={"aws_region": "eu-west-1"},
        credential_provider=f,  # type: ignore[arg-type]
    )

    with contextlib.suppress(Exception):
        q.collect()

    # Note: We get called 2 times per attempt
    if call_count != 4:
        raise AssertionError(call_count)


def test_is_aws_cloud() -> None:
    assert _is_aws_cloud(
        scheme="https",
        first_scan_path="https://bucket.s3.eu-west-1.amazonaws.com/key",
    )

    # Slash in front of amazonaws.com
    assert not _is_aws_cloud(
        scheme="https",
        first_scan_path="https://bucket/.s3.eu-west-1.amazonaws.com/key",
    )

    assert not _is_aws_cloud(
        scheme="https",
        first_scan_path="https://bucket?.s3.eu-west-1.amazonaws.com/key",
    )

    # Legacy global endpoint
    assert not _is_aws_cloud(
        scheme="https", first_scan_path="https://bucket.s3.amazonaws.com/key"
    )

    # Has query parameters (e.g. presigned URL).
    assert not _is_aws_cloud(
        scheme="https",
        first_scan_path="https://bucket.s3.eu-west-1.amazonaws.com/key?",
    )
