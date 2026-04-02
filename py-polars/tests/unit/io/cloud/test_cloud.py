from __future__ import annotations

import contextlib
import subprocess
import sys
from functools import partial
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.io.cloud._utils import _is_aws_cloud

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch


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
        result = scan_function(path_str, storage_options={"max_retries": 0})
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


def test_storage_options_retry_config(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    capture = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
import contextlib
import os

import polars as pl

os.environ["POLARS_VERBOSE"] = "1"
os.environ["POLARS_CLOUD_MAX_RETRIES"] = "1"
os.environ["POLARS_CLOUD_RETRY_TIMEOUT_MS"] = "1"
os.environ["POLARS_CLOUD_RETRY_INIT_BACKOFF_MS"] = "2"
os.environ["POLARS_CLOUD_RETRY_MAX_BACKOFF_MS"] = "10373"
os.environ["POLARS_CLOUD_RETRY_BASE_MULTIPLIER"] = "6.28"

q = pl.scan_parquet(
    "s3://.../...",
    storage_options={"aws_endpoint_url": "https://localhost:333"},
    credential_provider=None,
)

with contextlib.suppress(OSError):
    q.collect()

""",
        ],
        stderr=subprocess.STDOUT,
    ).decode()

    assert (
        """\
init_backoff: 2ms, \
max_backoff: 10.373s, \
base: 6.28 }, \
max_retries: 1, \
retry_timeout: 1ms"""
        in capture
    )

    q = pl.scan_parquet(
        "s3://.../...",
        storage_options={
            "file_cache_ttl": 7,
            "max_retries": 0,
            "retry_timeout_ms": 23,
            "retry_init_backoff_ms": 24,
            "retry_max_backoff_ms": 9875,
            "retry_base_multiplier": 3.14159,
            "aws_endpoint_url": "https://localhost:333",
        },
        credential_provider=None,
    )

    capfd.readouterr()

    with pytest.raises(OSError):
        q.collect()

    capture = capfd.readouterr().err

    assert "file_cache_ttl: 7" in capture

    assert (
        """\
init_backoff: 24ms, \
max_backoff: 9.875s, \
base: 3.14159 }, \
max_retries: 0, \
retry_timeout: 23ms"""
        in capture
    )
