from __future__ import annotations

import contextlib
import warnings
from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


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


@pytest.mark.parametrize(
    "function",
    [
        partial(pl.scan_parquet, "file:///dummy"),
        partial(pl.scan_csv, "file:///dummy"),
        partial(pl.scan_ipc, "file:///dummy"),
        partial(pl.scan_ndjson, "file:///dummy"),
        partial(pl.read_parquet, "file:///dummy"),
        partial(pl.read_ndjson, "file:///dummy"),
        partial(pl.DataFrame().write_parquet, "file:///dummy"),
        partial(pl.DataFrame().write_csv, "file:///dummy"),
        partial(pl.DataFrame().write_ipc, "file:///dummy"),
        partial(pl.LazyFrame().sink_parquet, "file:///dummy"),
        partial(pl.LazyFrame().sink_csv, "file:///dummy"),
        partial(pl.LazyFrame().sink_ipc, "file:///dummy"),
        partial(pl.LazyFrame().sink_ndjson, "file:///dummy"),
    ],
)
def test_storage_options_retries(
    function: Any,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

    with pytest.raises(
        DeprecationWarning,
        match=r"the `retries` parameter was deprecated in 1.37.1; specify 'max_retries' in `storage_options` instead",
    ):
        function(retries=7)

    capfd.readouterr()

    with warnings.catch_warnings(), contextlib.suppress(OSError):
        warnings.simplefilter("ignore")
        function(retries=7)

    capture = capfd.readouterr().err
    assert "max_retries: 7" in capture

    with contextlib.suppress(OSError):
        function(storage_options={"max_retries": 13})
    capture = capfd.readouterr().err
    assert "max_retries: 13" in capture

    with pytest.raises(
        ValueError, match=r"invalid value for 'max_retries': '1' \(expected int\)"
    ):
        function(storage_options={"max_retries": "1"})

    capfd.readouterr()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            function(file_cache_ttl=13)
        except OSError:
            pass
        except TypeError:
            return

    capture = capfd.readouterr().err
    assert "file_cache_ttl: 13" in capture

    with (
        pytest.raises(
            DeprecationWarning,
            match=r"the `file_cache_ttl` parameter was deprecated in 1.37.1; specify 'file_cache_ttl' in `storage_options` instead",
        ),
        contextlib.suppress(OSError),
    ):
        function(file_cache_ttl=13)


def test_storage_options_retry_config(
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

    monkeypatch.setenv("POLARS_CLOUD_MAX_RETRIES", "53")
    monkeypatch.setenv("POLARS_CLOUD_RETRY_TIMEOUT_MS", "10371")
    monkeypatch.setenv("POLARS_CLOUD_RETRY_INIT_BACKOFF_MS", "10372")
    monkeypatch.setenv("POLARS_CLOUD_RETRY_MAX_BACKOFF_MS", "10373")
    monkeypatch.setenv("POLARS_CLOUD_RETRY_BASE_MULTIPLIER", "6.28")

    capfd.readouterr()
    pl.scan_parquet("", storage_options={})
    capture = capfd.readouterr().err

    assert (
        """\
max_retries: 53, \
retry_timeout: 10.371s, \
retry_init_backoff: 10.372s, \
retry_max_backoff: 10.373s, \
retry_base_multiplier: TotalOrdWrap(6.28)"""
        in capture
    )

    capfd.readouterr()
    pl.scan_parquet(
        "",
        storage_options={
            "file_cache_ttl": 7,
            "max_retries": 3,
            "retry_timeout_ms": 9873,
            "retry_init_backoff_ms": 9874,
            "retry_max_backoff_ms": 9875,
            "retry_base_multiplier": 3.14159,
        },
    )
    capture = capfd.readouterr().err

    assert "file_cache_ttl: 7" in capture

    assert (
        """\
max_retries: 3, \
retry_timeout: 9.873s, \
retry_init_backoff: 9.874s, \
retry_max_backoff: 9.875s, \
retry_base_multiplier: TotalOrdWrap(3.14159)"""
        in capture
    )
