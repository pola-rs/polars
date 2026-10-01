"""Tests for the path reported in object-store error messages."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from tests.unit.io.cloud.moto_server import CountingS3

pytestmark = pytest.mark.slow()


def upload(s3: CountingS3, key: str) -> None:
    f = io.BytesIO()
    pl.DataFrame({"a": range(100)}).write_parquet(f)
    s3.client.put_object(Bucket="bucket", Key=key, Body=f.getvalue())


def test_error_reports_failing_path_not_cached_store_path(s3: CountingS3) -> None:
    # Object stores are cached per (bucket, cloud options), so the store for
    # `s3://bucket` outlives the object that built it.
    storage_options: dict[str, Any] = {**s3.storage_options, "max_retries": 0}

    upload(s3, "first.parquet")
    upload(s3, "nested/dir/second.parquet")

    pl.read_parquet("s3://bucket/first.parquet", storage_options=storage_options)

    def fail_second(environ: dict[str, Any]) -> Any:
        return "500" if "second.parquet" in environ.get("PATH_INFO", "") else None

    s3.on_request = fail_second

    with pytest.raises(OSError) as exc_info:
        pl.read_parquet(
            "s3://bucket/nested/dir/second.parquet", storage_options=storage_options
        )

    msg = str(exc_info.value)
    assert "(path: s3://bucket/nested/dir/second.parquet)" in msg
    assert "first.parquet" not in msg
