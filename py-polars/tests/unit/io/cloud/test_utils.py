from __future__ import annotations

import pytest

from polars.io._utils import _is_supported_cloud


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("s3://bucket/file.tmp", True),
        ("s3a://bucket/file.tmp", True),
        ("gs://bucket/file.tmp", True),
        ("gcs://bucket/file.tmp", True),
        ("abfs://container@account/file.tmp", True),
        ("abfss://container@account/file.tmp", True),
        ("azure://container@account/file.tmp", True),
        ("az://container@account/file.tmp", True),
        ("adl://account/file.tmp", True),
        ("file:///local/file.tmp", True),
        ("/local/file.tmp", False),
    ],
)
def test_is_cloud_url(url: str, expected: bool) -> None:
    assert _is_supported_cloud(url) is expected
