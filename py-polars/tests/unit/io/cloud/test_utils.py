from pathlib import Path

import pytest
from polars.io._utils import _is_supported_cloud


@pytest.mark.parametrize(
    ("possible_url", "expected_result"),
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
        ("/local/file.tmp", False,),
        ("file:///local/file.tmp", False,),
    ],
)
def test_is_cloud_url(possible_url: str | Path, expected_result: bool) -> None:
    assert _is_supported_cloud(possible_url) is expected_result
