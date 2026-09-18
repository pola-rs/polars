"""Tests for the suffix range request used to read cloud file metadata."""

from __future__ import annotations

import io
import threading
from collections import Counter
from typing import TYPE_CHECKING, Any

import boto3
import pytest
from moto.server import (  # type: ignore[attr-defined]
    DomainDispatcherApplication,
    create_backend_app,
)
from werkzeug.serving import make_server

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterator

pytestmark = pytest.mark.slow()

ACCESS_KEY = "accesskey"
SECRET_KEY = "secretkey"
REGION = "us-east-1"


class CountingS3:
    """Moto S3 server counting the requests it serves, keyed by method and range."""

    def __init__(self) -> None:
        app = DomainDispatcherApplication(create_backend_app)
        self.requests: Counter[str] = Counter()
        lock = threading.Lock()

        def counting_app(environ: dict[str, Any], start_response: Any) -> Any:
            key = f"{environ['REQUEST_METHOD']} {environ.get('HTTP_RANGE', '')}"
            with lock:
                self.requests[key] += 1
            return app(environ, start_response)

        self.server = make_server("127.0.0.1", 0, counting_app, threaded=True)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.client = boto3.client(
            "s3",
            endpoint_url=self.endpoint,
            region_name=REGION,
            aws_access_key_id=ACCESS_KEY,
            aws_secret_access_key=SECRET_KEY,
        )

    @property
    def storage_options(self) -> dict[str, str]:
        return {
            "aws_access_key_id": ACCESS_KEY,
            "aws_secret_access_key": SECRET_KEY,
            "aws_region": REGION,
            "aws_endpoint_url": self.endpoint,
            "aws_allow_http": "true",
        }

    def n_requests(self, prefix: str) -> int:
        return sum(v for k, v in self.requests.items() if k.startswith(prefix))


@pytest.fixture
def s3() -> Iterator[CountingS3]:
    s3 = CountingS3()
    threading.Thread(target=s3.server.serve_forever, daemon=True).start()
    s3.client.create_bucket(Bucket="bucket")
    yield s3
    s3.server.shutdown()


def upload_parquet(s3: CountingS3, key: str, **kwargs: Any) -> tuple[int, int]:
    """Upload a parquet file, returning its size and the size of its footer."""
    df = pl.DataFrame({"a": range(5000), "b": ["x"] * 5000, "c": [1.5] * 5000})
    f = io.BytesIO()
    df.write_parquet(f, **kwargs)
    body = f.getvalue()
    s3.client.put_object(Bucket="bucket", Key=key, Body=body)
    # The last 8 bytes hold the metadata length followed by the magic bytes.
    return len(body), int.from_bytes(body[-8:-4], "little") + 8


def test_scan_parquet_metadata_without_head_request(s3: CountingS3) -> None:
    upload_parquet(s3, "hit.parquet")

    df = pl.scan_parquet(
        "s3://bucket/hit.parquet", storage_options=s3.storage_options
    ).collect()

    assert df.height == 5000
    # The metadata is fetched by a single suffix range request, whose response also
    # carries the file size that a HEAD request would otherwise be needed for.
    assert s3.n_requests("HEAD") == 0
    assert s3.n_requests("GET bytes=-") == 1


def test_scan_parquet_metadata_prefetch_too_small(
    s3: CountingS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    file_size, footer_size = upload_parquet(s3, "miss.parquet", row_group_size=500)
    prefetch_size = 1024
    assert footer_size > prefetch_size

    monkeypatch.setenv("POLARS_CLOUD_FOOTER_READ_SIZE", str(prefetch_size))

    df = pl.scan_parquet(
        "s3://bucket/miss.parquet", storage_options=s3.storage_options
    ).collect()

    assert df.height == 5000
    assert df["a"].sum() == sum(range(5000))
    # The footer did not fit in the prefetch, so it is re-fetched in full. Still no
    # HEAD: the file size is known from the first response.
    assert s3.n_requests("HEAD") == 0
    assert s3.requests[f"GET bytes=-{prefetch_size}"] == 1
    assert s3.requests[f"GET bytes={file_size - footer_size}-{file_size - 1}"] == 1
