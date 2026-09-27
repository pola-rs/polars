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
    from collections.abc import Callable, Iterator

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.slow()

ACCESS_KEY = "accesskey"
SECRET_KEY = "secretkey"
REGION = "us-east-1"


class CountingS3:
    """Moto S3 server with request counts and an arrival-ordered log.

    `requests` counts by method and range; `log` records (method, bucket/key, range).
    """

    def __init__(self) -> None:
        app = DomainDispatcherApplication(create_backend_app)
        self.requests: Counter[str] = Counter()
        self.log: list[tuple[str, str, str]] = []
        # Run before moto; any non-None return value triggers an empty HTTP 500.
        self.on_request: Callable[[dict[str, Any]], Any] | None = None
        lock = threading.Lock()

        def counting_app(environ: dict[str, Any], start_response: Any) -> Any:
            method = environ["REQUEST_METHOD"]
            range_header = environ.get("HTTP_RANGE", "")
            with lock:
                self.requests[f"{method} {range_header}"] += 1
                self.log.append(
                    (method, environ.get("PATH_INFO", "").lstrip("/"), range_header)
                )
            if self.on_request is not None:
                response = self.on_request(environ)
                if response is not None:
                    start_response("500 Internal Server Error", [])
                    return [b""]
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

    def range_get_keys(self, log: list[tuple[str, str, str]]) -> set[str]:
        """Return keys fetched by range GET, counting multi-request footers once."""
        return {key for method, key, rng in log if method == "GET" and rng}

    def since(self, mark: int) -> list[tuple[str, str, str]]:
        return self.log[mark:]


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


def upload_rows(s3: CountingS3, key: str, n: int) -> int:
    """Upload `n` rows as Parquet and return the file size."""
    f = io.BytesIO()
    pl.DataFrame({"x": range(n)}).write_parquet(f, row_group_size=500)
    body = f.getvalue()
    s3.client.put_object(Bucket="bucket", Key=key, Body=body)
    return len(body)


class S3ParquetDataset:
    """Test dataset provider for Parquet files on S3."""

    def __init__(self, s3: CountingS3, keys: list[str], sizes: list[int]) -> None:
        self.keys = keys
        self.sizes = sizes
        self.storage_options = s3.storage_options
        self.pl_schema = pl.Schema({"x": pl.Int64})

    def schema(self) -> Any:
        return self.pl_schema.to_arrow()

    def to_dataset_scan(self, **_kwargs: Any) -> tuple[pl.LazyFrame, str]:
        # Known sizes, as the native Iceberg and Delta scans provide.
        lf = pl.scan_parquet(
            [f"s3://bucket/{key}" for key in self.keys],
            schema=self.pl_schema,
            storage_options=self.storage_options,
            _source_sizes=self.sizes,
        )
        return lf, "v1"

    def new_lazyframe(self, resolve_heavy_sources: int | None) -> pl.LazyFrame:
        from polars._plr import PyLazyFrame
        from polars._utils.wrap import wrap_ldf

        return wrap_ldf(
            PyLazyFrame.new_from_dataset_object(
                self, resolve_heavy_sources=resolve_heavy_sources
            )
        )


def test_resolve_heavy_sources_reads_only_the_retained_footers(
    s3: CountingS3,
) -> None:
    # Planning must fetch exactly the footers it ends up retaining.
    keys = [f"t/{i}.parquet" for i in range(3)]
    sizes = [
        upload_rows(s3, key, n) for key, n in zip(keys, [20, 4000, 20], strict=True)
    ]

    dataset = S3ParquetDataset(s3, keys, sizes)
    lf = dataset.new_lazyframe(resolve_heavy_sources=4)

    mark = len(s3.log)
    retained = lf._ldf._retained_parquet_footers()
    planning = s3.since(mark)

    # Partial metadata requires source 0; source 1 is heavy.
    assert retained == [[(0, 1), (1, 8)]]
    assert s3.range_get_keys(planning) == {f"bucket/{keys[i]}" for i, _ in retained[0]}


@pytest.mark.parametrize("mode", ["none", "row_counts"])
def test_resolve_modes_without_footers(
    s3: CountingS3, plmonkeypatch: PlMonkeyPatch, mode: str
) -> None:
    # Disable splitting reads, but preserve schema and row-count resolution.
    keys = [f"t/{i}.parquet" for i in range(3)]
    sizes = [
        upload_rows(s3, key, n) for key, n in zip(keys, [20, 4000, 20], strict=True)
    ]
    paths = [f"s3://bucket/{key}" for key in keys]
    all_keys = {f"bucket/{key}" for key in keys}

    plmonkeypatch.setenv("POLARS_RESOLVE_METADATA_LEVEL", mode)

    ordinary = pl.scan_parquet(
        paths,
        storage_options=s3.storage_options,
        _source_sizes=sizes,
        _resolve_heavy_sources=4,
    )
    mark = len(s3.log)
    assert ordinary._ldf._retained_parquet_footers() == [[(0, 1)]]
    # Both retain only source 0's footer; `row_counts` also reads every row count.
    expected = {f"bucket/{keys[0]}"} if mode == "none" else all_keys
    assert s3.range_get_keys(s3.since(mark)) == expected

    provided = pl.scan_parquet(
        paths,
        schema=pl.Schema({"x": pl.Int64}),
        storage_options=s3.storage_options,
        _source_sizes=sizes,
        _resolve_heavy_sources=4,
    )
    mark = len(s3.log)
    assert provided._ldf._retained_parquet_footers() == [[]]
    assert s3.range_get_keys(s3.since(mark)) == set()

    dataset = S3ParquetDataset(s3, keys, sizes).new_lazyframe(resolve_heavy_sources=4)
    mark = len(s3.log)
    assert dataset._ldf._retained_parquet_footers() == [[]]
    assert s3.range_get_keys(s3.since(mark)) == set()


def test_dataset_footer_waves_overlap(s3: CountingS3) -> None:
    # The barrier requires both datasets' heavy-footer reads to overlap.
    datasets = []
    heavy_keys = set()
    for name in ["a", "b"]:
        keys = [f"{name}/{i}.parquet" for i in range(3)]
        sizes = [
            upload_rows(s3, key, n) for key, n in zip(keys, [20, 4000, 20], strict=True)
        ]
        heavy_keys.add(f"bucket/{keys[1]}")
        datasets.append(S3ParquetDataset(s3, keys, sizes))

    barrier = threading.Barrier(2, timeout=10)
    lock = threading.Lock()
    seen: set[str] = set()
    passed: list[str] = []
    broken: list[str] = []

    def hook(environ: dict[str, Any]) -> Any:
        key = environ.get("PATH_INFO", "").lstrip("/")
        if key not in heavy_keys or not environ.get("HTTP_RANGE"):
            return None
        with lock:
            # Only the first request per file participates in the barrier.
            if key in seen:
                return None
            seen.add(key)
        try:
            barrier.wait()
        except threading.BrokenBarrierError:
            # Record the failure even if a later retry succeeds.
            broken.append(key)
            return "500"
        passed.append(key)
        return None

    s3.on_request = hook

    lf = pl.concat([d.new_lazyframe(resolve_heavy_sources=4) for d in datasets])
    retained = lf._ldf._retained_parquet_footers()

    assert broken == []
    assert sorted(passed) == sorted(heavy_keys)
    # Both scans keep source 0 and their heavy file's eight row groups.
    assert retained == [[(0, 1), (1, 8)], [(0, 1), (1, 8)]]
