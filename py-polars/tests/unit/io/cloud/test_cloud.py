from __future__ import annotations

import contextlib
import gzip
import io
import itertools
import os
import random
import re
import subprocess
import sys
import time
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest

import polars as pl
from polars.exceptions import ArgumentRemovedError
from polars.io.cloud._utils import _is_aws_cloud
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Iterator

    from tests.conftest import PlMonkeyPatch
    from tests.unit.io.cloud.conftest import CountingS3


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

    # Note: We get called once per attempt, and the store is rebuilt once on error.
    if call_count != 2:
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


@pytest.mark.skipif(sys.platform == "win32", reason="polars/#28961")
@pytest.mark.slow
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


def test_huggingface_token_env_var() -> None:
    try:
        subprocess.check_output(
            [
                sys.executable,
                "-c",
                """\
import polars as pl

pl.scan_csv("hf://...").collect()
""",
            ],
            env={
                **os.environ,
                "POLARS_VERBOSE_SENSITIVE": "1",
                "HF_TOKEN": "of news",
            },
            stderr=subprocess.STDOUT,
        )
    except subprocess.CalledProcessError as err:
        capture = err.stdout
    else:
        raise AssertionError

    assert b"Bearer of news" in capture


@pytest.mark.parametrize(
    "func",
    [
        pl.read_parquet,
        pl.read_parquet_metadata,
        pl.scan_parquet,
        pl.scan_ipc,
        pl.read_ndjson,
        pl.scan_ndjson,
    ],
)
def test_retries_removed(func: Any) -> None:
    msg = 'Pass {"max_retries": n} via `storage_options` instead.'
    with pytest.raises(ArgumentRemovedError, match=re.escape(msg)):
        func("s3://.../...", retries=3)


@pytest.mark.parametrize(
    "func",
    [pl.scan_ipc, pl.read_ndjson, pl.scan_ndjson],
)
def test_file_cache_ttl_removed(func: Any) -> None:
    msg = "The file cache is no longer supported."
    with pytest.raises(ArgumentRemovedError, match=re.escape(msg)):
        func("s3://.../...", file_cache_ttl=7)


def test_scan_ipc_cache_removed() -> None:
    msg = "The file cache is no longer supported."
    with pytest.raises(ArgumentRemovedError, match=re.escape(msg)):
        pl.scan_ipc("s3://.../...", cache=True)  # type: ignore[call-arg]


def _scan_s3_endpoint(endpoint: str, **storage_options: Any) -> pl.LazyFrame:
    return pl.scan_parquet(
        "s3://bucket/x.parquet",
        storage_options={
            "aws_endpoint_url": endpoint,
            "aws_allow_http": "true",
            "aws_access_key_id": "a",
            "aws_secret_access_key": "b",
            "aws_region": "us-east-1",
            "max_retries": 0,
            **storage_options,
        },
        credential_provider=None,
    )


def _parquet_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_parquet(buf)
    return buf.getvalue()


@contextlib.contextmanager
def _http_file_server(
    files: dict[str, bytes],
    delay: float = 0.0,
    *,
    ranges: bool = True,
    truncate_first: int = 0,
) -> Iterator[tuple[str, list[tuple[int, str]]]]:
    """Serve `files` by raw request target, with ranges, after an optional delay.

    Other targets get a 404. Logs (client port, target) per request. With
    `ranges=False`, the `Range` header is ignored. The first `truncate_first` GET
    bodies are cut off after one byte and the connection is closed.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests: list[tuple[int, str]] = []
    n_truncated = 0

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _serve(self, *, body: bool) -> None:
            requests.append((self.client_address[1], self.path))
            time.sleep(delay)
            data = files.get(self.path)
            if data is None:
                self.send_response(404)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return

            start, end = 0, len(data)
            if ranges and (range_ := self.headers.get("Range")) is not None:
                first, last = range_.removeprefix("bytes=").split("-")
                if first:
                    start = int(first)
                    end = min(int(last) + 1, len(data)) if last else len(data)
                else:
                    start = max(len(data) - int(last), 0)
                self.send_response(206)
                self.send_header(
                    "Content-Range", f"bytes {start}-{end - 1}/{len(data)}"
                )
            else:
                self.send_response(200)
            self.send_header("Content-Length", str(end - start))
            self.end_headers()
            nonlocal n_truncated
            if body and n_truncated < truncate_first:
                n_truncated += 1
                self.wfile.write(data[start : start + 1])
                self.close_connection = True
            elif body:
                self.wfile.write(data[start:end])

        def do_GET(self) -> None:
            self._serve(body=True)

        def do_HEAD(self) -> None:
            self._serve(body=False)

        def log_message(self, format: str, *args: Any) -> None:
            pass

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        # A short poll interval keeps `shutdown()` from waiting up to 0.5s.
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )
        thread.start()
        try:
            yield f"http://127.0.0.1:{server.server_address[1]}", requests
        finally:
            server.shutdown()
            thread.join()


@pytest.mark.slow
def test_scan_ndjson_http_empty_file_range_ignored() -> None:
    # The server answers the ranged GET of an empty file with a 200.
    with _http_file_server({"/empty.ndjson": b""}, ranges=False) as (endpoint, _):
        out = pl.scan_ndjson(
            f"{endpoint}/empty.ndjson", schema={"a": pl.Int64}
        ).collect()

    assert_frame_equal(out, pl.DataFrame(schema={"a": pl.Int64}))


@pytest.mark.slow
def test_scan_ndjson_http_prefix_metrics_retried_body(
    plmonkeypatch: PlMonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    # The body of the initial range request fails once and succeeds on the retry.
    # The request is accounted once.
    data = b'{"a":1}\n' * 1000
    plmonkeypatch.setenv("POLARS_LOG_METRICS", "1")

    with _http_file_server({"/a.ndjson": data}, truncate_first=1) as (
        endpoint,
        requests,
    ):
        capfd.readouterr()
        out = pl.scan_ndjson(f"{endpoint}/a.ndjson", schema={"a": pl.Int64}).collect()
        capture = capfd.readouterr().err

    assert_frame_equal(out, pl.DataFrame({"a": [1] * 1000}))
    assert [target for _, target in requests] == ["/a.ndjson", "/a.ndjson"]
    [line] = (x for x in capture.splitlines() if x.startswith("multi-scan"))
    assert f"total_bytes_requested={len(data)}," in line


@pytest.mark.slow
def test_scan_http_same_host() -> None:
    targets = [
        "/a.parquet",
        "/dir/b%2541.parquet",
        "/c.parquet?X-Amz-Signature=x%2Fy&e=1",
        "/d.parquet?q",
        # Not representable as an object path; gets its own store.
        "/x//e.parquet",
    ]
    dfs = [pl.DataFrame({"a": [i]}) for i in range(len(targets))]
    files = {t: _parquet_bytes(df) for t, df in zip(targets, dfs, strict=True)}

    with _http_file_server(files) as (endpoint, requests):
        for target, df in zip(targets, dfs, strict=True):
            assert_frame_equal(pl.scan_parquet(endpoint + target).collect(), df)

    # URLs are requested verbatim.
    assert {target for _, target in requests} == set(targets)
    # Files on one host share the cached store's connection pool.
    assert len({port for port, _ in requests}) < len(targets)


@pytest.mark.slow
def test_scan_http_same_host_error() -> None:
    df = pl.DataFrame({"a": [1]})
    files = {"/a.parquet?token=A": _parquet_bytes(df)}

    with _http_file_server(files) as (endpoint, requests):
        q = pl.scan_parquet(f"{endpoint}/a.parquet?token=A")
        assert_frame_equal(q.collect(), df)
        n = len(requests)

        q = pl.scan_parquet(f"{endpoint}/missing.parquet")
        with pytest.raises(FileNotFoundError, match=r"missing\.parquet") as exc:
            q.collect()

    # The error names the failing URL, not the one the host's store was built for.
    assert "a.parquet" not in str(exc.value)
    # An error does not drop the host's connection pool.
    assert {port for port, _ in requests[n:]} <= {port for port, _ in requests[:n]}


@pytest.mark.slow
@pytest.mark.parametrize("cache_size", [0, 1, 2])
def test_http_store_cache_size(cache_size: int, plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_HTTP_STORE_CACHE_SIZE", str(cache_size))
    df = pl.DataFrame({"a": [1]})
    files = {"/a.parquet": _parquet_bytes(df)}

    with (
        _http_file_server(files) as (a, a_requests),
        _http_file_server(files) as (b, _),
    ):
        assert_frame_equal(pl.scan_parquet(f"{a}/a.parquet").collect(), df)
        n = len(a_requests)
        for endpoint in [b, a]:
            assert_frame_equal(pl.scan_parquet(f"{endpoint}/a.parquet").collect(), df)

    # Host `a` keeps its store and connections only if the cache also fits `b`.
    first = {port for port, _ in a_requests[:n]}
    last = {port for port, _ in a_requests[n:]}
    assert (last <= first) == (cache_size >= 2)


@pytest.mark.slow
def test_cloud_connection_refused_error() -> None:
    q = _scan_s3_endpoint("http://127.0.0.1:1")

    with pytest.raises(ConnectionRefusedError, match="Caused by:") as exc:
        q.collect()

    assert "tcp connect error" in str(exc.value)


@pytest.mark.slow
def test_cloud_not_found_error() -> None:
    with _http_file_server({}) as (endpoint, _):
        q = _scan_s3_endpoint(endpoint)
        with pytest.raises(FileNotFoundError, match=re.escape("x.parquet")):
            q.collect()


@pytest.mark.slow
def test_cloud_timeout_error() -> None:
    with _http_file_server({}, delay=2.0) as (endpoint, _):
        q = _scan_s3_endpoint(endpoint, timeout="100ms", max_retries=1)
        with pytest.raises(TimeoutError, match="after 1 retries") as exc:
            q.collect()

    assert "operation timed out" in str(exc.value)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("sink", "scan"),
    [
        (pl.LazyFrame.sink_ipc, pl.scan_ipc),
        (pl.LazyFrame.sink_parquet, pl.scan_parquet),
    ],
)
def test_sink_single_put_28356(
    s3: CountingS3, sink: Any, scan: Any, plmonkeypatch: PlMonkeyPatch
) -> None:
    # S3 minimum part size.
    plmonkeypatch.setenv("POLARS_UPLOAD_CHUNK_SIZE", str(5 * 1024 * 1024))

    requests: list[str] = []

    def record(environ: dict[str, Any]) -> None:
        # E.g. "PUT", "POST uploads", "PUT partNumber", "POST uploadId".
        query = environ["QUERY_STRING"].partition("=")[0]
        requests.append(f"{environ['REQUEST_METHOD']} {query}".strip())

    s3.on_request = record

    small = pl.DataFrame({"x": range(10)})
    sink(small.lazy(), "s3://bucket/small", storage_options=s3.storage_options)

    assert requests == ["PUT"]

    # Incompressible, so that it spans multiple parts.
    large = pl.DataFrame({"x": pl.int_range(1_000_000, eager=True).hash()})
    requests.clear()
    sink(large.lazy(), "s3://bucket/large", storage_options=s3.storage_options)

    assert requests[0] == "POST uploads"
    assert set(requests[1:-1]) == {"PUT partNumber"}
    assert requests[-1] == "POST uploadId"

    for df, key in [(small, "small"), (large, "large")]:
        out = scan(f"s3://bucket/{key}", storage_options=s3.storage_options)
        assert_frame_equal(out.collect(), df)


@pytest.mark.slow
@pytest.mark.parametrize("decoy", [False, True])
def test_scan_csv_mixed_buckets_29758(s3: CountingS3, decoy: bool) -> None:
    # Unique keys: file cache entries are keyed by URI and persist across runs.
    x, y = f"x_{uuid4()}.csv", f"y_{uuid4()}.csv"
    s3.client.create_bucket(Bucket="bucket-2")
    s3.client.put_object(Bucket="bucket", Key=x, Body=b"a\n1\n")
    s3.client.put_object(Bucket="bucket-2", Key=y, Body=b"a\n2\n3\n4\n")
    if decoy:
        s3.client.put_object(Bucket="bucket", Key=y, Body=b"a\n9\n9\n")

    paths = [f"s3://bucket/{x}", f"s3://bucket-2/{y}"]
    expected = pl.DataFrame({"a": [1, 2, 3, 4]})

    lf = pl.scan_csv(paths, storage_options=s3.storage_options)
    assert_frame_equal(lf.collect(), expected)
    assert lf.select(pl.len()).collect().item() == 4

    lf = pl.scan_csv(
        paths, infer_schema_length=None, storage_options=s3.storage_options
    )
    assert_frame_equal(lf.collect(), expected)


@pytest.mark.slow
def test_scan_ndjson_mixed_buckets_29758(s3: CountingS3) -> None:
    x, y = f"x_{uuid4()}.ndjson", f"y_{uuid4()}.ndjson"
    s3.client.create_bucket(Bucket="bucket-2")
    s3.client.put_object(Bucket="bucket", Key=x, Body=b'{"a":1}\n')
    s3.client.put_object(Bucket="bucket-2", Key=y, Body=b'{"a":2}\n')
    s3.client.put_object(Bucket="bucket", Key=y, Body=b'{"a":9}\n')

    paths = [f"s3://bucket/{x}", f"s3://bucket-2/{y}"]
    expected = pl.DataFrame({"a": [1, 2]})

    lf = pl.scan_ndjson(paths, storage_options=s3.storage_options)
    assert_frame_equal(lf.collect(), expected)
    assert lf.select(pl.len()).collect().item() == 2

    lf = pl.scan_ndjson(
        paths, infer_schema_length=None, storage_options=s3.storage_options
    )
    assert_frame_equal(lf.collect(), expected)


def _assert_fetched_once(
    requests: list[tuple[str, str, str]], key: str, size: int
) -> list[tuple[int, int]]:
    """Assert the GET ranges of `key` cover `0..size` once, in contiguous pieces."""
    ranges = []
    for method, k, range_ in requests:
        if k == f"bucket/{key}":
            assert method == "GET", (key, method, range_)
            first, last = range_.removeprefix("bytes=").split("-")
            ranges.append((int(first), min(int(last) + 1, size)))
    assert ranges, f"no GET ranges recorded for {key}"
    ranges.sort()
    assert ranges[0][0] == 0, (key, ranges)
    for (_, end), (start, _) in itertools.pairwise(ranges):
        assert start == end, (key, ranges)
    assert ranges[-1][1] == size, (key, ranges)
    return ranges


@pytest.mark.slow
def test_scan_ndjson_small_files_single_request(s3: CountingS3) -> None:
    small = [f"small_{i}_{uuid4()}.ndjson" for i in range(5)]
    for i, key in enumerate(small):
        s3.client.put_object(Bucket="bucket", Key=key, Body=f'{{"a":{i}}}\n'.encode())

    empty = f"empty_{uuid4()}.ndjson"
    s3.client.put_object(Bucket="bucket", Key=empty, Body=b"")

    compressed = f"compressed_{uuid4()}.ndjson.gz"
    s3.client.put_object(
        Bucket="bucket", Key=compressed, Body=gzip.compress(b'{"a":5}\n')
    )

    # Larger than the initial fetch, reading continues after the fetched prefix.
    large = f"large_{uuid4()}.ndjson"
    n_large = 150_000
    s3.client.put_object(
        Bucket="bucket",
        Key=large,
        Body="".join(f'{{"a":{i}}}\n' for i in range(n_large)).encode(),
    )

    paths = [f"s3://bucket/{k}" for k in [*small, empty, compressed, large]]
    lf = pl.scan_ndjson(
        paths, schema={"a": pl.Int64}, storage_options=s3.storage_options
    )

    mark = len(s3.log)
    out = lf.collect()

    expected = pl.DataFrame({"a": [*range(5), 5, *range(n_large)]})
    assert_frame_equal(out, expected)

    # Small files are read with a single range request, without a HEAD.
    prefix_len = 256 * 1024
    requests = s3.since(mark)
    for key in small:
        assert [(m, r) for m, k, r in requests if k == f"bucket/{key}"] == [
            ("GET", f"bytes=0-{prefix_len - 1}")
        ]

    assert [(m, r) for m, k, r in requests if k == f"bucket/{compressed}"] == [
        ("GET", f"bytes=0-{prefix_len - 1}")
    ]

    # An empty object rejects any range, the size then comes from a HEAD. The
    # rejected request is retried once, as any failed request.
    assert [(m, r) for m, k, r in requests if k == f"bucket/{empty}"] == [
        ("GET", f"bytes=0-{prefix_len - 1}"),
        ("GET", f"bytes=0-{prefix_len - 1}"),
        ("HEAD", ""),
    ]

    # The large file continues after its prefix, every byte is fetched once.
    large_size = s3.client.head_object(Bucket="bucket", Key=large)["ContentLength"]
    ranges = _assert_fetched_once(requests, large, large_size)
    assert ranges[0] == (0, prefix_len)


@pytest.mark.slow
def test_scan_ndjson_prefix_continuation(
    s3: CountingS3, plmonkeypatch: PlMonkeyPatch
) -> None:
    # Many small chunks with few prefetch permits, after the prefix.
    plmonkeypatch.setenv("POLARS_NDJSON_CHUNK_SIZE", str(64 * 1024))
    plmonkeypatch.setenv("POLARS_NDJSON_CHUNK_PREFETCH_LIMIT", "2")

    rng = random.Random(0)
    values = [rng.getrandbits(62) for _ in range(120_000)]
    body = "".join(f'{{"a":{v}}}\n' for v in values).encode()
    # Random values compress poorly, so the gzip file is larger than the prefix.
    compressed_body = gzip.compress(body)
    assert len(compressed_body) > 256 * 1024

    plain = f"plain_{uuid4()}.ndjson"
    compressed = f"compressed_{uuid4()}.ndjson.gz"
    small = f"small_{uuid4()}.ndjson"
    s3.client.put_object(Bucket="bucket", Key=plain, Body=body)
    s3.client.put_object(Bucket="bucket", Key=compressed, Body=compressed_body)
    s3.client.put_object(Bucket="bucket", Key=small, Body=b'{"a":-1}\n')

    lf = pl.scan_ndjson(
        [f"s3://bucket/{k}" for k in [plain, small, compressed]],
        schema={"a": pl.Int64},
        storage_options=s3.storage_options,
    )
    expected = pl.DataFrame({"a": [*values, -1, *values]})

    mark = len(s3.log)
    assert_frame_equal(lf.collect(), expected)
    requests = s3.since(mark)
    assert len(_assert_fetched_once(requests, plain, len(body))) > 3
    assert len(_assert_fetched_once(requests, compressed, len(compressed_body))) > 3

    expected = expected.with_row_index()
    lf = lf.with_row_index()
    assert_frame_equal(lf.head(130_000).collect(), expected.head(130_000))
    assert_frame_equal(lf.tail(130_000).collect(), expected.tail(130_000))
    assert_frame_equal(
        lf.slice(119_000, 2_000).collect(), expected.slice(119_000, 2_000)
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    ("n_rows", "compress"),
    [
        (10_000, False),  # Smaller than the prefix, read from memory.
        (150_000, False),  # Prefix, then streamed.
        # Resolved by the multi-scan from a row count into a positive slice.
        (150_000, True),
    ],
)
def test_scan_ndjson_single_file_negative_slice(
    s3: CountingS3, n_rows: int, compress: bool
) -> None:
    # An uncompressed single source receives the negative slice in the reader, which
    # reads it in reverse.
    body = "".join(f'{{"a":{i}}}\n' for i in range(n_rows)).encode()
    assert (len(body) > 256 * 1024) == (n_rows > 10_000)
    key = f"neg_{uuid4()}.ndjson" + (".gz" if compress else "")
    s3.client.put_object(
        Bucket="bucket", Key=key, Body=gzip.compress(body) if compress else body
    )

    lf = pl.scan_ndjson(
        f"s3://bucket/{key}", schema={"a": pl.Int64}, storage_options=s3.storage_options
    ).with_row_index(offset=7)
    expected = pl.DataFrame({"a": range(n_rows)}).with_row_index(offset=7)

    assert_frame_equal(lf.tail(1_000).collect(), expected.tail(1_000))
    assert_frame_equal(lf.slice(-5_000, 2_000).collect(), expected.slice(-5_000, 2_000))


@pytest.mark.slow
def test_scan_ndjson_in_memory_file_between_streamed_files(s3: CountingS3) -> None:
    # A file read from memory sits between two streamed files. The third file must
    # not start prefetching before the first spawned all its prefetches, which
    # deadlocked on the shared prefetch permits. A subprocess turns a hang into a
    # timeout.
    body = "".join(f'{{"a":{i}}}\n' for i in range(1_000_000)).encode()
    keys = [f"{i}_{uuid4()}.ndjson" for i in range(3)]
    for key, data in zip(keys, [body, b"", body], strict=True):
        s3.client.put_object(Bucket="bucket", Key=key, Body=data)

    paths = [f"s3://bucket/{k}" for k in keys]
    code = f"""\
import polars as pl

lf = pl.scan_ndjson(
    {paths!r}, schema={{"a": pl.Int64}}, storage_options={s3.storage_options!r}
)
assert lf.collect().height == 2_000_000
"""
    env = {
        **os.environ,
        "POLARS_MAX_THREADS": "2",
        "POLARS_MAX_CONCURRENT_SCANS": "3",
        "POLARS_NDJSON_CHUNK_SIZE": str(1024 * 1024),
        "POLARS_NDJSON_CHUNK_PREFETCH_LIMIT": "1",
    }
    subprocess.run([sys.executable, "-c", code], env=env, check=True, timeout=60)
