from __future__ import annotations

import contextlib
import io
import os
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
    files: dict[str, bytes], delay: float = 0.0
) -> Iterator[tuple[str, list[tuple[int, str]]]]:
    """Serve `files` by raw request target, with ranges, after an optional delay.

    Other targets get a 404. Logs (client port, target) per request.
    """
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    requests: list[tuple[int, str]] = []

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
            if (range_ := self.headers.get("Range")) is not None:
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
            if body:
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
