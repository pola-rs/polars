from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path, PosixPath
from typing import TYPE_CHECKING, Any, TypeVar, cast

import pytest

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Callable, Generator


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cloud-distributed",
        action="store_true",
        default=False,
        help="Run all queries by default of the distributed engine",
    )


_xdist_crash_config: pytest.Config | None = None
_xdist_crash_log_dir: tempfile.TemporaryDirectory[str] | None = None

# How much of a crashed worker's captured output we echo, in bytes.
_XDIST_CRASH_OUTPUT_LIMIT = 64 * 1024


def _crash_log_path(log_dir: str | Path, worker_id: str, stream: str) -> Path:
    return Path(log_dir) / f"{worker_id}.{stream}"


def _capture_to_file(capman: Any, stream: str, path: Path) -> None:
    """Redirect pytest's fd-level capture of `stream` to a persistent file."""
    capture = getattr(capman._global_capturing, stream, None)
    tmpfile = getattr(capture, "tmpfile", None)
    targetfd = getattr(capture, "targetfd", None)
    if tmpfile is None or targetfd is None:
        return  # Not capturing this stream at fd level (`-s`, `--capture=sys`).

    # This is rather hacky but the tempfile pytest creates is passed to multiple
    # places before now, so it would be too late to replace that file object
    # with something we control. So we instead change the underlying file
    # descriptors to point out our path of choice.
    with path.open("wb+", buffering=0) as f:
        os.dup2(f.fileno(), tmpfile.fileno())
        os.dup2(f.fileno(), targetfd)


def pytest_configure(config: pytest.Config) -> None:
    # Stash the config so `pytest_handlecrashitem` (whose hookspec only receives
    # crashitem/report/sched) can reach the capture manager to bypass capturing.
    global _xdist_crash_config
    _xdist_crash_config = config

    workerinput = getattr(config, "workerinput", None)
    if workerinput is None:
        return  # Controller (or no xdist).

    log_dir = workerinput.get("polars_crash_log_dir")
    worker_id = workerinput.get("workerid")
    capman = config.pluginmanager.getplugin("capturemanager")
    if log_dir is None or worker_id is None or capman is None:
        return

    try:
        for stream in ("out", "err"):
            _capture_to_file(
                capman, stream, _crash_log_path(log_dir, worker_id, stream)
            )
    except Exception as exc:
        # Never let this break an otherwise fine test run.
        print(f"failed to redirect fd capture: {exc!r}", file=sys.stderr)


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node: Any) -> None:
    """Tell each xdist worker where to persist its captured output."""
    global _xdist_crash_log_dir
    if _xdist_crash_log_dir is None:
        _xdist_crash_log_dir = tempfile.TemporaryDirectory(
            prefix="polars-xdist-crash-", ignore_cleanup_errors=True
        )
    node.workerinput["polars_crash_log_dir"] = _xdist_crash_log_dir.name


def pytest_unconfigure(config: pytest.Config) -> None:
    global _xdist_crash_log_dir
    if _xdist_crash_log_dir is not None:
        _xdist_crash_log_dir.cleanup()
        _xdist_crash_log_dir = None


def _read_crash_output(worker_id: str, stream: str) -> str:
    if _xdist_crash_log_dir is None:
        return ""
    path = _crash_log_path(_xdist_crash_log_dir.name, worker_id, stream)
    try:
        data = path.read_bytes()
    except OSError:
        return ""
    if len(data) > _XDIST_CRASH_OUTPUT_LIMIT:
        data = b"<truncated>\n" + data[-_XDIST_CRASH_OUTPUT_LIMIT:]
    return data.decode("utf-8", errors="replace").strip()


@pytest.hookimpl(optionalhook=True)
def pytest_handlecrashitem(
    crashitem: str, report: pytest.TestReport, sched: object
) -> None:
    """Log which test an xdist worker was running when it crashed."""
    try:
        worker = getattr(report, "node", None)
        worker_id = getattr(getattr(worker, "gateway", None), "id", "?")
        lines = [f"ERROR: xdist worker {worker_id} crashed while running {crashitem}"]
        for stream, name in (("out", "stdout"), ("err", "stderr")):
            output = _read_crash_output(worker_id, stream)
            if output:
                lines.append(f"--- {worker_id} {name} ---\n{output}")

        def emit() -> None:
            print("\n".join(lines), file=sys.stderr, flush=True)

        # Suspend pytest's output capturing so the message reaches the real
        # streams instead of being buffered (and possibly lost) on crash.
        capman: Any = None
        if _xdist_crash_config is not None:
            capman = _xdist_crash_config.pluginmanager.getplugin("capturemanager")
        if capman is not None:
            with capman.global_and_fixture_disabled():
                emit()
        else:
            emit()
    except Exception as exc:
        # Never let logging failures mask the underlying crash.
        print(f"pytest_handlecrashitem logging failed: {exc!r}", file=sys.stderr)


@pytest.fixture(autouse=True)
def _patched_cloud(
    request: pytest.FixtureRequest, plmonkeypatch: PlMonkeyPatch
) -> None:
    if request.config.getoption("--cloud-distributed"):
        import signal
        import uuid
        from pathlib import Path

        from polars_cloud import ClusterContext, DirectQuery, set_compute_context

        TIMEOUT_SECS = 20

        T = TypeVar("T")

        def with_timeout(f: Callable[[], T]) -> T:
            def handler(signum: Any, frame: Any) -> None:
                msg = "test timed out"
                raise TimeoutError(msg)

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(TIMEOUT_SECS)

            return f()

        ctx = ClusterContext(uri="http://localhost")
        set_compute_context(ctx)

        prev_collect = pl.LazyFrame.collect

        def cloud_collect(lf: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
            # issue: cloud client should use pl.QueryOptFlags()
            if "optimizations" in kwargs:
                kwargs.pop("optimizations")
            if "engine" in kwargs:
                kwargs.pop("engine")

            return prev_collect(
                with_timeout(
                    lambda: lf.remote(plan_type="plain").distributed().execute()
                ).lazy()
            )

        class LazyExe:
            def __init__(
                self,
                query: DirectQuery,
                prev_tgt: io.BytesIO | io.StringIO | io.TextIOBase | None,
                path: str | Path,
            ) -> None:
                self.query = query

                self.prev_tgt = prev_tgt
                self.path = path

            def collect(self) -> pl.DataFrame:
                # 1. Actually execute the query.
                with_timeout(lambda: self.query.await_result())

                # 2. If our target was different, write the result into our target
                #    transparently.
                if self.prev_tgt is not None:
                    if isinstance(self.prev_tgt, (io.StringIO, io.TextIOBase)):
                        with Path(self.path).open("r") as f:
                            self.prev_tgt.write(f.read())
                    else:
                        with Path(self.path).open("rb") as f:
                            self.prev_tgt.write(f.read())

                    # delete the temporary file
                    Path(self.path).unlink()

                # Sinks always return an empty DataFrame.
                return pl.DataFrame({})

        def io_to_path(s: io.IOBase, ext: str) -> Path:
            path = Path(f"/tmp/pc-{uuid.uuid4()!s}.{ext}")

            with Path.open(path, "wb") as f:
                bs = s.read()
                if isinstance(bs, str):
                    bs = bytes(bs, encoding="utf-8")
                f.write(bs)
            s.seek(0, 2)
            return path

        def prepare_scan_sources(src: Any) -> str | Path | list[str | Path]:
            if isinstance(src, io.IOBase):
                src = io_to_path(src, ext)
            elif isinstance(src, bytes):
                src = io_to_path(io.BytesIO(src), ext)
            elif isinstance(src, list):
                for i in range(len(src)):
                    if isinstance(src[i], io.IOBase):
                        src[i] = io_to_path(src[i], ext)
                    elif isinstance(src[i], bytes):
                        src[i] = io_to_path(io.BytesIO(src[i]), ext)

            assert isinstance(src, (str, Path, list)) or (
                isinstance(src, list) and all(isinstance(x, (str, Path)) for x in src)
            )

            return src

        def create_cloud_scan(ext: str) -> Callable[..., pl.LazyFrame]:
            prev_scan = getattr(pl, f"scan_{ext}")
            prev_scan = cast("Callable[..., pl.LazyFrame]", prev_scan)

            def _(
                source: io.BytesIO | io.StringIO | str | Path, *args: Any, **kwargs: Any
            ) -> pl.LazyFrame:
                source = prepare_scan_sources(source)  # type: ignore[assignment]
                return prev_scan(source, *args, **kwargs)

            return _

        def create_read(ext: str) -> Callable[..., pl.DataFrame]:
            prev_read = getattr(pl, f"read_{ext}")
            prev_read = cast("Callable[..., pl.DataFrame]", prev_read)

            def _(
                source: io.BytesIO | str | Path, *args: Any, **kwargs: Any
            ) -> pl.DataFrame:
                if ext == "parquet" and kwargs.get("use_pyarrow", False):
                    return prev_read(source, *args, **kwargs)

                src = prepare_scan_sources(source)
                return prev_read(src, *args, **kwargs)

            return _

        def create_cloud_sink(
            ext: str, unsupported: list[str]
        ) -> Callable[..., pl.LazyFrame | None]:
            prev_sink = getattr(pl.LazyFrame, f"sink_{ext}")
            prev_sink = cast("Callable[..., pl.LazyFrame | None]", prev_sink)

            def _(lf: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.LazyFrame | None:
                # The cloud client sinks to a "placeholder-path".
                if args[0] == "placeholder-path" or isinstance(args[0], pl.PartitionBy):
                    prev_lazy = kwargs.get("lazy", False)
                    kwargs["lazy"] = True
                    lf = prev_sink(lf, *args, **kwargs)  # type: ignore[assignment]

                    class SimpleLazyExe:
                        def __init__(self, query: pl.LazyFrame) -> None:
                            self._ldf = query._ldf
                            self.query = query

                        def collect(self, *args: Any, **kwargs: Any) -> pl.DataFrame:
                            return prev_collect(self.query, *args, **kwargs)  # type: ignore[no-any-return]

                    slf = SimpleLazyExe(lf)
                    if prev_lazy:
                        return slf  # type: ignore[return-value]

                    slf.collect(
                        optimizations=kwargs.get("optimizations", pl.QueryOptFlags()),
                    )
                    return None

                prev_tgt = None
                if isinstance(
                    args[0], (io.BytesIO, io.StringIO, io.TextIOBase)
                ) or callable(getattr(args[0], "write", None)):
                    prev_tgt = args[0]
                    args = (f"/tmp/pc-{uuid.uuid4()!s}.{ext}",) + args[1:]
                elif isinstance(args[0], PosixPath):
                    args = (str(args[0]),) + args[1:]

                lazy = kwargs.pop("lazy", False)

                # these are all the unsupported flags
                for u in unsupported:
                    _ = kwargs.pop(u, None)

                kwargs["sink_to_single_file"] = "True"

                sink = getattr(
                    lf.remote(plan_type="plain").distributed(), f"sink_{ext}"
                )
                q = sink(*args, **kwargs)
                assert isinstance(q, DirectQuery)
                query = LazyExe(
                    q,
                    prev_tgt,
                    args[0],
                )

                if lazy:
                    return query  # type: ignore[return-value]

                # If the sink is not lazy, we are expected to collect it.
                query.collect()
                return None

            return _

        # fix: these need to become supported somehow
        BASE_UNSUPPORTED = ["engine", "optimizations", "mkdir", "retries"]
        for ext in ["parquet", "csv", "ipc", "ndjson"]:
            plmonkeypatch.setattr(f"polars.scan_{ext}", create_cloud_scan(ext))
            plmonkeypatch.setattr(f"polars.read_{ext}", create_read(ext))
            plmonkeypatch.setattr(
                f"polars.LazyFrame.sink_{ext}",
                create_cloud_sink(ext, BASE_UNSUPPORTED),
            )

        plmonkeypatch.setattr("polars.LazyFrame.collect", cloud_collect)
        plmonkeypatch.setenv("POLARS_SKIP_CLIENT_CHECK", "1")


class PlMonkeyPatch(pytest.MonkeyPatch):  # type: ignore[misc]
    """A wrapper of pytest.MonkeyPatch that updates Polars when an env var changes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def setenv(self, name: str, value: str, prepend: str | None = None) -> None:
        super().setenv(name, value, prepend)
        if name.startswith("POLARS_"):
            pl.Config.reload_env_vars()

    def delenv(self, name: str, raising: bool = True) -> None:
        super().delenv(name, raising)
        if name.startswith("POLARS_"):
            pl.Config.reload_env_vars()

    def undo(self) -> None:
        super().undo()
        pl.Config.reload_env_vars()


@pytest.fixture
def plmonkeypatch() -> Generator[PlMonkeyPatch, None, None]:
    """A wrapper of pytest.plmonkeypatch that updates Polars when an env var changes."""
    mpatch = PlMonkeyPatch()
    yield mpatch
    mpatch.undo()
