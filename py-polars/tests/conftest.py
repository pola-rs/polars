from __future__ import annotations

import io

import pytest

import polars as pl
from polars._typing import PartitioningScheme


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--cloud-distributed",
        action="store_true",
        default=False,
        help="Run all queries by default of the distributed engine",
    )


@pytest.fixture(autouse=True)
def _patched_cloud(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    if request.config.getoption("--cloud-distributed"):
        import signal
        import uuid
        from pathlib import Path

        from polars_cloud import ComputeContext, ComputeContextStatus

        TIMEOUT_SECS = 4

        def with_timeout(f):
            def handler(signum, frame):
                msg = "test timed out"
                raise TimeoutError(msg)

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(TIMEOUT_SECS)

            return f()

        class PatchedComputeContext(ComputeContext):
            def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
                self._interactive = True
                self._compute_address = "localhost:5051"
                self._compute_public_key = b""
                self._insecure = True
                self._compute_id = uuid.uuid4()

            def get_status(self: ComputeContext) -> ComputeContextStatus:
                """Get the status of the compute cluster."""
                return ComputeContextStatus.RUNNING

        monkeypatch.setattr(
            "polars_cloud.context.compute.ComputeContext.__init__",
            PatchedComputeContext.__init__,
        )
        monkeypatch.setattr(
            "polars_cloud.context.compute.ComputeContext.get_status",
            PatchedComputeContext.get_status,
        )

        prev = {
            name: getattr(pl.LazyFrame, name)
            for name in [
                "collect",
                "sink_parquet",
            ]
        }

        prevpl = {name: getattr(pl, name) for name in ["scan_parquet"]}

        def cloud_collect(lf: pl.LazyFrame, *args, **kwargs) -> pl.DataFrame:
            # issue: cloud client should use pl.QueryOptFlags()
            if "optimizations" in kwargs:
                kwargs.pop("optimizations")
            if "engine" in kwargs:
                kwargs.pop("engine")
            df = prev["collect"](
                with_timeout(lambda: lf.remote().distributed().collect(*args, **kwargs))
            )
            return df

        class LazyExe:
            def __init__(self, query, prev_tgt, path) -> None:
                self.query = query

                self.prev_tgt = prev_tgt
                self.path = path

            def collect(self) -> pl.DataFrame:
                res = with_timeout(lambda: self.query.await_result())
                if self.prev_tgt is not None:
                    with Path.open(self.path, "rb") as f:
                        self.prev_tgt.write(f.read())
                    Path(self.path).unlink()
                return res

        def io_to_path(s: io.BytesIO, ext: str) -> Path:
            path = Path(f"/tmp/pc-{uuid.uuid4()!s}.{ext}")

            offset = s.seek()
            with Path.open(path, "wb") as f:
                f.write(s.read())
            s.seek(offset)
            return path

        def cloud_scan_parquet(src, *args, **kwargs) -> pl.LazyFrame | None:
            if isinstance(src, io.BytesIO):
                src = io_to_path(src)
            elif isinstance(src, list):
                for i in range(len(src)):
                    if isinstance(src, io.BytesIO):
                        src[i] = io_to_path(src[i])

            return prevpl["scan_parquet"](src, *args, **kwargs)

        def cloud_sink_parquet(
            lf: pl.LazyFrame, *args, **kwargs
        ) -> pl.LazyFrame | None:
            if args[0] == "placeholder-path" or isinstance(args[0], PartitioningScheme):
                return prev["sink_parquet"](lf, *args, **kwargs)

            prev_tgt = None
            if isinstance(args[0], io.BytesIO):
                prev_tgt = args[0]
                args = (f"/tmp/pc-{uuid.uuid4()!s}.parquet",) + args[1:]

            lazy = kwargs.pop("lazy", False)
            _engine = kwargs.pop("engine", "auto")  # fix: unsupported
            _optimizations = kwargs.pop("optimizations", None)  # fix: unsupported
            _metadata = kwargs.pop("metadata", None)  # fix: unsupported
            _mkdir = kwargs.pop("mkdir", False)  # fix: unsupported
            _retries = kwargs.pop("retries", None)  # fix: unsupported
            query = LazyExe(
                lf.remote().distributed().sink_parquet(*args, **kwargs),
                prev_tgt,
                args[0],
            )

            if not lazy:
                return query.collect()

            return

        monkeypatch.setattr("polars.scan_parquet", cloud_scan_parquet)
        monkeypatch.setattr("polars.LazyFrame.collect", cloud_collect)
        monkeypatch.setattr("polars.LazyFrame.sink_parquet", cloud_sink_parquet)

        monkeypatch.setenv("POLARS_SKIP_CLIENT_CHECK", "1")
