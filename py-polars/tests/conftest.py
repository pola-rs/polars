from __future__ import annotations

import io
from typing import Any, Callable, TypeVar, cast

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

        from polars_cloud import ComputeContext, ComputeContextStatus, InteractiveQuery

        TIMEOUT_SECS = 20

        T = TypeVar("T")

        def with_timeout(f: Callable[[], T]) -> T:
            def handler(signum: Any, frame: Any) -> None:
                msg = "test timed out"
                raise TimeoutError(msg)

            signal.signal(signal.SIGALRM, handler)
            signal.alarm(TIMEOUT_SECS)

            return f()

        class PatchedComputeContext(ComputeContext):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
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

        prev_collect = pl.LazyFrame.collect

        def cloud_collect(lf: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.DataFrame:
            # issue: cloud client should use pl.QueryOptFlags()
            if "optimizations" in kwargs:
                kwargs.pop("optimizations")
            if "engine" in kwargs:
                kwargs.pop("engine")

            return prev_collect(
                with_timeout(
                    lambda: lf.remote(plan_type="plain")
                    .distributed()
                    .execute()
                    .await_result()
                ).lazy()
            )

        class LazyExe:
            def __init__(
                self, query: InteractiveQuery, prev_tgt: io.BytesIO | None, path: Path
            ) -> None:
                self.query = query

                self.prev_tgt = prev_tgt
                self.path = path

            def collect(self) -> pl.DataFrame:
                lf = with_timeout(lambda: self.query.await_result().lazy())
                res = prev_collect(lf)
                if self.prev_tgt is not None:
                    with Path.open(self.path, "rb") as f:
                        self.prev_tgt.write(f.read())

                    # delete the temporary file
                    Path(self.path).unlink()
                return res

        def io_to_path(s: io.BytesIO, ext: str) -> Path:
            path = Path(f"/tmp/pc-{uuid.uuid4()!s}.{ext}")

            offset = s.seek(0, 1)
            with Path.open(path, "wb") as f:
                f.write(s.read())
            s.seek(offset)
            return path

        def prepare_scan_sources(src: Any) -> str | Path | list[str | Path]:
            if isinstance(src, io.BytesIO):
                src = io_to_path(src, ext)
            elif isinstance(src, bytes):
                src = io_to_path(io.BytesIO(src), ext)
            elif isinstance(src, list):
                for i in range(len(src)):
                    if isinstance(src[i], io.BytesIO):
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
                src: io.BytesIO | str | Path, *args: Any, **kwargs: Any
            ) -> pl.LazyFrame:
                src = prepare_scan_sources(src)  # type: ignore[assignment]
                return prev_scan(src, *args, **kwargs)  # type: ignore[no-any-return]

            return _

        def create_read(ext: str) -> Callable[..., pl.DataFrame]:
            prev_read = getattr(pl, f"read_{ext}")
            prev_read = cast("Callable[..., pl.DataFrame]", prev_read)

            def _(
                src: io.BytesIO | str | Path, *args: Any, **kwargs: Any
            ) -> pl.DataFrame:
                src = prepare_scan_sources(src)  # type: ignore[assignment]
                return prev_read(src, *args, **kwargs)  # type: ignore[no-any-return]

            return _

        def create_cloud_sink(
            ext: str, unsupported: list[str]
        ) -> Callable[..., pl.LazyFrame | None]:
            prev_sink = getattr(pl.LazyFrame, f"sink_{ext}")
            prev_sink = cast("Callable[..., pl.LazyFrame | None]", prev_sink)

            def _(lf: pl.LazyFrame, *args: Any, **kwargs: Any) -> pl.LazyFrame | None:
                # The cloud client sinks to a "placeholder-path".
                if args[0] == "placeholder-path" or isinstance(
                    args[0], PartitioningScheme
                ):
                    return prev_sink(lf, *args, **kwargs)  # type: ignore[no-any-return]

                prev_tgt = None
                if isinstance(args[0], io.BytesIO):
                    prev_tgt = args[0]
                    args = (f"/tmp/pc-{uuid.uuid4()!s}.{ext}",) + args[1:]

                lazy = kwargs.pop("lazy", False)

                # these are all the unsupported flags
                for u in unsupported:
                    _ = kwargs.pop(u, None)

                sink = getattr(
                    lf.remote(plan_type="plain").distributed(), f"sink_{ext}"
                )
                q = sink(*args, **kwargs)
                assert isinstance(q, InteractiveQuery)
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
        for ext, unsupported in [
            ("parquet", ["metadata"]),
            ("csv", []),
            ("ipc", []),
            ("ndjson", []),
        ]:
            monkeypatch.setattr(f"polars.scan_{ext}", create_cloud_scan(ext))
            monkeypatch.setattr(f"polars.read_{ext}", create_read(ext))
            monkeypatch.setattr(
                f"polars.LazyFrame.sink_{ext}",
                create_cloud_sink(ext, BASE_UNSUPPORTED + unsupported),
            )

        monkeypatch.setattr("polars.LazyFrame.collect", cloud_collect)
        monkeypatch.setenv("POLARS_SKIP_CLIENT_CHECK", "1")
