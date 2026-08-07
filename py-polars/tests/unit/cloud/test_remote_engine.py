"""Tests for `RemoteEngine` dispatch.

These use a stub `polars_cloud` module so that they run without the real package
(and without a cluster) being available.
"""

from __future__ import annotations

import inspect
import io
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import polars as pl
from polars.lazyframe.engine_config import _REMOTE_SINK_PARAMS


class FakeQuery:
    """Stands in for `polars_cloud`'s `DirectQuery` / `ProxyQuery`."""

    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self._calls = calls

    def await_result(self, **kwargs: Any) -> str:
        self._calls.append(("await_result", kwargs))
        return "result"


class FakeExecuteRemote:
    """Stands in for `polars_cloud`'s `ExecuteRemote`."""

    def __init__(self, calls: list[tuple[Any, ...]], tag: str) -> None:
        self._calls = calls
        self.tag = tag

    def execute(self, **kwargs: Any) -> str:
        self._calls.append((self.tag, "execute", kwargs))
        return "query-result"

    def _sink(self, name: str, uri: Any, kwargs: dict[str, Any]) -> FakeQuery:
        self._calls.append((self.tag, name, uri, kwargs))
        return FakeQuery(self._calls)

    def sink_parquet(self, uri: Any, **kwargs: Any) -> FakeQuery:
        return self._sink("sink_parquet", uri, kwargs)

    def sink_csv(self, uri: Any, **kwargs: Any) -> FakeQuery:
        return self._sink("sink_csv", uri, kwargs)

    def sink_ipc(self, uri: Any, **kwargs: Any) -> FakeQuery:
        return self._sink("sink_ipc", uri, kwargs)


class FakeLazyFrameRemote(FakeExecuteRemote):
    """Stands in for `polars_cloud`'s `LazyFrameRemote`."""

    def distributed(self, **kwargs: Any) -> FakeExecuteRemote:
        self._calls.append(("distributed", kwargs))
        return FakeExecuteRemote(self._calls, "distributed")

    def single_node(self) -> FakeExecuteRemote:
        self._calls.append(("single_node",))
        return FakeExecuteRemote(self._calls, "single-node")

    def labels(self, labels: list[str]) -> FakeLazyFrameRemote:
        self._calls.append(("labels", labels))
        return self

    def sink_csv(self, uri: Any, **kwargs: Any) -> FakeQuery:
        if "maintain_order" in kwargs:
            msg = "LazyFrameRemote.sink_csv does not accept maintain_order"
            raise TypeError(msg)
        return super().sink_csv(uri, **kwargs)

    def sink_ipc(self, uri: Any, **kwargs: Any) -> FakeQuery:
        if "maintain_order" in kwargs:
            msg = "LazyFrameRemote.sink_ipc does not accept maintain_order"
            raise TypeError(msg)
        return super().sink_ipc(uri, **kwargs)


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Install a stub `polars_cloud` and record what the engine dispatches to it."""
    recorded: list[tuple[Any, ...]] = []

    def fake_remote(self: pl.LazyFrame, context: Any = None, **kwargs: Any) -> Any:
        recorded.append(("remote", context, kwargs))
        return FakeLazyFrameRemote(recorded, "auto")

    monkeypatch.setitem(sys.modules, "polars_cloud", ModuleType("polars_cloud"))
    monkeypatch.setattr(pl.LazyFrame, "remote", fake_remote, raising=False)
    return recorded


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.LazyFrame({"a": [1, 2, 3]})


def test_remote_engine_requires_polars_cloud(
    lf: pl.LazyFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    # a `None` entry in `sys.modules` makes the import fail as if not installed
    monkeypatch.setitem(sys.modules, "polars_cloud", None)  # type: ignore[arg-type]
    with pytest.raises(ImportError, match="remote engine requested"):
        lf.execute(engine=pl.RemoteEngine())


def test_execute_dispatches(calls: list[tuple[Any, ...]], lf: pl.LazyFrame) -> None:
    assert lf.execute(engine=pl.RemoteEngine()) == "query-result"  # type: ignore[comparison-overlap]
    assert calls[0][0] == "remote"
    assert calls[0][2] == {
        "plan_type": "dot",
        "n_retries": 0,
        "engine": "auto",
        "scaling_mode": "auto",
    }
    assert calls[1][:2] == ("auto", "execute")


def test_scaling_mode_auto_defers_to_polars_cloud(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame
) -> None:
    lf.execute(engine=pl.RemoteEngine())
    assert not any(c[0] in ("distributed", "single_node") for c in calls)


def test_scaling_mode_distributed(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame
) -> None:
    lf.execute(engine=pl.RemoteEngine(scaling_mode="distributed", max_workers=4))
    assert ("distributed", {"max_workers": 4}) in calls
    assert calls[-1][:2] == ("distributed", "execute")


def test_distributed_options_imply_distributed(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame
) -> None:
    lf.execute(engine=pl.RemoteEngine(max_workers=4))
    assert ("distributed", {"max_workers": 4}) in calls


def test_scaling_mode_single_node(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame
) -> None:
    lf.execute(engine=pl.RemoteEngine(scaling_mode="single-node"))
    assert calls[0][2]["scaling_mode"] == "single-node"
    assert not any(c[0] in ("distributed", "single_node") for c in calls)
    assert calls[-1][:2] == ("auto", "execute")


def test_single_node_rejects_distributed_options() -> None:
    with pytest.raises(ValueError, match="not supported with"):
        pl.RemoteEngine(scaling_mode="single-node", max_workers=4)


def test_invalid_scaling_mode() -> None:
    with pytest.raises(ValueError, match="invalid `scaling_mode`"):
        pl.RemoteEngine(scaling_mode="everywhere")  # type: ignore[arg-type]


def test_labels(calls: list[tuple[Any, ...]], lf: pl.LazyFrame) -> None:
    lf.execute(engine=pl.RemoteEngine(labels="etl"))
    assert ("labels", ["etl"]) in calls


@pytest.mark.parametrize(
    ("method", "uri"),
    [
        ("sink_parquet", "s3://bucket/out/"),
        ("sink_csv", "s3://bucket/out.csv"),
        ("sink_ipc", "s3://bucket/out.ipc"),
    ],
)
def test_sink_dispatches_and_blocks(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame, method: str, uri: str
) -> None:
    assert getattr(lf, method)(uri, engine=pl.RemoteEngine()) is None
    sink_call = next(c for c in calls if c[1] == method)
    assert sink_call[2] == uri
    assert calls[-1][0] == "await_result"


def test_sink_forwards_options(calls: list[tuple[Any, ...]], lf: pl.LazyFrame) -> None:
    lf.sink_parquet(
        "s3://bucket/out/",
        compression="lz4",
        row_group_size=1024,
        maintain_order=False,
        engine=pl.RemoteEngine(),
    )
    kwargs = next(c for c in calls if c[1] == "sink_parquet")[3]
    assert kwargs["compression"] == "lz4"
    assert kwargs["row_group_size"] == 1024
    assert kwargs["maintain_order"] is False


@pytest.mark.parametrize(
    ("method", "kwargs"),
    [
        ("sink_parquet", {"lazy": True}),
        ("sink_parquet", {"mkdir": True}),
        ("sink_parquet", {"sync_on_close": "data"}),
        ("sink_csv", {"compression": "gzip"}),
        ("sink_csv", {"check_extension": False}),
        ("sink_csv", {"maintain_order": False}),
        ("sink_ipc", {"record_batch_size": 100}),
        ("sink_ipc", {"maintain_order": False}),
    ],
)
def test_sink_rejects_unsupported_options(
    calls: list[tuple[Any, ...]],
    lf: pl.LazyFrame,
    method: str,
    kwargs: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="not supported by the remote engine"):
        getattr(lf, method)("s3://bucket/out/", engine=pl.RemoteEngine(), **kwargs)


@pytest.mark.parametrize("method", list(_REMOTE_SINK_PARAMS))
def test_sink_params_match_signature(method: str) -> None:
    """Every argument of a remote-capable sink is either forwarded or rejected."""
    spec = _REMOTE_SINK_PARAMS[method]
    signature = inspect.signature(getattr(pl.LazyFrame, method)).parameters
    arguments = set(signature) - {"self", "path", "engine"}

    assert arguments == set(spec.forward) | set(spec.unsupported)
    assert {name: signature[name].default for name in spec.unsupported} == dict(
        spec.unsupported
    )


@pytest.mark.parametrize("path", [io.BytesIO(), Path("local/path")])
def test_sink_rejects_non_uri_target(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame, path: Any
) -> None:
    with pytest.raises(TypeError, match="can only sink to a URI"):
        lf.sink_parquet(path, engine=pl.RemoteEngine())


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    "call",
    [
        lambda lf, e: lf.collect(engine=e),
        lambda lf, e: lf.collect_async(engine=e),
        lambda lf, e: lf.profile(engine=e),
        lambda lf, e: lf.sink_ndjson("s3://b/out.ndjson", engine=e),
        lambda lf, e: lf.sink_batches(print, engine=e),
        lambda lf, e: next(lf.collect_batches(engine=e)),
        lambda lf, e: pl.collect_all([lf], engine=e),
    ],
)
def test_unsupported_entry_points_raise(
    lf: pl.LazyFrame, call: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "polars_cloud", ModuleType("polars_cloud"))
    with pytest.raises(NotImplementedError, match="not supported by the remote engine"):
        call(lf, pl.RemoteEngine())


def test_plan_methods_still_work(lf: pl.LazyFrame) -> None:
    engine = pl.RemoteEngine(engine="streaming")
    assert lf.explain(engine=engine)
    assert lf.show_graph(engine=engine, plan_stage="ir", raw_output=True)


def test_eager_operations_stay_local(lf: pl.LazyFrame, tmp_path: Path) -> None:
    """`collect` is the primitive eager `DataFrame` ops are built on."""
    parquet_path = tmp_path / "data.parquet"
    ipc_path = tmp_path / "data.ipc"
    pl.DataFrame({"a": [1]}).write_parquet(parquet_path)
    pl.DataFrame({"a": [1]}).write_ipc(ipc_path)

    with pl.Config(engine_affinity=pl.RemoteEngine()):
        assert pl.DataFrame({"a": [1, 2]}).filter(pl.col("a") > 1).height == 1
        assert pl.read_csv(io.BytesIO(b"a,b\n1,2\n")).height == 1
        assert pl.read_parquet(parquet_path).height == 1
        assert pl.read_ipc(str(tmp_path / "*.ipc")).height == 1
        assert pl.read_lines(b"one\ntwo\n").height == 2
        assert (
            pl.concat([pl.DataFrame({"a": [1]}), pl.DataFrame({"a": [2]})]).height == 2
        )
        # explicitly asking for a local engine also still works
        assert lf.collect(engine="in-memory").height == 3


def test_affinity_dispatches(calls: list[tuple[Any, ...]], lf: pl.LazyFrame) -> None:
    with pl.Config(engine_affinity=pl.RemoteEngine(scaling_mode="distributed")):
        lf.execute()
    assert calls[0][2]["scaling_mode"] == "distributed"
    assert calls[-1][:2] == ("auto", "execute")
