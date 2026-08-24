"""
Tests for `RemoteEngine` dispatch.

These stub out `polars_cloud` so they run without the package (and without a
cluster).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal


class FakeQuery:
    """Stands in for `polars_cloud`'s `DirectQuery`."""

    def __init__(self, calls: list[tuple[Any, ...]]) -> None:
        self._calls = calls

    def await_result(self, **kwargs: Any) -> str:
        self._calls.append(("await_result", kwargs))
        return "result"


class FakeExecuteRemote:
    """Stands in for `polars_cloud`'s `ExecuteRemote`."""

    def __init__(
        self, calls: list[tuple[Any, ...]], tag: str, lf: pl.LazyFrame
    ) -> None:
        self._calls = calls
        self.tag = tag
        self._lf = lf

    def execute(self, **kwargs: Any) -> Any:
        self._calls.append((self.tag, "execute", kwargs))
        return FakeQueryResult(self._lf)

    def _sink(self, name: str, uri: Any, kwargs: dict[str, Any]) -> FakeQuery:
        self._calls.append((self.tag, name, uri, kwargs))
        return FakeQuery(self._calls)

    def sink_parquet(self, uri: Any, **kwargs: Any) -> FakeQuery:
        return self._sink("sink_parquet", uri, kwargs)

    def sink_csv(self, uri: Any, **kwargs: Any) -> FakeQuery:
        if "maintain_order" in kwargs:
            msg = "LazyFrameRemote.sink_csv does not accept maintain_order"
            raise TypeError(msg)
        return self._sink("sink_csv", uri, kwargs)

    def sink_ipc(self, uri: Any, **kwargs: Any) -> FakeQuery:
        if "maintain_order" in kwargs:
            msg = "LazyFrameRemote.sink_ipc does not accept maintain_order"
            raise TypeError(msg)
        return self._sink("sink_ipc", uri, kwargs)


class FakeQueryResult:
    """Stands in for the `QueryResult` that Polars Cloud hands back."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def lazy(self) -> pl.LazyFrame:
        return self._lf


class FakeLazyFrameRemote(FakeExecuteRemote):
    """Stands in for `polars_cloud`'s `LazyFrameRemote`."""

    def distributed(self, **kwargs: Any) -> FakeExecuteRemote:
        self._calls.append(("distributed", kwargs))
        return FakeExecuteRemote(self._calls, "distributed", self._lf)

    def labels(self, labels: list[str]) -> FakeLazyFrameRemote:
        self._calls.append(("labels", labels))
        return self


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, ...]]:
    """Install a stub `polars_cloud` and record what the engine dispatches to it."""
    recorded: list[tuple[Any, ...]] = []

    def fake_remote(self: pl.LazyFrame, context: Any = None, **kwargs: Any) -> Any:
        recorded.append(("remote", context, kwargs))
        return FakeLazyFrameRemote(recorded, "auto", self)

    monkeypatch.setitem(sys.modules, "polars_cloud", ModuleType("polars_cloud"))
    monkeypatch.setattr(pl.LazyFrame, "remote", fake_remote, raising=False)
    return recorded


@pytest.fixture
def _stub_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "polars_cloud", ModuleType("polars_cloud"))


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.LazyFrame({"a": [1, 2, 3]})


def test_requires_polars_cloud(monkeypatch: pytest.MonkeyPatch) -> None:
    # a `None` entry in `sys.modules` makes the import fail as if not installed
    monkeypatch.setitem(sys.modules, "polars_cloud", None)  # type: ignore[arg-type]
    with pytest.raises(ImportError, match="remote engine requested"):
        pl.RemoteEngine()


@pytest.mark.usefixtures("_stub_cloud")
def test_invalid_scaling_mode() -> None:
    with pytest.raises(ValueError, match=r"invalid `scaling_mode`"):
        pl.RemoteEngine(scaling_mode="everywhere")  # type: ignore[arg-type]


@pytest.mark.usefixtures("_stub_cloud")
def test_single_node_rejects_distributed_options() -> None:
    with pytest.raises(ValueError, match="not supported with"):
        pl.RemoteEngine(scaling_mode="single-node", max_workers=4)


@pytest.mark.usefixtures("_stub_cloud")
def test_is_an_engine_with_a_distinct_name() -> None:
    engine = pl.RemoteEngine()
    assert isinstance(engine, pl.Engine)
    assert engine.name == "remote"
    # plans render for the engine the workers are asked to prefer
    assert engine.plan_engine == "auto"
    assert pl.RemoteEngine(engine="streaming").plan_engine == "streaming"


@pytest.mark.usefixtures("_stub_cloud")
def test_worker_engine_does_not_follow_global_affinity() -> None:
    affinity = pl.RemoteEngine(engine="streaming")
    with pl.Config(engine_affinity=affinity):
        assert pl.RemoteEngine().plan_engine == "auto"


@pytest.mark.usefixtures("_stub_cloud")
def test_worker_engine_rejects_engine_object() -> None:
    # we can not serialize engine objects for use with `RemoteEngine`
    with pytest.raises(ValueError, match="Invalid engine argument"):
        pl.RemoteEngine(engine=pl.StreamingEngine())  # type: ignore[arg-type]


def test_execute_dispatches(calls: list[tuple[Any, ...]], lf: pl.LazyFrame) -> None:
    lf.execute(engine=pl.RemoteEngine())
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
    assert not any(c[0] == "distributed" for c in calls)


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


def test_labels(calls: list[tuple[Any, ...]], lf: pl.LazyFrame) -> None:
    lf.execute(engine=pl.RemoteEngine(labels="etl"))
    assert ("labels", ["etl"]) in calls


def test_collect_pulls_the_result_back_and_warns(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame
) -> None:
    with pytest.warns(UserWarning, match="transfers the entire result"):
        result = lf.collect(engine=pl.RemoteEngine())

    assert_frame_equal(result, pl.DataFrame({"a": [1, 2, 3]}))
    assert calls[1][:2] == ("auto", "execute")


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


@pytest.mark.parametrize("path", [io.BytesIO(), Path("local/path")])
def test_sink_rejects_non_uri_target(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame, path: Any
) -> None:
    with pytest.raises(TypeError, match="can only sink to a URI"):
        lf.sink_parquet(path, engine=pl.RemoteEngine())


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize(
    ("operation", "call"),
    [
        ("collect_async", lambda lf, e: lf.collect_async(engine=e)),
        ("collect_batches", lambda lf, e: lf.collect_batches(engine=e)),
        ("collect_all", lambda lf, e: pl.collect_all([lf], engine=e)),
        # a Python callback cannot run on a worker
        ("sink_batches", lambda lf, e: lf.sink_batches(print, engine=e)),
        # Polars Cloud exposes no NDJSON sink
        (
            "sink_ndjson",
            lambda lf, e: lf.sink_ndjson("s3://bucket/out.ndjson", engine=e),
        ),
    ],
)
@pytest.mark.usefixtures("_stub_cloud")
def test_unsupported_entry_points_raise(
    lf: pl.LazyFrame, operation: str, call: Any
) -> None:
    with pytest.raises(NotImplementedError, match=rf"`{operation}`.*RemoteEngine"):
        call(lf, pl.RemoteEngine())


@pytest.mark.usefixtures("_stub_cloud")
def test_plan_methods_never_reach_polars_cloud(lf: pl.LazyFrame) -> None:
    engine = pl.RemoteEngine(engine="streaming")
    assert lf.explain(engine=engine)
    assert lf.show_graph(engine=engine, plan_stage="ir", raw_output=True)
    # the plan matches the engine the workers would use, not the remote engine
    assert lf.explain(engine=engine) == lf.explain(engine="streaming")


@pytest.mark.usefixtures("_stub_cloud")
def test_eager_operations_stay_local(tmp_path: Path) -> None:
    # None of these may reach `polars_cloud`: `LazyFrame.remote` is not stubbed here,
    # so any dispatch would fail on the missing `pc.LazyFrameRemote`.
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.write_parquet(tmp_path / "data.parquet")
    df.write_ipc(tmp_path / "data.ipc")

    with pl.Config(engine_affinity=pl.RemoteEngine()):
        assert df.filter(pl.col("a") > 1).height == 2
        assert df.top_k(2, by="a").height == 2
        assert df.group_by("a").agg(pl.col("b").sum()).height == 3
        assert df.group_by("a").head(1).height == 3
        assert df[::2].height == 2
        assert pl.read_csv(io.BytesIO(b"a,b\n1,2\n")).height == 1
        assert pl.read_parquet(tmp_path / "data.parquet").height == 3
        assert pl.read_ipc(tmp_path / "data.ipc").height == 3
        assert pl.read_ndjson(b'{"a":1}\n').height == 1
        assert pl.read_lines(b"one\ntwo\n").height == 2
        assert pl.concat([df, df]).height == 6
        assert len(pl.align_frames(df, df, on="a")) == 2
        assert pl.sql("SELECT * FROM df", eager=True).height == 3
        # explicitly asking for a local engine also still works
        assert pl.LazyFrame({"a": [1]}).collect(engine="in-memory").height == 1


@pytest.mark.usefixtures("_stub_cloud")
def test_affinity_dispatches_lazy_queries(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame
) -> None:
    with pl.Config(engine_affinity=pl.RemoteEngine(scaling_mode="distributed")):
        lf.execute()
    assert calls[0][2]["scaling_mode"] == "distributed"
    assert calls[-1][:2] == ("auto", "execute")


def test_describe_honors_remote_affinity(
    calls: list[tuple[Any, ...]], lf: pl.LazyFrame
) -> None:
    with pl.Config(engine_affinity=pl.RemoteEngine()):
        result = lf.describe(percentiles=[])

    assert result["statistic"].to_list() == [
        "count",
        "null_count",
        "mean",
        "std",
        "min",
        "max",
    ]
    assert any(call[:2] == ("auto", "execute") for call in calls)
