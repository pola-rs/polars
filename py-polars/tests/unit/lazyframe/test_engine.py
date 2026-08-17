"""Tests for engine selection and dispatch."""

from __future__ import annotations

import asyncio
import inspect
import os
import pickle
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars import _plr as plr
from polars.exceptions import UnstableWarning
from polars.lazyframe import engine_config
from polars.lazyframe.engine import _LocalEngine
from polars.lazyframe.engine_config import _eager_engine, _select_engine
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path

    from polars._typing import EngineType


@pytest.fixture
def lf() -> pl.LazyFrame:
    return pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


def test_explain_streaming_flag_reaches_optimizer() -> None:
    # `explain` uses the engine only to set `optflags.streaming`; a sort+head is one
    # of the few plans the streaming flag actually rewrites. This is the regression
    # net for the `engine == "streaming"` comparison in `LazyFrame.explain`.
    lf = pl.LazyFrame({"a": [3, 1, 2]}).sort("a").head(2)
    assert lf.explain(engine="streaming") != lf.explain(engine="in-memory")


def test_collect_async_streaming_warns_unstable(lf: pl.LazyFrame) -> None:
    async def run() -> None:
        with pytest.warns(
            UnstableWarning, match="streaming mode is considered unstable"
        ):
            await lf.collect_async(engine="streaming")

    pl.Config().warn_unstable(True)
    try:
        asyncio.run(run())
    finally:
        pl.Config().warn_unstable(False)


def test_sink_forwards_optimizations(tmp_path: Path, lf: pl.LazyFrame) -> None:
    # Non-lazy sinks used to apply the caller's optimizations to the sink plan and
    # then call `collect(engine=...)` without passing them on; `forward_old_opt_flags`
    # substituted the defaults and `collect` re-applied them, replacing the flags
    # wholesale. `Engine._finish_sink` now forwards them.
    seen: list[pl.QueryOptFlags] = []

    class RecordingEngine(pl.InMemoryEngine):
        def collect(  # type: ignore[override]
            self, lf: pl.LazyFrame, *, optimizations: pl.QueryOptFlags, **kwargs: Any
        ) -> Any:
            seen.append(optimizations)
            return super().collect(lf, optimizations=optimizations, **kwargs)

    optimizations = pl.QueryOptFlags(predicate_pushdown=False)
    lf.sink_parquet(
        tmp_path / "out.parquet", engine=RecordingEngine(), optimizations=optimizations
    )

    assert len(seen) == 1
    assert seen[0] is optimizations
    assert not seen[0].predicate_pushdown


def test_collect_all_async_honors_engine_affinity(
    monkeypatch: pytest.MonkeyPatch, lf: pl.LazyFrame
) -> None:
    seen: list[Any] = []
    original = plr.collect_all_with_callback

    def spy(lfs: Any, engine: Any, optflags: Any, callback: Any) -> Any:
        seen.append(engine)
        return original(lfs, engine, optflags, callback)

    monkeypatch.setattr(plr, "collect_all_with_callback", spy)
    monkeypatch.setenv("POLARS_ENGINE_AFFINITY", "streaming")
    plr.config_reload_env_var("POLARS_ENGINE_AFFINITY")

    async def run() -> None:
        await pl.collect_all_async([lf])

    asyncio.run(run())

    assert seen == ["streaming"]


def test_collect_batches_accepts_gpu_engine_object(lf: pl.LazyFrame) -> None:
    expected = lf.collect()
    assert_frame_equal(pl.concat(lf.collect_batches(engine=pl.GPUEngine())), expected)
    assert_frame_equal(pl.concat(lf.collect_batches(engine="gpu")), expected)


@pytest.mark.parametrize(
    ("spelling", "expected"),
    [
        ("auto", "auto"),
        ("in-memory", "in-memory"),
        # legacy alias, still accepted by Rust
        ("cpu", "in-memory"),
        ("streaming", "streaming"),
        ("gpu", "gpu"),
        (pl.InMemoryEngine(), "in-memory"),
        (pl.StreamingEngine(), "streaming"),
        (pl.GPUEngine(), "gpu"),
    ],
)
def test_select_engine(spelling: EngineType, expected: str) -> None:
    selected = _select_engine(spelling)
    assert isinstance(selected, pl.Engine)
    assert selected.name == expected


def test_select_engine_is_idempotent() -> None:
    for spelling in ("auto", "in-memory", "streaming"):
        once = _select_engine(spelling)  # type: ignore[arg-type]
        assert _select_engine(once) is once


def test_select_engine_invalid_raises() -> None:
    with pytest.raises(ValueError, match="Invalid engine argument"):
        _select_engine("bogus")  # type: ignore[arg-type]


def test_select_engine_honors_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_ENGINE_AFFINITY", "streaming")
    plr.config_reload_env_var("POLARS_ENGINE_AFFINITY")

    assert _select_engine("auto").name == "streaming"
    # an explicit engine still wins over the affinity
    assert _select_engine("in-memory").name == "in-memory"


def test_eager_engine_ignores_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_ENGINE_AFFINITY", "streaming")
    plr.config_reload_env_var("POLARS_ENGINE_AFFINITY")

    # internal eager operations always run in-memory
    assert _eager_engine().name == "in-memory"
    with pl.Config(engine_affinity=pl.StreamingEngine()):
        assert _eager_engine().name == "in-memory"


def test_engine_is_abstract() -> None:
    with pytest.raises(TypeError):
        pl.Engine()  # type: ignore[abstract]


def test_engine_has_no_context_manager_protocol() -> None:
    # cudf-polars ships GPUEngine subclasses (RayEngine, DaskEngine, SPMDEngine)
    # that are used as context managers; `Engine` must not shadow them.
    assert not hasattr(pl.Engine, "__enter__")
    assert not hasattr(pl.Engine, "__exit__")


def test_gpu_engine_is_an_engine() -> None:
    assert isinstance(pl.GPUEngine(), pl.Engine)
    assert engine_config.GPUEngine is pl.GPUEngine


def test_gpu_engine_construction_unchanged() -> None:
    # `crates/polars-python/src/cloud_server.rs` constructs this from Rust.
    engine = pl.GPUEngine(raise_on_fail=True)
    assert engine.config == {"raise_on_fail": True}
    assert engine.name == "gpu"
    assert engine.monitoring is False

    engine = pl.GPUEngine(monitoring=False)
    assert engine.config == {"raise_on_fail": False}
    assert engine.monitoring is False


def test_gpu_engine_stays_hashable_and_picklable() -> None:
    # Defining `__eq__` on `Engine` without `__hash__` would break both.
    engine = pl.GPUEngine(device=1)
    assert hash(engine) is not None
    assert pickle.loads(pickle.dumps(engine)).config == engine.config

    assert (
        pickle.loads(pickle.dumps(pl.GPUEngine(monitoring=False))).monitoring is False
    )


def test_named_engines_are_unmonitored_singletons() -> None:
    # Named engines resolve to shared instances, so nothing may mutate `monitoring`.
    for name in ("streaming", "in-memory"):
        selected = _select_engine(name)  # type: ignore[arg-type]
        assert isinstance(selected, _LocalEngine)
        assert selected.monitoring is None
        assert _select_engine(name) is selected  # type: ignore[arg-type]


class _CountingEngine(pl.Engine):
    def __init__(self) -> None:
        self.collected = 0

    @property
    def name(self) -> str:
        return "counting"

    def collect(self, lf: pl.LazyFrame, **kwargs: Any) -> Any:
        self.collected += 1
        return pl.InMemoryEngine().collect(lf, **kwargs)

    def execute(self, lf: pl.LazyFrame, **kwargs: Any) -> Any:
        return pl.InMemoryEngine().execute(lf, **kwargs)


class _LocalCountingEngine(_LocalEngine):
    def __init__(self) -> None:
        self.collected = 0

    @property
    def name(self) -> str:
        return "in-memory"

    def collect(self, lf: pl.LazyFrame, **kwargs: Any) -> Any:
        self.collected += 1
        return super().collect(lf, **kwargs)


def test_engine_core_contract_is_three_members() -> None:
    # `name`/`collect`/`execute` are what every backend must answer for. Everything
    # else -- including the sinks -- is a capability a backend may not have, so it
    # raises rather than blocking construction.
    assert {"name", "collect", "execute"} == set(pl.Engine.__abstractmethods__)

    class Incomplete(pl.Engine):
        @property
        def name(self) -> str:
            return "incomplete"

    with pytest.raises(TypeError, match=r"abstract methods.*collect"):
        Incomplete()  # type: ignore[abstract]


def test_minimal_engine_supports_collect_and_execute(lf: pl.LazyFrame) -> None:
    engine = _CountingEngine()
    expected = lf.collect()

    assert_frame_equal(lf.collect(engine=engine), expected)
    assert_frame_equal(lf.execute(engine=engine).lazy().collect(), expected)
    assert engine.collected == 1


def test_in_process_engine_inherits_working_sinks(
    tmp_path: Path, lf: pl.LazyFrame
) -> None:
    engine = _LocalCountingEngine()
    path = tmp_path / "out.parquet"
    lf.sink_parquet(path, engine=engine)

    assert_frame_equal(pl.read_parquet(path), lf.collect())
    assert engine.collected == 1


@pytest.mark.parametrize(
    ("operation", "call"),
    [
        ("collect_batches", lambda lf, e: lf.collect_batches(engine=e)),
        ("collect_async", lambda lf, e: lf.collect_async(engine=e)),
        ("collect_all", lambda lf, e: pl.collect_all([lf], engine=e)),
        ("sink_parquet", lambda lf, e: lf.sink_parquet("out.parquet", engine=e)),
        ("sink_csv", lambda lf, e: lf.sink_csv("out.csv", engine=e)),
        ("sink_batches", lambda lf, e: lf.sink_batches(print, engine=e)),
    ],
)
def test_capability_gated_operations_raise(
    lf: pl.LazyFrame, operation: str, call: Callable[[pl.LazyFrame, pl.Engine], Any]
) -> None:
    engine = _CountingEngine()
    with pytest.raises(NotImplementedError, match=rf"`{operation}`.*_CountingEngine"):
        call(lf, engine)


@pytest.mark.parametrize(
    ("name", "engine"),
    [("in-memory", pl.InMemoryEngine()), ("streaming", pl.StreamingEngine())],
)
def test_string_and_object_spellings_agree(
    tmp_path: Path, lf: pl.LazyFrame, name: EngineType, engine: pl.Engine
) -> None:
    assert_frame_equal(lf.collect(engine=name), lf.collect(engine=engine))
    assert lf.explain(engine=name) == lf.explain(engine=engine)
    assert lf.show_graph(
        engine=name, plan_stage="physical", raw_output=True
    ) == lf.show_graph(engine=engine, plan_stage="physical", raw_output=True)
    assert_frame_equal(
        pl.collect_all([lf], engine=name)[0], pl.collect_all([lf], engine=engine)[0]
    )
    assert_frame_equal(
        pl.concat(lf.collect_batches(engine=name)),
        pl.concat(lf.collect_batches(engine=engine)),
    )

    by_name, by_object = tmp_path / "name.parquet", tmp_path / "object.parquet"
    lf.sink_parquet(by_name, engine=name)
    lf.sink_parquet(by_object, engine=engine)
    assert_frame_equal(pl.read_parquet(by_name), pl.read_parquet(by_object))


SINKS = ["sink_parquet", "sink_csv", "sink_ipc", "sink_ndjson", "sink_batches"]


@pytest.mark.parametrize("method_name", SINKS)
def test_engine_sink_signature_matches_lazyframe(method_name: str) -> None:
    lf_params = list(
        inspect.signature(getattr(pl.LazyFrame, method_name)).parameters.values()
    )
    engine_params = list(
        inspect.signature(getattr(pl.Engine, method_name)).parameters.values()
    )

    assert "engine" in {p.name for p in lf_params}
    assert "engine" not in {p.name for p in engine_params}

    # LazyFrame: (self, <target>, *, ..., engine, ...)
    # Engine:    (self, lf, <target>, *, ...)
    assert engine_params[1].name == "lf"
    expected = [p for p in lf_params[1:] if p.name != "engine"]
    actual = engine_params[2:]

    def key(p: inspect.Parameter) -> tuple[str, Any]:
        return (p.name, p.kind)

    assert [key(p) for p in actual] == [key(p) for p in expected], (
        f"{method_name} drifted: LazyFrame has {[p.name for p in expected]}, "
        f"Engine has {[p.name for p in actual]}"
    )
    assert all(p.default is inspect.Parameter.empty for p in actual), (
        f"{method_name} declares defaults on `Engine`; they belong on `LazyFrame` "
        f"only: {[p.name for p in actual if p.default is not inspect.Parameter.empty]}"
    )


def test_engine_covers_every_lazyframe_sink() -> None:
    # `sink_delta` and `sink_iceberg` are adapters backed by external writers. They
    # use `engine` to produce rows but do not delegate to an `Engine.sink_*` method.
    lf_sinks = {
        name
        for name in dir(pl.LazyFrame)
        if name.startswith("sink_") and name not in {"sink_delta", "sink_iceberg"}
    }
    assert lf_sinks == set(SINKS)


@pytest.fixture
def _restore_affinity() -> Iterator[None]:
    yield
    pl.Config.restore_defaults()


@pytest.mark.usefixtures("_restore_affinity")
def test_object_engine_affinity_is_used_for_auto() -> None:
    engine = pl.GPUEngine(device=1, raise_on_fail=True)
    pl.Config.set_engine_affinity(engine)

    assert _select_engine("auto") is engine
    # an explicit engine still wins
    assert _select_engine("in-memory").name == "in-memory"


@pytest.mark.usefixtures("_restore_affinity")
def test_object_engine_affinity_clears_the_env_var() -> None:
    # the two spellings are mutually exclusive: an object cannot be represented by
    # POLARS_ENGINE_AFFINITY, and a name must not be shadowed by a stale object.
    pl.Config.set_engine_affinity("streaming")
    pl.Config.set_engine_affinity(pl.GPUEngine())
    assert os.environ.get("POLARS_ENGINE_AFFINITY") is None

    pl.Config.set_engine_affinity("streaming")
    assert os.environ["POLARS_ENGINE_AFFINITY"] == "streaming"
    assert _select_engine("auto").name == "streaming"


@pytest.mark.usefixtures("_restore_affinity")
def test_object_engine_affinity_is_cleared_by_restore_defaults() -> None:
    pl.Config.set_engine_affinity(pl.StreamingEngine())
    pl.Config.restore_defaults()
    assert _select_engine("auto").name == "auto"


@pytest.mark.usefixtures("_restore_affinity")
def test_object_engine_affinity_is_scoped_by_context_manager() -> None:
    engine = pl.GPUEngine(device=1)
    pl.Config.set_engine_affinity(engine)

    with pl.Config(engine_affinity=pl.StreamingEngine()):
        assert _select_engine("auto").name == "streaming"
    assert _select_engine("auto") is engine

    with pl.Config(engine_affinity="in-memory"):
        assert _select_engine("auto").name == "in-memory"
    assert _select_engine("auto") is engine


@pytest.mark.usefixtures("_restore_affinity")
def test_object_engine_affinity_is_dropped_by_save_load() -> None:
    # `save` records environment variables only, so a round-trip cannot preserve an
    # object affinity -- it clears it rather than silently keeping a stale one.
    pl.Config.set_engine_affinity(pl.StreamingEngine())
    pl.Config.load(pl.Config.save())
    assert _select_engine("auto").name == "auto"


@pytest.mark.usefixtures("_restore_affinity")
def test_object_engine_affinity_drives_collect(lf: pl.LazyFrame) -> None:
    calls = []

    class RecordingEngine(pl.InMemoryEngine):
        def collect(self, lf: pl.LazyFrame, **kwargs: Any) -> Any:
            calls.append("collect")
            return super().collect(lf, **kwargs)

    pl.Config.set_engine_affinity(RecordingEngine())
    assert_frame_equal(
        lf.collect(),
        pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]}).collect(engine="in-memory"),
    )
    assert calls == ["collect"]


def test_runtime_only_options_are_scoped_by_config() -> None:
    import polars.io.cloud.credential_provider._builder as builder

    assert builder.DEFAULT_CREDENTIAL_PROVIDER == "auto"
    with pl.Config(default_credential_provider=None):
        assert builder.DEFAULT_CREDENTIAL_PROVIDER is None
    assert builder.DEFAULT_CREDENTIAL_PROVIDER == "auto"

    gpu = pl.GPUEngine(device=1)
    pl.Config.set_engine_affinity(gpu)
    try:
        with pl.Config(engine_affinity=pl.StreamingEngine()):
            assert _select_engine("auto").name == "streaming"
        assert _select_engine("auto") is gpu
    finally:
        pl.Config.restore_defaults()


def test_restore_defaults_resets_runtime_only_options() -> None:
    import polars.io.cloud.credential_provider._builder as builder

    pl.Config.set_engine_affinity(pl.StreamingEngine())
    pl.Config.set_default_credential_provider(None)
    pl.Config.restore_defaults()

    assert _select_engine("auto").name == "auto"
    assert builder.DEFAULT_CREDENTIAL_PROVIDER == "auto"
