from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import uuid
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

import polars as pl
import polars._plr as plr
from polars.lazyframe.engine import StreamingEngine
from tests.unit.conftest import mock_module_import

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def fake_cloud_observer() -> tuple[ModuleType, MagicMock]:
    module = ModuleType("polars_cloud")
    observer = MagicMock(name="QueryCloudObserver()")
    module.QueryCloudObserver = MagicMock(  # type: ignore[attr-defined]
        name="QueryCloudObserver", return_value=observer
    )
    module.authenticate = MagicMock(name="authenticate")  # type: ignore[attr-defined]
    return module, observer


@pytest.fixture(autouse=True)
def _reset_monitoring() -> Iterator[None]:
    with pl.Config(restore_defaults=True):
        yield
    plr.set_query_monitoring(False)


def _sample_lf() -> pl.LazyFrame:
    return (
        pl.LazyFrame({"a": [1, 2, 3, 4, 5], "b": ["x", "x", "y", "z", "z"]})
        .group_by("b")
        .agg(pl.col("a").sum())
    )


def test_config_enable_monitoring() -> None:
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        assert "POLARS_ENGINE_AFFINITY" not in os.environ

        pl.Config.enable_monitoring()
        module.authenticate.assert_called_once()
        assert os.environ["POLARS_QUERY_MONITORING"] == "1"
        assert os.environ["POLARS_ENGINE_AFFINITY"] == "streaming"

        # no engine argument: monitoring switches the affinity to streaming
        _sample_lf().collect()
        _sample_lf().collect()
        assert observer.on_query_started.call_count == 2

        # disabling stops monitoring; the affinity is left as it is
        pl.Config.enable_monitoring(False)
        assert "POLARS_QUERY_MONITORING" not in os.environ
        _sample_lf().collect()

    assert observer.on_query_started.call_count == 2


def test_collect_calls_observer() -> None:
    """Running a query with monitoring on invokes the observer callbacks."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        result = _sample_lf().collect(engine="streaming")

    assert result.shape == (3, 2)
    observer.on_query_started.assert_called_once()
    observer.on_query_planned.assert_called_once()
    observer.on_query_planned.return_value.close.assert_called_once()
    observer.on_query_failed.assert_not_called()

    started_id = observer.on_query_started.call_args.args[0]
    planned_id = observer.on_query_planned.call_args.args[0]
    assert isinstance(started_id, uuid.UUID)
    assert started_id == planned_id


def test_config_scope_monitoring() -> None:
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        assert "POLARS_ENGINE_AFFINITY" not in os.environ

        with pl.Config(enable_monitoring=True):
            assert os.environ["POLARS_QUERY_MONITORING"] == "1"
            assert os.environ["POLARS_ENGINE_AFFINITY"] == "streaming"
            _sample_lf().collect()
        observer.on_query_started.assert_called_once()

        # leaving the scope stops monitoring
        _sample_lf().collect()

        assert "POLARS_QUERY_MONITORING" not in os.environ
        assert "POLARS_ENGINE_AFFINITY" not in os.environ

    observer.on_query_started.assert_called_once()


def test_engine_object_follows_config() -> None:
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        _sample_lf().collect(engine=StreamingEngine())

    observer.on_query_started.assert_called_once()


def test_engine_object_without_config_is_not_monitored() -> None:
    """An engine with `monitoring=None` follows the Config; here it is off."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        _sample_lf().collect(engine=StreamingEngine())

    module.QueryCloudObserver.assert_not_called()
    observer.on_query_started.assert_not_called()


def test_engine_monitoring_overrides_config_off() -> None:
    """`monitoring=True` monitors the query without the Config being enabled."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        engine = StreamingEngine(monitoring=True)
        module.authenticate.assert_called_once()

        _sample_lf().collect(engine=engine)

    observer.on_query_started.assert_called_once()
    # the engine must not touch global state
    assert "POLARS_QUERY_MONITORING" not in os.environ


def test_engine_monitoring_overrides_config_on() -> None:
    """`monitoring=False` exempts the query while the Config is enabled."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        _sample_lf().collect(engine=StreamingEngine(monitoring=False))

    # the observer is registered by the Config, but never invoked for this query
    observer.on_query_started.assert_not_called()


def test_engine_monitoring_requires_polars_cloud(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "polars_cloud", None)
    with pytest.raises(ModuleNotFoundError, match="polars_cloud"):
        StreamingEngine(monitoring=True)


def test_in_memory_engine_monitoring() -> None:
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        _sample_lf().collect(engine=pl.InMemoryEngine(monitoring=True))

    observer.on_query_started.assert_called_once()
    observer.on_query_planned.assert_called_once()


def test_engine_affinity_object_carries_monitoring() -> None:
    """A configured engine object monitors every query in its scope."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        with pl.Config(engine_affinity=StreamingEngine(monitoring=True)):
            _sample_lf().collect()
        observer.on_query_started.assert_called_once()

        # leaving the scope restores the unmonitored affinity
        _sample_lf().collect()

    observer.on_query_started.assert_called_once()
    assert "POLARS_QUERY_MONITORING" not in os.environ


def _run_collect(lf: pl.LazyFrame, engine: StreamingEngine) -> None:
    lf.collect(engine=engine)


def _run_collect_all(lf: pl.LazyFrame, engine: StreamingEngine) -> None:
    pl.collect_all([lf], engine=engine)


def _run_collect_batches(lf: pl.LazyFrame, engine: StreamingEngine) -> None:
    list(lf.collect_batches(engine=engine))


def _run_collect_async(lf: pl.LazyFrame, engine: StreamingEngine) -> None:
    async def run() -> None:
        await lf.collect_async(engine=engine)

    asyncio.run(run())


def _run_collect_all_async(lf: pl.LazyFrame, engine: StreamingEngine) -> None:
    async def run() -> None:
        await pl.collect_all_async([lf], engine=engine)

    asyncio.run(run())


def _run_sink(lf: pl.LazyFrame, engine: StreamingEngine) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        lf.sink_parquet(Path(tmp) / "out.parquet", engine=engine)


@pytest.mark.parametrize(
    "run",
    [
        _run_collect,
        _run_collect_all,
        _run_collect_batches,
        _run_collect_async,
        _run_collect_all_async,
        _run_sink,
    ],
)
def test_monitoring_covers_local_execution_paths(
    run: Callable[[pl.LazyFrame, StreamingEngine], None],
) -> None:
    """Every way of executing a query locally reports to the observer."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        run(_sample_lf(), StreamingEngine(monitoring=True))

    observer.on_query_started.assert_called_once()
    observer.on_query_planned.assert_called_once()


def test_planned_payload_decodes() -> None:
    """The IR and physical plan handed to the observer are valid msgpack."""
    msgpack = pytest.importorskip("msgpack")
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        _sample_lf().collect(engine="streaming")

    _, _, ir_bytes, phys_bytes = observer.on_query_planned.call_args.args
    ir = msgpack.unpackb(ir_bytes, raw=False)
    phys = msgpack.unpackb(phys_bytes, raw=False)

    assert isinstance(ir, list)
    assert len(ir) > 0
    assert {"id", "properties"} <= set(ir[0].keys())
    assert isinstance(phys, list)
    assert len(phys) > 0


def test_metrics_handle_snapshot() -> None:
    """The metrics handle snapshots to msgpack rows after the query runs."""
    msgpack = pytest.importorskip("msgpack")
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        _sample_lf().collect(engine="streaming")

    # The handle stays valid after the query, so snapshot reflects real work.
    handle = observer.on_query_planned.call_args.args[1]
    rows = msgpack.unpackb(handle.snapshot_query_metrics(), raw=False)

    assert isinstance(rows, list)
    assert len(rows) > 0
    expected_keys = {"phys_node_key", "rows_sent", "rows_received", "done"}
    assert expected_keys <= set(rows[0].keys())
    assert any(r["done"] for r in rows)
    assert sum(r["rows_sent"] for r in rows) > 0


def test_on_query_failed_called() -> None:
    """A failing query reports `on_query_failed` with the error message."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        with pytest.raises(pl.exceptions.PolarsError):
            pl.LazyFrame({"a": [1, 2, 3]}).select(pl.col("does_not_exist")).collect(
                engine="streaming"
            )

    observer.on_query_started.assert_called_once()
    observer.on_query_failed.assert_called_once()
    query_id, err = observer.on_query_failed.call_args.args
    assert isinstance(query_id, uuid.UUID)
    assert isinstance(err, str)
    assert err != ""


def test_no_monitoring_no_observer() -> None:
    """With monitoring off the observer class is never constructed or called."""
    module, _ = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        _sample_lf().collect(engine="streaming")

    module.QueryCloudObserver.assert_not_called()


def test_in_memory_engine_planned_without_physical() -> None:
    """The in-memory engine is observed with an IR-only planned query.

    It has no streaming physical plan, so `phys_bytes` decodes to null, but the
    full started/planned/close span still fires.
    """
    msgpack = pytest.importorskip("msgpack")
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        _sample_lf().collect(engine="in-memory")

    observer.on_query_started.assert_called_once()
    observer.on_query_planned.assert_called_once()
    observer.on_query_planned.return_value.close.assert_called_once()

    _, _, ir_bytes, phys_bytes = observer.on_query_planned.call_args.args
    assert len(msgpack.unpackb(ir_bytes, raw=False)) > 0
    assert msgpack.unpackb(phys_bytes, raw=False) is None
