from __future__ import annotations

import os
import uuid
from types import ModuleType
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

import polars as pl
import polars._plr as plr
from polars.lazyframe.engine_config import StreamingEngine
from tests.unit.conftest import mock_module_import

if TYPE_CHECKING:
    from collections.abc import Iterator


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
    module, _observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        module.authenticate.assert_called_once()
        assert os.environ["POLARS_QUERY_MONITORING"] == "1"
        assert os.environ["POLARS_ENGINE_AFFINITY"] == "streaming"

        pl.Config.enable_monitoring(False)
        assert "POLARS_QUERY_MONITORING" not in os.environ


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


def test_streaming_engine_object_enables_monitoring() -> None:
    """`StreamingEngine(monitoring=True)` enables monitoring without the env var."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        _sample_lf().collect(engine=StreamingEngine(monitoring=True))

    observer.on_query_started.assert_called_once()
    observer.on_query_planned.assert_called_once()
    observer.on_query_planned.return_value.close.assert_called_once()


def test_streaming_engine_monitoring_false_overrides_env() -> None:
    """`StreamingEngine(monitoring=False)` disables monitoring (engine flag wins)."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        _sample_lf().collect(engine=StreamingEngine(monitoring=False))

    module.QueryCloudObserver.assert_not_called()
    observer.on_query_started.assert_not_called()


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
    rows = msgpack.unpackb(handle.snapshot(), raw=False)

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
