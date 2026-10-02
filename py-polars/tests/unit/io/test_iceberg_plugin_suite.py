"""
Re-runs the Iceberg test suite (`test_iceberg.py`) with the plugin planner.

Scans resolved by the engine (`IcebergScanResolver.to_dataset_scan`) are planned with
the `polars_iceberg` plugin. Tests calling `_to_dataset_scan_impl()` directly inspect
the PyIceberg planner's intermediate data and keep using it, as do scans with
`reader_override="pyiceberg"`.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest

pytest.importorskip("polars_iceberg")

import polars.io.iceberg._dataset as iceberg_dataset
from tests.unit.io.test_iceberg import *  # noqa: F403

_in_engine_call = threading.local()
_to_dataset_scan = iceberg_dataset.IcebergScanResolver.to_dataset_scan


def _to_dataset_scan_with_plugin(self: Any, **kwargs: Any) -> Any:
    _in_engine_call.value = True
    try:
        return _to_dataset_scan(self, **kwargs)
    finally:
        _in_engine_call.value = False


@pytest.fixture(autouse=True)
def _route_engine_scans_to_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        iceberg_dataset,
        "use_plugin_planner",
        lambda: getattr(_in_engine_call, "value", False),
    )
    monkeypatch.setattr(
        iceberg_dataset.IcebergScanResolver,
        "to_dataset_scan",
        _to_dataset_scan_with_plugin,
    )
