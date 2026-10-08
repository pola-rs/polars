"""
Re-runs the Iceberg test suite (`test_iceberg.py`) with the plugin planner.

Scans resolved by the engine (`IcebergScanResolver.to_dataset_scan`) are planned with
the `polars_iceberg` plugin. Tests calling `_to_dataset_scan_impl()` directly inspect
the PyIceberg planner's intermediate data and keep using it, as do scans with
`reader_override="pyiceberg"`.
"""

from __future__ import annotations

import functools
import threading
from typing import Any

import pytest

pytest.importorskip("polars_iceberg")

import polars.io.iceberg._dataset as iceberg_dataset
from tests.unit.io.test_iceberg import *  # noqa: F403

# These tests count the reads of PyIceberg's FileIO, which the plugin planner does not
# use: it reads through Polars' storage, cached on the Rust side (tested in
# `test_iceberg_plugin.py`).
_XFAIL_METADATA_FILE_CACHE = [
    "test_scan_iceberg_metadata_file_cache",
    "test_scan_iceberg_metadata_file_cache_disabled",
    "test_scan_iceberg_metadata_file_cache_incremental",
]


def _xfail_copy(f: Any, reason: str) -> Any:
    # Wrap so the mark does not leak onto the test in `test_iceberg.py`.
    @functools.wraps(f)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return f(*args, **kwargs)

    return pytest.mark.xfail(reason=reason, strict=True)(wrapper)


for _name in _XFAIL_METADATA_FILE_CACHE:
    globals()[_name] = _xfail_copy(
        globals()[_name], "plugin planner does not read through the PyIceberg FileIO"
    )

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
