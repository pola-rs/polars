from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from tests.unit.conftest import IS_WASM

if IS_WASM:
    pytest.skip(
        "the sql feature is not enabled on Emscripten/Pyodide builds.",
        allow_module_level=True,
    )


@pytest.fixture
def io_files_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files"


@pytest.fixture
def df_distinct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "category": ["A", "A", "B", "B", "B", "C", "C", None, None, "A"],
            "subcategory": ["x", "x", "y", "y", "z", "x", "y", "x", "y", "x"],
            "value": [100, 100, 200, 200, 300, 400, 500, 600, 700, 100],
            "status": [
                "active",
                "active",
                "active",
                "inactive",
                "active",
                "inactive",
                "active",
                "active",
                "inactive",
                "active",
            ],
            "score": [10, 20, 30, 30, 40, 50, 60, 70, 80, 10],
        }
    )
