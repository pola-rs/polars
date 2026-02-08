from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.unit.conftest import IS_WASM

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture
def io_files_path() -> Path:
    return Path(__file__).parent / "files"


# Doing this with a very rudimentary way by checking over
# file paths because I am too lazy for this right now
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if not IS_WASM:
        return

    io_root = Path(__file__).resolve().parent
    skip_marker = pytest.mark.skip(
        reason="IO features (csv/ipc/parquet/json/etc.) are not enabled on Emscripten/Pyodide builds."
    )
    for item in items:
        try:
            item_path = Path(str(item.fspath)).resolve()
        except Exception:
            continue
        if item_path.is_relative_to(io_root):
            item.add_marker(skip_marker)


def format_file_uri(absolute_local_path: str | Path) -> str:
    absolute_local_path = str(absolute_local_path)

    if sys.platform == "win32":
        assert absolute_local_path[0].isalpha()
        assert absolute_local_path[1] == ":"
        p = absolute_local_path.replace("\\", "/")
        return f"file:///{p}"

    assert absolute_local_path.startswith("/")
    return f"file://{absolute_local_path}"


def normalize_path_separator_pl(s: Any) -> Any:
    if sys.platform == "win32":
        return s.str.replace_all("\\", "/", literal=True)

    return s
