from __future__ import annotations

from pathlib import Path

import pytest
import sys

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture
def io_files_path() -> Path:
    return Path(__file__).parent / "files"


def format_file_uri(absolute_local_path: str | Path) -> str:
    absolute_local_path = str(absolute_local_path)

    if sys.platform == "win32":
        assert absolute_local_path[0].isalpha() and absolute_local_path[1] == ":"
        return f"file:///{absolute_local_path.replace('\\', '/')}"

    assert absolute_local_path.startswith("/")
    return f"file://{absolute_local_path}"


def normalize_path_separator_pl(s: Any) -> Any:
    if sys.platform == "win32":
        return s.str.replace_all("\\", "/", literal=True)

    return s
