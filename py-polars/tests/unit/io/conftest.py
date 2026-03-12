from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from typing import Any


@pytest.fixture
def io_files_path() -> Path:
    return Path(__file__).parent / "files"


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
