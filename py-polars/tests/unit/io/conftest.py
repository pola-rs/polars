from __future__ import annotations

from pathlib import Path

import pytest
import sys


@pytest.fixture
def io_files_path() -> Path:
    return Path(__file__).parent / "files"


def format_file_uri(absolute_local_path: str | Path) -> str:
    if sys.platform == "win32":
        assert absolute_local_path[0].isalpha() and absolute_local_path[1] == ":"
        return f"file:///{absolute_local_path}"

    assert absolute_local_path.startswith("/")
    return f"file://{absolute_local_path}"
