import os
from pathlib import Path

import pytest

io_test_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "unit", "io")
)


@pytest.fixture()
def io_files_path() -> Path:
    current_dir = os.path.abspath(os.path.dirname(__file__))
    return Path(current_dir).parent / "unit" / "io" / "files"
