import os
from pathlib import Path

import pytest

IO_TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "unit", "io")
)


@pytest.fixture()
def io_files_dir() -> Path:
    return Path(IO_TEST_DIR) / "files"
