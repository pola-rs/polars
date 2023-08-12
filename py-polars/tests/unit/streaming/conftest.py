from pathlib import Path

import pytest


@pytest.fixture()
def io_files_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files"
