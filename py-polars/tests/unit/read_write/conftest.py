from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture()
def io_files_path() -> Path:
    current_dir = os.path.dirname(__file__)
    return Path(current_dir) / "files"
