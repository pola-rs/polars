from pathlib import Path

import pytest

from tests.unit.conftest import IS_WASM

if IS_WASM:
    pytest.skip(
        "the streaming feature is not enabled on Emscripten/Pyodide builds.",
        allow_module_level=True,
    )


@pytest.fixture
def io_files_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files"
