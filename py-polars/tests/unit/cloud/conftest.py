from __future__ import annotations

import pytest

from tests.unit.conftest import IS_WASM

if IS_WASM:
    pytest.skip(
        "cloud features are not enabled on Emscripten/Pyodide builds",
        allow_module_level=True,
    )
