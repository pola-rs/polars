from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.plugins import (
    _is_dynamic_lib,
    _resolve_plugin_path,
    _serialize_kwargs,
    register_plugin_function,
)


@pytest.mark.write_disk()
def test_register_plugin_function_invalid_plugin_path(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    plugin_path = tmp_path / "lib.so"
    plugin_path.touch()

    expr = register_plugin_function(
        plugin_path=plugin_path, function_name="hello", args=5
    )

    with pytest.raises(ComputeError, match="error loading dynamic library"):
        pl.select(expr)


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (None, b""),
        ({}, b""),
        (
            {"hi": 0},
            b"\x80\x05\x95\x0b\x00\x00\x00\x00\x00\x00\x00}\x94\x8c\x02hi\x94K\x00s.",
        ),
    ],
)
def test_serialize_kwargs(input: dict[str, Any] | None, expected: bytes) -> None:
    assert _serialize_kwargs(input) == expected


@pytest.mark.write_disk()
def test_resolve_plugin_path(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    (tmp_path / "lib1.so").touch()
    (tmp_path / "__init__.py").touch()

    expected = tmp_path / "lib1.so"

    result = _resolve_plugin_path(tmp_path)
    assert result == expected
    result = _resolve_plugin_path(tmp_path / "lib1.so")
    assert result == expected
    result = _resolve_plugin_path(str(tmp_path))
    assert result == expected
    result = _resolve_plugin_path(str(tmp_path / "lib1.so"))
    assert result == expected


@pytest.mark.write_disk()
def test_resolve_plugin_path_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "__init__.py").touch()

    with pytest.raises(FileNotFoundError, match="no dynamic library found"):
        _resolve_plugin_path(tmp_path)


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (Path("lib.so"), True),
        (Path("lib.pyd"), True),
        (Path("lib.dll"), True),
        (Path("lib.py"), False),
    ],
)
def test_is_dynamic_lib(path: Path, expected: bool, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    full_path = tmp_path / path
    full_path.touch()
    assert _is_dynamic_lib(full_path) is expected


@pytest.mark.write_disk()
def test_is_dynamic_lib_dir(tmp_path: Path) -> None:
    path = Path("lib.so")
    full_path = tmp_path / path

    full_path.mkdir(exist_ok=True)
    (full_path / "hello.txt").touch()

    assert _is_dynamic_lib(full_path) is False
