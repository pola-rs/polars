from pathlib import Path

import pytest

from polars.plugins import _get_dynamic_lib_location


def test_get_dynamic_lib_location(tmpdir: Path) -> None:
    (tmpdir / "lib1.so").write_text("", encoding="utf-8")
    (tmpdir / "__init__.py").write_text("", encoding="utf-8")
    result = _get_dynamic_lib_location(tmpdir)
    assert result == str(tmpdir / "lib1.so")
    result = _get_dynamic_lib_location(tmpdir / "lib1.so")
    assert result == str(tmpdir / "lib1.so")
    result = _get_dynamic_lib_location(str(tmpdir))
    assert result == str(tmpdir / "lib1.so")
    result = _get_dynamic_lib_location(str(tmpdir / "lib1.so"))
    assert result == str(tmpdir / "lib1.so")


def test_get_dynamic_lib_location_raises(tmpdir: Path) -> None:
    (tmpdir / "__init__.py").write_text("", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="no dynamic library found"):
        _get_dynamic_lib_location(tmpdir)
