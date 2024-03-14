from pathlib import Path

import pytest

from polars.plugins import _get_dynamic_lib_location, _is_dynamic_lib


@pytest.mark.write_disk()
def test_get_dynamic_lib_location(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    (tmp_path / "lib1.so").touch()
    (tmp_path / "__init__.py").touch()

    expected = tmp_path / "lib1.so"

    result = _get_dynamic_lib_location(tmp_path)
    assert result == expected
    result = _get_dynamic_lib_location(tmp_path / "lib1.so")
    assert result == expected
    result = _get_dynamic_lib_location(str(tmp_path))
    assert result == expected
    result = _get_dynamic_lib_location(str(tmp_path / "lib1.so"))
    assert result == expected


@pytest.mark.write_disk()
def test_get_dynamic_lib_location_raises(tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "__init__.py").touch()

    with pytest.raises(FileNotFoundError, match="no dynamic library found"):
        _get_dynamic_lib_location(tmp_path)


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
