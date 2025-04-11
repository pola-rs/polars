from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, AnyStr

import pytest

from polars._utils.various import issue_warning, normalize_filepath
from polars.exceptions import PerformanceWarning

if TYPE_CHECKING:
    import os


def test_issue_warning() -> None:
    msg = "hello"
    with pytest.warns(PerformanceWarning, match=msg):
        issue_warning(msg, PerformanceWarning)


class TestNormalizeFilepath:
    """Test coverage for `polars/_utils/various.py::normalize_filepath()`."""

    class CustomPathLike:
        """Implementation of the `os.PathLike` protocol."""

        def __init__(self, path: str | bytes) -> None:
            self.path = path

        def __fspath__(self) -> str | bytes:
            return self.path

    @pytest.mark.parametrize(
        ("path", "expected_path", "expected_type"),
        [
            ("/dummy/file.py", "/dummy/file.py", str),
            ("~/dummy/file.py", str(Path.home() / "dummy" / "file.py"), str),
            (b"/dummy/file.py", b"/dummy/file.py", bytes),
            (b"~/dummy/file.py", bytes(Path.home() / "dummy" / "file.py"), bytes),
            (Path("/dummy/file.py"), "/dummy/file.py", str),
            (Path("~/dummy/file.py"), str(Path.home() / "dummy" / "file.py"), str),
            (CustomPathLike("/dummy/file.py"), "/dummy/file.py", str),
            (
                CustomPathLike("~/dummy/file.py"),
                str(Path.home() / "dummy" / "file.py"),
                str,
            ),
            (CustomPathLike(b"/dummy/file.py"), b"/dummy/file.py", bytes),
            (
                CustomPathLike(b"~/dummy/file.py"),
                bytes(Path.home() / "dummy" / "file.py"),
                bytes,
            ),
        ],
    )
    def test_normalize_filepath(
        self,
        path: AnyStr | os.PathLike[AnyStr],
        expected_path: str | bytes,
        expected_type: type[str | bytes],
    ) -> None:
        normalized = normalize_filepath(path)
        assert normalized == expected_path
        assert type(normalized) is expected_type
        assert "~" not in str(normalized)
