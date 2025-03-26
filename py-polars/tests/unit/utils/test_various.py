from pathlib import Path

import pytest

from polars._utils.various import issue_warning, normalize_filepath
from polars.exceptions import PerformanceWarning


def test_issue_warning() -> None:
    msg = "hello"
    with pytest.warns(PerformanceWarning, match=msg):
        issue_warning(msg, PerformanceWarning)


class TestNormalizeFilepath:
    """Test coverage for `polars/_utils/various.py::normalize_filepath()`."""

    def test_normalize_filepath_str(self) -> None:
        path = "/dummy/file.py"
        normalized = normalize_filepath(path)
        assert type(normalized) is str
        assert normalized == path

    def test_normalize_filepath_str_normalizes(self) -> None:
        path = "~/dummy/file.py"
        normalized = normalize_filepath(path)
        assert type(normalized) is str
        assert normalized != path

    def test_normalize_filepath_bytes(self) -> None:
        path = b"/dummy/file.py"
        normalized = normalize_filepath(path)
        assert type(normalized) is bytes
        assert normalized == path

    def test_normalize_filepath_bytes_normalizes(self) -> None:
        path = b"~/dummy/file.py"
        normalized = normalize_filepath(path)
        assert type(normalized) is bytes
        assert normalized != path

    def test_normalize_filepath_pathlib(self) -> None:
        path = Path("/") / "dummy" / "file.py"
        normalized = normalize_filepath(path)
        assert type(normalized) is str
        assert normalized == str(path)

    def test_normalize_filepath_pathlib_normalizes(self) -> None:
        path = Path("~") / "dummy" / "file.py"
        normalized = normalize_filepath(path)
        assert type(normalized) is str
        assert normalized != str(path)
        assert normalized == str(Path.home() / "dummy" / "file.py")

    def test_normalize_filepath_custom_pathlike(self) -> None:
        class CustomPathLike:
            def __fspath__(self) -> bytes:
                return b"/dummy/file.py"

        path = CustomPathLike()
        normalized = normalize_filepath(path)
        assert type(normalized) is bytes
        assert normalized == b"/dummy/file.py"

    def test_normalize_filepath_custom_pathlike_normalizes(self) -> None:
        class CustomPathLike:
            def __fspath__(self) -> bytes:
                return b"~/dummy/file.py"

        path = CustomPathLike()
        normalized = normalize_filepath(path)
        assert type(normalized) is bytes
        assert normalized != b"~/dummy/file.py"
        assert normalized == bytes(Path.home() / "dummy" / "file.py")
