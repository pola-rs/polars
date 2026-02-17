"""Run all Python code snippets."""

import os
import runpy
import sys
import warnings
from collections.abc import Iterator
from pathlib import Path

with warnings.catch_warnings():
    # matplotlib had some deprecated calls into pyparsing
    warnings.simplefilter("ignore", DeprecationWarning)
    import matplotlib as mpl
import pytest

# Do not show plots
mpl.use("Agg")

# Get paths to Python code snippets
repo_root = Path(__file__).parent.parent.parent.parent
python_snippets_dir = repo_root / "docs" / "source" / "src" / "python"
snippet_paths = list(python_snippets_dir.rglob("*.py"))

# Skip visualization snippets
snippet_paths = [p for p in snippet_paths if "visualization" not in str(p)]

# Skip UDF section on Python 3.13 as numba does not support it yet
if sys.version_info >= (3, 13):
    snippet_paths = [p for p in snippet_paths if "user-defined-functions" not in str(p)]


@pytest.fixture(scope="module")
def _change_test_dir() -> Iterator[None]:
    """Change path to repo root to accommodate data paths in code snippets."""
    current_path = Path().resolve()
    os.chdir(repo_root)
    yield
    os.chdir(current_path)


@pytest.mark.docs
@pytest.mark.parametrize("path", snippet_paths)
@pytest.mark.usefixtures("_change_test_dir")
def test_run_python_snippets(path: Path) -> None:
    runpy.run_path(str(path))
