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

skip_paths = {
    python_snippets_dir / "user-guide" / "visualization",  # no plots in CI
    python_snippets_dir / "user-guide" / "io" / "hugging-face.py",  # rate limited
    python_snippets_dir / "user-guide" / "expressions" / "aggregation.py",  # rate limited (fetches from HuggingFace)
}
# numba does not support Python 3.13 yet
if sys.version_info >= (3, 13):
    skip_paths.add(python_snippets_dir / "user-guide" / "user-defined-functions")

snippet_paths = [p for p in snippet_paths if not any(p.is_relative_to(s) for s in skip_paths)]


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
