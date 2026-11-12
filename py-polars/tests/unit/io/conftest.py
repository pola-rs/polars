from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from typing import Any

    from pyiceberg.catalog import Catalog


@pytest.fixture
def io_files_path() -> Path:
    return Path(__file__).parent / "files"


def format_file_uri(absolute_local_path: str | Path) -> str:
    absolute_local_path = str(absolute_local_path)

    if sys.platform == "win32":
        assert absolute_local_path[0].isalpha()
        assert absolute_local_path[1] == ":"
        p = absolute_local_path.replace("\\", "/")
        return f"file:///{p}"

    assert absolute_local_path.startswith("/")
    return f"file://{absolute_local_path}"


def normalize_path_separator_pl(s: Any) -> Any:
    if sys.platform == "win32":
        return s.str.replace_all("\\", "/", literal=True)

    return s


@pytest.fixture
def reset_iceberg_catalog_resolver() -> Iterator[None]:
    """Restore the default Iceberg catalog resolver after a test that may set one."""
    from polars.io.iceberg._catalog_resolver import set_iceberg_catalog_resolver

    yield
    set_iceberg_catalog_resolver(None)


def recording_catalog_resolver(
    catalog: Catalog,
) -> tuple[Callable[[str], Catalog], list[str]]:
    """A resolver that returns `catalog`, and the names it has been called with."""
    names: list[str] = []

    def resolve(name: str) -> Catalog:
        names.append(name)
        return catalog

    return resolve, names


def failing_catalog_resolver(name: str) -> Catalog:
    """A resolver for tests in which no catalog name may be resolved."""
    pytest.fail(f"resolver called for {name!r}")
