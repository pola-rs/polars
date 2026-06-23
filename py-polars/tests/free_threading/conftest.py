from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests.free_threading._free_threading import (
    assert_gil_disabled,
    is_free_threaded_python,
)

if TYPE_CHECKING:
    from collections.abc import Generator


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    if is_free_threaded_python():
        return

    skip_free_threaded = pytest.mark.skip(reason="requires free-threaded CPython")
    for item in items:
        item.add_marker(skip_free_threaded)


@pytest.fixture(autouse=True)
def guard_gil_disabled() -> Generator[None, None, None]:
    assert_gil_disabled()
    yield
    assert_gil_disabled()
