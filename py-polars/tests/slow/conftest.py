import os

import pytest

IO_TEST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "unit", "io")
)


@pytest.fixture
def io_test_dir() -> str:
    return IO_TEST_DIR
