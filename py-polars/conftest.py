import pytest

import polars


@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    doctest_namespace["pl"] = polars
