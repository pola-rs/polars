import pytest

import polars as pl
from tests.benchmark.data import generate_group_by_data


@pytest.fixture(scope="session")
def groupby_data() -> pl.DataFrame:
    return generate_group_by_data(10_000, 100, null_ratio=0.05)
