import pytest

import polars as pl
from tests.benchmark.datagen_groupby import generate_group_by_data


@pytest.fixture(scope="module")
def groupby_data() -> pl.DataFrame:
    return generate_group_by_data(1000, 10, null_ratio=0.1)
