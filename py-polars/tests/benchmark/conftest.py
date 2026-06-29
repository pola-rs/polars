import pytest

import polars as pl
from tests.benchmark.data import generate_group_by_data


@pytest.fixture(scope="session")
def groupby_data() -> pl.DataFrame:
    return generate_group_by_data(10_000, 100, null_ratio=0.05)


@pytest.fixture(scope="session")
def high_cardinality_groupby_data() -> pl.DataFrame:
    n = 200_000
    return pl.DataFrame(
        {
            "k": range(n),
            "v": [i % 97 for i in range(n)],
        }
    )
