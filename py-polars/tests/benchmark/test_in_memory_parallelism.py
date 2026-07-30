import pytest

import polars as pl

pytestmark = pytest.mark.benchmark()


@pytest.fixture(scope="module")
def single_chunk_df() -> pl.DataFrame:
    return pl.DataFrame({"a": range(5_000_000)})


def test_filter_single_chunk(single_chunk_df: pl.DataFrame) -> None:
    predicate = (pl.col("a") * 3 % 7) == 0
    single_chunk_df.lazy().filter(predicate).collect(engine="in-memory")


def test_select_single_chunk(single_chunk_df: pl.DataFrame) -> None:
    expression = ((pl.col("a") * 3 + 1) % 7).alias("x")
    single_chunk_df.lazy().select(expression).collect(engine="in-memory")


def test_with_columns_single_chunk(single_chunk_df: pl.DataFrame) -> None:
    expression = ((pl.col("a") * 3 + 1) % 7).alias("x")
    single_chunk_df.lazy().with_columns(expression).collect(engine="in-memory")
