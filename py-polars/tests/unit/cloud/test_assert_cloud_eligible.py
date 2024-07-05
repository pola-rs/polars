import pytest

import polars as pl
from polars._utils.cloud import assert_cloud_eligible


@pytest.mark.parametrize(
    "lf",
    [
        pl.scan_parquet("data.parquet").select("a"),
    ],
)
def test_assert_cloud_eligible_pass(lf: pl.LazyFrame) -> None:
    assert_cloud_eligible(lf)


@pytest.mark.parametrize(
    "lf",
    [
        pl.scan_parquet("data.parquet").select(
            pl.col("a").map_elements(lambda x: sum(x))
        ),
    ],
)
def test_assert_cloud_eligible_fail(lf: pl.LazyFrame) -> None:
    with pytest.raises(
        AssertionError, match="logical plan ineligible for execution on Polars Cloud"
    ):
        assert_cloud_eligible(lf)
