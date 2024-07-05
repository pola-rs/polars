import pytest

import polars as pl
from polars._utils.cloud import assert_cloud_eligible

CLOUD_PATH = "s3://my-nonexistent-bucket/dataset"


@pytest.mark.parametrize(
    "lf",
    [
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        .select("a", "b")
        .filter(pl.col("a") == pl.lit(1)),
    ],
)
def test_assert_cloud_eligible(lf: pl.LazyFrame) -> None:
    assert_cloud_eligible(lf)


@pytest.mark.parametrize(
    "lf",
    [
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).select(
            pl.col("a").map_elements(lambda x: sum(x))
        ),
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).select(
            pl.col("b").map_batches(lambda x: sum(x))
        ),
        pl.LazyFrame({"a": [{"x": 1, "y": 2}]}).select(
            pl.col("a").name.map(lambda x: x.upper())
        ),
        pl.LazyFrame({"a": [{"x": 1, "y": 2}]}).select(
            pl.col("a").name.map_fields(lambda x: x.upper())
        ),
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).map_batches(lambda x: sum(x)),
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        .group_by("a")
        .map_groups(lambda x: sum(x), schema={"b": pl.Int64}),
    ],
)
def test_assert_cloud_eligible_fail_on_udf(lf: pl.LazyFrame) -> None:
    with pytest.raises(
        AssertionError, match="logical plan ineligible for execution on Polars Cloud"
    ):
        assert_cloud_eligible(lf)


@pytest.mark.parametrize(
    "lf",
    [
        pl.scan_parquet("data.parquet"),
        pl.scan_ndjson("data.parquet"),
        pl.scan_csv("data.csv"),
        # pl.scan_ipc("data.feather"),
        # pl.scan_pyarrow_dataset("")
    ],
)
def test_assert_cloud_eligible_fail_on_local_data_source(lf: pl.LazyFrame) -> None:
    with pytest.raises(
        AssertionError, match="logical plan ineligible for execution on Polars Cloud"
    ):
        assert_cloud_eligible(lf)


# @pytest.mark.parametrize(
#     "lf",
#     [
#         pl.scan_iceberg(CLOUD_PATH),
#         pl.scan_delta(CLOUD_PATH),
#     ],
# )
# def test_assert_cloud_eligible_unsupported_scan(lf: pl.LazyFrame) -> None:
#     with pytest.raises(
#         AssertionError, match="logical plan ineligible for execution on Polars Cloud"
#     ):
#         assert_cloud_eligible(lf)
