from io import BytesIO
from pathlib import Path

import pytest

import polars as pl
from polars._utils.cloud import prepare_cloud_plan
from polars.exceptions import ComputeError, InvalidOperationError

CLOUD_SOURCE = "s3://my-nonexistent-bucket/dataset"
DST = "s3://my-nonexistent-bucket/output"


@pytest.mark.parametrize(
    "lf",
    [
        pl.scan_parquet(CLOUD_SOURCE)
        .select("c", pl.lit(2))
        .with_row_index()
        .sink_parquet(DST, lazy=True),
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        .select("a", "b")
        .filter(pl.col("a") == pl.lit(1))
        .sink_parquet(DST, lazy=True),
    ],
)
def test_prepare_cloud_plan(lf: pl.LazyFrame) -> None:
    result = prepare_cloud_plan(lf)
    assert isinstance(result, bytes)

    deserialized = pl.LazyFrame.deserialize(BytesIO(result))
    assert isinstance(deserialized, pl.LazyFrame)


@pytest.mark.parametrize(
    "lf",
    [
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        .select(pl.col("a").map_elements(lambda x: sum(x)))
        .sink_parquet(DST, lazy=True),
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        .select(pl.col("b").map_batches(lambda x: sum(x)))
        .sink_parquet(DST, lazy=True),
        pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
        .map_batches(lambda x: x)
        .sink_parquet(DST, lazy=True),
        pl.scan_parquet(CLOUD_SOURCE)
        .filter(pl.col("a") < pl.lit(1).map_elements(lambda x: x + 1))
        .sink_parquet(DST, lazy=True),
        pl.LazyFrame({"a": [[1, 2], [3, 4, 5]], "b": [3, 4]})
        .select(pl.col("a").map_elements(lambda x: sum(x), return_dtype=pl.Int64))
        .sink_parquet(DST, lazy=True),
    ],
)
def test_prepare_cloud_plan_udf(lf: pl.LazyFrame) -> None:
    result = prepare_cloud_plan(lf)
    assert isinstance(result, bytes)

    deserialized = pl.LazyFrame.deserialize(BytesIO(result))
    assert isinstance(deserialized, pl.LazyFrame)


def test_prepare_cloud_plan_optimization_toggle() -> None:
    lf = pl.LazyFrame({"a": [1, 2], "b": [3, 4]}).sink_parquet(DST, lazy=True)

    result = prepare_cloud_plan(
        lf, optimizations=pl.QueryOptFlags(projection_pushdown=False)
    )
    assert isinstance(result, bytes)

    # TODO: How to check that this optimization was toggled correctly?
    deserialized = pl.LazyFrame.deserialize(BytesIO(result))
    assert isinstance(deserialized, pl.LazyFrame)


@pytest.mark.parametrize(
    "lf",
    [
        pl.scan_parquet("data.parquet").sink_parquet(DST, lazy=True),
        pl.scan_ndjson(Path("data.ndjson")).sink_parquet(DST, lazy=True),
        pl.scan_csv("data-*.csv").sink_parquet(DST, lazy=True),
        pl.scan_ipc(["data-1.feather", "data-2.feather"]).sink_parquet(DST, lazy=True),
    ],
)
def test_prepare_cloud_plan_fail_on_local_data_source(lf: pl.LazyFrame) -> None:
    with pytest.raises(
        InvalidOperationError,
        match="logical plan ineligible for execution on Polars Cloud",
    ):
        prepare_cloud_plan(lf)


@pytest.mark.parametrize(
    "lf",
    [
        pl.LazyFrame({"a": [{"x": 1, "y": 2}]})
        .select(pl.col("a").name.map(lambda x: x.upper()))
        .sink_parquet(DST, lazy=True),
        pl.LazyFrame({"a": [{"x": 1, "y": 2}]})
        .select(pl.col("a").name.map_fields(lambda x: x.upper()))
        .sink_parquet(DST, lazy=True),
    ],
)
def test_prepare_cloud_plan_fail_on_serialization(lf: pl.LazyFrame) -> None:
    with pytest.raises(ComputeError, match="serialization not supported"):
        prepare_cloud_plan(lf)
