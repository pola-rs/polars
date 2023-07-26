import pytest

import polars as pl


def test_err_on_s3_glob() -> None:
    with pytest.raises(
        ValueError,
        match=r"globbing patterns not supported when scanning non-local files",
    ):
        pl.scan_parquet(
            "s3://saturn-public-data/nyc-taxi/data/yellow_tripdata_2019-1.*.parquet"
        )
