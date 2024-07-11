import pytest

import polars as pl
from polars.exceptions import ComputeError


@pytest.mark.slow()
@pytest.mark.parametrize("format", ["parquet", "csv", "ipc"])
def test_scan_retries_zero(format: str) -> None:
    path_str = f"s3://my-nonexistent-bucket/data.{format}"
    scan_function = getattr(pl, f"scan_{format}")

    with pytest.raises(ComputeError):
        scan_function(path_str, retries=0).collect()
