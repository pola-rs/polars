from functools import partial

import pytest

import polars as pl
from polars.exceptions import ComputeError


@pytest.mark.slow
@pytest.mark.parametrize("format", ["parquet", "csv", "ndjson", "ipc"])
def test_scan_nonexistent_cloud_path_17444(format: str) -> None:
    # https://github.com/pola-rs/polars/issues/17444

    path_str = f"s3://my-nonexistent-bucket/data.{format}"
    scan_function = getattr(pl, f"scan_{format}")
    # Prevent automatic credential provideder instantiation, otherwise CI may fail with
    # * pytest.PytestUnraisableExceptionWarning:
    #   * Exception ignored:
    #     * ResourceWarning: unclosed socket
    scan_function = partial(scan_function, credential_provider=None)

    # Just calling the scan function should not raise any errors
    if format == "ndjson":
        # NDJSON does not have a `retries` parameter yet - so use the default
        result = scan_function(path_str)
    else:
        result = scan_function(path_str, retries=0)
    assert isinstance(result, pl.LazyFrame)

    # Upon collection, it should fail
    with pytest.raises(ComputeError):
        result.collect()
