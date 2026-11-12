from __future__ import annotations

import io
import re
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch
    from tests.unit.io.cloud.conftest import CountingS3


@pytest.mark.slow
@pytest.mark.parametrize("partitioned", [False, True])
def test_sink_error_stops_input(
    s3: CountingS3, plmonkeypatch: PlMonkeyPatch, partitioned: bool
) -> None:
    plmonkeypatch.setenv("POLARS_CLOUD_FOOTER_READ_SIZE", "512")
    plmonkeypatch.setenv("POLARS_NUM_READERS_PRE_INIT", "1")
    plmonkeypatch.setenv("POLARS_MAX_CONCURRENT_SCANS", "1")
    plmonkeypatch.setenv("POLARS_MAX_OPEN_SINKS", "1")
    plmonkeypatch.setenv("POLARS_INFLIGHT_SINK_MORSEL_LIMIT", "1")
    plmonkeypatch.setenv("POLARS_UPLOAD_CHUNK_SIZE", "1024")

    data = io.BytesIO()
    pl.DataFrame({"x": range(1024)}).write_parquet(data, compression="uncompressed")
    keys = [f"input/{i}.parquet" for i in range(32)]
    for key in keys:
        s3.client.put_object(Bucket="bucket", Key=key, Body=data.getvalue())

    def fail_upload(environ: dict[str, Any]) -> bool | None:
        if environ["REQUEST_METHOD"] == "PUT":
            return True
        return None

    s3.on_request = fail_upload
    mark = len(s3.log)
    source = pl.scan_parquet(
        [f"s3://bucket/{key}" for key in keys], storage_options=s3.storage_options
    )
    target = (
        pl.PartitionBy("s3://bucket/output/", max_rows_per_file=1024)
        if partitioned
        else "s3://bucket/output.parquet"
    )
    with pytest.raises(
        (OSError, pl.exceptions.ComputeError), match="500 Internal Server Error"
    ):
        source.sink_parquet(
            target,
            compression="uncompressed",
            row_group_size=1024,
            storage_options={**s3.storage_options, "max_retries": 0},
            engine="streaming",
        )

    read_keys = {
        key
        for method, key, byte_range in s3.since(mark)
        if method == "GET"
        and (match := re.fullmatch(r"bytes=(\d+)-(\d+)", byte_range))
        and int(match[2]) - int(match[1]) + 1 > len(data.getvalue()) // 2
    }
    assert 0 < len(read_keys) < len(keys)
