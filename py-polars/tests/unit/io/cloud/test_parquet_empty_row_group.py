"""Tests for reading Parquet files with empty row groups from cloud storage."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.unit.io.cloud.conftest import CountingS3

pytestmark = pytest.mark.slow()


def test_scan_parquet_empty_row_group(s3: CountingS3, tmp_path: Path) -> None:
    # The zero-size column chunks of an empty row group can all start at offset 0. Fetched
    # by start offset, their byte ranges used to collapse into one and trip an assertion.
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    schema = pa.schema([("i", pa.int64())] + [(f"b{i}", pa.bool_()) for i in range(4)])
    table = pa.table(
        {"i": list(range(10)), **{f"b{i}": [True] * 10 for i in range(4)}},
        schema=schema,
    )
    path = tmp_path / "empty_row_group.parquet"
    with pq.ParquetWriter(path, schema) as writer:
        writer.write_table(table)
        writer.write_table(schema.empty_table())
    assert pq.ParquetFile(path).metadata.num_row_groups == 2

    s3.client.put_object(
        Bucket="bucket", Key="empty_row_group.parquet", Body=path.read_bytes()
    )
    expected = pl.read_parquet(path)
    assert expected.shape == (10, 5)

    lf = pl.scan_parquet(
        "s3://bucket/empty_row_group.parquet", storage_options=s3.storage_options
    )
    assert_frame_equal(lf.collect(), expected)
    assert_frame_equal(lf.select("i", "b0").collect(), expected.select("i", "b0"))
