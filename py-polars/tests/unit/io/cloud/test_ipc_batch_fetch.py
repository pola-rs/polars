"""Tests for grouping the record batch fetches of remote IPC scans."""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.ipc
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from tests.unit.io.cloud.conftest import CountingS3

pytestmark = pytest.mark.slow()

N_ROWS = 20_000
BATCH_SIZE = 100


def frame() -> pl.DataFrame:
    return pl.DataFrame({"a": range(N_ROWS), "b": ["x" * 100] * N_ROWS})


def upload_polars_ipc(s3: CountingS3, key: str) -> None:
    f = io.BytesIO()
    frame().write_ipc(f, compression="uncompressed", record_batch_size=BATCH_SIZE)
    # Larger than the footer prefetch, so record batches are fetched separately.
    assert len(f.getvalue()) > 256 * 1024
    s3.client.put_object(Bucket="bucket", Key=key, Body=f.getvalue())


def range_gets(s3: CountingS3, mark: int) -> list[str]:
    return [rng for method, _, rng in s3.since(mark) if method == "GET" and rng]


def test_scan_ipc_groups_record_batch_fetches(
    s3: CountingS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A budget and group size large enough for all 200 record batches in one group.
    monkeypatch.setenv("POLARS_RECORD_BATCH_PREFETCH_SIZE", "1024")
    monkeypatch.setenv("POLARS_RECORD_BATCH_GROUP_MAX_BYTES", str(8 << 20))
    upload_polars_ipc(s3, "grouped.arrow")

    mark = len(s3.log)
    out = pl.scan_ipc(
        "s3://bucket/grouped.arrow", storage_options=s3.storage_options
    ).collect(engine="streaming")

    assert_frame_equal(out, frame())
    # The footer, then all record batches in a single request.
    assert len(range_gets(s3, mark)) == 2


def test_scan_ipc_grouped_fetch_tiny_budget(
    s3: CountingS3, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A budget smaller than a group must not deadlock.
    monkeypatch.setenv("POLARS_RECORD_BATCH_PREFETCH_SIZE", "1")
    monkeypatch.setenv("POLARS_RECORD_BATCH_PREFETCH_KBYTES_BUDGET", "1")
    upload_polars_ipc(s3, "tiny_budget.arrow")

    out = pl.scan_ipc(
        "s3://bucket/tiny_budget.arrow", storage_options=s3.storage_options
    ).collect(engine="streaming")

    assert_frame_equal(out, frame())


def test_scan_ipc_grouped_fetch_slice_with_row_index(s3: CountingS3) -> None:
    upload_polars_ipc(s3, "sliced.arrow")

    out = (
        pl.scan_ipc("s3://bucket/sliced.arrow", storage_options=s3.storage_options)
        .with_row_index()
        .slice(5_050, 3_000)
        .collect(engine="streaming")
    )

    assert_frame_equal(out, frame().with_row_index().slice(5_050, 3_000))


def test_scan_ipc_metadata_only_fetches_are_not_merged(s3: CountingS3) -> None:
    # Without Polars metadata, a count reads each record batch's metadata only.
    table = frame().to_arrow()
    f = io.BytesIO()
    with pa.ipc.new_file(f, table.schema) as writer:
        for batch in table.to_batches(max_chunksize=BATCH_SIZE):
            writer.write_batch(batch)
    s3.client.put_object(Bucket="bucket", Key="pyarrow.arrow", Body=f.getvalue())

    mark = len(s3.log)
    out = (
        pl.scan_ipc("s3://bucket/pyarrow.arrow", storage_options=s3.storage_options)
        .select(pl.len())
        .collect(engine="streaming")
    )

    assert out.item() == N_ROWS
    # Merging the metadata ranges would also download the record batch bodies.
    for rng in range_gets(s3, mark)[1:]:
        start, end = map(int, rng.removeprefix("bytes=").split("-"))
        assert end - start < 4096
