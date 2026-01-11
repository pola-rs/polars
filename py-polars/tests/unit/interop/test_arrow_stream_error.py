"""Test for https://github.com/pola-rs/polars/issues/25966."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

import polars as pl
from polars.exceptions import ComputeError

if TYPE_CHECKING:
    from collections.abc import Iterator


def test_arrow_stream_error_raises_compute_error() -> None:
    error_msg = "simulated error"

    def failing_generator() -> Iterator[pa.RecordBatch]:
        raise pa.ArrowInvalid(error_msg)
        yield

    schema = pa.schema([("col", pa.int64())])
    reader = pa.RecordBatchReader.from_batches(schema, failing_generator())

    class FailingStream:
        def __arrow_c_stream__(self, requested_schema: object = None) -> object:
            return reader.__arrow_c_stream__(requested_schema)

    with pytest.raises(ComputeError, match=error_msg):
        pl.from_arrow(FailingStream())
