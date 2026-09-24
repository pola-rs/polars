from __future__ import annotations

from datetime import time
from typing import TYPE_CHECKING

import numpy as np

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

# Test files are from the apache/parquet-testing repository, and were written by
# the Arrow C++ encryption tests (see cpp/src/parquet/encryption/test_encryption_util.cc
# in the apache/arrow repository for details).
FOOTER_KEY = b"0123456789012345"
NUM_ROWS = 50

JULIAN_DAY_OF_EPOCH = 2_440_588
MICROS_PER_DAY = 86_400 * 1_000_000


def expected_data() -> pl.DataFrame:
    n = NUM_ROWS
    return pl.DataFrame(
        {
            "boolean_field": pl.Series([i % 2 == 0 for i in range(n)]),
            "int32_field": pl.Series([time(microsecond=i * 1000) for i in range(n)]),
            "int64_field": pl.Series(
                [[2 * i * 10**12, (2 * i + 1) * 10**12] for i in range(n)],
                dtype=pl.List(pl.Int64),
            ),
            # INT96 values are [nanos low bits, nanos high bits, Julian day] =
            # [i, i + 1, i + 2]. These overflow when read as nanoseconds, so
            # are read as microseconds.
            "int96_field": pl.Series(
                [
                    (i + 2 - JULIAN_DAY_OF_EPOCH) * MICROS_PER_DAY
                    + (((i + 1) << 32) + i) // 1000
                    for i in range(n)
                ],
                dtype=pl.Int64,
            ).cast(pl.Datetime("us")),
            "float_field": pl.Series(np.arange(n, dtype=np.float32) * np.float32(1.1)),
            "double_field": pl.Series([i * 1.1111111 for i in range(n)]),
            "ba_field": pl.Series(
                [f"parquet{i:03}".encode() if i % 2 == 0 else None for i in range(n)],
                dtype=pl.Binary,
            ),
            "flba_field": pl.Series([bytes([i]) * 10 for i in range(n)]),
        }
    )


def test_read_uniform_encryption(io_files_path: Path) -> None:
    path = io_files_path / "parquet-encryption" / "uniform_encryption.parquet.encrypted"
    expected = expected_data()

    # TODO: Pass decryption properties with FOOTER_KEY once supported.
    df = pl.read_parquet(path, schema=expected.schema)

    assert_frame_equal(df, expected)
