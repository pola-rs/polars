"""Tests for the testing infrastructure."""

import pyarrow as pa

import polars as pl


def test_memory_usage(memory_usage):
    """
    The ``memory_usage`` fixture gives somewhat accurate results.
    """
    # Memory from Python is tracked:
    b = b"X" * 1_300_000
    del b
    peak_bytes = memory_usage.get_peak()
    assert 1_300_000 <= peak_bytes <= 2_000_000
    memory_usage.reset_tracking()
    assert memory_usage.get_peak() < 1_000_000

    # Memory from Polars is tracked:
    df = pl.DataFrame({"x": pl.arange(0, 1_000_000, eager=True, dtype=pl.Int64)})
    del df
    peak_bytes = memory_usage.get_peak()
    assert 8_000_000 <= peak_bytes < 8_500_000

    memory_usage.reset_tracking()
    assert memory_usage.get_peak() < 1_000_000

    # Memory from pyarrow is tracked:
    b = b"X" * 1_300_000
    old_peak = memory_usage.get_peak()
    table = pa.Table.from_pylist([{"value": b}])
    del table
    del b
    new_peak = memory_usage.get_peak()
    assert new_peak - old_peak >= 1_300_000
