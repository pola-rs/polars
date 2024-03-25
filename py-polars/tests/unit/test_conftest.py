"""Tests for the testing infrastructure."""

import numpy as np

import polars as pl
from tests.unit.conftest import MemoryUsage


def test_memory_usage(memory_usage_without_pyarrow: MemoryUsage) -> None:
    """The ``memory_usage`` fixture gives somewhat accurate results."""
    memory_usage = memory_usage_without_pyarrow

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

    # Memory from NumPy is tracked:
    arr = np.ones((1_400_000,), dtype=np.uint8)
    del arr
    peak = memory_usage.get_peak()
    assert 1_400_000 < peak < 1_500_000
