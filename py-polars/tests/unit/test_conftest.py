"""Tests for the testing infrastructure."""

import pytest


@pytest.mark.xfail
def test_memory_usage() -> None:
    pytest.fail(reason="Disabled for now")
    # """The ``memory_usage`` fixture gives somewhat accurate results."""
    # memory_usage = memory_usage_without_pyarrow
    # assert memory_usage.get_current() < 100_000
    # assert memory_usage.get_peak() < 100_000
    #
    # # Memory from Python is tracked:
    # b = b"X" * 1_300_000
    # assert 1_300_000 <= memory_usage.get_current() <= 2_000_000
    # assert 1_300_000 <= memory_usage.get_peak() <= 2_000_000
    # del b
    # assert memory_usage.get_current() <= 500_000
    # assert 1_300_000 <= memory_usage.get_peak() <= 2_000_000
    # memory_usage.reset_tracking()
    # assert memory_usage.get_current() < 100_000
    # assert memory_usage.get_peak() < 100_000
    #
    # # Memory from Polars is tracked:
    # df = pl.DataFrame({"x": pl.arange(0, 1_000_000, eager=True, dtype=pl.Int64)})
    # del df
    # peak_bytes = memory_usage.get_peak()
    # assert 8_000_000 <= peak_bytes < 8_500_000
    #
    # memory_usage.reset_tracking()
    # assert memory_usage.get_peak() < 1_000_000
    #
    # # Memory from NumPy is tracked:
    # arr = np.ones((1_400_000,), dtype=np.uint8)
    # del arr
    # peak = memory_usage.get_peak()
    # assert 1_400_000 < peak < 1_500_000
