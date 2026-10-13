"""Tests for the testing infrastructure."""

import pytest

import polars as pl


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


@pytest.fixture
def _lying_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every `DataFrame.schema` disagree with what `collect_schema()` resolves."""
    monkeypatch.setattr(
        pl.DataFrame, "schema", property(lambda self: pl.Schema({"a": pl.String}))
    )


@pytest.mark.usefixtures("_lying_schema")
def test_lazy_schema_check_fires() -> None:
    with pytest.raises(AssertionError, match="Schemas are different"):
        pl.LazyFrame({"a": [1, 2, 3]}).collect()


@pytest.mark.may_fail_lazy_schema  # reason: deliberate
@pytest.mark.usefixtures("_lying_schema")
def test_lazy_schema_check_marker_suppresses() -> None:
    pl.LazyFrame({"a": [1, 2, 3]}).collect()


def test_lazy_schema_check_leaves_collect_errors_alone() -> None:
    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        pl.LazyFrame({"a": [1]}).select(pl.col("nope")).collect()
