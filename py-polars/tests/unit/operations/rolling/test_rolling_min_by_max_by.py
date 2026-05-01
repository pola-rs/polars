"""Tests for rolling min_by / max_by optimization (O(n) via monotonic deque)."""

import polars as pl
from polars.testing import assert_frame_equal


def test_rolling_min_by_basic() -> None:
    """Basic rolling min_by: select the value whose by is minimal."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 30, 40, 50],
            "by": [5, 1, 4, 2, 3],
        }
    )
    result = df.rolling(index_column="idx", period="3i").agg(
        pl.col("values").min_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 20, 20, 40],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_max_by_basic() -> None:
    """Basic rolling max_by."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 30, 40, 50],
            "by": [5, 1, 4, 2, 3],
        }
    )
    result = df.rolling(index_column="idx", period="3i").agg(
        pl.col("values").max_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 10, 10, 30, 30],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_min_by_dynamic_window() -> None:
    """Rolling min_by with period='5i' (larger window)."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5, 6, 7],
            "values": [10, 20, 30, 40, 50, 60, 70],
            "by": [7, 3, 6, 1, 5, 2, 4],
        }
    )
    result = df.rolling(index_column="idx", period="5i").agg(
        pl.col("values").min_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5, 6, 7],
            "values": [10, 20, 20, 40, 40, 40, 40],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_min_by_with_null_in_by() -> None:
    """Null in the by column should be excluded from argmin."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "values": [10, 20, 30, 40],
            "by": [5, None, 3, 1],
        }
    )
    result = df.rolling(index_column="idx", period="3i").agg(
        pl.col("values").min_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "values": [10, 10, 30, 40],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_min_by_matches_naive() -> None:
    """Verify the optimized path matches naive (group_by) path byte-for-byte."""
    import random

    random.seed(42)
    n = 100
    df = pl.DataFrame(
        {
            "idx": list(range(n)),
            "values": [random.randint(0, 1000) for _ in range(n)],
            "by": [random.randint(0, 1000) for _ in range(n)],
        }
    )

    result_rolling = df.rolling(index_column="idx", period="5i").agg(
        pl.col("values").min_by("by")
    )

    naive_values = []
    for i in range(n):
        start = max(0, i - 4)
        window_by = df["by"][start : i + 1].to_list()
        window_vals = df["values"][start : i + 1].to_list()
        min_idx = window_by.index(min(window_by))
        naive_values.append(window_vals[min_idx])

    expected = pl.DataFrame(
        {
            "idx": list(range(n)),
            "values": naive_values,
        }
    )
    assert_frame_equal(result_rolling, expected)


def test_rolling_max_by_matches_naive() -> None:
    """Verify max_by optimized path matches naive."""
    import random

    random.seed(123)
    n = 50
    df = pl.DataFrame(
        {
            "idx": list(range(n)),
            "values": [random.randint(0, 1000) for _ in range(n)],
            "by": [random.randint(0, 1000) for _ in range(n)],
        }
    )

    result_rolling = df.rolling(index_column="idx", period="7i").agg(
        pl.col("values").max_by("by")
    )

    naive_values = []
    for i in range(n):
        start = max(0, i - 6)
        window_by = df["by"][start : i + 1].to_list()
        window_vals = df["values"][start : i + 1].to_list()
        max_idx = window_by.index(max(window_by))
        naive_values.append(window_vals[max_idx])

    expected = pl.DataFrame(
        {
            "idx": list(range(n)),
            "values": naive_values,
        }
    )
    assert_frame_equal(result_rolling, expected)


def test_rolling_min_by_with_null_in_values() -> None:
    """Null in the values column: the gathered value should be null if selected."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "values": [10, None, 30, 40],
            "by": [5, 1, 3, 2],
        }
    )
    result = df.rolling(index_column="idx", period="3i").agg(
        pl.col("values").min_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "values": [10, None, None, None],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_min_by_duplicate_by_values() -> None:
    """Duplicate by values: pick first occurrence (same as slow path)."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 30, 40, 50],
            "by": [3, 1, 1, 2, 1],
        }
    )
    result = df.rolling(index_column="idx", period="3i").agg(
        pl.col("values").min_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 20, 20, 30],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_min_by_enum_by_column() -> None:
    """Enum by column uses fast path correctly."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 30, 40, 50],
            "by": pl.Series(["c", "a", "b", "a", "c"]).cast(pl.Enum(["a", "b", "c"])),
        }
    )
    result = df.rolling(index_column="idx", period="3i").agg(
        pl.col("values").min_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 20, 20, 40],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_min_by_categorical_by_column() -> None:
    """Categorical by column falls back to slow path."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 30, 40, 50],
            "by": pl.Series(["c", "a", "b", "a", "c"]).cast(pl.Categorical),
        }
    )
    result = df.rolling(index_column="idx", period="3i").agg(
        pl.col("values").min_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 10, 10, 20, 50],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_min_by_float_by_column() -> None:
    """Rolling min_by with float by column."""
    df = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "values": [10, 20, 30, 40],
            "by": [1.5, 0.5, 2.5, 0.1],
        }
    )
    result = df.rolling(index_column="idx", period="3i").agg(
        pl.col("values").min_by("by")
    )
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "values": [10, 20, 20, 40],
        }
    )
    assert_frame_equal(result, expected)
