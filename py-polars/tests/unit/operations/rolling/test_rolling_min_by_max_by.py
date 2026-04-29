"""Tests for rolling min_by / max_by optimization (O(n) via monotonic deque)."""

import polars as pl
from polars.testing import assert_frame_equal


def test_rolling_min_by_basic() -> None:
    """Basic rolling min_by: select the value column element whose by column is minimal."""
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
    # Window contents (idx-based, period=3i means 3 rows ending at current):
    # idx=1: by=[5]         -> min_by=5  -> values=10
    # idx=2: by=[5,1]       -> min_by=1  -> values=20
    # idx=3: by=[5,1,4]     -> min_by=1  -> values=20
    # idx=4: by=[1,4,2]     -> min_by=1  -> values=20
    # idx=5: by=[4,2,3]     -> min_by=2  -> values=40
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
    # idx=1: by=[5]         -> max_by=5  -> values=10
    # idx=2: by=[5,1]       -> max_by=5  -> values=10
    # idx=3: by=[5,1,4]     -> max_by=5  -> values=10
    # idx=4: by=[1,4,2]     -> max_by=4  -> values=30
    # idx=5: by=[4,2,3]     -> max_by=4  -> values=30
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
    # idx=1: by=[7]             -> min=7 -> values=10
    # idx=2: by=[7,3]           -> min=3 -> values=20
    # idx=3: by=[7,3,6]         -> min=3 -> values=20
    # idx=4: by=[7,3,6,1]       -> min=1 -> values=40
    # idx=5: by=[7,3,6,1,5]     -> min=1 -> values=40
    # idx=6: by=[3,6,1,5,2]     -> min=1 -> values=40
    # idx=7: by=[6,1,5,2,4]     -> min=1 -> values=40
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
    # idx=1: by=[5]         -> min=5    -> values=10
    # idx=2: by=[5,null]    -> min=5    -> values=10
    # idx=3: by=[5,null,3]  -> min=3    -> values=30
    # idx=4: by=[null,3,1]  -> min=1    -> values=40
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

    # Rolling uses the optimized path
    result_rolling = df.rolling(index_column="idx", period="5i").agg(
        pl.col("values").min_by("by")
    )

    # Naive: manual construction
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
    # idx=1: by=[5]       -> min=5 -> values=10
    # idx=2: by=[5,1]     -> min=1 -> values=null
    # idx=3: by=[5,1,3]   -> min=1 -> values=null
    # idx=4: by=[1,3,2]   -> min=1 -> values=null
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "values": [10, None, None, None],
        }
    )
    assert_frame_equal(result, expected)


def test_rolling_min_by_duplicate_by_values() -> None:
    """Duplicate values in by column: should pick first occurrence (same as slow path)."""
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
    # idx=1: by=[3]       -> min=3 -> values=10
    # idx=2: by=[3,1]     -> min=1 -> values=20
    # idx=3: by=[3,1,1]   -> min=1 -> values=20 (first occurrence)
    # idx=4: by=[1,1,2]   -> min=1 -> values=20 (first occurrence)
    # idx=5: by=[1,2,1]   -> min=1 -> values=30 (first occurrence in window)
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4, 5],
            "values": [10, 20, 20, 20, 30],
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
    # idx=1: by=[1.5]          -> min=1.5 -> values=10
    # idx=2: by=[1.5,0.5]      -> min=0.5 -> values=20
    # idx=3: by=[1.5,0.5,2.5]  -> min=0.5 -> values=20
    # idx=4: by=[0.5,2.5,0.1]  -> min=0.1 -> values=40
    expected = pl.DataFrame(
        {
            "idx": [1, 2, 3, 4],
            "values": [10, 20, 20, 40],
        }
    )
    assert_frame_equal(result, expected)
