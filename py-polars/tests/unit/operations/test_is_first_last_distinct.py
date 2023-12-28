import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_is_first_distinct() -> None:
    lf = pl.LazyFrame({"a": [4, 1, 4]})
    result = lf.select(pl.col("a").is_first_distinct()).collect()["a"]
    expected = pl.Series("a", [True, True, False])
    assert_series_equal(result, expected)


def test_is_first_distinct_bool_bit_chunk_index_calc() -> None:
    # The fast path activates on sizes >=64 and processes in chunks of 64-bits.
    # It calculates the indexes using the bit counts, which needs to be from the
    # correct side.
    assert pl.arange(0, 64, eager=True).filter(
        pl.Series([True] + 63 * [False]).is_first_distinct()
    ).to_list() == [0, 1]

    assert pl.arange(0, 64, eager=True).filter(
        pl.Series([False] + 63 * [True]).is_first_distinct()
    ).to_list() == [0, 1]

    assert pl.arange(0, 64, eager=True).filter(
        pl.Series(2 * [True] + 2 * [False] + 60 * [None]).is_first_distinct()
    ).to_list() == [0, 2, 4]

    assert pl.arange(0, 64, eager=True).filter(
        pl.Series(2 * [False] + 2 * [None] + 60 * [True]).is_first_distinct()
    ).to_list() == [0, 2, 4]


def test_is_first_distinct_struct() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3, 2, None, 2, 1], "b": [0, 2, 3, 2, None, 2, 0]})
    result = lf.select(pl.struct("a", "b").is_first_distinct())
    expected = pl.LazyFrame({"a": [True, True, True, False, True, False, False]})
    assert_frame_equal(result, expected)


def test_is_first_distinct_list() -> None:
    lf = pl.LazyFrame({"a": [[1, 2], [3], [1, 2], [4, 5], [4, 5]]})
    result = lf.select(pl.col("a").is_first_distinct())
    expected = pl.LazyFrame({"a": [True, True, False, True, False]})
    assert_frame_equal(result, expected)


def test_is_first_distinct_various() -> None:
    # numeric
    s = pl.Series([1, 1, None, 2, None, 3, 3])
    expected = [True, False, True, True, False, True, False]
    assert s.is_first_distinct().to_list() == expected
    # str
    s = pl.Series(["x", "x", None, "y", None, "z", "z"])
    expected = [True, False, True, True, False, True, False]
    assert s.is_first_distinct().to_list() == expected
    # boolean
    s = pl.Series([True, True, None, False, None, False, False])
    expected = [True, False, True, True, False, False, False]
    assert s.is_first_distinct().to_list() == expected
    # struct
    s = pl.Series(
        [
            {"x": 1, "y": 2},
            {"x": 1, "y": 2},
            None,
            {"x": 2, "y": 1},
            None,
            {"x": 3, "y": 2},
            {"x": 3, "y": 2},
        ]
    )
    expected = [True, False, True, True, False, True, False]
    assert s.is_first_distinct().to_list() == expected
    # list
    s = pl.Series([[1, 2], [1, 2], None, [2, 3], None, [3, 4], [3, 4]])
    expected = [True, False, True, True, False, True, False]
    assert s.is_first_distinct().to_list() == expected


def test_is_last_distinct() -> None:
    # numeric
    s = pl.Series([1, 1, None, 2, None, 3, 3])
    expected = [False, True, False, True, True, False, True]
    assert s.is_last_distinct().to_list() == expected
    # str
    s = pl.Series(["x", "x", None, "y", None, "z", "z"])
    expected = [False, True, False, True, True, False, True]
    assert s.is_last_distinct().to_list() == expected
    # boolean
    s = pl.Series([True, True, None, False, None, False, False])
    expected = [False, True, False, False, True, False, True]
    assert s.is_last_distinct().to_list() == expected
    # struct
    s = pl.Series(
        [
            {"x": 1, "y": 2},
            {"x": 1, "y": 2},
            None,
            {"x": 2, "y": 1},
            None,
            {"x": 3, "y": 2},
            {"x": 3, "y": 2},
        ]
    )
    expected = [False, True, False, True, True, False, True]
    assert s.is_last_distinct().to_list() == expected
    # list
    s = pl.Series([[1, 2], [1, 2], None, [2, 3], None, [3, 4], [3, 4]])
    expected = [False, True, False, True, True, False, True]
    assert s.is_last_distinct().to_list() == expected


@pytest.mark.parametrize("dtypes", [pl.Int32, pl.String, pl.Boolean, pl.List(pl.Int32)])
def test_is_first_last_distinct_all_null(dtypes: pl.PolarsDataType) -> None:
    s = pl.Series([None, None, None], dtype=dtypes)
    assert s.is_first_distinct().to_list() == [True, False, False]
    assert s.is_last_distinct().to_list() == [False, False, True]


def test_is_first_last_deprecated() -> None:
    with pytest.deprecated_call():
        pl.col("a").is_first()
    with pytest.deprecated_call():
        pl.col("a").is_last()
