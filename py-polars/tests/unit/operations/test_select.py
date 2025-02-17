import polars as pl
from polars.testing import assert_frame_equal


def test_select_by_col_list(fruits_cars: pl.DataFrame) -> None:
    ldf = fruits_cars.lazy()
    result = ldf.select(pl.col(["A", "B"]).sum())
    expected = pl.LazyFrame({"A": 15, "B": 15})
    assert_frame_equal(result, expected)


def test_select_args_kwargs() -> None:
    ldf = pl.LazyFrame({"foo": [1, 2], "bar": [3, 4], "ham": ["a", "b"]})

    # Single column name
    result = ldf.select("foo")
    expected = pl.LazyFrame({"foo": [1, 2]})
    assert_frame_equal(result, expected)

    # Column names as list
    result = ldf.select(["foo", "bar"])
    expected = pl.LazyFrame({"foo": [1, 2], "bar": [3, 4]})
    assert_frame_equal(result, expected)

    # Column names as positional arguments
    result, expected = ldf.select("foo", "bar", "ham"), ldf
    assert_frame_equal(result, expected)

    # Keyword arguments
    result = ldf.select(oof="foo")
    expected = pl.LazyFrame({"oof": [1, 2]})
    assert_frame_equal(result, expected)

    # Mixed
    result = ldf.select("bar", "foo", oof="foo")
    expected = pl.LazyFrame({"bar": [3, 4], "foo": [1, 2], "oof": [1, 2]})
    assert_frame_equal(result, expected)


def test_select_empty() -> None:
    result = pl.select()
    expected = pl.DataFrame()
    assert_frame_equal(result, expected)


def test_select_none() -> None:
    result = pl.select(None)
    expected = pl.select(pl.lit(None))
    assert_frame_equal(result, expected)


def test_select_none_combined() -> None:
    other = pl.lit(1).alias("one")

    result = pl.select(None, other)
    expected = pl.select(pl.lit(None), other)
    assert_frame_equal(result, expected)

    result = pl.select(other, None)
    expected = pl.select(other, pl.lit(None))
    assert_frame_equal(result, expected)


def test_select_empty_list() -> None:
    result = pl.select([])
    expected = pl.DataFrame()
    assert_frame_equal(result, expected)


def test_select_named_inputs_reserved() -> None:
    result = pl.select(inputs=1.0, structify=pl.lit("x"))
    expected = pl.DataFrame({"inputs": [1.0], "structify": ["x"]})
    assert_frame_equal(result, expected)
