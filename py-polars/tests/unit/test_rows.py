import pytest

import polars as pl
from polars.exceptions import NoRowsReturned, TooManyRowsReturned


def test_iterrows() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [None, False, None]})

    it = df.iterrows()
    assert next(it) == (1, None)
    assert next(it) == (2, False)
    assert next(it) == (3, None)
    with pytest.raises(StopIteration):
        next(it)


def test_row_tuple() -> None:
    df = pl.DataFrame({"a": ["foo", "bar", "2"], "b": [1, 2, 3], "c": [1.0, 2.0, 3.0]})

    # return row by index
    assert df.row(0) == ("foo", 1, 1.0)
    assert df.row(1) == ("bar", 2, 2.0)
    assert df.row(-1) == ("2", 3, 3.0)

    # return row by predicate
    assert df.row(by_predicate=pl.col("a") == "bar") == ("bar", 2, 2.0)
    assert df.row(by_predicate=pl.col("b").is_in([2, 4, 6])) == ("bar", 2, 2.0)

    # expected error conditions
    with pytest.raises(TooManyRowsReturned):
        df.row(by_predicate=pl.col("b").is_in([1, 3, 5]))

    with pytest.raises(NoRowsReturned):
        df.row(by_predicate=pl.col("a") == "???")

    # cannot set both 'index' and 'by_predicate'
    with pytest.raises(ValueError):
        df.row(0, by_predicate=pl.col("a") == "bar")

    # must call 'by_predicate' by keyword
    with pytest.raises(TypeError):
        df.row(None, pl.col("a") == "bar")  # type: ignore[misc]

    # cannot pass predicate into 'index'
    with pytest.raises(TypeError):
        df.row(pl.col("a") == "bar")  # type: ignore[arg-type]

    # at least one of 'index' and 'by_predicate' must be set
    with pytest.raises(ValueError):
        df.row()


def test_rows() -> None:
    df = pl.DataFrame({"a": [1, 2], "b": [1, 2]})
    assert df.rows() == [(1, 1), (2, 2)]
    assert df.reverse().rows() == [(2, 2), (1, 1)]
