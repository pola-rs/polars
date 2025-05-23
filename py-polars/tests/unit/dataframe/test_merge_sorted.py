import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_merge_sorted_valid_inputs() -> None:
    df0 = pl.DataFrame(
        {
            "name": ["steve", "elise", "bob"],
            "age": [42, 44, 18],
        },
    ).sort("age")

    df1 = pl.DataFrame(
        {
            "name": ["anna", "megan", "steve", "thomas"],
            "age": [21, 33, 42, 20],
        },
    ).sort("age")

    out = df0.merge_sorted(df1, key="age")

    expected = pl.DataFrame(
        {
            "name": ["bob", "thomas", "anna", "megan", "steve", "steve", "elise"],
            "age": [18, 20, 21, 33, 42, 42, 44],
        },
    )

    assert_frame_equal(out, expected)


def test_merge_sorted_bad_input_type() -> None:
    a = pl.DataFrame({"x": [1, 2, 3]})
    b = pl.DataFrame({"x": [4, 5, 6]})

    with pytest.raises(
        TypeError,
        match="expected `other` .*to be a 'DataFrame'.* not 'Series'",
    ):
        a.merge_sorted(pl.Series(b), key="x")  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="expected `other` .*to be a 'DataFrame'.* not 'LazyFrame'",
    ):
        a.merge_sorted(b.lazy(), key="x")  # type: ignore[arg-type]

    class DummyDataFrameSubclass(pl.DataFrame):
        pass

    b = DummyDataFrameSubclass(b)

    a.merge_sorted(b, key="x")
