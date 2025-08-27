import pytest
from hypothesis import given

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric.strategies import series


def test_merge_sorted_valid_inputs() -> None:
    df0 = pl.LazyFrame(
        {
            "name": ["steve", "elise", "bob"],
            "age": [42, 44, 18],
        },
    ).sort("age")

    df1 = pl.LazyFrame(
        {
            "name": ["anna", "megan", "steve", "thomas"],
            "age": [21, 33, 42, 20],
        },
    ).sort("age")

    out = df0.merge_sorted(df1, key="age").collect()

    expected = pl.DataFrame(
        {
            "name": ["bob", "thomas", "anna", "megan", "steve", "steve", "elise"],
            "age": [18, 20, 21, 33, 42, 42, 44],
        },
    )

    assert_frame_equal(out, expected)


def test_merge_sorted_bad_input_type() -> None:
    a = pl.LazyFrame({"x": [1, 2, 3]})
    b = pl.LazyFrame({"x": [4, 5, 6]})

    with pytest.raises(
        TypeError,
        match="expected `other` .*to be a 'LazyFrame'.* not 'Series'",
    ):
        a.merge_sorted(pl.Series(b.collect()), key="x")  # type: ignore[arg-type]

    with pytest.raises(
        TypeError,
        match="expected `other` .*to be a 'LazyFrame'.* not 'DataFrame'",
    ):
        a.merge_sorted(b.collect(), key="x")  # type: ignore[arg-type]

    class DummyLazyFrameSubclass(pl.LazyFrame):
        pass

    b = DummyLazyFrameSubclass(b.collect())
    a.merge_sorted(b, key="x")


@given(s=series(name="x"))
def test_merge_sorted_parametric(s: pl.Series) -> None:
    lf = s.sort().to_frame().lazy()

    lf = lf.merge_sorted(lf, "x")

    assert_frame_equal(
        lf.collect(engine="in-memory"),
        lf.collect(engine="streaming"),
    )
