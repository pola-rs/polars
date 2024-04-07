import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal


def test_melt() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})
    for _idv, _vv in (("A", ("B", "C")), (cs.string(), cs.integer())):
        melted_eager = df.melt(id_vars="A", value_vars=["B", "C"])
        assert all(melted_eager["value"] == [1, 3, 5, 2, 4, 6])

        melted_lazy = df.lazy().melt(id_vars="A", value_vars=["B", "C"])
        assert all(melted_lazy.collect()["value"] == [1, 3, 5, 2, 4, 6])

    melted = df.melt(id_vars="A", value_vars="B")
    assert all(melted["value"] == [1, 3, 5])
    n = 3

    for melted in [df.melt(), df.lazy().melt().collect()]:
        assert melted["variable"].to_list() == ["A"] * n + ["B"] * n + ["C"] * n
        assert melted["value"].to_list() == [
            "a",
            "b",
            "c",
            "1",
            "3",
            "5",
            "2",
            "4",
            "6",
        ]

    for melted in [
        df.melt(value_name="foo", variable_name="bar"),
        df.lazy().melt(value_name="foo", variable_name="bar").collect(),
    ]:
        assert melted["bar"].to_list() == ["A"] * n + ["B"] * n + ["C"] * n
        assert melted["foo"].to_list() == [
            "a",
            "b",
            "c",
            "1",
            "3",
            "5",
            "2",
            "4",
            "6",
        ]


def test_melt_duplicates() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})

    with pytest.raises(pl.DuplicateError):
        df.melt(id_vars=["B", "B"])
    with pytest.raises(pl.DuplicateError):
        df.lazy().melt(id_vars=["B", "B"]).collect()

    with pytest.raises(pl.DuplicateError):
        df.melt(id_vars=["A", "B"], value_vars=["C", "A"])
    with pytest.raises(pl.DuplicateError):
        df.lazy().melt(id_vars=["A", "B"], value_vars=["C", "A"]).collect()


def test_melt_missing_columns() -> None:
    df = pl.DataFrame({"A": ["a", "b", "c"], "B": [1, 3, 5], "C": [2, 4, 6]})

    with pytest.raises(pl.ColumnNotFoundError):
        df.melt(id_vars=["A", "D"])
    with pytest.raises(pl.ColumnNotFoundError):
        df.melt(id_vars=["A"], value_vars=["D"])

    with pytest.raises(pl.ColumnNotFoundError):
        df.lazy().melt(id_vars=["A", "D"]).collect()
    with pytest.raises(pl.ColumnNotFoundError):
        df.lazy().melt(id_vars=["A"], value_vars=["D"]).collect()


def test_melt_projection_pd_7747() -> None:
    df = pl.LazyFrame(
        {
            "number": [1, 2, 1, 2, 1],
            "age": [40, 30, 21, 33, 45],
            "weight": [100, 103, 95, 90, 110],
        }
    )
    result = (
        df.with_columns(pl.col("age").alias("wgt"))
        .melt(id_vars="number", value_vars="wgt")
        .select("number", "value")
        .collect()
    )
    expected = pl.DataFrame(
        {
            "number": [1, 2, 1, 2, 1],
            "value": [40, 30, 21, 33, 45],
        }
    )
    assert_frame_equal(result, expected)


# https://github.com/pola-rs/polars/issues/10075
def test_melt_no_value_vars() -> None:
    lf = pl.LazyFrame({"a": [1, 2, 3]})

    result = lf.melt("a")

    expected = pl.LazyFrame(
        schema={"a": pl.Int64, "variable": pl.String, "value": pl.Null}
    )
    assert_frame_equal(result, expected)


# https://github.com/pola-rs/polars/issues/13493
def test_melt_missing_column_13493() -> None:
    with pytest.raises(pl.ColumnNotFoundError):
        (
            pl.DataFrame({"col0": [0], "col1": ["a"], "col2": ["b"]})
            .lazy()
            .melt(id_vars="row_nr")
            .collect()
        )
