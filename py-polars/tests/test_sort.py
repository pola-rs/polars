from typing import List, Union

import pytest

import polars as pl


def test_sort_dates_multiples() -> None:
    df = pl.DataFrame(
        [
            pl.Series(
                "date",
                [
                    "2021-01-01 00:00:00",
                    "2021-01-01 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-02 00:00:00",
                    "2021-01-03 00:00:00",
                ],
            ).str.strptime(pl.Datetime, "%Y-%m-%d %T"),
            pl.Series("values", [5, 4, 3, 2, 1]),
        ]
    )

    expected = [4, 5, 2, 3, 1]

    # datetime
    out: pl.DataFrame = df.sort(["date", "values"])
    assert out["values"].to_list() == expected

    # Date
    out = df.with_column(pl.col("date").cast(pl.Date)).sort(["date", "values"])
    assert out["values"].to_list() == expected


def test_sort_by() -> None:
    df = pl.DataFrame(
        {"a": [1, 2, 3, 4, 5], "b": [1, 1, 1, 2, 2], "c": [2, 3, 1, 2, 1]}
    )

    by: List[Union[pl.Expr, str]]
    for by in [["b", "c"], [pl.col("b"), "c"]]:  # type: ignore
        out = df.select([pl.col("a").sort_by(by)])
        assert out["a"].to_list() == [3, 1, 2, 5, 4]

    out = df.select([pl.col("a").sort_by(by, reverse=[False])])
    assert out["a"].to_list() == [3, 1, 2, 5, 4]

    out = df.select([pl.col("a").sort_by(by, reverse=[True])])
    assert out["a"].to_list() == [4, 5, 2, 1, 3]

    out = df.select([pl.col("a").sort_by(by, reverse=[True, False])])
    assert out["a"].to_list() == [5, 4, 3, 1, 2]

    # by can also be a single column
    out = df.select([pl.col("a").sort_by("b", reverse=[False])])
    assert out["a"].to_list() == [1, 2, 3, 4, 5]


def test_sort_in_place() -> None:
    df = pl.DataFrame({"a": [1, 3, 2, 4, 5]})
    ret = df.sort("a", in_place=True)
    result = df["a"].to_list()
    expected = [1, 2, 3, 4, 5]
    assert result == expected
    assert ret is None


def test_sort_by_exprs() -> None:
    # make sure that the expression does not overwrite columns in the dataframe
    df = pl.DataFrame({"a": [1, 2, -1, -2]})
    out = df.sort(pl.col("a").abs()).to_series()

    assert out.to_list() == [1, -1, 2, -2]


def test_argsort_nulls() -> None:
    a = pl.Series("a", [1.0, 2.0, 3.0, None, None])
    assert a.argsort(nulls_last=True).to_list() == [0, 1, 2, 4, 3]
    assert a.argsort(nulls_last=False).to_list() == [3, 4, 0, 1, 2]

    assert a.to_frame().sort(by="a", nulls_last=False).to_series().to_list() == [
        None,
        None,
        1.0,
        2.0,
        3.0,
    ]
    assert a.to_frame().sort(by="a", nulls_last=True).to_series().to_list() == [
        1.0,
        2.0,
        3.0,
        None,
        None,
    ]
    with pytest.raises(ValueError):
        a.to_frame().sort(by=["a", "b"], nulls_last=True)


def test_argsort_window_functions() -> None:
    df = pl.DataFrame({"Id": [1, 1, 2, 2, 3, 3], "Age": [1, 2, 3, 4, 5, 6]})
    out = df.select(
        [
            pl.col("Age").arg_sort().over("Id").alias("arg_sort"),
            pl.argsort_by("Age").over("Id").alias("argsort_by"),
        ]
    )

    assert (
        out["arg_sort"].to_list() == out["argsort_by"].to_list() == [0, 1, 0, 1, 0, 1]
    )
