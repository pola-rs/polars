import pytest

import polars as pl

pytestmark = pytest.mark.xdist_group("streaming")


def test_streaming_nested_categorical() -> None:
    assert (
        pl.LazyFrame({"numbers": [1, 1, 2], "cat": [["str"], ["foo"], ["bar"]]})
        .with_columns(pl.col("cat").cast(pl.List(pl.Categorical)))
        .group_by("numbers")
        .agg(pl.col("cat").first())
        .sort("numbers")
    ).collect(streaming=True).to_dict(as_series=False) == {
        "numbers": [1, 2],
        "cat": [["str"], ["bar"]],
    }


def test_streaming_cat_14933() -> None:
    # https://github.com/pola-rs/polars/issues/14933

    df1 = pl.LazyFrame({"a": pl.Series([0], dtype=pl.UInt32)})
    df2 = pl.LazyFrame(
        [
            pl.Series("a", [0, 1], dtype=pl.UInt32),
            pl.Series("l", [None, None], dtype=pl.Categorical(ordering="physical")),
        ]
    )
    result = df1.join(df2, on="a", how="left")
    expected = {"a": [0], "l": [None]}
    assert result.collect(streaming=True).to_dict(as_series=False) == expected
