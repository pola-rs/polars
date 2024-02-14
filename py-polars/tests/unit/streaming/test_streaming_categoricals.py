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
