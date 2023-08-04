import pytest

import polars as pl


@pytest.mark.slow()
def test_streaming_groupby_sorted_fast_path_nulls_10273() -> None:
    df = pl.Series(
        name="x",
        values=(
            *(i for i in range(4) for _ in range(100)),
            *(None for _ in range(100)),
        ),
    ).to_frame()

    assert (
        df.set_sorted("x")
        .lazy()
        .groupby("x")
        .agg(pl.count())
        .collect(streaming=True)
        .sort("x")
    ).to_dict(False) == {"x": [None, 0, 1, 2, 3], "count": [100, 100, 100, 100, 100]}
