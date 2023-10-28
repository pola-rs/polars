import numpy as np

import polars as pl
from polars.testing import assert_frame_equal


def test_arr_min_max() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(inner=pl.Int64, width=2))
    assert s.arr.max().to_list() == [2, 4]
    assert s.arr.min().to_list() == [1, 3]


def test_arr_sum() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(inner=pl.Int64, width=2))
    assert s.arr.sum().to_list() == [3, 7]


def test_arr_unique() -> None:
    df = pl.DataFrame(
        {"a": pl.Series("a", [[1, 1], [4, 3]], dtype=pl.Array(inner=pl.Int64, width=2))}
    )

    out = df.select(pl.col("a").arr.unique(maintain_order=True))
    expected = pl.DataFrame({"a": [[1], [4, 3]]})
    assert_frame_equal(out, expected)


def test_array_to_numpy() -> None:
    s = pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(inner=pl.Int64, width=2))
    assert (s.to_numpy() == np.array([[1, 2], [3, 4], [5, 6]])).all()
