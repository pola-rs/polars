import numpy as np

import polars as pl


def test_arr_min_max() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(width=2, inner=pl.Int64))
    assert s.arr.max().to_list() == [2, 4]
    assert s.arr.min().to_list() == [1, 3]


def test_arr_sum() -> None:
    s = pl.Series("a", [[1, 2], [4, 3]], dtype=pl.Array(width=2, inner=pl.Int64))
    assert s.arr.sum().to_list() == [3, 7]


def test_array_to_numpy() -> None:
    s = pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(width=2, inner=pl.Int64))
    assert (s.to_numpy() == np.array([[1, 2], [3, 4], [5, 6]])).all()
