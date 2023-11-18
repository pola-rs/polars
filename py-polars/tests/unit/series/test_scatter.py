import numpy as np
import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_scatter() -> None:
    s = pl.Series("s", [1, 2, 3])

    # no-op (empty sequences)
    for x in (
        (),
        [],
        pl.Series(),
        pl.Series(dtype=pl.Int8),
        np.array([]),
        np.ndarray(shape=(0, 0)),
    ):
        s.scatter(x, 8)  # type: ignore[arg-type]
        assert s.to_list() == [1, 2, 3]

    # set new values, one index at a time
    s.scatter(0, 8)
    s.scatter([1], None)
    assert s.to_list() == [8, None, 3]

    # set new value at multiple indexes in one go
    s.scatter([0, 2], None)
    assert s.to_list() == [None, None, None]

    # try with different series dtype
    s = pl.Series("s", ["a", "b", "c"])
    s.scatter((1, 2), "x")
    assert s.to_list() == ["a", "x", "x"]
    assert s.scatter([0, 2], 0.12345).to_list() == ["0.12345", "x", "0.12345"]

    # set multiple values values
    s = pl.Series(["z", "z", "z"])
    assert s.scatter([0, 1], ["a", "b"]).to_list() == ["a", "b", "z"]
    s = pl.Series([True, False, True])
    assert s.scatter([0, 1], [False, True]).to_list() == [False, True, True]

    # set negative indices
    a = pl.Series(range(5))
    a[-2] = None
    a[-5] = None
    assert a.to_list() == [None, 1, 2, None, 4]

    with pytest.raises(pl.OutOfBoundsError):
        a[-100] = None


def test_set_at_idx_deprecated() -> None:
    s = pl.Series("s", [1, 2, 3])
    with pytest.deprecated_call():
        result = s.set_at_idx(1, 10)
    expected = pl.Series("s", [1, 10, 3])
    assert_series_equal(result, expected)
