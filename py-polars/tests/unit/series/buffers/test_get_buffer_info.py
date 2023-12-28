import pytest

import polars as pl


def test_get_buffer_info() -> None:
    # not much to test on the ptr value itself.
    s = pl.Series([1, None, 3])

    ptr = s._get_buffer_info()[0]
    assert isinstance(ptr, int)
    s2 = s.append(pl.Series([1, 2]))

    ptr2 = s2.rechunk()._get_buffer_info()[0]
    assert ptr != ptr2

    for dtype in list(pl.FLOAT_DTYPES) + list(pl.INTEGER_DTYPES):
        assert pl.Series([1, 2, 3], dtype=dtype)._s._get_buffer_info()[0] > 0


def test_get_buffer_info_chunked() -> None:
    s1 = pl.Series([1, 2])
    s2 = pl.Series([3, 4])
    s = pl.concat([s1, s2], rechunk=False)

    with pytest.raises(pl.ComputeError):
        s._get_buffer_info()
