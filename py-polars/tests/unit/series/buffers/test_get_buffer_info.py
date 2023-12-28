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
