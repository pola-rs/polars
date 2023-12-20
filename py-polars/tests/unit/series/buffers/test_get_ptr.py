import polars as pl


def test_ptr() -> None:
    # not much to test on the ptr value itself.
    s = pl.Series([1, None, 3])

    ptr = s._get_ptr()[2]
    assert isinstance(ptr, int)
    s2 = s.append(pl.Series([1, 2]))

    ptr2 = s2.rechunk()._get_ptr()[2]
    assert ptr != ptr2

    for dtype in pl.FLOAT_DTYPES | pl.INTEGER_DTYPES:
        assert pl.Series([1, 2, 3], dtype=dtype)._s.get_ptr()[2] > 0
