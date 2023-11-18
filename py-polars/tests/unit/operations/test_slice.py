import polars as pl
from polars.testing import assert_frame_equal, assert_frame_not_equal


def test_tail_union() -> None:
    assert (
        pl.concat(
            [
                pl.LazyFrame({"a": [1, 2]}),
                pl.LazyFrame({"a": [3, 4]}),
                pl.LazyFrame({"a": [5, 6]}),
            ]
        )
        .tail(1)
        .collect()
    ).to_dict(as_series=False) == {"a": [6]}


def test_python_slicing_data_frame() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", "c"]})
    expected = pl.DataFrame({"a": [2, 3], "b": ["b", "c"]})
    for slice_params in (
        [1, 10],  # slice > len(df)
        [1, 2],  # slice == len(df)
        [1],  # optional len
    ):
        assert_frame_equal(df.slice(*slice_params), expected)

    for py_slice in (
        slice(1, 2),
        slice(0, 2, 2),
        slice(3, -3, -1),
        slice(1, None, -2),
        slice(-1, -3, -1),
        slice(-3, None, -3),
    ):
        # confirm frame slice matches python slice
        assert df[py_slice].rows() == df.rows()[py_slice]


def test_python_slicing_series() -> None:
    s = pl.Series(name="a", values=[0, 1, 2, 3, 4, 5], dtype=pl.UInt8)
    for srs_slice, expected in (
        [s.slice(2, 3), [2, 3, 4]],
        [s.slice(4, 1), [4]],
        [s.slice(4, None), [4, 5]],
        [s.slice(3), [3, 4, 5]],
        [s.slice(-2), [4, 5]],
    ):
        assert srs_slice.to_list() == expected  # type: ignore[attr-defined]

    for py_slice in (
        slice(1, 2),
        slice(0, 2, 2),
        slice(3, -3, -1),
        slice(1, None, -2),
        slice(-1, -3, -1),
        slice(-3, None, -3),
    ):
        # confirm series slice matches python slice
        assert s[py_slice].to_list() == s.to_list()[py_slice]


def test_python_slicing_lazy_frame() -> None:
    ldf = pl.LazyFrame({"a": [1, 2, 3, 4], "b": ["a", "b", "c", "d"]})
    expected = pl.LazyFrame({"a": [3, 4], "b": ["c", "d"]})
    for slice_params in (
        [2, 10],  # slice > len(df)
        [2, 4],  # slice == len(df)
        [2],  # optional len
    ):
        assert_frame_equal(ldf.slice(*slice_params), expected)

    for py_slice in (
        slice(1, 2),
        slice(0, 3, 2),
        slice(-3, None),
        slice(None, 2, 2),
        slice(3, None, -1),
        slice(1, None, -2),
    ):
        # confirm frame slice matches python slice
        assert ldf[py_slice].collect().rows() == ldf.collect().rows()[py_slice]

    assert_frame_equal(ldf[::-1], ldf.reverse())
    assert_frame_equal(ldf[::-2], ldf.reverse().gather_every(2))


def test_head_tail_limit() -> None:
    df = pl.DataFrame({"a": range(10), "b": range(10)})

    assert df.head(5).rows() == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
    assert_frame_equal(df.limit(5), df.head(5))
    assert df.tail(5).rows() == [(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
    assert_frame_not_equal(df.head(5), df.tail(5))

    # check if it doesn't fail when out of bounds
    assert df.head(100).height == 10
    assert df.limit(100).height == 10
    assert df.tail(100).height == 10

    # limit is an alias of head
    assert_frame_equal(df.head(5), df.limit(5))

    # negative values
    assert df.head(-7).rows() == [(0, 0), (1, 1), (2, 2)]
    assert len(df.head(-2)) == 8
    assert df.tail(-8).rows() == [(8, 8), (9, 9)]
    assert len(df.tail(-6)) == 4

    # negative values out of bounds
    assert len(df.head(-12)) == 0
    assert len(df.limit(-12)) == 0
    assert len(df.tail(-12)) == 0
