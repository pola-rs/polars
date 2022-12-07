from __future__ import annotations

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_compare_series_value_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.Series([2, 3, 4])
    with pytest.raises(AssertionError, match="Series are different\n\nValue mismatch"):
        assert_series_equal(srs1, srs2)


def test_compare_series_empty_equal() -> None:
    srs1 = pl.Series([])
    srs2 = pl.Series(())
    assert_series_equal(srs1, srs2)


def test_compare_series_nans_assert_equal() -> None:
    # note: NaN values do not _compare_ equal, but should _assert_ equal (by default)
    nan = float("NaN")

    srs1 = pl.Series([1.0, 2.0, nan, 4.0, None, 6.0])
    srs2 = pl.Series([1.0, nan, 3.0, 4.0, None, 6.0])
    srs3 = pl.Series([1.0, 2.0, 3.0, 4.0, None, 6.0])

    for srs in (srs1, srs2, srs3):
        assert_series_equal(srs, srs)
        assert_series_equal(srs, srs, check_exact=True)

    with pytest.raises(AssertionError):
        assert_series_equal(srs1, srs1, nans_compare_equal=False)
    with pytest.raises(AssertionError):
        assert_series_equal(srs1, srs1, nans_compare_equal=False, check_exact=True)

    for check_exact, nans_equal in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        if check_exact:
            check_msg = "Exact value mismatch"
        else:
            check_msg = f"Value mismatch.*nans_compare_equal={nans_equal}"

        with pytest.raises(AssertionError, match=check_msg):
            assert_series_equal(
                srs1, srs2, check_exact=check_exact, nans_compare_equal=nans_equal
            )
        with pytest.raises(AssertionError, match=check_msg):
            assert_series_equal(
                srs1, srs3, check_exact=check_exact, nans_compare_equal=nans_equal
            )

    srs4 = pl.Series([1.0, 2.0, 3.0, 4.0, None, 6.0])
    srs5 = pl.Series([1.0, 2.0, 3.0, 4.0, nan, 6.0])
    srs6 = pl.Series([1, 2, 3, 4, None, 6])

    assert_series_equal(srs4, srs6, check_dtype=False)
    with pytest.raises(AssertionError):
        assert_series_equal(srs5, srs6, check_dtype=False)


def test_compare_series_nulls() -> None:
    srs1 = pl.Series([1, 2, None])
    srs2 = pl.Series([1, 2, None])
    assert_series_equal(srs1, srs2)

    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.Series([1, None, None])
    with pytest.raises(AssertionError, match="Value mismatch"):
        assert_series_equal(srs1, srs2)
    with pytest.raises(AssertionError, match="Exact value mismatch"):
        assert_series_equal(srs1, srs2, check_exact=True)


def test_compare_series_value_mismatch_string() -> None:
    srs1 = pl.Series(["hello", "no"])
    srs2 = pl.Series(["hello", "yes"])
    with pytest.raises(
        AssertionError, match="Series are different\n\nExact value mismatch"
    ):
        assert_series_equal(srs1, srs2)


def test_compare_series_type_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.DataFrame({"col1": [2, 3, 4]})
    with pytest.raises(AssertionError, match="Series are different\n\nType mismatch"):
        assert_series_equal(srs1, srs2)  # type: ignore[arg-type]

    srs3 = pl.Series([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match="Series are different\n\nDtype mismatch"):
        assert_series_equal(srs1, srs3)


def test_compare_series_name_mismatch() -> None:
    srs1 = pl.Series(values=[1, 2, 3], name="srs1")
    srs2 = pl.Series(values=[1, 2, 3], name="srs2")
    with pytest.raises(AssertionError, match="Series are different\n\nName mismatch"):
        assert_series_equal(srs1, srs2)


def test_compare_series_shape_mismatch() -> None:
    srs1 = pl.Series(values=[1, 2, 3, 4], name="srs1")
    srs2 = pl.Series(values=[1, 2, 3], name="srs2")
    with pytest.raises(AssertionError, match="Series are different\n\nShape mismatch"):
        assert_series_equal(srs1, srs2)


def test_compare_series_value_exact_mismatch() -> None:
    srs1 = pl.Series([1.0, 2.0, 3.0])
    srs2 = pl.Series([1.0, 2.0 + 1e-7, 3.0])
    with pytest.raises(
        AssertionError, match="Series are different\n\nExact value mismatch"
    ):
        assert_series_equal(srs1, srs2, check_exact=True)


def test_compare_frame_equal_nans() -> None:
    # NaN values do not _compare_ equal, but should _assert_ as equal here
    nan = float("NaN")

    df1 = pl.DataFrame(
        data={"x": [1.0, nan], "y": [nan, 2.0]},
        columns=[("x", pl.Float32), ("y", pl.Float64)],
    )
    assert_frame_equal(df1, df1, check_exact=True)

    df2 = pl.DataFrame(
        data={"x": [1.0, nan], "y": [None, 2.0]},
        columns=[("x", pl.Float32), ("y", pl.Float64)],
    )
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2, check_exact=True)


def test_assert_frame_equal_pass() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(df1, df2)


def test_assert_frame_equal_types() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    srs1 = pl.Series(values=[1, 2], name="a")
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, srs1)  # type: ignore[arg-type]


def test_assert_frame_equal_length_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [1, 2]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch2() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch_order() -> None:
    df1 = pl.DataFrame({"b": [3, 4], "a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, df2)
    assert_frame_equal(df1, df2, check_column_names=False)


def test_assert_series_equal_int_overflow() -> None:
    # internally may call 'abs' if not check_exact, which can overflow on signed int
    s0 = pl.Series([-128], dtype=pl.Int8)
    s1 = pl.Series([0, -128], dtype=pl.Int8)
    s2 = pl.Series([1, -128], dtype=pl.Int8)

    for check_exact in (True, False):
        assert_series_equal(s0, s0, check_exact=check_exact)
        with pytest.raises(AssertionError):
            assert_series_equal(s1, s2, check_exact=check_exact)
