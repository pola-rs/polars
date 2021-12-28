import pytest

import polars as pl


def test_compare_series_value_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.Series([2, 3, 4])
    with pytest.raises(AssertionError, match="Series are different\n\nValue mismatch"):
        pl.testing.assert_series_equal(srs1, srs2)


def test_compare_series_nulls_are_equal() -> None:
    srs1 = pl.Series([1, 2, None])
    srs2 = pl.Series([1, 2, None])
    pl.testing.assert_series_equal(srs1, srs2)


def test_compare_series_value_mismatch_string() -> None:
    srs1 = pl.Series(["hello", "no"])
    srs2 = pl.Series(["hello", "yes"])
    with pytest.raises(
        AssertionError, match="Series are different\n\nExact value mismatch"
    ):
        pl.testing.assert_series_equal(srs1, srs2)


def test_compare_series_type_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.DataFrame({"col1": [2, 3, 4]})
    with pytest.raises(AssertionError, match="Series are different\n\nType mismatch"):
        pl.testing.assert_series_equal(srs1, srs2)  # type: ignore

    srs3 = pl.Series([1.0, 2.0, 3.0])
    with pytest.raises(AssertionError, match="Series are different\n\nDtype mismatch"):
        pl.testing.assert_series_equal(srs1, srs3)


def test_compare_series_name_mismatch() -> None:
    srs1 = pl.Series(values=[1, 2, 3], name="srs1")
    srs2 = pl.Series(values=[1, 2, 3], name="srs2")
    with pytest.raises(AssertionError, match="Series are different\n\nName mismatch"):
        pl.testing.assert_series_equal(srs1, srs2)


def test_compare_series_shape_mismatch() -> None:
    srs1 = pl.Series(values=[1, 2, 3, 4], name="srs1")
    srs2 = pl.Series(values=[1, 2, 3], name="srs2")
    with pytest.raises(AssertionError, match="Series are different\n\nShape mismatch"):
        pl.testing.assert_series_equal(srs1, srs2)


def test_compare_series_value_exact_mismatch() -> None:
    srs1 = pl.Series([1.0, 2.0, 3.0])
    srs2 = pl.Series([1.0, 2.0 + 1e-7, 3.0])
    with pytest.raises(
        AssertionError, match="Series are different\n\nExact value mismatch"
    ):
        pl.testing.assert_series_equal(srs1, srs2, check_exact=True)


def test_assert_frame_equal_pass() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2]})
    pl.testing.assert_frame_equal(df1, df2)


def test_assert_frame_equal_types() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    srs1 = pl.Series(values=[1, 2], name="a")
    with pytest.raises(AssertionError):
        pl.testing.assert_frame_equal(df1, srs1)  # type: ignore


def test_assert_frame_equal_length_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(AssertionError):
        pl.testing.assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"b": [1, 2]})
    with pytest.raises(AssertionError):
        pl.testing.assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch2() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(AssertionError):
        pl.testing.assert_frame_equal(df1, df2)


def test_assert_frame_equal_column_mismatch_order() -> None:
    df1 = pl.DataFrame({"b": [3, 4], "a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(AssertionError):
        pl.testing.assert_frame_equal(df1, df2)
