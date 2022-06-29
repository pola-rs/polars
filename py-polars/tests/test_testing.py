import pytest
from hypothesis import given, settings
from hypothesis.strategies import sampled_from

import polars as pl
from polars.testing import (
    assert_frame_equal,
    assert_series_equal,
    column,
    columns,
    dataframes,
    series,
    strategy_dtypes,
)

TEMPORAL_DTYPES = [pl.Datetime, pl.Date, pl.Time, pl.Duration]


def test_compare_series_value_mismatch() -> None:
    srs1 = pl.Series([1, 2, 3])
    srs2 = pl.Series([2, 3, 4])
    with pytest.raises(AssertionError, match="Series are different\n\nValue mismatch"):
        assert_series_equal(srs1, srs2)


def test_compare_series_empty_equal() -> None:
    srs1 = pl.Series([])
    srs2 = pl.Series(())
    assert_series_equal(srs1, srs2)


def test_compare_series_nulls_are_equal() -> None:
    srs1 = pl.Series([1, 2, None])
    srs2 = pl.Series([1, 2, None])
    assert_series_equal(srs1, srs2)


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
        assert_series_equal(srs1, srs2)  # type: ignore

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


def test_assert_frame_equal_pass() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    df2 = pl.DataFrame({"a": [1, 2]})
    assert_frame_equal(df1, df2)


def test_assert_frame_equal_types() -> None:
    df1 = pl.DataFrame({"a": [1, 2]})
    srs1 = pl.Series(values=[1, 2], name="a")
    with pytest.raises(AssertionError):
        assert_frame_equal(df1, srs1)  # type: ignore


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


@given(df=dataframes(), lf=dataframes(lazy=True), srs=series())
@settings(max_examples=10)
def test_strategy_classes(df: pl.DataFrame, lf: pl.LazyFrame, srs: pl.Series) -> None:
    assert isinstance(df, pl.DataFrame)
    assert isinstance(lf, pl.LazyFrame)
    assert isinstance(srs, pl.Series)


@given(
    df1=dataframes(cols=5, size=5),
    df2=dataframes(min_cols=10, max_cols=20, min_size=5, max_size=25),
    s1=series(size=5),
    s2=series(min_size=5, max_size=25, name="col"),
)
def test_strategy_shape(
    df1: pl.DataFrame, df2: pl.DataFrame, s1: pl.Series, s2: pl.Series
) -> None:
    assert df1.shape == (5, 5)
    assert df1.columns == ["col0", "col1", "col2", "col3", "col4"]

    assert 10 <= len(df2.columns) <= 20
    assert 5 <= len(df2) <= 25

    assert s1.len() == 5
    assert 5 <= s2.len() <= 25
    assert s1.name == ""
    assert s2.name == "col"


@given(
    lf=dataframes(
        # generate lazyframes with at least one row
        lazy=True,
        min_size=1,
        # test mix & match of bulk-assigned cols with custom cols
        cols=columns(["a", "b"], dtype=pl.UInt8, unique=True),
        include_cols=[
            column("c", dtype=pl.Boolean),
            column("d", strategy=sampled_from(["x", "y", "z"])),
        ],
    )
)
def test_strategy_frame_columns(lf: pl.LazyFrame) -> None:
    assert lf.schema == {"a": pl.UInt8, "b": pl.UInt8, "c": pl.Boolean, "d": pl.Utf8}
    assert lf.columns == ["a", "b", "c", "d"]
    df = lf.collect()

    # confirm uint cols bounds
    uint8_max = (2**8) - 1
    assert df["a"].min() >= 0
    assert df["b"].min() >= 0
    assert df["a"].max() <= uint8_max
    assert df["b"].max() <= uint8_max

    # confirm uint cols uniqueness
    assert df["a"].is_unique().all()
    assert df["b"].is_unique().all()

    # boolean col
    assert all(isinstance(v, bool) for v in df["c"].to_list())

    # string col, entries selected from custom values
    xyz = {"x", "y", "z"}
    assert all(v in xyz for v in df["d"].to_list())


@given(
    df=dataframes(allowed_dtypes=TEMPORAL_DTYPES, max_size=1),
    lf=dataframes(excluded_dtypes=TEMPORAL_DTYPES, max_size=1, lazy=True),
    s1=series(max_size=1),
    s2=series(dtype=pl.Boolean, max_size=1),
    s3=series(allowed_dtypes=TEMPORAL_DTYPES, max_size=1),
    s4=series(excluded_dtypes=TEMPORAL_DTYPES, max_size=1),
)
def test_strategy_dtypes(
    df: pl.DataFrame,
    lf: pl.LazyFrame,
    s1: pl.Series,
    s2: pl.Series,
    s3: pl.Series,
    s4: pl.Series,
) -> None:
    # dataframe, lazyframe
    assert all(tp in TEMPORAL_DTYPES for tp in df.dtypes)
    assert all(tp not in TEMPORAL_DTYPES for tp in lf.dtypes)

    # series
    assert s1.dtype in strategy_dtypes
    assert s2.dtype == pl.Boolean
    assert s3.dtype in TEMPORAL_DTYPES
    assert s4.dtype not in TEMPORAL_DTYPES
