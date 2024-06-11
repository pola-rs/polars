from datetime import datetime

import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_equals() -> None:
    s1 = pl.Series("a", [1.0, 2.0, None], pl.Float64)
    s2 = pl.Series("a", [1, 2, None], pl.Int64)

    assert s1.equals(s2) is True
    assert s1.equals(s2, check_dtypes=True) is False
    assert s1.equals(s2, null_equal=False) is False

    df = pl.DataFrame(
        {"dtm": [datetime(2222, 2, 22, 22, 22, 22)]},
        schema_overrides={"dtm": pl.Datetime(time_zone="UTC")},
    ).with_columns(
        s3=pl.col("dtm").dt.convert_time_zone("Europe/London"),
        s4=pl.col("dtm").dt.convert_time_zone("Asia/Tokyo"),
    )
    s3 = df["s3"].rename("b")
    s4 = df["s4"].rename("b")

    assert s3.equals(s4) is False
    assert s3.equals(s4, check_dtypes=True) is False
    assert s3.equals(s4, null_equal=False) is False
    assert s3.dt.convert_time_zone("Asia/Tokyo").equals(s4) is True


def test_series_equals_check_names() -> None:
    s1 = pl.Series("foo", [1, 2, 3])
    s2 = pl.Series("bar", [1, 2, 3])
    assert s1.equals(s2) is True
    assert s1.equals(s2, check_names=True) is False


def test_eq_list_cmp_list() -> None:
    s = pl.Series([[1], [1, 2]])
    result = s == [1, 2]
    expected = pl.Series([False, True])
    assert_series_equal(result, expected)


def test_eq_list_cmp_int() -> None:
    s = pl.Series([[1], [1, 2]])
    with pytest.raises(
        TypeError, match="cannot convert Python type 'int' to List\\(Int64\\)"
    ):
        s == 1  # noqa: B015


def test_eq_array_cmp_list() -> None:
    s = pl.Series([[1, 3], [1, 2]], dtype=pl.Array(pl.Int16, 2))
    result = s == [1, 2]
    expected = pl.Series([False, True])
    assert_series_equal(result, expected)


def test_eq_array_cmp_int() -> None:
    s = pl.Series([[1, 3], [1, 2]], dtype=pl.Array(pl.Int16, 2))
    with pytest.raises(
        TypeError,
        match="cannot convert Python type 'int' to Array\\(Int16, shape=\\(2,\\)\\)",
    ):
        s == 1  # noqa: B015


def test_eq_list() -> None:
    s = pl.Series([1, 1])

    result = s == [1, 2]
    expected = pl.Series([True, False])
    assert_series_equal(result, expected)

    result = s == 1
    expected = pl.Series([True, True])
    assert_series_equal(result, expected)


def test_eq_missing_expr() -> None:
    s = pl.Series([1, None])
    result = s.eq_missing(pl.lit(1))

    assert isinstance(result, pl.Expr)
    result_evaluated = pl.select(result).to_series()
    expected = pl.Series([True, False])
    assert_series_equal(result_evaluated, expected)


def test_ne_missing_expr() -> None:
    s = pl.Series([1, None])
    result = s.ne_missing(pl.lit(1))

    assert isinstance(result, pl.Expr)
    result_evaluated = pl.select(result).to_series()
    expected = pl.Series([False, True])
    assert_series_equal(result_evaluated, expected)


def test_series_equals_strict_deprecated() -> None:
    s1 = pl.Series("a", [1.0, 2.0, None], pl.Float64)
    s2 = pl.Series("a", [1, 2, None], pl.Int64)
    with pytest.deprecated_call():
        assert not s1.equals(s2, strict=True)  # type: ignore[call-arg]
