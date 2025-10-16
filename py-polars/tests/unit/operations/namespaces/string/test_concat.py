from datetime import datetime

import pytest

import polars as pl
from polars.testing import assert_series_equal


def test_str_join() -> None:
    s = pl.Series(["1", None, "2", None])
    # propagate null
    assert_series_equal(
        s.str.join("-", ignore_nulls=False), pl.Series([None], dtype=pl.String)
    )
    # ignore null
    assert_series_equal(s.str.join(), pl.Series(["12"]))

    # str None/null is ok
    s = pl.Series(["1", "None", "2", "null"])
    assert_series_equal(
        s.str.join("-", ignore_nulls=False), pl.Series(["1-None-2-null"])
    )
    assert_series_equal(s.str.join("-"), pl.Series(["1-None-2-null"]))


def test_str_join2() -> None:
    df = pl.DataFrame({"foo": [1, None, 2, None]})

    out = df.select(pl.col("foo").str.join(ignore_nulls=False))
    assert out.item() is None

    out = df.select(pl.col("foo").str.join())
    assert out.item() == "12"


def test_str_join_all_null() -> None:
    s = pl.Series([None, None, None], dtype=pl.String)
    assert_series_equal(
        s.str.join(ignore_nulls=False), pl.Series([None], dtype=pl.String)
    )
    assert_series_equal(s.str.join(ignore_nulls=True), pl.Series([""]))


def test_str_join_empty_list() -> None:
    s = pl.Series([], dtype=pl.String)
    assert_series_equal(s.str.join(ignore_nulls=False), pl.Series([""]))
    assert_series_equal(s.str.join(ignore_nulls=True), pl.Series([""]))


def test_str_join_empty_list2() -> None:
    s = pl.Series([], dtype=pl.String)
    df = pl.DataFrame({"foo": s})
    result = df.select(pl.col("foo").str.join()).item()
    expected = ""
    assert result == expected


def test_str_join_empty_list_agg_context() -> None:
    df = pl.DataFrame(data={"i": [1], "v": [None]}, schema_overrides={"v": pl.String})
    result = df.group_by("i").agg(pl.col("v").drop_nulls().str.join())["v"].item()
    expected = ""
    assert result == expected


def test_str_join_datetime() -> None:
    df = pl.DataFrame({"d": [datetime(2020, 1, 1), None, datetime(2022, 1, 1)]})
    out = df.select(pl.col("d").str.join("|", ignore_nulls=True))
    assert out.item() == "2020-01-01 00:00:00.000000|2022-01-01 00:00:00.000000"
    out = df.select(pl.col("d").str.join("|", ignore_nulls=False))
    assert out.item() is None


def test_str_concat_deprecated() -> None:
    s = pl.Series(["1", None, "2", None])
    with pytest.deprecated_call():
        result = s.str.concat()
    expected = pl.Series(["1-2"])
    assert_series_equal(result, expected)
