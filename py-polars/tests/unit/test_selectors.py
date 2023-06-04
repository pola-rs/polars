import pytest

import polars as pl
import polars.selectors as s


@pytest.fixture()
def df() -> pl.DataFrame:
    # set up an empty dataframe with plenty of columns of various dtypes
    df = pl.DataFrame(
        schema={
            "abc": pl.UInt16,
            "bbb": pl.UInt32,
            "cde": pl.Float64,
            "def": pl.Float32,
            "eee": pl.Boolean,
            "fgg": pl.Boolean,
            "ghi": pl.Time,
            "JJK": pl.Date,
            "Lmn": pl.Duration,
            "opp": pl.Datetime("ms"),
            "qqR": pl.Utf8,
        },
    )
    return df


def test_selector_all(df: pl.DataFrame) -> None:
    assert df.schema == df.select(s.all()).schema
    assert {} == df.select(~s.all()).schema
    assert df.schema == df.select(~(~s.all())).schema


def test_selector_by_dtype(df: pl.DataFrame) -> None:
    assert df.select(s.by_dtype(pl.UInt16, pl.Boolean)).schema == {
        "abc": pl.UInt16,
        "eee": pl.Boolean,
        "fgg": pl.Boolean,
    }
    assert df.select(~s.by_dtype(pl.INTEGER_DTYPES, pl.TEMPORAL_DTYPES)).schema == {
        "cde": pl.Float64,
        "def": pl.Float32,
        "eee": pl.Boolean,
        "fgg": pl.Boolean,
        "qqR": pl.Utf8,
    }


def test_selector_by_name(df: pl.DataFrame) -> None:
    assert df.select(s.by_name("abc", "cde")).columns == [
        "abc",
        "cde",
    ]
    assert df.select(~s.by_name("abc", "cde", "ghi", "Lmn", "opp", "eee")).columns == [
        "bbb",
        "def",
        "fgg",
        "JJK",
        "qqR",
    ]


def test_selector_contains(df: pl.DataFrame) -> None:
    assert df.select(s.contains("b")).columns == ["abc", "bbb"]
    assert df.select(s.contains(("e", "g"))).columns == [
        "cde",
        "def",
        "eee",
        "fgg",
        "ghi",
    ]
    assert df.select(~s.contains(("b", "e", "g"))).columns == [
        "JJK",
        "Lmn",
        "opp",
        "qqR",
    ]


def test_selector_datetime(df: pl.DataFrame) -> None:
    assert df.select(s.datetime()).schema == {"opp": pl.Datetime("ms")}
    assert df.select(s.datetime("ns")).schema == {}

    all_columns = set(df.columns)
    assert set(df.select(~s.datetime()).columns) == all_columns - {"opp"}


def test_selector_ends_with(df: pl.DataFrame) -> None:
    assert df.select(s.ends_with("e")).columns == ["cde", "eee"]
    assert df.select(s.ends_with("e", "g", "i", "n", "p")).columns == [
        "cde",
        "eee",
        "fgg",
        "ghi",
        "Lmn",
        "opp",
    ]
    assert df.select(~s.ends_with("b", "e", "g", "i", "n", "p")).columns == [
        "abc",
        "def",
        "JJK",
        "qqR",
    ]


def test_selector_first_last(df: pl.DataFrame) -> None:
    assert df.select(s.first()).columns == ["abc"]
    assert df.select(s.last()).columns == ["qqR"]


def test_selector_float(df: pl.DataFrame) -> None:
    assert df.select(s.float()).schema == {
        "cde": pl.Float64,
        "def": pl.Float32,
    }
    all_columns = set(df.columns)
    assert set(df.select(~s.float()).columns) == (all_columns - {"cde", "def"})


def test_selector_integer(df: pl.DataFrame) -> None:
    assert df.select(s.integer()).schema == {
        "abc": pl.UInt16,
        "bbb": pl.UInt32,
    }
    all_columns = set(df.columns)
    assert set(df.select(~s.integer()).columns) == (all_columns - {"abc", "bbb"})


def test_selector_matches(df: pl.DataFrame) -> None:
    assert df.select(s.matches(r"^(?i)[E-N]{3}$")).columns == [
        "eee",
        "fgg",
        "ghi",
        "JJK",
        "Lmn",
    ]
    assert df.select(~s.matches(r"^(?i)[E-N]{3}$")).columns == [
        "abc",
        "bbb",
        "cde",
        "def",
        "opp",
        "qqR",
    ]


def test_selector_numeric(df: pl.DataFrame) -> None:
    assert df.select(s.numeric()).schema == {
        "abc": pl.UInt16,
        "bbb": pl.UInt32,
        "cde": pl.Float64,
        "def": pl.Float32,
    }
    assert df.select(s.numeric().exclude(pl.UInt16)).schema == {
        "bbb": pl.UInt32,
        "cde": pl.Float64,
        "def": pl.Float32,
    }
    all_columns = set(df.columns)
    assert set(df.select(~s.numeric()).columns) == (
        all_columns - {"abc", "bbb", "cde", "def"}
    )


def test_selector_startswith(df: pl.DataFrame) -> None:
    assert df.select(s.starts_with("a")).columns == ["abc"]
    assert df.select(s.starts_with("d", "e", "f", "g", "h", "i", "j")).columns == [
        "def",
        "eee",
        "fgg",
        "ghi",
    ]
    assert df.select(~s.starts_with("d", "e", "f", "g", "h", "i", "j")).columns == [
        "abc",
        "bbb",
        "cde",
        "JJK",
        "Lmn",
        "opp",
        "qqR",
    ]


def test_selector_temporal(df: pl.DataFrame) -> None:
    assert df.select(s.temporal()).schema == {
        "ghi": pl.Time,
        "JJK": pl.Date,
        "Lmn": pl.Duration,
        "opp": pl.Datetime("ms"),
    }
    all_columns = set(df.columns)
    assert set(df.select(~s.temporal()).columns) == (
        all_columns - {"ghi", "JJK", "Lmn", "opp"}
    )
