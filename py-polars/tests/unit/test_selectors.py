import pytest

import polars as pl
import polars.selectors as cs


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
    assert df.schema == df.select(cs.all()).schema
    assert {} == df.select(~cs.all()).schema
    assert df.schema == df.select(~(~cs.all())).schema


def test_selector_by_dtype(df: pl.DataFrame) -> None:
    assert df.select(cs.by_dtype(pl.UInt16, pl.Boolean)).schema == {
        "abc": pl.UInt16,
        "eee": pl.Boolean,
        "fgg": pl.Boolean,
    }
    assert df.select(~cs.by_dtype(pl.INTEGER_DTYPES, pl.TEMPORAL_DTYPES)).schema == {
        "cde": pl.Float64,
        "def": pl.Float32,
        "eee": pl.Boolean,
        "fgg": pl.Boolean,
        "qqR": pl.Utf8,
    }


def test_selector_by_name(df: pl.DataFrame) -> None:
    assert df.select(cs.by_name("abc", "cde")).columns == [
        "abc",
        "cde",
    ]
    assert df.select(~cs.by_name("abc", "cde", "ghi", "Lmn", "opp", "eee")).columns == [
        "bbb",
        "def",
        "fgg",
        "JJK",
        "qqR",
    ]


def test_selector_contains(df: pl.DataFrame) -> None:
    assert df.select(cs.contains("b")).columns == ["abc", "bbb"]
    assert df.select(cs.contains(("e", "g"))).columns == [
        "cde",
        "def",
        "eee",
        "fgg",
        "ghi",
    ]
    assert df.select(~cs.contains(("b", "e", "g"))).columns == [
        "JJK",
        "Lmn",
        "opp",
        "qqR",
    ]


def test_selector_datetime(df: pl.DataFrame) -> None:
    assert df.select(cs.datetime()).schema == {"opp": pl.Datetime("ms")}
    assert df.select(cs.datetime("ns")).schema == {}

    all_columns = set(df.columns)
    assert set(df.select(~cs.datetime()).columns) == all_columns - {"opp"}


def test_selector_ends_with(df: pl.DataFrame) -> None:
    assert df.select(cs.ends_with("e")).columns == ["cde", "eee"]
    assert df.select(cs.ends_with("e", "g", "i", "n", "p")).columns == [
        "cde",
        "eee",
        "fgg",
        "ghi",
        "Lmn",
        "opp",
    ]
    assert df.select(~cs.ends_with("b", "e", "g", "i", "n", "p")).columns == [
        "abc",
        "def",
        "JJK",
        "qqR",
    ]


def test_selector_first_last(df: pl.DataFrame) -> None:
    assert df.select(cs.first()).columns == ["abc"]
    assert df.select(cs.last()).columns == ["qqR"]


def test_selector_float(df: pl.DataFrame) -> None:
    assert df.select(cs.float()).schema == {
        "cde": pl.Float64,
        "def": pl.Float32,
    }
    all_columns = set(df.columns)
    assert set(df.select(~cs.float()).columns) == (all_columns - {"cde", "def"})


def test_selector_integer(df: pl.DataFrame) -> None:
    assert df.select(cs.integer()).schema == {
        "abc": pl.UInt16,
        "bbb": pl.UInt32,
    }
    all_columns = set(df.columns)
    assert set(df.select(~cs.integer()).columns) == (all_columns - {"abc", "bbb"})


def test_selector_matches(df: pl.DataFrame) -> None:
    assert df.select(cs.matches(r"^(?i)[E-N]{3}$")).columns == [
        "eee",
        "fgg",
        "ghi",
        "JJK",
        "Lmn",
    ]
    assert df.select(~cs.matches(r"^(?i)[E-N]{3}$")).columns == [
        "abc",
        "bbb",
        "cde",
        "def",
        "opp",
        "qqR",
    ]


def test_selector_numeric(df: pl.DataFrame) -> None:
    assert df.select(cs.numeric()).schema == {
        "abc": pl.UInt16,
        "bbb": pl.UInt32,
        "cde": pl.Float64,
        "def": pl.Float32,
    }
    assert df.select(cs.numeric().exclude(pl.UInt16)).schema == {
        "bbb": pl.UInt32,
        "cde": pl.Float64,
        "def": pl.Float32,
    }
    all_columns = set(df.columns)
    assert set(df.select(~cs.numeric()).columns) == (
        all_columns - {"abc", "bbb", "cde", "def"}
    )


def test_selector_startswith(df: pl.DataFrame) -> None:
    assert df.select(cs.starts_with("a")).columns == ["abc"]
    assert df.select(cs.starts_with("d", "e", "f", "g", "h", "i", "j")).columns == [
        "def",
        "eee",
        "fgg",
        "ghi",
    ]
    assert df.select(~cs.starts_with("d", "e", "f", "g", "h", "i", "j")).columns == [
        "abc",
        "bbb",
        "cde",
        "JJK",
        "Lmn",
        "opp",
        "qqR",
    ]


def test_selector_temporal(df: pl.DataFrame) -> None:
    assert df.select(cs.temporal()).schema == {
        "ghi": pl.Time,
        "JJK": pl.Date,
        "Lmn": pl.Duration,
        "opp": pl.Datetime("ms"),
    }
    all_columns = set(df.columns)
    assert set(df.select(~cs.temporal()).columns) == (
        all_columns - {"ghi", "JJK", "Lmn", "opp"}
    )
