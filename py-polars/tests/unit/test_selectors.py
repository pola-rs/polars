import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal


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
    assert df.select(cs.contains(("ee", "x"))).columns == ["eee"]


def test_selector_datetime(df: pl.DataFrame) -> None:
    assert df.select(cs.datetime()).schema == {"opp": pl.Datetime("ms")}
    assert df.select(cs.datetime("ns")).schema == {}

    all_columns = set(df.columns)
    assert set(df.select(~cs.datetime()).columns) == all_columns - {"opp"}


def test_selector_ends_with(df: pl.DataFrame) -> None:
    assert df.select(cs.ends_with("e")).columns == ["cde", "eee"]
    assert df.select(cs.ends_with("ee")).columns == ["eee"]
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

    all_columns = set(df.columns)
    assert set(df.select(~cs.first()).columns) == (all_columns - {"abc"})
    assert set(df.select(~cs.last()).columns) == (all_columns - {"qqR"})


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
    assert df.select(cs.starts_with("ee")).columns == ["eee"]
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


def test_selector_expansion() -> None:
    df = pl.DataFrame({name: [] for name in "abcde"})

    s1 = pl.all().meta._as_selector()
    s2 = pl.col(["a", "b"])
    s = s1.meta._selector_sub(s2)
    assert df.select(s).columns == ["c", "d", "e"]

    s1 = pl.col("^a|b$").meta._as_selector()
    s = s1.meta._selector_add(pl.col(["d", "e"]))
    assert df.select(s).columns == ["a", "b", "d", "e"]

    s = s.meta._selector_sub(pl.col("d"))
    assert df.select(s).columns == ["a", "b", "e"]

    # add a duplicate, this tests if they are pruned
    s = s.meta._selector_add(pl.col("a"))
    assert df.select(s).columns == ["a", "b", "e"]

    s1 = pl.col(["a", "b", "c"])
    s2 = pl.col(["b", "c", "d"])

    s = s1.meta._as_selector()
    s = s.meta._selector_and(s2)
    assert df.select(s).columns == ["b", "c"]


def test_selector_repr() -> None:
    assert repr(cs.all() - cs.first()) == "cs.all() - cs.first()"
    assert repr(~cs.starts_with("a", "b")) == "~cs.starts_with('a', 'b')"
    assert repr(cs.float() | cs.by_name("x")) == "cs.float() | cs.by_name('x')"
    assert (
        repr(cs.integer() & cs.matches("z")) == "cs.integer() & cs.matches(pattern='z')"
    )


def test_selector_sets(df: pl.DataFrame) -> None:
    # or
    assert df.select(cs.temporal() | cs.string() | cs.starts_with("e")).schema == {
        "eee": pl.Boolean,
        "ghi": pl.Time,
        "JJK": pl.Date,
        "Lmn": pl.Duration,
        "opp": pl.Datetime("ms"),
        "qqR": pl.Utf8,
    }

    # and
    assert df.select(cs.temporal() & cs.matches("opp|JJK")).schema == {
        "JJK": pl.Date,
        "opp": pl.Datetime("ms"),
    }

    # SET A - SET B
    assert df.select(cs.temporal() - cs.matches("opp|JJK")).schema == {
        "ghi": pl.Time,
        "Lmn": pl.Duration,
    }

    # COMPLEMENT SET
    assert df.select(~cs.by_dtype([pl.Duration, pl.Time])).schema == {
        "abc": pl.UInt16,
        "bbb": pl.UInt32,
        "cde": pl.Float64,
        "def": pl.Float32,
        "eee": pl.Boolean,
        "fgg": pl.Boolean,
        "JJK": pl.Date,
        "opp": pl.Datetime("ms"),
        "qqR": pl.Utf8,
    }


def test_selector_dispatch_default_operator() -> None:
    df = pl.DataFrame({"a": [1, 1], "b": [2, 2], "abc": [3, 3]})
    out = df.select((cs.numeric() & ~cs.by_name("abc")) + 1)
    expected = pl.DataFrame(
        {
            "a": [2, 2],
            "b": [3, 3],
        }
    )
    assert_frame_equal(out, expected)


def test_selector_expr_dispatch() -> None:
    df = pl.DataFrame(
        data={
            "colx": [float("inf"), -1, float("nan"), 25],
            "coly": [1, float("-inf"), 10, float("nan")],
        },
        schema={"colx": pl.Float64, "coly": pl.Float32},
    )
    expected = pl.DataFrame(
        data={
            "colx": [0.0, -1.0, 0.0, 25.0],
            "coly": [1.0, 0.0, 10.0, 0.0],
        },
        schema={"colx": pl.Float64, "coly": pl.Float32},
    )

    # basic selector-broadcast expression
    assert_frame_equal(
        expected,
        df.with_columns(
            pl.when(cs.float().is_finite()).then(cs.float()).otherwise(0.0).keep_name()
        ),
    )

    # inverted selector-broadcast expression
    assert_frame_equal(
        expected,
        df.with_columns(
            pl.when(~cs.float().is_finite()).then(0.0).otherwise(cs.float()).keep_name()
        ),
    )

    # check that "as_expr" behaves, both explicitly and implicitly
    for nan_or_inf in (
        cs.float().is_nan().as_expr() | cs.float().is_infinite().as_expr(),  # type: ignore[attr-defined]
        cs.float().is_nan().as_expr() | cs.float().is_infinite(),  # type: ignore[attr-defined]
        cs.float().is_nan() | cs.float().is_infinite(),
    ):
        assert_frame_equal(
            expected,
            df.with_columns(
                pl.when(nan_or_inf).then(0.0).otherwise(cs.float()).keep_name()
            ).fill_null(0),
        )
