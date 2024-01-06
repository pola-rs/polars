from typing import Any

import pytest

import polars as pl
import polars.selectors as cs
from polars.selectors import expand_selector, is_selector
from polars.testing import assert_frame_equal


def assert_repr_equals(item: Any, expected: str) -> None:
    """Assert that the repr of an item matches the expected string."""
    if not isinstance(expected, str):
        raise TypeError(f"'expected' must be a string; found {type(expected)}")
    assert repr(item) == expected


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
            "qqR": pl.String,
        },
    )
    return df


def test_selector_all(df: pl.DataFrame) -> None:
    assert df.schema == df.select(cs.all()).schema
    assert {} == df.select(~cs.all()).schema
    assert df.schema == df.select(~(~cs.all())).schema
    assert df.select(cs.all() & pl.col("abc")).schema == {"abc": pl.UInt16}


def test_selector_by_dtype(df: pl.DataFrame) -> None:
    assert df.select(cs.by_dtype(pl.UInt16) | cs.boolean()).schema == {
        "abc": pl.UInt16,
        "eee": pl.Boolean,
        "fgg": pl.Boolean,
    }
    assert df.select(~cs.by_dtype(pl.INTEGER_DTYPES, pl.TEMPORAL_DTYPES)).schema == {
        "cde": pl.Float64,
        "def": pl.Float32,
        "eee": pl.Boolean,
        "fgg": pl.Boolean,
        "qqR": pl.String,
    }


def test_selector_by_name(df: pl.DataFrame) -> None:
    for selector in (
        cs.by_name("abc", "cde"),
        cs.by_name("abc") | pl.col("cde"),
    ):
        assert df.select(selector).columns == [
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

    df = pl.DataFrame(
        schema={
            "d1": pl.Datetime("ns", "Asia/Tokyo"),
            "d2": pl.Datetime("ns", "UTC"),
            "d3": pl.Datetime("us", "UTC"),
            "d4": pl.Datetime("us"),
            "d5": pl.Datetime("ms"),
        },
    )
    assert df.select(cs.datetime()).columns == ["d1", "d2", "d3", "d4", "d5"]
    assert df.select(~cs.datetime()).schema == {}

    assert df.select(cs.datetime(["ms", "ns"])).columns == ["d1", "d2", "d5"]
    assert df.select(cs.datetime(["ms", "ns"], time_zone="*")).columns == ["d1", "d2"]

    assert df.select(~cs.datetime(["ms", "ns"])).columns == ["d3", "d4"]
    assert df.select(~cs.datetime(["ms", "ns"], time_zone="*")).columns == [
        "d3",
        "d4",
        "d5",
    ]
    assert df.select(
        cs.datetime(time_zone=["UTC", "Asia/Tokyo", "Europe/London"])
    ).columns == ["d1", "d2", "d3"]

    assert df.select(cs.datetime(time_zone="*")).columns == ["d1", "d2", "d3"]
    assert df.select(cs.datetime("ns", time_zone="*")).columns == ["d1", "d2"]
    assert df.select(cs.datetime(time_zone="UTC")).columns == ["d2", "d3"]
    assert df.select(cs.datetime("us", time_zone="UTC")).columns == ["d3"]
    assert df.select(cs.datetime(time_zone="Asia/Tokyo")).columns == ["d1"]
    assert df.select(cs.datetime("us", time_zone="Asia/Tokyo")).columns == []
    assert df.select(cs.datetime(time_zone=None)).columns == ["d4", "d5"]
    assert df.select(cs.datetime("ns", time_zone=None)).columns == []

    assert df.select(~cs.datetime(time_zone="*")).columns == ["d4", "d5"]
    assert df.select(~cs.datetime("ns", time_zone="*")).columns == ["d3", "d4", "d5"]
    assert df.select(~cs.datetime(time_zone="UTC")).columns == ["d1", "d4", "d5"]
    assert df.select(~cs.datetime("us", time_zone="UTC")).columns == [
        "d1",
        "d2",
        "d4",
        "d5",
    ]
    assert df.select(~cs.datetime(time_zone="Asia/Tokyo")).columns == [
        "d2",
        "d3",
        "d4",
        "d5",
    ]
    assert df.select(~cs.datetime("us", time_zone="Asia/Tokyo")).columns == [
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
    ]
    assert df.select(~cs.datetime(time_zone=None)).columns == ["d1", "d2", "d3"]
    assert df.select(~cs.datetime("ns", time_zone=None)).columns == [
        "d1",
        "d2",
        "d3",
        "d4",
        "d5",
    ]
    assert df.select(cs.datetime("ns")).columns == ["d1", "d2"]
    assert df.select(cs.datetime("us")).columns == ["d3", "d4"]
    assert df.select(cs.datetime("ms")).columns == ["d5"]

    # bonus check; significantly more verbose, but equivalent to a selector -
    assert (
        df.select(
            pl.all().exclude(
                pl.Datetime("ms", time_zone="*"), pl.Datetime("ns", time_zone="*")
            )
        ).columns
        == df.select(~cs.datetime(["ms", "ns"], time_zone="*")).columns
    )


def test_select_decimal(df: pl.DataFrame) -> None:
    assert df.select(cs.decimal()).columns == []
    df = pl.DataFrame(
        schema={
            "zz0": pl.Float64,
            "zz1": pl.Decimal,
            "zz2": pl.Decimal(10, 10),
        }
    )
    assert df.select(cs.numeric()).columns == ["zz0", "zz1", "zz2"]
    assert df.select(cs.decimal()).columns == ["zz1", "zz2"]
    assert df.select(~cs.decimal()).columns == ["zz0"]


def test_selector_drop(df: pl.DataFrame) -> None:
    dfd = df.drop(cs.numeric(), cs.temporal())
    assert dfd.columns == ["eee", "fgg", "qqR"]


def test_selector_duration(df: pl.DataFrame) -> None:
    assert df.select(cs.duration("ms")).columns == []
    assert df.select(cs.duration(["ms", "ns"])).columns == []
    assert expand_selector(df, cs.duration()) == ("Lmn",)

    df = pl.DataFrame(
        schema={
            "d1": pl.Duration("ns"),
            "d2": pl.Duration("us"),
            "d3": pl.Duration("ms"),
        },
    )
    assert expand_selector(df, cs.duration()) == ("d1", "d2", "d3")
    assert expand_selector(df, cs.duration("us")) == ("d2",)
    assert expand_selector(df, cs.duration(["ms", "ns"])) == ("d1", "d3")


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


def test_selector_miscellaneous(df: pl.DataFrame) -> None:
    assert df.select(cs.string()).columns == ["qqR"]
    assert df.select(cs.categorical()).columns == []

    test_schema = {
        "abc": pl.String,
        "mno": pl.Binary,
        "tuv": pl.Object,
        "xyz": pl.Categorical,
    }
    assert expand_selector(test_schema, cs.binary()) == ("mno",)
    assert expand_selector(test_schema, ~cs.binary()) == ("abc", "tuv", "xyz")
    assert expand_selector(test_schema, cs.object()) == ("tuv",)
    assert expand_selector(test_schema, ~cs.object()) == ("abc", "mno", "xyz")
    assert expand_selector(test_schema, cs.categorical()) == ("xyz",)
    assert expand_selector(test_schema, ~cs.categorical()) == ("abc", "mno", "tuv")


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
    assert df.select(cs.time()).schema == {"ghi": pl.Time}
    assert df.select(cs.date() | cs.time()).schema == {"ghi": pl.Time, "JJK": pl.Date}


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
    assert_repr_equals(cs.all() - cs.first(), "(cs.all() - cs.first())")
    assert_repr_equals(~cs.starts_with("a", "b"), "~cs.starts_with('a', 'b')")
    assert_repr_equals(cs.float() | cs.by_name("x"), "(cs.float() | cs.by_name('x'))")
    assert_repr_equals(
        cs.integer() & cs.matches("z"), "(cs.integer() & cs.matches(pattern='z'))"
    )
    assert_repr_equals(
        cs.temporal() | cs.by_dtype(pl.String) & cs.string(include_categorical=False),
        "(cs.temporal() | (cs.by_dtype(dtypes=[String]) & cs.string(include_categorical=False)))",
    )


def test_selector_sets(df: pl.DataFrame) -> None:
    # or
    assert df.select(cs.temporal() | cs.string() | cs.starts_with("e")).schema == {
        "eee": pl.Boolean,
        "ghi": pl.Time,
        "JJK": pl.Date,
        "Lmn": pl.Duration,
        "opp": pl.Datetime("ms"),
        "qqR": pl.String,
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

    # equivalent (though more verbose) to the above, using `exclude`
    assert df.select(cs.exclude(~cs.temporal() | cs.matches("opp|JJK"))).schema == {
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
        "qqR": pl.String,
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
            pl.when(cs.float().is_finite()).then(cs.float()).otherwise(0.0).name.keep()
        ),
    )

    # inverted selector-broadcast expression
    assert_frame_equal(
        expected,
        df.with_columns(
            pl.when(~cs.float().is_finite()).then(0.0).otherwise(cs.float()).name.keep()
        ),
    )

    # check that "as_expr" behaves, both explicitly and implicitly
    for nan_or_inf in (
        cs.float().is_nan().as_expr() | cs.float().is_infinite().as_expr(),
        cs.float().is_nan().as_expr() | cs.float().is_infinite(),
        cs.float().is_nan() | cs.float().is_infinite(),
    ):
        assert_frame_equal(
            expected,
            df.with_columns(
                pl.when(nan_or_inf).then(0.0).otherwise(cs.float()).name.keep()
            ).fill_null(0),
        )


def test_regex_expansion_group_by_9947() -> None:
    df = pl.DataFrame({"g": [3], "abc": [1], "abcd": [3]})
    assert df.group_by("g").agg(pl.col("^ab.*$")).columns == ["g", "abc", "abcd"]


def test_regex_expansion_exclude_10002() -> None:
    df = pl.DataFrame({"col_1": [1, 2, 3], "col_2": [2, 4, 3]})
    expected = {"col_1": [10, 20, 30], "col_2": [0.2, 0.4, 0.3]}

    assert (
        df.select(
            pl.col("^col_.*$").exclude("col_2").mul(10),
            pl.col("^col_.*$").exclude("col_1") / 10,
        ).to_dict(as_series=False)
        == expected
    )


def test_is_selector() -> None:
    assert is_selector(cs.numeric())
    assert is_selector(cs.by_dtype(pl.UInt32) | pl.col("xyz"))
    assert not is_selector(pl.col("cde"))
    assert not is_selector(None)


def test_selector_or() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2, 3],
            "float": [1.0, 2.0, 3.0],
            "str": ["x", "y", "z"],
        }
    ).with_row_number("rn")

    result = df.select(cs.by_name("rn") | ~cs.numeric())

    expected = pl.DataFrame(
        {"rn": [0, 1, 2], "str": ["x", "y", "z"]}, schema_overrides={"rn": pl.UInt32}
    )
    assert_frame_equal(result, expected)
