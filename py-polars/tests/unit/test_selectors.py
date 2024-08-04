import sys
from collections import OrderedDict
from datetime import datetime
from typing import Any

import pytest

import polars as pl
import polars.selectors as cs
from polars._typing import SelectorType
from polars.dependencies import _ZONEINFO_AVAILABLE
from polars.exceptions import ColumnNotFoundError
from polars.selectors import expand_selector, is_selector
from polars.testing import assert_frame_equal
from tests.unit.conftest import INTEGER_DTYPES, TEMPORAL_DTYPES

if sys.version_info >= (3, 9):
    from zoneinfo import ZoneInfo
elif _ZONEINFO_AVAILABLE:
    # Import from submodule due to typing issue with backports.zoneinfo package:
    # https://github.com/pganssle/zoneinfo/issues/125
    from backports.zoneinfo._zoneinfo import ZoneInfo


def assert_repr_equals(item: Any, expected: str) -> None:
    """Assert that the repr of an item matches the expected string."""
    if not isinstance(expected, str):
        msg = f"'expected' must be a string; found {type(expected)}"
        raise TypeError(msg)
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
    assert df.select(~cs.all()).schema == {}
    assert df.schema == df.select(~(~cs.all())).schema
    assert df.select(cs.all() & pl.col("abc")).schema == {"abc": pl.UInt16}


def test_selector_alpha() -> None:
    df = pl.DataFrame(
        schema=["Hello 123", "こんにちは (^_^)", "مرحبا", "你好!", "World"],
    )
    # alphabetical-only (across all languages)
    assert expand_selector(df, cs.alpha()) == ("مرحبا", "World")
    assert expand_selector(df, cs.alpha(ascii_only=True)) == ("World",)
    assert expand_selector(df, ~cs.alpha()) == (
        "Hello 123",
        "こんにちは (^_^)",
        "你好!",
    )
    assert expand_selector(df, ~cs.alpha(ignore_spaces=True)) == (
        "Hello 123",
        "こんにちは (^_^)",
        "你好!",
    )

    # alphanumeric-only (across all languages)
    assert expand_selector(df, cs.alphanumeric(True)) == ("World",)
    assert expand_selector(df, ~cs.alphanumeric()) == (
        "Hello 123",
        "こんにちは (^_^)",
        "你好!",
    )
    assert expand_selector(df, ~cs.alphanumeric(True, ignore_spaces=True)) == (
        "こんにちは (^_^)",
        "مرحبا",
        "你好!",
    )
    assert expand_selector(df, cs.alphanumeric(ignore_spaces=True)) == (
        "Hello 123",
        "مرحبا",
        "World",
    )
    assert expand_selector(df, ~cs.alphanumeric(ignore_spaces=True)) == (
        "こんにちは (^_^)",
        "你好!",
    )


def test_selector_by_dtype(df: pl.DataFrame) -> None:
    assert df.select(cs.boolean() | cs.by_dtype(pl.UInt16)).schema == OrderedDict(
        {
            "abc": pl.UInt16,
            "eee": pl.Boolean,
            "fgg": pl.Boolean,
        }
    )
    assert df.select(
        ~cs.by_dtype(*INTEGER_DTYPES, *TEMPORAL_DTYPES)
    ).schema == pl.Schema(
        {
            "cde": pl.Float64(),
            "def": pl.Float32(),
            "eee": pl.Boolean(),
            "fgg": pl.Boolean(),
            "qqR": pl.String(),
        }
    )
    assert df.select(cs.by_dtype()).schema == {}
    assert df.select(cs.by_dtype([])).schema == {}

    # expected errors
    with pytest.raises(TypeError):
        df.select(cs.by_dtype(999))  # type: ignore[arg-type]


def test_selector_by_index(df: pl.DataFrame) -> None:
    # one or more +ve indexes
    assert df.select(cs.by_index(0)).columns == ["abc"]
    assert df.select(pl.nth(0, 1, 2)).columns == ["abc", "bbb", "cde"]
    assert df.select(cs.by_index(0, 1, 2)).columns == ["abc", "bbb", "cde"]

    # one or more -ve indexes
    assert df.select(cs.by_index(-1)).columns == ["qqR"]
    assert df.select(cs.by_index(-3, -2, -1)).columns == ["Lmn", "opp", "qqR"]

    # range objects
    assert df.select(cs.by_index(range(3))).columns == ["abc", "bbb", "cde"]
    assert df.select(cs.by_index(0, range(-3, 0))).columns == [
        "abc",
        "Lmn",
        "opp",
        "qqR",
    ]

    # exclude by index
    assert df.select(~cs.by_index(range(0, df.width, 2))).columns == [
        "bbb",
        "def",
        "fgg",
        "JJK",
        "opp",
    ]

    # expected errors
    with pytest.raises(ColumnNotFoundError):
        df.select(cs.by_index(999))

    for invalid in ("one", ["two", "three"]):
        with pytest.raises(TypeError):
            df.select(cs.by_index(invalid))  # type: ignore[arg-type]


def test_selector_by_name(df: pl.DataFrame) -> None:
    for selector in (
        cs.by_name("abc", "cde"),
        cs.by_name("abc") | pl.col("cde"),
    ):
        assert df.select(selector).columns == ["abc", "cde"]

    assert df.select(~cs.by_name("abc", "cde", "ghi", "Lmn", "opp", "eee")).columns == [
        "bbb",
        "def",
        "fgg",
        "JJK",
        "qqR",
    ]
    assert df.select(cs.by_name()).columns == []
    assert df.select(cs.by_name([])).columns == []

    selected_cols = df.select(
        cs.by_name("???", "fgg", "!!!", require_all=False)
    ).columns
    assert selected_cols == ["fgg"]

    for missing_column in ("missing", "???"):
        assert df.select(cs.by_name(missing_column, require_all=False)).columns == []

    # check "by_name & col"
    for selector_expr, expected in (
        (cs.by_name("abc", "cde") & pl.col("ghi"), ["abc", "cde", "ghi"]),
        (pl.col("ghi") & cs.by_name("cde", "abc"), ["ghi", "cde", "abc"]),
    ):
        assert df.select(selector_expr).columns == expected

    # expected errors
    with pytest.raises(ColumnNotFoundError, match="xxx"):
        df.select(cs.by_name("xxx", "fgg", "!!!"))

    with pytest.raises(ColumnNotFoundError):
        df.select(cs.by_name("stroopwafel"))

    with pytest.raises(TypeError):
        df.select(cs.by_name(999))  # type: ignore[arg-type]


def test_selector_contains(df: pl.DataFrame) -> None:
    assert df.select(cs.contains("b")).columns == ["abc", "bbb"]
    assert df.select(cs.contains(("e", "g"))).columns == [  # type: ignore[arg-type]
        "cde",
        "def",
        "eee",
        "fgg",
        "ghi",
    ]
    assert df.select(~cs.contains("b", "e", "g")).columns == [
        "JJK",
        "Lmn",
        "opp",
        "qqR",
    ]
    assert df.select(cs.contains("ee", "x")).columns == ["eee"]

    # expected errors
    with pytest.raises(TypeError):
        df.select(cs.contains(999))  # type: ignore[arg-type]


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

    # expected errors
    with pytest.raises(TypeError):
        df.select(cs.datetime(999))  # type: ignore[arg-type]


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


def test_selector_digit() -> None:
    df = pl.DataFrame(schema=["Portfolio", "Year", "2000", "2010", "2020", "✌️"])
    assert expand_selector(df, cs.digit()) == ("2000", "2010", "2020")
    assert expand_selector(df, ~cs.digit()) == ("Portfolio", "Year", "✌️")

    df = pl.DataFrame({"১৯৯৯": [1999], "২০৭৭": [2077], "3000": [3000]})
    assert expand_selector(df, cs.digit()) == tuple(df.columns)
    assert expand_selector(df, cs.digit(ascii_only=True)) == ("3000",)
    assert expand_selector(df, (cs.digit() - cs.digit(True))) == ("১৯৯৯", "২০৭৭")


def test_selector_drop(df: pl.DataFrame) -> None:
    dfd = df.drop(cs.numeric(), cs.temporal())
    assert dfd.columns == ["eee", "fgg", "qqR"]

    df = pl.DataFrame([["x"], [1]], schema={"foo": pl.String, "foo_right": pl.Int8})
    assert df.drop(cs.ends_with("_right")).schema == {"foo": pl.String()}


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

    # expected errors
    with pytest.raises(TypeError):
        df.select(cs.ends_with(999))  # type: ignore[arg-type]


def test_selector_expand() -> None:
    schema = {
        "id": pl.Int64,
        "desc": pl.String,
        "count": pl.UInt32,
        "value": pl.Float64,
    }

    expanded = cs.expand_selector(schema, cs.numeric() - cs.unsigned_integer())
    assert expanded == ("id", "value")

    with pytest.raises(TypeError, match="expected a selector"):
        cs.expand_selector(schema, pl.exclude("id", "count"))

    with pytest.raises(TypeError, match="expected a selector"):
        cs.expand_selector(schema, pl.col("value") // 100)

    expanded = cs.expand_selector(schema, pl.exclude("id", "count"), strict=False)
    assert expanded == ("desc", "value")

    expanded = cs.expand_selector(schema, cs.numeric().exclude("id"), strict=False)
    assert expanded == ("count", "value")


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
    # expected errors
    with pytest.raises(TypeError):
        df.select(cs.starts_with(999))  # type: ignore[arg-type]


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


def test_selector_temporal_13665() -> None:
    df = pl.DataFrame(
        data={"utc": [datetime(1950, 7, 5), datetime(2099, 12, 31)]},
        schema={"utc": pl.Datetime(time_zone="UTC")},
    ).with_columns(
        idx=pl.int_range(0, 2),
        utc=pl.col("utc").dt.replace_time_zone(None),
        tokyo=pl.col("utc").dt.convert_time_zone("Asia/Tokyo"),
        hawaii=pl.col("utc").dt.convert_time_zone("US/Hawaii"),
    )
    for selector in (cs.datetime(), cs.datetime("us"), cs.temporal()):
        assert df.select(selector).to_dict(as_series=False) == {
            "utc": [
                datetime(1950, 7, 5, 0, 0),
                datetime(2099, 12, 31, 0, 0),
            ],
            "tokyo": [
                datetime(1950, 7, 5, 10, 0, tzinfo=ZoneInfo(key="Asia/Tokyo")),
                datetime(2099, 12, 31, 9, 0, tzinfo=ZoneInfo(key="Asia/Tokyo")),
            ],
            "hawaii": [
                datetime(1950, 7, 4, 14, 0, tzinfo=ZoneInfo(key="US/Hawaii")),
                datetime(2099, 12, 30, 14, 0, tzinfo=ZoneInfo(key="US/Hawaii")),
            ],
        }


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
        cs.integer() & cs.matches("z"),
        "(cs.integer() & cs.matches(pattern='z'))",
    )
    assert_repr_equals(
        cs.by_name("baz", "moose", "foo", "bear"),
        "cs.by_name('baz', 'moose', 'foo', 'bear')",
    )
    assert_repr_equals(
        cs.by_name("baz", "moose", "foo", "bear", require_all=False),
        "cs.by_name('baz', 'moose', 'foo', 'bear', require_all=False)",
    )
    assert_repr_equals(
        cs.temporal() | cs.by_dtype(pl.String) & cs.string(include_categorical=False),
        "(cs.temporal() | (cs.by_dtype(dtypes=[String]) & cs.string(include_categorical=False)))",
    )


def test_selector_sets(df: pl.DataFrame) -> None:
    # or
    assert df.select(
        cs.temporal() | cs.string() | cs.starts_with("e")
    ).schema == OrderedDict(
        {
            "eee": pl.Boolean,
            "ghi": pl.Time,
            "JJK": pl.Date,
            "Lmn": pl.Duration,
            "opp": pl.Datetime("ms"),
            "qqR": pl.String,
        }
    )

    # and
    assert df.select(cs.temporal() & cs.matches("opp|JJK")).schema == OrderedDict(
        {
            "JJK": pl.Date,
            "opp": pl.Datetime("ms"),
        }
    )

    # SET A - SET B
    assert df.select(cs.temporal() - cs.matches("opp|JJK")).schema == OrderedDict(
        {
            "ghi": pl.Time,
            "Lmn": pl.Duration,
        }
    )

    # equivalent (though more verbose) to the above, using `exclude`
    assert df.select(
        cs.exclude(~cs.temporal() | cs.matches("opp|JJK"))
    ).schema == OrderedDict(
        {
            "ghi": pl.Time,
            "Lmn": pl.Duration,
        }
    )

    frame = pl.DataFrame({"colx": [0, 1, 2], "coly": [3, 4, 5], "colz": [6, 7, 8]})
    sub_expr = cs.matches("[yz]$") - pl.col("colx")  # << shouldn't behave as set
    assert frame.select(sub_expr).rows() == [(3, 6), (3, 6), (3, 6)]

    with pytest.raises(TypeError, match=r"unsupported .* \('Expr' - 'Selector'\)"):
        df.select(pl.col("colx") - cs.matches("[yz]$"))

    # complement
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

    # exclusive or
    for selected in (
        df.select((cs.matches("e|g")) ^ cs.numeric()),
        df.select((cs.contains("b", "g")) ^ pl.col("eee")),
    ):
        assert selected.schema == OrderedDict(
            {
                "abc": pl.UInt16,
                "bbb": pl.UInt32,
                "eee": pl.Boolean,
                "fgg": pl.Boolean,
                "ghi": pl.Time,
            }
        )


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


def test_regex_expansion_group_by_9947() -> None:
    df = pl.DataFrame({"g": [3], "abc": [1], "abcd": [3]})
    assert df.group_by("g").agg(pl.col("^ab.*$")).columns == ["g", "abc", "abcd"]


def test_regex_expansion_exclude_10002() -> None:
    df = pl.DataFrame({"col_1": [1, 2, 3], "col_2": [2, 4, 3]})
    expected = pl.DataFrame({"col_1": [10, 20, 30], "col_2": [0.2, 0.4, 0.3]})

    assert_frame_equal(
        df.select(
            pl.col("^col_.*$").exclude("col_2").mul(10),
            pl.col("^col_.*$").exclude("col_1") / 10,
        ),
        expected,
    )


def test_is_selector() -> None:
    # only actual/compound selectors should pass this check
    assert is_selector(cs.numeric())
    assert is_selector(cs.by_dtype(pl.UInt32) | pl.col("xyz"))

    # expressions (and literals, etc) should fail
    assert not is_selector(pl.col("xyz"))
    assert not is_selector(cs.numeric().name.suffix(":num"))
    assert not is_selector(cs.date() + pl.col("time"))
    assert not is_selector(None)
    assert not is_selector("x")

    schema = {"x": pl.Int64, "y": pl.Float64}
    with pytest.raises(TypeError):
        expand_selector(schema, 999)

    with pytest.raises(TypeError):
        expand_selector(schema, "colname")


def test_selector_or() -> None:
    df = pl.DataFrame(
        {
            "int": [1, 2, 3],
            "float": [1.0, 2.0, 3.0],
            "str": ["x", "y", "z"],
        }
    ).with_row_index("idx")

    result = df.select(cs.by_name("idx") | ~cs.numeric())

    expected = pl.DataFrame(
        {"idx": [0, 1, 2], "str": ["x", "y", "z"]},
        schema_overrides={"idx": pl.UInt32},
    )
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    "selector",
    [
        (cs.string() | cs.numeric()),
        (cs.numeric() | cs.string()),
        ~(~cs.numeric() & ~cs.string()),
        ~(~cs.string() & ~cs.numeric()),
        (cs.signed_integer() ^ cs.contains("b", "e", "q")) - cs.starts_with("e"),
    ],
)
def test_selector_result_order(df: pl.DataFrame, selector: SelectorType) -> None:
    # ensure that selector results always match schema column-order
    assert df.select(selector).schema == OrderedDict(
        {
            "abc": pl.UInt16,
            "bbb": pl.UInt32,
            "cde": pl.Float64,
            "def": pl.Float32,
            "qqR": pl.String,
        }
    )
