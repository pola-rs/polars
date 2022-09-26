from __future__ import annotations

import pytest

import polars as pl


def test_extract_binary() -> None:
    df = pl.DataFrame({"foo": ["aron", "butler", "charly", "david"]})
    out = df.filter(pl.col("foo").str.extract("^(a)", 1) == "a").to_series()
    assert out[0] == "aron"


def test_auto_explode() -> None:
    df = pl.DataFrame(
        [pl.Series("val", ["A", "B", "C", "D"]), pl.Series("id", [1, 1, 2, 2])]
    )
    pl.col("val").str.concat(delimiter=",")
    grouped = (
        df.groupby("id")
        .agg(pl.col("val").str.concat(delimiter=",").alias("grouped"))
        .get_column("grouped")
    )
    assert grouped.dtype == pl.Utf8


def test_contains() -> None:
    df = pl.DataFrame(
        data=[(1, "some * * text"), (2, "(with) special\n * chars"), (3, "**etc...?$")],
        columns=["idx", "text"],
    )
    for pattern, as_literal, expected in (
        (r"\* \*", False, [True, False, False]),
        (r"* *", True, [True, False, False]),
        (r"^\(", False, [False, True, False]),
        (r"^\(", True, [False, False, False]),
        (r"(", True, [False, True, False]),
        (r"e", False, [True, True, True]),
        (r"e", True, [True, True, True]),
        (r"^\S+$", False, [False, False, True]),
        (r"\?\$", False, [False, False, True]),
        (r"?$", True, [False, False, True]),
    ):
        # series
        assert (
            expected == df["text"].str.contains(pattern, literal=as_literal).to_list()
        )
        # frame select
        assert (
            expected
            == df.select(pl.col("text").str.contains(pattern, literal=as_literal))[
                "text"
            ].to_list()
        )
        # frame filter
        assert sum(expected) == len(
            df.filter(pl.col("text").str.contains(pattern, literal=as_literal))
        )


def test_null_comparisons() -> None:
    s = pl.Series("s", [None, "str", "a"])
    assert (s.shift() == s).null_count() == 0
    assert (s.shift() != s).null_count() == 0


def test_replace() -> None:
    df = pl.DataFrame(
        data=[(1, "* * text"), (2, "(with) special\n * chars **etc...?$")],
        columns=["idx", "text"],
        orient="row",
    )
    for pattern, replacement, as_literal, expected in (
        (r"\*", "-", False, ["- * text", "(with) special\n - chars **etc...?$"]),
        (r"*", "-", True, ["- * text", "(with) special\n - chars **etc...?$"]),
        (r"^\(", "[", False, ["* * text", "[with) special\n * chars **etc...?$"]),
        (r"^\(", "[", True, ["* * text", "(with) special\n * chars **etc...?$"]),
        (r"t$", "an", False, ["* * texan", "(with) special\n * chars **etc...?$"]),
        (r"t$", "an", True, ["* * text", "(with) special\n * chars **etc...?$"]),
    ):
        # series
        assert (
            expected
            == df["text"]
            .str.replace(pattern, replacement, literal=as_literal)
            .to_list()
        )
        # expr
        assert (
            expected
            == df.select(
                pl.col("text").str.replace(pattern, replacement, literal=as_literal)
            )["text"].to_list()
        )


def test_replace_all() -> None:
    df = pl.DataFrame(
        data=[(1, "* * text"), (2, "(with) special * chars **etc...?$")],
        columns=["idx", "text"],
        orient="row",
    )
    for pattern, replacement, as_literal, expected in (
        (r"\*", "-", False, ["- - text", "(with) special - chars --etc...?$"]),
        (r"*", "-", True, ["- - text", "(with) special - chars --etc...?$"]),
        (r"\W", "", False, ["text", "withspecialcharsetc"]),
        (r".?$", "", True, ["* * text", "(with) special * chars **etc.."]),
        (
            r"(\b)[\w\s]{2,}(\b)",
            "$1(blah)$3",
            False,
            ["* * (blah)", "((blah)) (blah) * (blah) **(blah)...?$"],
        ),
    ):
        # series
        assert (
            expected
            == df["text"]
            .str.replace_all(pattern, replacement, literal=as_literal)
            .to_list()
        )
        # expr
        assert (
            expected
            == df.select(
                pl.col("text").str.replace_all(pattern, replacement, literal=as_literal)
            )["text"].to_list()
        )
        # invalid regex (but valid literal - requires "literal=True")
        with pytest.raises(pl.ComputeError):
            df["text"].str.replace_all("*", "")


def test_replace_expressions() -> None:
    df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"], "value": ["A", "B"]})
    out = df.select([pl.col("foo").str.replace(pl.col("foo").first(), pl.col("value"))])
    assert out.to_dict(False) == {"foo": ["A", "xyz 678 910t"]}
    out = df.select([pl.col("foo").str.replace(pl.col("foo").last(), "value")])
    assert out.to_dict(False) == {"foo": ["123 bla 45 asd", "value"]}

    df = pl.DataFrame(
        {"foo": ["1 bla 45 asd", "xyz 6t"], "pat": [r"\d", r"\W"], "value": ["A", "B"]}
    )
    out = df.select([pl.col("foo").str.replace_all(pl.col("pat").first(), "value")])
    assert out.to_dict(False) == {"foo": ["value bla valuevalue asd", "xyz valuet"]}


def test_extract_all_count() -> None:
    df = pl.DataFrame({"foo": ["123 bla 45 asd", "xyz 678 910t"]})
    assert (
        df.select(
            [
                pl.col("foo").str.extract_all(r"a").alias("extract"),
                pl.col("foo").str.count_match(r"a").alias("count"),
            ]
        ).to_dict(False)
    ) == {"extract": [["a", "a"], None], "count": [2, 0]}

    assert df["foo"].str.extract_all(r"a").dtype == pl.List
    assert df["foo"].str.count_match(r"a").dtype == pl.UInt32


def test_zfill() -> None:
    df = pl.DataFrame(
        {
            "num": [-10, -1, 0, 1, 10, 100, 1000, 10000, 100000, 1000000, None],
        }
    )
    out = [
        "-0010",
        "-0001",
        "00000",
        "00001",
        "00010",
        "00100",
        "01000",
        "10000",
        "100000",
        "1000000",
        None,
    ]
    assert (
        df.with_column(pl.col("num").cast(str).str.zfill(5)).to_series().to_list()
        == out
    )
    assert df["num"].cast(str).str.zfill(5).to_list() == out


def test_ljust_and_rjust() -> None:
    df = pl.DataFrame({"a": ["foo", "longer_foo", "longest_fooooooo", "hi"]})
    assert (
        df.select(
            [
                pl.col("a").str.rjust(10).alias("rjust"),
                pl.col("a").str.rjust(10).str.lengths().alias("rjust_len"),
                pl.col("a").str.ljust(10).alias("ljust"),
                pl.col("a").str.ljust(10).str.lengths().alias("ljust_len"),
            ]
        ).to_dict(False)
    ) == {
        "rjust": ["       foo", "longer_foo", "longest_fooooooo", "        hi"],
        "rjust_len": [10, 10, 16, 10],
        "ljust": ["foo       ", "longer_foo", "longest_fooooooo", "hi        "],
        "ljust_len": [10, 10, 16, 10],
    }


def test_format_empty_df() -> None:
    df = pl.DataFrame(
        [
            pl.Series("val1", [], dtype=pl.Categorical),
            pl.Series("val2", [], dtype=pl.Categorical),
        ]
    ).select(
        [
            pl.format("{}:{}", pl.col("val1"), pl.col("val2")).alias("cat"),
        ]
    )
    assert df.shape == (0, 1)
    assert df.dtypes == [pl.Utf8]


def test_starts_ends_with() -> None:
    assert pl.DataFrame({"a": ["hamburger", "nuts", "lollypop"]}).select(
        [
            pl.col("a").str.ends_with("pop").alias("pop"),
            pl.col("a").str.starts_with("ham").alias("ham"),
        ]
    ).to_dict(False) == {"pop": [False, False, True], "ham": [True, False, False]}
