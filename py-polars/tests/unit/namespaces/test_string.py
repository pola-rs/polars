from __future__ import annotations

from typing import cast

import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_str_slice() -> None:
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    assert df["a"].str.slice(-3).to_list() == ["bar", "foo"]
    assert df.select([pl.col("a").str.slice(2, 4)])["a"].to_list() == ["obar", "rfoo"]


def test_str_concat() -> None:
    s = pl.Series(["1", None, "2"])
    result = s.str.concat()
    expected = pl.Series(["1-null-2"])
    assert_series_equal(result, expected)


def test_str_concat2() -> None:
    df = pl.DataFrame({"foo": [1, None, 2]})
    df = df.select(pl.col("foo").str.concat("-"))
    assert cast(str, df.item()) == "1-null-2"


def test_str_lengths() -> None:
    s = pl.Series(["Café", None, "345", "東京"])
    expected = pl.Series([5, None, 3, 6], dtype=pl.UInt32)
    assert_series_equal(s.str.lengths(), expected)


def test_str_n_chars() -> None:
    s = pl.Series(["Café", None, "345", "東京"])
    expected = pl.Series([4, None, 3, 2], dtype=pl.UInt32)
    assert_series_equal(s.str.n_chars(), expected)


def test_str_contains() -> None:
    s = pl.Series(["messi", "ronaldo", "ibrahimovic"])
    expected = pl.Series([True, False, False])
    assert_series_equal(s.str.contains("mes"), expected)


def test_str_encode() -> None:
    s = pl.Series(["foo", "bar", None])
    hex_encoded = pl.Series(["666f6f", "626172", None])
    base64_encoded = pl.Series(["Zm9v", "YmFy", None])
    assert_series_equal(s.str.encode("hex"), hex_encoded)
    assert_series_equal(s.str.encode("base64"), base64_encoded)
    with pytest.raises(ValueError):
        s.str.encode("utf8")  # type: ignore[arg-type]


def test_str_decode() -> None:
    hex_encoded = pl.Series(["666f6f", "626172", None])
    base64_encoded = pl.Series(["Zm9v", "YmFy", None])
    expected = pl.Series([b"foo", b"bar", None])

    assert_series_equal(hex_encoded.str.decode("hex"), expected)
    assert_series_equal(base64_encoded.str.decode("base64"), expected)


def test_str_decode_exception() -> None:
    s = pl.Series(["not a valid", "626172", None])
    with pytest.raises(pl.ComputeError):
        s.str.decode(encoding="hex")
    with pytest.raises(pl.ComputeError):
        s.str.decode(encoding="base64")
    with pytest.raises(ValueError):
        s.str.decode("utf8")  # type: ignore[arg-type]


def test_hex_decode_return_dtype() -> None:
    data = {"a": ["68656c6c6f", "776f726c64"]}
    expr = pl.col("a").str.decode("hex")

    df = pl.DataFrame(data).select(expr)
    assert df.schema == {"a": pl.Binary}

    ldf = pl.LazyFrame(data).select(expr)
    assert ldf.schema == {"a": pl.Binary}


def test_base64_decode_return_dtype() -> None:
    data = {"a": ["Zm9v", "YmFy"]}
    expr = pl.col("a").str.decode("base64")

    df = pl.DataFrame(data).select(expr)
    assert df.schema == {"a": pl.Binary}

    ldf = pl.LazyFrame(data).select(expr)
    assert ldf.schema == {"a": pl.Binary}


def test_str_replace_str_replace_all() -> None:
    s = pl.Series(["hello", "world", "test", "rooted"])
    expected = pl.Series(["hell0", "w0rld", "test", "r0oted"])
    assert_series_equal(s.str.replace("o", "0"), expected)

    expected = pl.Series(["hell0", "w0rld", "test", "r00ted"])
    assert_series_equal(s.str.replace_all("o", "0"), expected)


def test_str_replace_n_single() -> None:
    s = pl.Series(["aba", "abaa"])

    assert s.str.replace("a", "b", n=1).to_list() == ["bba", "bbaa"]
    assert s.str.replace("a", "b", n=2).to_list() == ["bbb", "bbba"]
    assert s.str.replace("a", "b", n=3).to_list() == ["bbb", "bbbb"]


def test_str_replace_n_same_length() -> None:
    # pat and val have the same length
    # this triggers a fast path
    s = pl.Series(["abfeab", "foobarabfooabab"])
    assert s.str.replace("ab", "AB", n=1).to_list() == ["ABfeab", "foobarABfooabab"]
    assert s.str.replace("ab", "AB", n=2).to_list() == ["ABfeAB", "foobarABfooABab"]
    assert s.str.replace("ab", "AB", n=3).to_list() == ["ABfeAB", "foobarABfooABAB"]


def test_str_to_lowercase() -> None:
    s = pl.Series(["Hello", "WORLD"])
    expected = pl.Series(["hello", "world"])
    assert_series_equal(s.str.to_lowercase(), expected)


def test_str_to_uppercase() -> None:
    s = pl.Series(["Hello", "WORLD"])
    expected = pl.Series(["HELLO", "WORLD"])
    assert_series_equal(s.str.to_uppercase(), expected)


def test_str_case_cyrillic() -> None:
    vals = ["Biтpyк", "Iвaн"]
    s = pl.Series(vals)
    assert s.str.to_lowercase().to_list() == [a.lower() for a in vals]
    assert s.str.to_uppercase().to_list() == [a.upper() for a in vals]


def test_str_parse_int() -> None:
    bin = pl.Series(["110", "101", "010"])
    assert_series_equal(bin.str.parse_int(2), pl.Series([6, 5, 2]).cast(pl.Int32))

    hex = pl.Series(["fa1e", "ff00", "cafe", "invalid", None])
    assert_series_equal(
        hex.str.parse_int(16, strict=False),
        pl.Series([64030, 65280, 51966, None, None]).cast(pl.Int32),
        check_exact=True,
    )

    with pytest.raises(pl.ComputeError):
        hex.str.parse_int(16)


def test_str_parse_int_df() -> None:
    df = pl.DataFrame(
        {
            "bin": ["110", "101", "-010", "invalid", None],
            "hex": ["fa1e", "ff00", "cafe", "invalid", None],
        }
    )
    out = df.with_columns(
        [
            pl.col("bin").str.parse_int(2, strict=False),
            pl.col("hex").str.parse_int(16, strict=False),
        ]
    )

    expected = pl.DataFrame(
        {
            "bin": [6, 5, -2, None, None],
            "hex": [64030, 65280, 51966, None, None],
        }
    )
    assert out.frame_equal(expected)

    with pytest.raises(pl.ComputeError):
        df.with_columns(
            [pl.col("bin").str.parse_int(2), pl.col("hex").str.parse_int(16)]
        )


def test_str_strip() -> None:
    s = pl.Series([" hello ", "world\t "])
    expected = pl.Series(["hello", "world"])
    assert_series_equal(s.str.strip(), expected)

    expected = pl.Series(["hell", "world"])
    assert_series_equal(s.str.strip().str.strip("o"), expected)

    expected = pl.Series(["ell", "rld\t"])
    assert_series_equal(s.str.strip(" hwo"), expected)


def test_str_lstrip() -> None:
    s = pl.Series([" hello ", "\t world"])
    expected = pl.Series(["hello ", "world"])
    assert_series_equal(s.str.lstrip(), expected)

    expected = pl.Series(["ello ", "world"])
    assert_series_equal(s.str.lstrip().str.lstrip("h"), expected)

    expected = pl.Series(["ello ", "\t world"])
    assert_series_equal(s.str.lstrip("hw "), expected)


def test_str_rstrip() -> None:
    s = pl.Series([" hello ", "world\t "])
    expected = pl.Series([" hello", "world"])
    assert_series_equal(s.str.rstrip(), expected)

    expected = pl.Series([" hell", "world"])
    assert_series_equal(s.str.rstrip().str.rstrip("o"), expected)

    expected = pl.Series([" he", "wor"])
    assert_series_equal(s.str.rstrip("odl \t"), expected)


def test_str_strip_whitespace() -> None:
    s = pl.Series("a", ["trailing  ", "  leading", "  both  "])

    expected = pl.Series("a", ["trailing", "  leading", "  both"])
    assert_series_equal(s.str.rstrip(), expected)

    expected = pl.Series("a", ["trailing  ", "leading", "both  "])
    assert_series_equal(s.str.lstrip(), expected)

    expected = pl.Series("a", ["trailing", "leading", "both"])
    assert_series_equal(s.str.strip(), expected)


def test_str_split() -> None:
    a = pl.Series("a", ["a, b", "a", "ab,c,de"])
    for out in [a.str.split(","), pl.select(pl.lit(a).str.split(",")).to_series()]:
        assert out[0].to_list() == ["a", " b"]
        assert out[1].to_list() == ["a"]
        assert out[2].to_list() == ["ab", "c", "de"]

    for out in [
        a.str.split(",", inclusive=True),
        pl.select(pl.lit(a).str.split(",", inclusive=True)).to_series(),
    ]:
        assert out[0].to_list() == ["a,", " b"]
        assert out[1].to_list() == ["a"]
        assert out[2].to_list() == ["ab,", "c,", "de"]


def test_json_extract_series() -> None:
    s = pl.Series(["[1, 2, 3]", None, "[4, 5, 6]"])
    expected = pl.Series([[1, 2, 3], None, [4, 5, 6]])
    dtype = pl.List(pl.Int64)
    assert_series_equal(s.str.json_extract(None), expected)
    assert_series_equal(s.str.json_extract(dtype), expected)

    s = pl.Series(['{"a": 1, "b": true}', None, '{"a": 2, "b": false}'])
    expected = pl.Series([{"a": 1, "b": True}, None, {"a": 2, "b": False}])
    dtype2 = pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)])
    assert_series_equal(s.str.json_extract(None), expected)
    assert_series_equal(s.str.json_extract(dtype2), expected)

    expected = pl.Series([{"a": 1}, None, {"a": 2}])
    dtype2 = pl.Struct([pl.Field("a", pl.Int64)])
    assert_series_equal(s.str.json_extract(dtype2), expected)

    s = pl.Series([], dtype=pl.Utf8)
    expected = pl.Series([], dtype=pl.List(pl.Int64))
    dtype = pl.List(pl.Int64)
    assert_series_equal(s.str.json_extract(dtype), expected)


def test_json_extract_lazy_expr() -> None:
    dtype = pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)])
    ldf = (
        pl.DataFrame({"json": ['{"a": 1, "b": true}', None, '{"a": 2, "b": false}']})
        .lazy()
        .select(pl.col("json").str.json_extract(dtype))
    )
    expected = pl.DataFrame(
        {"json": [{"a": 1, "b": True}, None, {"a": 2, "b": False}]}
    ).lazy()
    assert ldf.schema == {"json": dtype}
    assert_frame_equal(ldf, expected)


def test_jsonpath_single() -> None:
    s = pl.Series(['{"a":"1"}', None, '{"a":2}', '{"a":2.1}', '{"a":true}'])
    expected = pl.Series(["1", None, "2", "2.1", "true"])
    assert_series_equal(s.str.json_path_match("$.a"), expected)


def test_extract_regex() -> None:
    s = pl.Series(
        [
            "http://vote.com/ballon_dor?candidate=messi&ref=polars",
            "http://vote.com/ballon_dor?candidat=jorginho&ref=polars",
            "http://vote.com/ballon_dor?candidate=ronaldo&ref=polars",
        ]
    )
    expected = pl.Series(["messi", None, "ronaldo"])
    assert_series_equal(s.str.extract(r"candidate=(\w+)", 1), expected)


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
    # test strict/non strict
    s_txt = pl.Series(["123", "456", "789"])
    assert (
        pl.Series([None, None, None]).cast(pl.Boolean).to_list()
        == s_txt.str.contains("(not_valid_regex", literal=False, strict=False).to_list()
    )
    with pytest.raises(pl.ComputeError):
        s_txt.str.contains("(not_valid_regex", literal=False, strict=True)
    assert (
        pl.Series([True, False, False]).cast(pl.Boolean).to_list()
        == s_txt.str.contains("1", literal=False, strict=False).to_list()
    )

    df = pl.DataFrame(
        data=[(1, "some * * text"), (2, "(with) special\n * chars"), (3, "**etc...?$")],
        schema=["idx", "text"],
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


def test_contains_expr() -> None:
    df = pl.DataFrame(
        {
            "text": [
                "some text",
                "(with) special\n .* chars",
                "**etc...?$",
                None,
                "b",
                "invalid_regex",
            ],
            "pattern": [r"[me]", r".*", r"^\(", "a", None, "*"],
        }
    )

    assert df.select(
        [
            pl.col("text")
            .str.contains(pl.col("pattern"), literal=False, strict=False)
            .alias("contains"),
            pl.col("text")
            .str.contains(pl.col("pattern"), literal=True)
            .alias("contains_lit"),
        ]
    ).to_dict(False) == {
        "contains": [True, True, False, False, False, None],
        "contains_lit": [False, True, False, False, False, False],
    }

    with pytest.raises(pl.ComputeError):
        df.select(
            pl.col("text").str.contains(pl.col("pattern"), literal=False, strict=True)
        )


def test_replace() -> None:
    df = pl.DataFrame(
        data=[(1, "* * text"), (2, "(with) special\n * chars **etc...?$")],
        schema=["idx", "text"],
        orient="row",
    )
    for pattern, replacement, as_literal, expected in (
        (r"\*", "-", False, ["- * text", "(with) special\n - chars **etc...?$"]),
        (r"*", "-", True, ["- * text", "(with) special\n - chars **etc...?$"]),
        (r"^\(", "[", False, ["* * text", "[with) special\n * chars **etc...?$"]),
        (r"^\(", "[", True, ["* * text", "(with) special\n * chars **etc...?$"]),
        (r"t$", "an", False, ["* * texan", "(with) special\n * chars **etc...?$"]),
        (r"t$", "an", True, ["* * text", "(with) special\n * chars **etc...?$"]),
        (r"(with) special", "$1", True, ["* * text", "$1\n * chars **etc...?$"]),
        (
            r"\((with)\) special",
            ":$1:",
            False,
            ["* * text", ":with:\n * chars **etc...?$"],
        ),
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

    assert pl.Series(["."]).str.replace(".", "$0", literal=True)[0] == "$0"
    assert pl.Series(["(.)(?)"]).str.replace(".", "$1", literal=True)[0] == "($1)(?)"


def test_replace_all() -> None:
    df = pl.DataFrame(
        data=[(1, "* * text"), (2, "(with) special\n * chars **etc...?$")],
        schema=["idx", "text"],
        orient="row",
    )
    for pattern, replacement, as_literal, expected in (
        (r"\*", "-", False, ["- - text", "(with) special\n - chars --etc...?$"]),
        (r"*", "-", True, ["- - text", "(with) special\n - chars --etc...?$"]),
        (r"\W", "", False, ["text", "withspecialcharsetc"]),
        (r".?$", "", True, ["* * text", "(with) special\n * chars **etc.."]),
        (
            r"(with) special",
            "$1",
            True,
            ["* * text", "$1\n * chars **etc...?$"],
        ),
        (
            r"\((with)\) special",
            ":$1:",
            False,
            ["* * text", ":with:\n * chars **etc...?$"],
        ),
        (
            r"(\b)[\w\s]{2,}(\b)",
            "$1(blah)$3",
            False,
            ["* * (blah)", "((blah)) (blah)\n * (blah) **(blah)...?$"],
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

    assert (
        pl.Series([r"(.)(\?)(\?)"]).str.replace_all("\\?", "$0", literal=True)[0]
        == "(.)($0)($0)"
    )
    assert (
        pl.Series([r"(.)(\?)(\?)"]).str.replace_all("\\?", "$0", literal=False)[0]
        == "(.)(\\?)(\\?)"
    )


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


def test_extract_all_many() -> None:
    df = pl.DataFrame({"foo": ["ab", "abc", "abcd"], "re": ["a", "bc", "a.c"]})
    assert df["foo"].str.extract_all(df["re"]).to_list() == [["a"], ["bc"], ["abc"]]


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
        df.with_columns(pl.col("num").cast(str).str.zfill(5)).to_series().to_list()
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


def test_starts_ends_with() -> None:
    df = pl.DataFrame(
        {"a": ["hamburger", "nuts", "lollypop"], "sub": ["ham", "ts", None]}
    )

    assert df.select(
        [
            pl.col("a").str.ends_with("pop").alias("ends_pop"),
            pl.col("a").str.ends_with(pl.lit(None)).alias("ends_None"),
            pl.col("a").str.ends_with(pl.col("sub")).alias("ends_sub"),
            pl.col("a").str.starts_with("ham").alias("starts_ham"),
            pl.col("a").str.starts_with(pl.lit(None)).alias("starts_None"),
            pl.col("a").str.starts_with(pl.col("sub")).alias("starts_sub"),
        ]
    ).to_dict(False) == {
        "ends_pop": [False, False, True],
        "ends_None": [False, False, False],
        "ends_sub": [False, True, False],
        "starts_ham": [True, False, False],
        "starts_None": [False, False, False],
        "starts_sub": [True, False, False],
    }


def test_json_path_match_type_4905() -> None:
    df = pl.DataFrame({"json_val": ['{"a":"hello"}', None, '{"a":"world"}']})
    assert df.filter(
        pl.col("json_val").str.json_path_match("$.a").is_in(["hello"])
    ).to_dict(False) == {"json_val": ['{"a":"hello"}']}


def test_length_vs_nchars() -> None:
    df = pl.DataFrame({"s": ["café", "東京"]}).with_columns(
        [
            pl.col("s").str.lengths().alias("length"),
            pl.col("s").str.n_chars().alias("nchars"),
        ]
    )
    assert df.rows() == [("café", 5, 4), ("東京", 6, 2)]


def test_decode_strict() -> None:
    df = pl.DataFrame(
        {"strings": ["0IbQvTc3", "0J%2FQldCf0JA%3D", "0J%2FRgNC%2B0YHRgtC%2B"]}
    )
    assert df.select(pl.col("strings").str.decode("base64", strict=False)).to_dict(
        False
    ) == {"strings": [b"\xd0\x86\xd0\xbd77", None, None]}
    with pytest.raises(pl.ComputeError):
        df.select(pl.col("strings").str.decode("base64", strict=True))


def test_split() -> None:
    df = pl.DataFrame({"x": ["a_a", None, "b", "c_c_c"]})
    out = df.select([pl.col("x").str.split("_")])

    expected = pl.DataFrame(
        [
            {"x": ["a", "a"]},
            {"x": None},
            {"x": ["b"]},
            {"x": ["c", "c", "c"]},
        ]
    )

    assert_frame_equal(out, expected)
    assert_frame_equal(df["x"].str.split("_").to_frame(), expected)

    out = df.select([pl.col("x").str.split("_", inclusive=True)])

    expected = pl.DataFrame(
        [
            {"x": ["a_", "a"]},
            {"x": None},
            {"x": ["b"]},
            {"x": ["c_", "c_", "c"]},
        ]
    )

    assert_frame_equal(out, expected)
    assert_frame_equal(df["x"].str.split("_", inclusive=True).to_frame(), expected)


def test_split_exact() -> None:
    df = pl.DataFrame({"x": ["a_a", None, "b", "c_c"]})
    out = df.select([pl.col("x").str.split_exact("_", 2, inclusive=False)]).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a", None, "b", "c"],
            "field_1": ["a", None, None, "c"],
            "field_2": pl.Series([None, None, None, None], dtype=pl.Utf8),
        }
    )

    assert_frame_equal(out, expected)
    out2 = df["x"].str.split_exact("_", 2, inclusive=False).to_frame().unnest("x")
    assert_frame_equal(out2, expected)

    out = df.select([pl.col("x").str.split_exact("_", 1, inclusive=True)]).unnest("x")

    expected = pl.DataFrame(
        {"field_0": ["a_", None, "b", "c_"], "field_1": ["a", None, None, "c"]}
    )
    assert_frame_equal(out, expected)
    assert df["x"].str.split_exact("_", 1).dtype == pl.Struct
    assert df["x"].str.split_exact("_", 1, inclusive=False).dtype == pl.Struct


def test_splitn() -> None:
    df = pl.DataFrame({"x": ["a_a", None, "b", "c_c_c"]})
    out = df.select([pl.col("x").str.splitn("_", 2)]).unnest("x")

    expected = pl.DataFrame(
        {"field_0": ["a", None, "b", "c"], "field_1": ["a", None, None, "c_c"]}
    )

    assert_frame_equal(out, expected)
    assert_frame_equal(df["x"].str.splitn("_", 2).to_frame().unnest("x"), expected)


def test_titlecase() -> None:
    df = pl.DataFrame(
        {
            "sing": [
                "welcome to my world",
                "THERE'S NO TURNING BACK",
                "double  space",
                "and\ta\t tab",
            ]
        }
    )

    assert df.select(pl.col("sing").str.to_titlecase()).to_dict(False) == {
        "sing": [
            "Welcome To My World",
            "There's No Turning Back",
            "Double  Space",
            "And\tA\t Tab",
        ]
    }
