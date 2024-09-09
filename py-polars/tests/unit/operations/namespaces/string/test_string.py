from __future__ import annotations

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import (
    ColumnNotFoundError,
    ComputeError,
    InvalidOperationError,
    SchemaError,
)
from polars.testing import assert_frame_equal, assert_series_equal


def test_str_slice() -> None:
    df = pl.DataFrame({"a": ["foobar", "barfoo"]})
    assert df["a"].str.slice(-3).to_list() == ["bar", "foo"]
    assert df.select([pl.col("a").str.slice(2, 4)])["a"].to_list() == ["obar", "rfoo"]


def test_str_slice_expr() -> None:
    df = pl.DataFrame(
        {
            "a": ["foobar", None, "barfoo", "abcd", ""],
            "offset": [1, 3, None, -3, 2],
            "length": [3, 4, 2, None, 2],
        }
    )
    out = df.select(
        all_expr=pl.col("a").str.slice("offset", "length"),
        offset_expr=pl.col("a").str.slice("offset", 2),
        length_expr=pl.col("a").str.slice(0, "length"),
        length_none=pl.col("a").str.slice("offset", None),
        offset_length_lit=pl.col("a").str.slice(-3, 3),
        str_lit=pl.lit("qwert").str.slice("offset", "length"),
    )
    expected = pl.DataFrame(
        {
            "all_expr": ["oob", None, None, "bcd", ""],
            "offset_expr": ["oo", None, None, "bc", ""],
            "length_expr": ["foo", None, "ba", "abcd", ""],
            "length_none": ["oobar", None, None, "bcd", ""],
            "offset_length_lit": ["bar", None, "foo", "bcd", ""],
            "str_lit": ["wer", "rt", None, "ert", "er"],
        }
    )
    assert_frame_equal(out, expected)

    # negative length is not allowed
    with pytest.raises(InvalidOperationError):
        df.select(pl.col("a").str.slice(0, -1))


@pytest.mark.parametrize(
    ("input", "n", "output"),
    [
        (["012345", "", None], 0, ["", "", None]),
        (["012345", "", None], 2, ["01", "", None]),
        (["012345", "", None], -2, ["0123", "", None]),
        (["012345", "", None], 100, ["012345", "", None]),
        (["012345", "", None], -100, ["", "", None]),
    ],
)
def test_str_head(input: list[str], n: int, output: list[str]) -> None:
    assert pl.Series(input).str.head(n).to_list() == output


@pytest.mark.parametrize(
    ("input", "n", "output"),
    [
        ("你好世界", 0, ""),
        ("你好世界", 2, "你好"),
        ("你好世界", 999, "你好世界"),
        ("你好世界", -1, "你好世"),
        ("你好世界", -2, "你好"),
        ("你好世界", -999, ""),
    ],
)
def test_str_head_codepoints(input: str, n: int, output: str) -> None:
    assert pl.Series([input]).str.head(n).to_list() == [output]


def test_str_head_expr() -> None:
    s = "012345"
    df = pl.DataFrame(
        {"a": [s, s, s, s, s, s, "", None], "n": [0, 2, -2, 100, -100, None, 3, -2]}
    )
    out = df.select(
        n_expr=pl.col("a").str.head("n"),
        n_pos2=pl.col("a").str.head(2),
        n_neg2=pl.col("a").str.head(-2),
        n_pos100=pl.col("a").str.head(100),
        n_pos_neg100=pl.col("a").str.head(-100),
        n_pos_0=pl.col("a").str.head(0),
        str_lit=pl.col("a").str.head(pl.lit(2)),
        lit_expr=pl.lit(s).str.head("n"),
        lit_n=pl.lit(s).str.head(2),
    )
    expected = pl.DataFrame(
        {
            "n_expr": ["", "01", "0123", "012345", "", None, "", None],
            "n_pos2": ["01", "01", "01", "01", "01", "01", "", None],
            "n_neg2": ["0123", "0123", "0123", "0123", "0123", "0123", "", None],
            "n_pos100": [s, s, s, s, s, s, "", None],
            "n_pos_neg100": ["", "", "", "", "", "", "", None],
            "n_pos_0": ["", "", "", "", "", "", "", None],
            "str_lit": ["01", "01", "01", "01", "01", "01", "", None],
            "lit_expr": ["", "01", "0123", "012345", "", None, "012", "0123"],
            "lit_n": ["01", "01", "01", "01", "01", "01", "01", "01"],
        }
    )
    assert_frame_equal(out, expected)


@pytest.mark.parametrize(
    ("input", "n", "output"),
    [
        (["012345", "", None], 0, ["", "", None]),
        (["012345", "", None], 2, ["45", "", None]),
        (["012345", "", None], -2, ["2345", "", None]),
        (["012345", "", None], 100, ["012345", "", None]),
        (["012345", "", None], -100, ["", "", None]),
    ],
)
def test_str_tail(input: list[str], n: int, output: list[str]) -> None:
    assert pl.Series(input).str.tail(n).to_list() == output


@pytest.mark.parametrize(
    ("input", "n", "output"),
    [
        ("你好世界", 0, ""),
        ("你好世界", 2, "世界"),
        ("你好世界", 999, "你好世界"),
        ("你好世界", -1, "好世界"),
        ("你好世界", -2, "世界"),
        ("你好世界", -999, ""),
    ],
)
def test_str_tail_codepoints(input: str, n: int, output: str) -> None:
    assert pl.Series([input]).str.tail(n).to_list() == [output]


def test_str_tail_expr() -> None:
    s = "012345"
    df = pl.DataFrame(
        {"a": [s, s, s, s, s, s, "", None], "n": [0, 2, -2, 100, -100, None, 3, -2]}
    )
    out = df.select(
        n_expr=pl.col("a").str.tail("n"),
        n_pos2=pl.col("a").str.tail(2),
        n_neg2=pl.col("a").str.tail(-2),
        n_pos100=pl.col("a").str.tail(100),
        n_pos_neg100=pl.col("a").str.tail(-100),
        n_pos_0=pl.col("a").str.tail(0),
        str_lit=pl.col("a").str.tail(pl.lit(2)),
        lit_expr=pl.lit(s).str.tail("n"),
        lit_n=pl.lit(s).str.tail(2),
    )
    expected = pl.DataFrame(
        {
            "n_expr": ["", "45", "2345", "012345", "", None, "", None],
            "n_pos2": ["45", "45", "45", "45", "45", "45", "", None],
            "n_neg2": ["2345", "2345", "2345", "2345", "2345", "2345", "", None],
            "n_pos100": [s, s, s, s, s, s, "", None],
            "n_pos_neg100": ["", "", "", "", "", "", "", None],
            "n_pos_0": ["", "", "", "", "", "", "", None],
            "str_lit": ["45", "45", "45", "45", "45", "45", "", None],
            "lit_expr": ["", "45", "2345", "012345", "", None, "345", "2345"],
            "lit_n": ["45", "45", "45", "45", "45", "45", "45", "45"],
        }
    )
    assert_frame_equal(out, expected)


def test_str_slice_multibyte() -> None:
    ref = "你好世界"
    s = pl.Series([ref])

    # Pad the string to simplify (negative) offsets starting before/after the string.
    npad = 20
    padref = "_" * npad + ref + "_" * npad
    for start in range(-5, 6):
        for length in range(6):
            offset = npad + start if start >= 0 else npad + start + len(ref)
            correct = padref[offset : offset + length].strip("_")
            result = s.str.slice(start, length)
            expected = pl.Series([correct])
            assert_series_equal(result, expected)


def test_str_len_bytes() -> None:
    s = pl.Series(["Café", None, "345", "東京"])
    result = s.str.len_bytes()
    expected = pl.Series([5, None, 3, 6], dtype=pl.UInt32)
    assert_series_equal(result, expected)


def test_str_len_chars() -> None:
    s = pl.Series(["Café", None, "345", "東京"])
    result = s.str.len_chars()
    expected = pl.Series([4, None, 3, 2], dtype=pl.UInt32)
    assert_series_equal(result, expected)


def test_str_contains() -> None:
    s = pl.Series(["messi", "ronaldo", "ibrahimovic"])
    expected = pl.Series([True, False, False])
    assert_series_equal(s.str.contains("mes"), expected)


def test_count_match_literal() -> None:
    s = pl.Series(["12 dbc 3xy", "cat\\w", "1zy3\\d\\d", None])
    out = s.str.count_matches(r"\d", literal=True)
    expected = pl.Series([0, 0, 2, None], dtype=pl.UInt32)
    assert_series_equal(out, expected)

    out = s.str.count_matches(pl.Series([r"\w", r"\w", r"\d", r"\d"]), literal=True)
    expected = pl.Series([0, 1, 2, None], dtype=pl.UInt32)
    assert_series_equal(out, expected)


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
    with pytest.raises(ComputeError):
        s.str.decode(encoding="hex")
    with pytest.raises(ComputeError):
        s.str.decode(encoding="base64")
    with pytest.raises(ValueError):
        s.str.decode("utf8")  # type: ignore[arg-type]


@pytest.mark.parametrize("strict", [True, False])
def test_str_find(strict: bool) -> None:
    df = pl.DataFrame(
        data=[
            ("Dubai", 3564931, "b[ai]", "ai"),
            ("Abu Dhabi", 1807000, "b[ai]", " "),
            ("Sharjah", 1405000, "[ai]n", "s"),
            ("Al Ain", 846747, "[ai]n", ""),
            ("Ajman", 490035, "[ai]n", "ma"),
            ("Ras Al Khaimah", 191753, "a.+a", "Kha"),
            ("Fujairah", 118933, "a.+a", None),
            ("Umm Al Quwain", 59098, "a.+a", "wa"),
            (None, None, None, "n/a"),
        ],
        schema={
            "city": pl.String,
            "population": pl.Int32,
            "pat": pl.String,
            "lit": pl.String,
        },
        orient="row",
    )
    city, pop, pat, lit = (pl.col(c) for c in ("city", "population", "pat", "lit"))

    for match_lit in (True, False):
        res = df.select(
            find_a_regex=city.str.find("(?i)a", strict=strict),
            find_a_lit=city.str.find("a", literal=match_lit),
            find_00_lit=pop.cast(pl.String).str.find("00", literal=match_lit),
            find_col_lit=city.str.find(lit, strict=strict, literal=match_lit),
            find_col_pat=city.str.find(pat, strict=strict),
        )
        assert res.to_dict(as_series=False) == {
            "find_a_regex": [3, 0, 2, 0, 0, 1, 3, 4, None],
            "find_a_lit": [3, 6, 2, None, 3, 1, 3, 10, None],
            "find_00_lit": [None, 4, 4, None, 2, None, None, None, None],
            "find_col_lit": [3, 3, None, 0, 2, 7, None, 9, None],
            "find_col_pat": [2, 7, None, 4, 3, 1, 3, None, None],
        }


def test_str_find_invalid_regex() -> None:
    # test behaviour of 'strict' with invalid regular expressions
    df = pl.DataFrame({"txt": ["AbCdEfG"]})
    rx_invalid = "(?i)AB.))"

    with pytest.raises(ComputeError):
        df.with_columns(pl.col("txt").str.find(rx_invalid, strict=True))

    res = df.with_columns(pl.col("txt").str.find(rx_invalid, strict=False))
    assert res.item() is None


def test_str_find_escaped_chars() -> None:
    # test behaviour of 'literal=True' with special chars
    df = pl.DataFrame({"txt": ["123.*465", "x(x?)x"]})

    res = df.with_columns(
        x1=pl.col("txt").str.find("(x?)", literal=True),
        x2=pl.col("txt").str.find(".*4", literal=True),
        x3=pl.col("txt").str.find("(x?)"),
        x4=pl.col("txt").str.find(".*4"),
    )
    # ┌──────────┬──────┬──────┬─────┬──────┐
    # │ txt      ┆ x1   ┆ x2   ┆ x3  ┆ x4   │
    # │ ---      ┆ ---  ┆ ---  ┆ --- ┆ ---  │
    # │ str      ┆ u32  ┆ u32  ┆ u32 ┆ u32  │
    # ╞══════════╪══════╪══════╪═════╪══════╡
    # │ 123.*465 ┆ null ┆ 3    ┆ 0   ┆ 0    │
    # │ x(x?)x   ┆ 1    ┆ null ┆ 0   ┆ null │
    # └──────────┴──────┴──────┴─────┴──────┘
    assert_frame_equal(
        pl.DataFrame(
            {
                "txt": ["123.*465", "x(x?)x"],
                "x1": [None, 1],
                "x2": [3, None],
                "x3": [0, 0],
                "x4": [0, None],
            }
        ).cast({cs.signed_integer(): pl.UInt32}),
        res,
    )


def test_hex_decode_return_dtype() -> None:
    data = {"a": ["68656c6c6f", "776f726c64"]}
    expr = pl.col("a").str.decode("hex")

    df = pl.DataFrame(data).select(expr)
    assert df.schema == {"a": pl.Binary}

    ldf = pl.LazyFrame(data).select(expr)
    assert ldf.collect_schema() == {"a": pl.Binary}


def test_base64_decode_return_dtype() -> None:
    data = {"a": ["Zm9v", "YmFy"]}
    expr = pl.col("a").str.decode("base64")

    df = pl.DataFrame(data).select(expr)
    assert df.schema == {"a": pl.Binary}

    ldf = pl.LazyFrame(data).select(expr)
    assert ldf.collect_schema() == {"a": pl.Binary}


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


def test_str_to_integer() -> None:
    bin = pl.Series(["110", "101", "010"])
    assert_series_equal(bin.str.to_integer(base=2), pl.Series([6, 5, 2]).cast(pl.Int64))

    hex = pl.Series(["fa1e", "ff00", "cafe", "invalid", None])
    assert_series_equal(
        hex.str.to_integer(base=16, strict=False),
        pl.Series([64030, 65280, 51966, None, None]).cast(pl.Int64),
        check_exact=True,
    )

    with pytest.raises(ComputeError):
        hex.str.to_integer(base=16)


def test_str_to_integer_base_expr() -> None:
    df = pl.DataFrame(
        {"str": ["110", "ff00", "234", None, "130"], "base": [2, 16, 10, 8, None]}
    )
    out = df.select(base_expr=pl.col("str").str.to_integer(base="base"))
    expected = pl.DataFrame({"base_expr": [6, 65280, 234, None, None]})
    assert_frame_equal(out, expected)

    # test strict raise
    df = pl.DataFrame({"str": ["110", "ff00", "cafe", None], "base": [2, 10, 10, 8]})

    with pytest.raises(ComputeError, match="failed for 2 value"):
        df.select(pl.col("str").str.to_integer(base="base"))


def test_str_to_integer_base_literal() -> None:
    df = pl.DataFrame(
        {
            "bin": ["110", "101", "-010", "invalid", None],
            "hex": ["fa1e", "ff00", "cafe", "invalid", None],
        }
    )
    result = df.with_columns(
        pl.col("bin").str.to_integer(base=2, strict=False),
        pl.col("hex").str.to_integer(base=16, strict=False),
    )

    expected = pl.DataFrame(
        {
            "bin": [6, 5, -2, None, None],
            "hex": [64030, 65280, 51966, None, None],
        }
    )
    assert_frame_equal(result, expected)

    with pytest.raises(ComputeError):
        df.with_columns(
            pl.col("bin").str.to_integer(base=2),
            pl.col("hex").str.to_integer(base=16),
        )


def test_str_strip_chars_expr() -> None:
    df = pl.DataFrame(
        {
            "s": [" hello ", "^^world^^", "&&hi&&", "  polars  ", None],
            "pat": [" ", "^", "&", None, "anything"],
        }
    )

    all_expr = df.select(
        pl.col("s").str.strip_chars(pl.col("pat")).alias("strip_chars"),
        pl.col("s").str.strip_chars_start(pl.col("pat")).alias("strip_chars_start"),
        pl.col("s").str.strip_chars_end(pl.col("pat")).alias("strip_chars_end"),
    )

    expected = pl.DataFrame(
        {
            "strip_chars": ["hello", "world", "hi", "polars", None],
            "strip_chars_start": ["hello ", "world^^", "hi&&", "polars  ", None],
            "strip_chars_end": [" hello", "^^world", "&&hi", "  polars", None],
        }
    )

    assert_frame_equal(all_expr, expected)

    strip_by_null = df.select(
        pl.col("s").str.strip_chars(None).alias("strip_chars"),
        pl.col("s").str.strip_chars_start(None).alias("strip_chars_start"),
        pl.col("s").str.strip_chars_end(None).alias("strip_chars_end"),
    )

    # only whitespace are striped.
    expected = pl.DataFrame(
        {
            "strip_chars": ["hello", "^^world^^", "&&hi&&", "polars", None],
            "strip_chars_start": ["hello ", "^^world^^", "&&hi&&", "polars  ", None],
            "strip_chars_end": [" hello", "^^world^^", "&&hi&&", "  polars", None],
        }
    )
    assert_frame_equal(strip_by_null, expected)


def test_str_strip_chars() -> None:
    s = pl.Series([" hello ", "world\t "])
    expected = pl.Series(["hello", "world"])
    assert_series_equal(s.str.strip_chars(), expected)

    expected = pl.Series(["hell", "world"])
    assert_series_equal(s.str.strip_chars().str.strip_chars("o"), expected)

    expected = pl.Series(["ell", "rld\t"])
    assert_series_equal(s.str.strip_chars(" hwo"), expected)


def test_str_strip_chars_start() -> None:
    s = pl.Series([" hello ", "\t world"])
    expected = pl.Series(["hello ", "world"])
    assert_series_equal(s.str.strip_chars_start(), expected)

    expected = pl.Series(["ello ", "world"])
    assert_series_equal(s.str.strip_chars_start().str.strip_chars_start("h"), expected)

    expected = pl.Series(["ello ", "\t world"])
    assert_series_equal(s.str.strip_chars_start("hw "), expected)


def test_str_strip_chars_end() -> None:
    s = pl.Series([" hello ", "world\t "])
    expected = pl.Series([" hello", "world"])
    assert_series_equal(s.str.strip_chars_end(), expected)

    expected = pl.Series([" hell", "world"])
    assert_series_equal(s.str.strip_chars_end().str.strip_chars_end("o"), expected)

    expected = pl.Series([" he", "wor"])
    assert_series_equal(s.str.strip_chars_end("odl \t"), expected)


def test_str_strip_whitespace() -> None:
    s = pl.Series("a", ["trailing  ", "  leading", "  both  "])

    expected = pl.Series("a", ["trailing", "  leading", "  both"])
    assert_series_equal(s.str.strip_chars_end(), expected)

    expected = pl.Series("a", ["trailing  ", "leading", "both  "])
    assert_series_equal(s.str.strip_chars_start(), expected)

    expected = pl.Series("a", ["trailing", "leading", "both"])
    assert_series_equal(s.str.strip_chars(), expected)


def test_str_strip_prefix_literal() -> None:
    s = pl.Series(["foo:bar", "foofoo:bar", "bar:bar", "foo", "", None])
    expected = pl.Series([":bar", "foo:bar", "bar:bar", "", "", None])
    assert_series_equal(s.str.strip_prefix("foo"), expected)
    # test null literal
    expected = pl.Series([None, None, None, None, None, None], dtype=pl.String)
    assert_series_equal(s.str.strip_prefix(pl.lit(None, dtype=pl.String)), expected)


def test_str_strip_prefix_suffix_expr() -> None:
    df = pl.DataFrame(
        {
            "s": ["foo-bar", "foobarbar", "barfoo", "", "anything", None],
            "prefix": ["foo", "foobar", "foo", "", None, "bar"],
            "suffix": ["bar", "barbar", "bar", "", None, "foo"],
        }
    )
    out = df.select(
        pl.col("s").str.strip_prefix(pl.col("prefix")).alias("strip_prefix"),
        pl.col("s").str.strip_suffix(pl.col("suffix")).alias("strip_suffix"),
    )
    assert out.to_dict(as_series=False) == {
        "strip_prefix": ["-bar", "bar", "barfoo", "", None, None],
        "strip_suffix": ["foo-", "foo", "barfoo", "", None, None],
    }


def test_str_strip_suffix() -> None:
    s = pl.Series(["foo:bar", "foo:barbar", "foo:foo", "bar", "", None])
    expected = pl.Series(["foo:", "foo:bar", "foo:foo", "", "", None])
    assert_series_equal(s.str.strip_suffix("bar"), expected)
    # test null literal
    expected = pl.Series([None, None, None, None, None, None], dtype=pl.String)
    assert_series_equal(s.str.strip_suffix(pl.lit(None, dtype=pl.String)), expected)


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


def test_json_decode_series() -> None:
    s = pl.Series(["[1, 2, 3]", None, "[4, 5, 6]"])
    expected = pl.Series([[1, 2, 3], None, [4, 5, 6]])
    dtype = pl.List(pl.Int64)
    assert_series_equal(s.str.json_decode(None), expected)
    assert_series_equal(s.str.json_decode(dtype), expected)

    s = pl.Series(['{"a": 1, "b": true}', None, '{"a": 2, "b": false}'])
    expected = pl.Series([{"a": 1, "b": True}, None, {"a": 2, "b": False}])
    dtype2 = pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)])
    assert_series_equal(s.str.json_decode(None), expected)
    assert_series_equal(s.str.json_decode(dtype2), expected)

    expected = pl.Series([{"a": 1}, None, {"a": 2}])
    dtype2 = pl.Struct([pl.Field("a", pl.Int64)])
    assert_series_equal(s.str.json_decode(dtype2), expected)

    s = pl.Series([], dtype=pl.String)
    expected = pl.Series([], dtype=pl.List(pl.Int64))
    dtype = pl.List(pl.Int64)
    assert_series_equal(s.str.json_decode(dtype), expected)


def test_json_decode_lazy_expr() -> None:
    dtype = pl.Struct([pl.Field("a", pl.Int64), pl.Field("b", pl.Boolean)])
    ldf = (
        pl.DataFrame({"json": ['{"a": 1, "b": true}', None, '{"a": 2, "b": false}']})
        .lazy()
        .select(pl.col("json").str.json_decode(dtype))
    )
    expected = pl.DataFrame(
        {"json": [{"a": 1, "b": True}, None, {"a": 2, "b": False}]}
    ).lazy()
    assert ldf.collect_schema() == {"json": dtype}
    assert_frame_equal(ldf, expected)


def test_json_decode_nested_struct() -> None:
    json = [
        '[{"key_1": "a"}]',
        '[{"key_1": "a2", "key_2": 2}]',
        '[{"key_1": "a3", "key_2": 3, "key_3": "c"}]',
    ]
    df = pl.DataFrame({"json_str": json})
    df_parsed = df.with_columns(
        pl.col("json_str").str.json_decode().alias("parsed_list_json")
    )

    expected_dtype = pl.List(
        pl.Struct(
            [
                pl.Field("key_1", pl.String),
                pl.Field("key_2", pl.Int64),
                pl.Field("key_3", pl.String),
            ]
        )
    )
    assert df_parsed.get_column("parsed_list_json").dtype == expected_dtype

    key_1_values = df_parsed.select(
        pl.col("parsed_list_json")
        .list.get(0)
        .struct.field("key_1")
        .alias("key_1_values")
    )
    expected_values = pl.Series("key_1_values", ["a", "a2", "a3"])
    assert_series_equal(key_1_values.get_column("key_1_values"), expected_values)


def test_json_decode_primitive_to_list_11053() -> None:
    df = pl.DataFrame(
        {
            "json": [
                '{"col1": ["123"], "col2": "123"}',
                '{"col1": ["xyz"], "col2": null}',
            ]
        }
    )
    schema = pl.Struct(
        {
            "col1": pl.List(pl.String),
            "col2": pl.List(pl.String),
        }
    )

    output = df.select(
        pl.col("json").str.json_decode(schema).alias("casted_json")
    ).unnest("casted_json")
    expected = pl.DataFrame({"col1": [["123"], ["xyz"]], "col2": [["123"], None]})
    assert_frame_equal(output, expected)


def test_jsonpath_single() -> None:
    s = pl.Series(['{"a":"1"}', None, '{"a":2}', '{"a":2.1}', '{"a":true}'])
    expected = pl.Series(["1", None, "2", "2.1", "true"])
    assert_series_equal(s.str.json_path_match("$.a"), expected)


def test_json_path_match() -> None:
    df = pl.DataFrame(
        {
            "str": [
                '{"a":"1"}',
                None,
                '{"b":2}',
                '{"a":2.1, "b": "hello"}',
                '{"a":true}',
            ],
            "pat": ["$.a", "$.a", "$.b", "$.b", None],
        }
    )
    out = df.select(
        all_expr=pl.col("str").str.json_path_match(pl.col("pat")),
        str_expr=pl.col("str").str.json_path_match("$.a"),
        pat_expr=pl.lit('{"a": 1.1, "b": 10}').str.json_path_match(pl.col("pat")),
    )
    expected = pl.DataFrame(
        {
            "all_expr": ["1", None, "2", "hello", None],
            "str_expr": ["1", None, None, "2.1", "true"],
            "pat_expr": ["1.1", "1.1", "10", "10", None],
        }
    )
    assert_frame_equal(out, expected)


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


def test_extract() -> None:
    df = pl.DataFrame(
        {
            "s": ["aron123", "12butler", "charly*", "~david", None],
            "pat": [r"^([a-zA-Z]+)", r"^(\d+)", None, "^(da)", r"(.*)"],
        }
    )

    out = df.select(
        all_expr=pl.col("s").str.extract(pl.col("pat"), 1),
        str_expr=pl.col("s").str.extract("^([a-zA-Z]+)", 1),
        pat_expr=pl.lit("aron123").str.extract(pl.col("pat")),
    )
    expected = pl.DataFrame(
        {
            "all_expr": ["aron", "12", None, None, None],
            "str_expr": ["aron", None, "charly", None, None],
            "pat_expr": ["aron", None, None, None, "aron123"],
        }
    )
    assert_frame_equal(out, expected)


def test_extract_binary() -> None:
    df = pl.DataFrame({"foo": ["aron", "butler", "charly", "david"]})
    out = df.filter(pl.col("foo").str.extract("^(a)", 1) == "a").to_series()
    assert out[0] == "aron"


def test_str_join_returns_scalar() -> None:
    df = pl.DataFrame(
        [pl.Series("val", ["A", "B", "C", "D"]), pl.Series("id", [1, 1, 2, 2])]
    )
    grouped = (
        df.group_by("id")
        .agg(pl.col("val").str.join(delimiter=",").alias("grouped"))
        .get_column("grouped")
    )
    assert grouped.dtype == pl.String


def test_contains() -> None:
    # test strict/non strict
    s_txt = pl.Series(["123", "456", "789"])
    assert (
        pl.Series([None, None, None]).cast(pl.Boolean).to_list()
        == s_txt.str.contains("(not_valid_regex", literal=False, strict=False).to_list()
    )
    with pytest.raises(ComputeError):
        s_txt.str.contains("(not_valid_regex", literal=False, strict=True)
    assert (
        pl.Series([True, False, False]).cast(pl.Boolean).to_list()
        == s_txt.str.contains("1", literal=False, strict=False).to_list()
    )

    df = pl.DataFrame(
        data=[(1, "some * * text"), (2, "(with) special\n * chars"), (3, "**etc...?$")],
        schema=["idx", "text"],
        orient="row",
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
        pl.col("text")
        .str.contains(pl.col("pattern"), literal=False, strict=False)
        .alias("contains"),
        pl.col("text")
        .str.contains(pl.col("pattern"), literal=True)
        .alias("contains_lit"),
    ).to_dict(as_series=False) == {
        "contains": [True, True, False, None, None, None],
        "contains_lit": [False, True, False, None, None, False],
    }

    with pytest.raises(ComputeError):
        df.select(
            pl.col("text").str.contains(pl.col("pattern"), literal=False, strict=True)
        )


@pytest.mark.parametrize(
    ("pattern", "case_insensitive", "expected"),
    [
        (["me"], False, True),
        (["Me"], False, False),
        (["Me"], True, True),
        (pl.Series(["me", "they"]), False, True),
        (pl.Series(["Me", "they"]), False, False),
        (pl.Series(["Me", "they"]), True, True),
        (["me", "they"], False, True),
        (["Me", "they"], False, False),
        (["Me", "they"], True, True),
    ],
)
def test_contains_any(
    pattern: pl.Series | list[str],
    case_insensitive: bool,
    expected: bool,
) -> None:
    df = pl.DataFrame({"text": ["Tell me what you want"]})
    # series
    assert (
        expected
        == df["text"]
        .str.contains_any(pattern, ascii_case_insensitive=case_insensitive)
        .item()
    )
    # expr
    assert (
        expected
        == df.select(
            pl.col("text").str.contains_any(
                pattern, ascii_case_insensitive=case_insensitive
            )
        )["text"].item()
    )
    # frame filter
    assert int(expected) == len(
        df.filter(
            pl.col("text").str.contains_any(
                pattern, ascii_case_insensitive=case_insensitive
            )
        )
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
        with pytest.raises(ComputeError):
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
    assert out.to_dict(as_series=False) == {"foo": ["A", "xyz 678 910t"]}
    out = df.select([pl.col("foo").str.replace(pl.col("foo").last(), "value")])
    assert out.to_dict(as_series=False) == {"foo": ["123 bla 45 asd", "value"]}

    df = pl.DataFrame(
        {"foo": ["1 bla 45 asd", "xyz 6t"], "pat": [r"\d", r"\W"], "value": ["A", "B"]}
    )
    out = df.select([pl.col("foo").str.replace_all(pl.col("pat").first(), "value")])
    assert out.to_dict(as_series=False) == {
        "foo": ["value bla valuevalue asd", "xyz valuet"]
    }


@pytest.mark.parametrize(
    ("pattern", "replacement", "case_insensitive", "expected"),
    [
        (["say"], "", False, "Tell me what you want"),
        (["me"], ["them"], False, "Tell them what you want"),
        (["who"], ["them"], False, "Tell me what you want"),
        (["me", "you"], "it", False, "Tell it what it want"),
        (["Me", "you"], "it", False, "Tell me what it want"),
        (["me", "you"], ["it"], False, "Tell it what it want"),
        (["me", "you"], ["you", "me"], False, "Tell you what me want"),
        (["me", "You", "them"], "it", False, "Tell it what you want"),
        (["Me", "you"], "it", True, "Tell it what it want"),
        (["me", "YOU"], ["you", "me"], True, "Tell you what me want"),
        (pl.Series(["me", "YOU"]), ["you", "me"], False, "Tell you what you want"),
        (pl.Series(["me", "YOU"]), ["you", "me"], True, "Tell you what me want"),
    ],
)
def test_replace_many(
    pattern: pl.Series | list[str],
    replacement: pl.Series | list[str] | str,
    case_insensitive: bool,
    expected: str,
) -> None:
    df = pl.DataFrame({"text": ["Tell me what you want"]})
    # series
    assert (
        expected
        == df["text"]
        .str.replace_many(pattern, replacement, ascii_case_insensitive=case_insensitive)
        .item()
    )
    # expr
    assert (
        expected
        == df.select(
            pl.col("text").str.replace_many(
                pattern,
                replacement,
                ascii_case_insensitive=case_insensitive,
            )
        ).item()
    )


@pytest.mark.parametrize(
    ("mapping", "case_insensitive", "expected"),
    [
        ({}, False, "Tell me what you want"),
        ({"me": "them"}, False, "Tell them what you want"),
        ({"who": "them"}, False, "Tell me what you want"),
        ({"me": "it", "you": "it"}, False, "Tell it what it want"),
        ({"Me": "it", "you": "it"}, False, "Tell me what it want"),
        ({"me": "you", "you": "me"}, False, "Tell you what me want"),
        ({}, True, "Tell me what you want"),
        ({"Me": "it", "you": "it"}, True, "Tell it what it want"),
        ({"me": "you", "YOU": "me"}, True, "Tell you what me want"),
    ],
)
def test_replace_many_mapping(
    mapping: dict[str, str],
    case_insensitive: bool,
    expected: str,
) -> None:
    df = pl.DataFrame({"text": ["Tell me what you want"]})
    # series
    assert (
        expected
        == df["text"]
        .str.replace_many(mapping, ascii_case_insensitive=case_insensitive)
        .item()
    )
    # expr
    assert (
        expected
        == df.select(
            pl.col("text").str.replace_many(
                mapping,
                ascii_case_insensitive=case_insensitive,
            )
        ).item()
    )


def test_replace_many_invalid_inputs() -> None:
    df = pl.DataFrame({"text": ["Tell me what you want"]})

    # Ensure a string as the first argument is parsed as a column name.
    with pytest.raises(ColumnNotFoundError, match="me"):
        df.select(pl.col("text").str.replace_many("me", "you"))

    with pytest.raises(SchemaError):
        df.select(pl.col("text").str.replace_many(1, 2))

    with pytest.raises(SchemaError):
        df.select(pl.col("text").str.replace_many([1], [2]))

    with pytest.raises(SchemaError):
        df.select(pl.col("text").str.replace_many(["me"], None))

    with pytest.raises(TypeError):
        df.select(pl.col("text").str.replace_many(["me"]))

    with pytest.raises(
        InvalidOperationError,
        match="expected the same amount of patterns as replacement strings",
    ):
        df.select(pl.col("text").str.replace_many(["a"], ["b", "c"]))

    s = df.to_series()

    with pytest.raises(ColumnNotFoundError, match="me"):
        s.str.replace_many("me", "you")  # type: ignore[arg-type]

    with pytest.raises(SchemaError):
        df.select(pl.col("text").str.replace_many(["me"], None))

    with pytest.raises(TypeError):
        df.select(pl.col("text").str.replace_many(["me"]))

    with pytest.raises(
        InvalidOperationError,
        match="expected the same amount of patterns as replacement strings",
    ):
        s.str.replace_many(["a"], ["b", "c"])


def test_extract_all_count() -> None:
    df = pl.DataFrame({"foo": ["123 bla 45 asd", "xaz 678 910t", "boo", None]})
    assert (
        df.select(
            pl.col("foo").str.extract_all(r"a").alias("extract"),
            pl.col("foo").str.count_matches(r"a").alias("count"),
        ).to_dict(as_series=False)
    ) == {"extract": [["a", "a"], ["a"], [], None], "count": [2, 1, 0, None]}

    assert df["foo"].str.extract_all(r"a").dtype == pl.List
    assert df["foo"].str.count_matches(r"a").dtype == pl.UInt32


def test_count_matches_many() -> None:
    df = pl.DataFrame(
        {
            "foo": ["123 bla 45 asd", "xyz 678 910t", None, "boo"],
            "bar": [r"\d", r"[a-z]", r"\d", None],
        }
    )
    assert (
        df.select(
            pl.col("foo").str.count_matches(pl.col("bar")).alias("count")
        ).to_dict(as_series=False)
    ) == {"count": [5, 4, None, None]}

    assert df["foo"].str.count_matches(df["bar"]).dtype == pl.UInt32

    # Test broadcast.
    broad = df.select(
        pl.col("foo").str.count_matches(pl.col("bar").first()).alias("count"),
        pl.col("foo").str.count_matches(pl.col("bar").last()).alias("count_null"),
    )
    assert broad.to_dict(as_series=False) == {
        "count": [5, 6, None, 0],
        "count_null": [None, None, None, None],
    }
    assert broad.schema == {"count": pl.UInt32, "count_null": pl.UInt32}


def test_extract_all_many() -> None:
    df = pl.DataFrame(
        {
            "foo": ["ab", "abc", "abcd", "foo", None, "boo"],
            "re": ["a", "bc", "a.c", "a", "a", None],
        }
    )
    assert df["foo"].str.extract_all(df["re"]).to_list() == [
        ["a"],
        ["bc"],
        ["abc"],
        [],
        None,
        None,
    ]

    # Test broadcast.
    broad = df.select(
        pl.col("foo").str.extract_all(pl.col("re").first()).alias("a"),
        pl.col("foo").str.extract_all(pl.col("re").last()).alias("null"),
    )
    assert broad.to_dict(as_series=False) == {
        "a": [["a"], ["a"], ["a"], [], None, []],
        "null": [None] * 6,
    }
    assert broad.schema == {"a": pl.List(pl.String), "null": pl.List(pl.String)}


def test_extract_groups() -> None:
    def _named_groups_builder(pattern: str, groups: dict[str, str]) -> str:
        return pattern.format(
            **{name: f"(?<{name}>{value})" for name, value in groups.items()}
        )

    expected = {
        "authority": ["ISO", "ISO/IEC/IEEE"],
        "spec_num": ["80000", "29148"],
        "part_num": ["1", None],
        "revision_year": ["2009", "2018"],
    }

    pattern = _named_groups_builder(
        r"{authority}\s{spec_num}(?:-{part_num})?(?::{revision_year})",
        {
            "authority": r"^ISO(?:/[A-Z]+)*",
            "spec_num": r"\d+",
            "part_num": r"\d+",
            "revision_year": r"\d{4}",
        },
    )

    df = pl.DataFrame({"iso_code": ["ISO 80000-1:2009", "ISO/IEC/IEEE 29148:2018"]})

    assert (
        df.select(pl.col("iso_code").str.extract_groups(pattern))
        .unnest("iso_code")
        .to_dict(as_series=False)
        == expected
    )

    assert df.select(pl.col("iso_code").str.extract_groups("")).to_dict(
        as_series=False
    ) == {"iso_code": [{"iso_code": None}, {"iso_code": None}]}

    assert df.select(
        pl.col("iso_code").str.extract_groups(r"\A(ISO\S*).*?(\d+)")
    ).to_dict(as_series=False) == {
        "iso_code": [{"1": "ISO", "2": "80000"}, {"1": "ISO/IEC/IEEE", "2": "29148"}]
    }

    assert df.select(
        pl.col("iso_code").str.extract_groups(r"\A(ISO\S*).*?(?<year>\d+)\z")
    ).to_dict(as_series=False) == {
        "iso_code": [
            {"1": "ISO", "year": "2009"},
            {"1": "ISO/IEC/IEEE", "year": "2018"},
        ]
    }

    assert pl.select(
        pl.lit(r"foobar").str.extract_groups(r"(?<foo>.{3})|(?<bar>...)")
    ).to_dict(as_series=False) == {"literal": [{"foo": "foo", "bar": None}]}


def test_starts_ends_with() -> None:
    df = pl.DataFrame(
        {
            "a": ["hamburger", "nuts", "lollypop", None],
            "sub": ["ham", "ts", None, "anything"],
        }
    )

    assert df.select(
        pl.col("a").str.ends_with("pop").alias("ends_pop"),
        pl.col("a").str.ends_with(pl.lit(None)).alias("ends_None"),
        pl.col("a").str.ends_with(pl.col("sub")).alias("ends_sub"),
        pl.col("a").str.starts_with("ham").alias("starts_ham"),
        pl.col("a").str.starts_with(pl.lit(None)).alias("starts_None"),
        pl.col("a").str.starts_with(pl.col("sub")).alias("starts_sub"),
    ).to_dict(as_series=False) == {
        "ends_pop": [False, False, True, None],
        "ends_None": [None, None, None, None],
        "ends_sub": [False, True, None, None],
        "starts_ham": [True, False, False, None],
        "starts_None": [None, None, None, None],
        "starts_sub": [True, False, None, None],
    }


def test_json_path_match_type_4905() -> None:
    df = pl.DataFrame({"json_val": ['{"a":"hello"}', None, '{"a":"world"}']})
    assert df.filter(
        pl.col("json_val").str.json_path_match("$.a").is_in(["hello"])
    ).to_dict(as_series=False) == {"json_val": ['{"a":"hello"}']}


def test_decode_strict() -> None:
    df = pl.DataFrame(
        {"strings": ["0IbQvTc3", "0J%2FQldCf0JA%3D", "0J%2FRgNC%2B0YHRgtC%2B"]}
    )
    result = df.select(pl.col("strings").str.decode("base64", strict=False))
    expected = {"strings": [b"\xd0\x86\xd0\xbd77", None, None]}
    assert result.to_dict(as_series=False) == expected

    with pytest.raises(ComputeError):
        df.select(pl.col("strings").str.decode("base64", strict=True))


def test_split() -> None:
    df = pl.DataFrame({"x": ["a_a", None, "b", "c_c_c", ""]})
    out = df.select([pl.col("x").str.split("_")])

    expected = pl.DataFrame(
        [
            {"x": ["a", "a"]},
            {"x": None},
            {"x": ["b"]},
            {"x": ["c", "c", "c"]},
            {"x": [""]},
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
            {"x": []},
        ]
    )

    assert_frame_equal(out, expected)
    assert_frame_equal(df["x"].str.split("_", inclusive=True).to_frame(), expected)

    out = df.select([pl.col("x").str.split("")])

    expected = pl.DataFrame(
        [
            {"x": ["a", "_", "a"]},
            {"x": None},
            {"x": ["b"]},
            {"x": ["c", "_", "c", "_", "c"]},
            {"x": []},
        ]
    )

    assert_frame_equal(out, expected)
    assert_frame_equal(df["x"].str.split("").to_frame(), expected)

    out = df.select([pl.col("x").str.split("", inclusive=True)])

    expected = pl.DataFrame(
        [
            {"x": ["a", "_", "a"]},
            {"x": None},
            {"x": ["b"]},
            {"x": ["c", "_", "c", "_", "c"]},
            {"x": []},
        ]
    )

    assert_frame_equal(out, expected)
    assert_frame_equal(df["x"].str.split("", inclusive=True).to_frame(), expected)

    plan = (
        df.lazy()
        .select(
            a=pl.col("x").str.split(" ", inclusive=False),
            b=pl.col("x").str.split_exact(" ", 1, inclusive=False),
        )
        .explain()
    )

    assert "str.split(" in plan
    assert "str.split_exact(" in plan

    plan = (
        df.lazy()
        .select(
            a=pl.col("x").str.split(" ", inclusive=True),
            b=pl.col("x").str.split_exact(" ", 1, inclusive=True),
        )
        .explain()
    )

    assert "str.split_inclusive(" in plan
    assert "str.split_exact_inclusive(" in plan


def test_split_expr() -> None:
    df = pl.DataFrame(
        {
            "x": ["a_a", None, "b", "c*c*c", "dddd", ""],
            "by": ["_", "#", "^", "*", "", ""],
        }
    )
    out = df.select([pl.col("x").str.split(pl.col("by"))])
    expected = pl.DataFrame(
        [
            {"x": ["a", "a"]},
            {"x": None},
            {"x": ["b"]},
            {"x": ["c", "c", "c"]},
            {"x": ["d", "d", "d", "d"]},
            {"x": []},
        ]
    )
    assert_frame_equal(out, expected)

    out = df.select([pl.col("x").str.split(pl.col("by"), inclusive=True)])
    expected = pl.DataFrame(
        [
            {"x": ["a_", "a"]},
            {"x": None},
            {"x": ["b"]},
            {"x": ["c*", "c*", "c"]},
            {"x": ["d", "d", "d", "d"]},
            {"x": []},
        ]
    )
    assert_frame_equal(out, expected)


def test_split_exact() -> None:
    df = pl.DataFrame({"x": ["a_a", None, "b", "c_c", ""]})
    out = df.select([pl.col("x").str.split_exact("_", 2, inclusive=False)]).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a", None, "b", "c", ""],
            "field_1": ["a", None, None, "c", None],
            "field_2": pl.Series([None, None, None, None, None], dtype=pl.String),
        }
    )

    assert_frame_equal(out, expected)
    out2 = df["x"].str.split_exact("_", 2, inclusive=False).to_frame().unnest("x")
    assert_frame_equal(out2, expected)

    out = df.select([pl.col("x").str.split_exact("_", 1, inclusive=True)]).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a_", None, "b", "c_", None],
            "field_1": ["a", None, None, "c", None],
        }
    )
    assert_frame_equal(out, expected)
    assert df["x"].str.split_exact("_", 1).dtype == pl.Struct
    assert df["x"].str.split_exact("_", 1, inclusive=False).dtype == pl.Struct

    out = df.select([pl.col("x").str.split_exact("", 1)]).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a", None, "b", "c", None],
            "field_1": ["_", None, None, "_", None],
        }
    )
    assert_frame_equal(out, expected)

    out = df.select([pl.col("x").str.split_exact("", 1, inclusive=True)]).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a", None, "b", "c", None],
            "field_1": ["_", None, None, "_", None],
        }
    )
    assert_frame_equal(out, expected)


def test_split_exact_expr() -> None:
    df = pl.DataFrame(
        {
            "x": ["a_a", None, "b", "c^c^c", "d#d", "eeee", ""],
            "by": ["_", "&", "$", "^", None, "", ""],
        }
    )

    out = df.select(
        pl.col("x").str.split_exact(pl.col("by"), 2, inclusive=False)
    ).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a", None, "b", "c", None, "e", None],
            "field_1": ["a", None, None, "c", None, "e", None],
            "field_2": pl.Series(
                [None, None, None, "c", None, "e", None], dtype=pl.String
            ),
        }
    )

    assert_frame_equal(out, expected)

    out2 = df.select(
        pl.col("x").str.split_exact(pl.col("by"), 2, inclusive=True)
    ).unnest("x")

    expected2 = pl.DataFrame(
        {
            "field_0": ["a_", None, "b", "c^", None, "e", None],
            "field_1": ["a", None, None, "c^", None, "e", None],
            "field_2": pl.Series(
                [None, None, None, "c", None, "e", None], dtype=pl.String
            ),
        }
    )
    assert_frame_equal(out2, expected2)


def test_splitn() -> None:
    df = pl.DataFrame({"x": ["a_a", None, "b", "c_c_c", ""]})
    out = df.select([pl.col("x").str.splitn("_", 2)]).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a", None, "b", "c", ""],
            "field_1": ["a", None, None, "c_c", None],
        }
    )

    assert_frame_equal(out, expected)
    assert_frame_equal(df["x"].str.splitn("_", 2).to_frame().unnest("x"), expected)

    out = df.select([pl.col("x").str.splitn("", 2)]).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a", None, "b", "c", None],
            "field_1": ["_a", None, None, "_c_c", None],
        }
    )

    assert_frame_equal(out, expected)
    assert_frame_equal(df["x"].str.splitn("", 2).to_frame().unnest("x"), expected)


def test_splitn_expr() -> None:
    df = pl.DataFrame(
        {
            "x": ["a_a", None, "b", "c^c^c", "d#d", "eeee", ""],
            "by": ["_", "&", "$", "^", None, "", ""],
        }
    )

    out = df.select(pl.col("x").str.splitn(pl.col("by"), 2)).unnest("x")

    expected = pl.DataFrame(
        {
            "field_0": ["a", None, "b", "c", None, "e", None],
            "field_1": ["a", None, None, "c^c", None, "eee", None],
        }
    )

    assert_frame_equal(out, expected)


def test_titlecase() -> None:
    df = pl.DataFrame(
        {
            "misc": [
                "welcome to my world",
                "double  space",
                "and\ta\t tab",
                "by jean-paul sartre, 'esq'",
                "SOMETIMES/life/gives/you/a/2nd/chance",
            ],
        }
    )
    expected = [
        "Welcome To My World",
        "Double  Space",
        "And\tA\t Tab",
        "By Jean-Paul Sartre, 'Esq'",
        "Sometimes/Life/Gives/You/A/2nd/Chance",
    ]
    actual = df.select(pl.col("misc").str.to_titlecase()).to_series()
    for ex, act in zip(expected, actual):
        assert ex == act, f"{ex} != {act}"

    df = pl.DataFrame(
        {
            "quotes": [
                "'e.t. phone home'",
                "you talkin' to me?",
                "i feel the need--the need for speed",
                "to infinity,and BEYOND!",
                "say 'what' again!i dare you - I\u00a0double-dare you!",
                "What.we.got.here... is#failure#to#communicate",
            ]
        }
    )
    expected_str = [
        "'E.T. Phone Home'",
        "You Talkin' To Me?",
        "I Feel The Need--The Need For Speed",
        "To Infinity,And Beyond!",
        "Say 'What' Again!I Dare You - I\u00a0Double-Dare You!",
        "What.We.Got.Here... Is#Failure#To#Communicate",
    ]
    expected_py = [s.title() for s in df["quotes"].to_list()]
    for ex_str, ex_py, act in zip(
        expected_str, expected_py, df["quotes"].str.to_titlecase()
    ):
        assert ex_str == act, f"{ex_str} != {act}"
        assert ex_py == act, f"{ex_py} != {act}"


def test_string_replace_with_nulls_10124() -> None:
    df = pl.DataFrame({"col1": ["S", "S", "S", None, "S", "S", "S", "S"]})

    assert df.select(
        pl.col("col1"),
        pl.col("col1").str.replace("S", "O", n=1).alias("n_1"),
        pl.col("col1").str.replace("S", "O", n=3).alias("n_3"),
    ).to_dict(as_series=False) == {
        "col1": ["S", "S", "S", None, "S", "S", "S", "S"],
        "n_1": ["O", "O", "O", None, "O", "O", "O", "O"],
        "n_3": ["O", "O", "O", None, "O", "O", "O", "O"],
    }


def test_string_extract_groups_lazy_schema_10305() -> None:
    df = pl.LazyFrame(
        data={
            "url": [
                "http://vote.com/ballon_dor?candidate=messi&ref=python",
                "http://vote.com/ballon_dor?candidate=weghorst&ref=polars",
                "http://vote.com/ballon_dor?error=404&ref=rust",
            ]
        }
    )
    pattern = r"candidate=(?<candidate>\w+)&ref=(?<ref>\w+)"
    df = df.select(captures=pl.col("url").str.extract_groups(pattern)).unnest(
        "captures"
    )

    assert df.collect_schema() == {"candidate": pl.String, "ref": pl.String}


def test_string_reverse() -> None:
    df = pl.DataFrame(
        {
            "text": [None, "foo", "bar", "i like pizza&#", None, "man\u0303ana"],
        }
    )
    expected = pl.DataFrame(
        [
            pl.Series(
                "text",
                [None, "oof", "rab", "#&azzip ekil i", None, "anan\u0303am"],
                dtype=pl.String,
            ),
        ]
    )

    result = df.select(pl.col("text").str.reverse())
    assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("data", "expected_data"),
    [
        (["", None, "a"], ["", None, "b"]),
        ([None, None, "a"], [None, None, "b"]),
        (["", "", ""], ["", "", ""]),
        ([None, None, None], [None, None, None]),
        (["a", "", None], ["b", "", None]),
    ],
)
def test_replace_lit_n_char_13385(
    data: list[str | None], expected_data: list[str | None]
) -> None:
    s = pl.Series(data, dtype=pl.String)
    res = s.str.replace("a", "b", literal=True)
    expected_s = pl.Series(expected_data, dtype=pl.String)
    assert_series_equal(res, expected_s)


def test_extract_many() -> None:
    df = pl.DataFrame({"values": ["discontent"]})
    patterns = ["winter", "disco", "onte", "discontent"]
    assert (
        df.with_columns(
            pl.col("values")
            .str.extract_many(patterns, overlapping=False)
            .alias("matches"),
            pl.col("values")
            .str.extract_many(patterns, overlapping=True)
            .alias("matches_overlapping"),
        )
    ).to_dict(as_series=False) == {
        "values": ["discontent"],
        "matches": [["disco"]],
        "matches_overlapping": [["disco", "onte", "discontent"]],
    }

    # many patterns
    df = pl.DataFrame(
        {
            "values": ["discontent", "rhapsody"],
            "patterns": [
                ["winter", "disco", "onte", "discontent"],
                ["rhap", "ody", "coalesce"],
            ],
        }
    )

    assert df.select(pl.col("values").str.extract_many("patterns")).to_dict(
        as_series=False
    ) == {"values": [["disco"], ["rhap", "ody"]]}
