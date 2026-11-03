from __future__ import annotations

import gzip
import io
import re
import zlib
from typing import TYPE_CHECKING

import pytest
import zstandard

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("separator", "data"),
    [
        (r"\s+", "  1 \t 2  \n 3    4\t\n"),
        (r"[,;]+", "1,,2\n3;4\n"),
        (r"(,|;)", "1,2\n3;4\n"),
        (r"(?<=\d)::", "1::2\n3::4\n"),
        (r"[\u2003\t]+", "1\u20032\n3\t4\n"),
        (re.compile(r"[,;]"), "1,2\n3;4\n"),
    ],
)
def test_regex_separator(separator: str | re.Pattern[str], data: str) -> None:
    result = pl.read_csv(data.encode(), separator=separator, has_header=False)
    expected = pl.DataFrame({"column_0": [1, 3], "column_1": [2, 4]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("kind", ["bytes", "string_io", "bytes_io", "path"])
def test_regex_sources(kind: str, tmp_path: Path) -> None:
    data = "a   b\n1\t2\n"
    source: Path | io.StringIO | io.BytesIO | bytes
    if kind == "path":
        source = tmp_path / "data.txt"
        source.write_text(data, encoding="utf8")
    elif kind == "string_io":
        source = io.StringIO(data)
    elif kind == "bytes_io":
        source = io.BytesIO(data.encode())
    else:
        source = data.encode()
    result = pl.read_csv(source, separator=r"\s+")
    assert_frame_equal(result, pl.DataFrame({"a": [1], "b": [2]}))


def test_regex_quoted_fields() -> None:
    data = 'name   note\n"a  b"  "x\t y"\n"c""d"  "line 1\nline 2"\n'
    result = pl.read_csv(data.encode(), separator=r"\s+")
    expected = pl.DataFrame(
        {"name": ["a  b", 'c"d'], "note": ["x\t y", "line 1\nline 2"]}
    )
    assert_frame_equal(result, expected)


def test_regex_quote_inside_unquoted_field() -> None:
    data = b'a b c\na"z z" 2\n'
    result = pl.read_csv(data, separator=r"\s+")
    assert_frame_equal(result, pl.read_csv(data, separator=" "))


def test_regex_quoted_separator() -> None:
    result = pl.read_csv(b'a::b\n"left::right"::"a,b"\n', separator=r"::")
    assert_frame_equal(result, pl.DataFrame({"a": ["left::right"], "b": ["a,b"]}))


def test_regex_empty_fields() -> None:
    result = pl.read_csv(b"a::b::c\n::2::\n", separator=r"::")
    assert result.row(0) == (None, 2, None)


def test_regex_literal_single_byte() -> None:
    result = pl.read_csv(b"a|b\n1|2\n", separator="|")
    assert_frame_equal(result, pl.DataFrame({"a": [1], "b": [2]}))


def test_regex_compiled_single_character() -> None:
    result = pl.read_csv(b"ab\n", separator=re.compile("."), has_header=False)
    assert result.row(0) == (None, None, None)


def test_regex_csv_options() -> None:
    result = pl.read_csv(
        b"metadata\na  b\n1  2\n3  4\n",
        separator=r"\s+",
        skip_rows=1,
        n_rows=1,
        schema_overrides={"b": pl.Float64},
        row_index_name="row",
        row_index_offset=5,
        columns=["b"],
    )
    expected = pl.DataFrame({"row": pl.Series([5], dtype=pl.UInt32), "b": [2.0]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_regex_comments_and_blank_lines(newline: str) -> None:
    data = newline.join(['# ignored  "', "a  b", "", "1  2", "# ignored", "3  4", ""])
    result = pl.read_csv(data.encode(), separator=r"\s+", comment_prefix="#")
    expected = pl.DataFrame({"a": [None, 1, 3], "b": [None, 2, 4]})
    assert_frame_equal(result, expected)


def test_regex_quote_disabled() -> None:
    result = pl.read_csv(b'a::b\n"x::y"\n', separator=r"::", quote_char=None)
    assert_frame_equal(result, pl.DataFrame({"a": ['"x'], "b": ['y"']}))


@pytest.mark.parametrize("encoding", ["utf8", "windows-1251"])
def test_regex_encoding(encoding: str, tmp_path: Path) -> None:
    path = tmp_path / "encoded.txt"
    path.write_bytes("name  n\nёж  1\n".encode(encoding))
    result = pl.read_csv(path, separator=r"\s+", encoding=encoding)
    assert_frame_equal(result, pl.DataFrame({"name": ["ёж"], "n": [1]}))


@pytest.mark.parametrize("kind", ["string_io", "text_file"])
def test_regex_decoded_text_encoding(kind: str, tmp_path: Path) -> None:
    data = "name  n\nёж  1\n"
    path = tmp_path / "decoded.txt"
    path.write_text(data, encoding="windows-1251")
    with (
        io.StringIO(data) if kind == "string_io" else path.open(encoding="windows-1251")
    ) as source:
        result = pl.read_csv(source, separator=r"\s+", encoding="windows-1251")
    assert_frame_equal(result, pl.DataFrame({"name": ["ёж"], "n": [1]}))


def test_regex_lossy_encoding() -> None:
    result = pl.read_csv(b"name  n\n\xff  1\n", separator=r"\s+", encoding="utf8-lossy")
    assert_frame_equal(result, pl.DataFrame({"name": ["�"], "n": [1]}))


@pytest.mark.parametrize("separator", [re.compile(""), re.compile(r"(?=b)"), r"a*"])
def test_regex_zero_width(separator: str | re.Pattern[str]) -> None:
    with pytest.raises(ValueError, match=r"empty|zero.width"):
        pl.read_csv(b"ab\n", separator=separator, has_header=False)


def test_regex_invalid_pattern() -> None:
    with pytest.raises(re.error):
        pl.read_csv(b"a b\n", separator="[abc")


@pytest.mark.parametrize("prefix", ["\ufeff  ", '\ufeff# comment "\n  '])
def test_regex_bom(prefix: str) -> None:
    data = f"{prefix}a  b\n  1  2\n".encode()
    result = pl.read_csv(data, separator=r"\s+", comment_prefix="#")
    assert_frame_equal(result, pl.DataFrame({"a": [1], "b": [2]}))


def test_regex_skip_lines_unmatched_quote() -> None:
    result = pl.read_csv(b'ignored "\na  b\n1  2\n', separator=r"\s+", skip_lines=1)
    assert_frame_equal(result, pl.DataFrame({"a": [1], "b": [2]}))


def test_regex_plain_zlib_header() -> None:
    result = pl.read_csv(b"x  y\n1  2\n", separator=r"\s+")
    assert_frame_equal(result, pl.DataFrame({"x": [1], "y": [2]}))


@pytest.mark.parametrize("compression", ["gzip", "zlib", "zstd"])
def test_regex_compression(compression: str) -> None:
    raw = b"a   b\n1\t2\n"
    if compression == "gzip":
        raw = gzip.compress(raw)
    elif compression == "zlib":
        raw = zlib.compress(raw)
    else:
        raw = zstandard.ZstdCompressor().compress(raw)
    result = pl.read_csv(raw, separator=r"\s+")
    assert_frame_equal(result, pl.DataFrame({"a": [1], "b": [2]}))


@pytest.mark.parametrize("source", [[b"a b\n1 2\n"], "*.txt"])
def test_regex_multiple_sources_unsupported(source: list[bytes] | str) -> None:
    with pytest.raises(ValueError, match="single source"):
        pl.read_csv(source, separator=r"\s+")


def test_regex_file_paths_unsupported() -> None:
    with pytest.raises(ValueError, match="single source"):
        pl.read_csv(b"a b\n1 2\n", separator=r"\s+", include_file_paths="file")
