from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import (
    InvalidOperationError,
)
from polars.testing import assert_frame_equal, assert_series_equal

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence
    from typing import Any

    from polars import Series

    FixtureRequest = Any


def _cat(values: Sequence[str | None]) -> Series:
    return pl.Series(values, dtype=pl.Categorical)


@pytest.fixture(params=[True, False], autouse=True)
def _setup_string_cache(
    request: FixtureRequest,
) -> Generator[Any, Any, Any]:
    """Setup fixture which runs each test with and without global string cache."""
    use_global = request.param
    if use_global:
        with pl.StringCache():
            yield
    else:
        yield


def test_str_slice() -> None:
    df = pl.DataFrame({"a": _cat(["foobar", "barfoo"])})
    assert df["a"].str.slice(-3).to_list() == ["bar", "foo"]
    assert df.select([pl.col("a").str.slice(2, 4)])["a"].to_list() == ["obar", "rfoo"]


def test_str_slice_expr() -> None:
    df = pl.DataFrame(
        {
            "a": _cat(["foobar", None, "barfoo", "abcd", ""]),
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
        str_lit=pl.lit("qwert", dtype=pl.Categorical).str.slice("offset", "length"),
    )
    expected = pl.DataFrame(
        {
            "all_expr": _cat(["oob", None, None, "bcd", ""]),
            "offset_expr": _cat(["oo", None, None, "bc", ""]),
            "length_expr": _cat(["foo", None, "ba", "abcd", ""]),
            "length_none": _cat(["oobar", None, None, "bcd", ""]),
            "offset_length_lit": _cat(["bar", None, "foo", "bcd", ""]),
            "str_lit": _cat(["wer", "rt", None, "ert", "er"]),
        }
    )
    if pl.using_string_cache():
        assert_frame_equal(out, expected)
    else:
        assert out.schema == expected.schema
        assert_frame_equal(out, expected, categorical_as_str=True)

    # negative length is not allowed
    with pytest.raises(InvalidOperationError):
        df.select(pl.col("a").str.slice(0, -1))


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
    assert _cat([input]).str.head(n).to_list() == [output]


def test_str_head_expr() -> None:
    s = "012345"
    df = pl.DataFrame(
        {
            "a": _cat([s, s, s, s, s, s, "", None]),
            "n": [0, 2, -2, 100, -100, None, 3, -2],
        }
    )
    out = df.select(
        n_expr=pl.col("a").str.head("n"),
        n_pos2=pl.col("a").str.head(2),
        n_neg2=pl.col("a").str.head(-2),
        n_pos100=pl.col("a").str.head(100),
        n_pos_neg100=pl.col("a").str.head(-100),
        n_pos_0=pl.col("a").str.head(0),
        str_lit=pl.col("a").str.head(pl.lit(2)),
        lit_expr=pl.lit(s, dtype=pl.Categorical).str.head("n"),
        lit_n=pl.lit(s, dtype=pl.Categorical).str.head(2),
    )
    expected = pl.DataFrame(
        {
            "n_expr": _cat(["", "01", "0123", "012345", "", None, "", None]),
            "n_pos2": _cat(["01", "01", "01", "01", "01", "01", "", None]),
            "n_neg2": _cat(["0123", "0123", "0123", "0123", "0123", "0123", "", None]),
            "n_pos100": _cat([s, s, s, s, s, s, "", None]),
            "n_pos_neg100": _cat(["", "", "", "", "", "", "", None]),
            "n_pos_0": _cat(["", "", "", "", "", "", "", None]),
            "str_lit": _cat(["01", "01", "01", "01", "01", "01", "", None]),
            "lit_expr": _cat(["", "01", "0123", "012345", "", None, "012", "0123"]),
            "lit_n": _cat(["01", "01", "01", "01", "01", "01", "01", "01"]),
        }
    )
    if pl.using_string_cache():
        assert_frame_equal(out, expected)
    else:
        assert out.schema == expected.schema
        assert_frame_equal(out, expected, categorical_as_str=True)


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
    assert _cat(input).str.tail(n).to_list() == output


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
    assert _cat([input]).str.tail(n).to_list() == [output]


def test_str_tail_expr() -> None:
    s = "012345"
    df = pl.DataFrame(
        {
            "a": _cat([s, s, s, s, s, s, "", None]),
            "n": [0, 2, -2, 100, -100, None, 3, -2],
        }
    )
    out = df.select(
        n_expr=pl.col("a").str.tail("n"),
        n_pos2=pl.col("a").str.tail(2),
        n_neg2=pl.col("a").str.tail(-2),
        n_pos100=pl.col("a").str.tail(100),
        n_pos_neg100=pl.col("a").str.tail(-100),
        n_pos_0=pl.col("a").str.tail(0),
        str_lit=pl.col("a").str.tail(pl.lit(2)),
        lit_expr=pl.lit(s, dtype=pl.Categorical).str.tail("n"),
        lit_n=pl.lit(s, dtype=pl.Categorical).str.tail(2),
    )
    expected = pl.DataFrame(
        {
            "n_expr": _cat(["", "45", "2345", "012345", "", None, "", None]),
            "n_pos2": _cat(["45", "45", "45", "45", "45", "45", "", None]),
            "n_neg2": _cat(["2345", "2345", "2345", "2345", "2345", "2345", "", None]),
            "n_pos100": _cat([s, s, s, s, s, s, "", None]),
            "n_pos_neg100": _cat(["", "", "", "", "", "", "", None]),
            "n_pos_0": _cat(["", "", "", "", "", "", "", None]),
            "str_lit": _cat(["45", "45", "45", "45", "45", "45", "", None]),
            "lit_expr": _cat(["", "45", "2345", "012345", "", None, "345", "2345"]),
            "lit_n": _cat(["45", "45", "45", "45", "45", "45", "45", "45"]),
        }
    )
    if pl.using_string_cache():
        assert_frame_equal(out, expected)
    else:
        assert out.schema == expected.schema
        assert_frame_equal(out, expected, categorical_as_str=True)


def test_str_slice_multibyte() -> None:
    ref = "你好世界"
    s = _cat([ref])

    # Pad the string to simplify (negative) offsets starting before/after the string.
    npad = 20
    padref = "_" * npad + ref + "_" * npad
    for start in range(-5, 6):
        for length in range(6):
            offset = npad + start if start >= 0 else npad + start + len(ref)
            correct = padref[offset : offset + length].strip("_")
            result = s.str.slice(start, length)
            expected = _cat([correct])
            if pl.using_string_cache():
                assert_series_equal(result, expected)
            else:
                assert result.dtype == expected.dtype
                assert_series_equal(result, expected, categorical_as_str=True)
