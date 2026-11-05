from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


@pytest.mark.parametrize(
    ("regex_op", "expected"),
    [
        ("RLIKE", [0, 3]),
        ("REGEXP", [0, 3]),
        ("NOT RLIKE", [1, 2, 4]),
        ("NOT REGEXP", [1, 2, 4]),
    ],
)
def test_regex_expr_match(regex_op: str, expected: list[int]) -> None:
    # note: the REGEXP and RLIKE operators can also use another
    # column/expression as the source of the match pattern
    df = pl.DataFrame(
        {
            "idx": [0, 1, 2, 3, 4],
            "str": ["ABC", "abc", "000", "A0C", "a0c"],
            "pat": ["^A", "^A", "^A", r"[AB]\d.*$", ".*xxx$"],
        }
    )
    with pl.SQLContext(df=df, eager=True) as ctx:
        out = ctx.execute(f"SELECT idx, str FROM df WHERE str {regex_op} pat")
        assert out.to_series().to_list() == expected


@pytest.mark.parametrize(
    ("op", "pattern", "expected"),
    [
        ("~", "^veg", "vegetables"),
        ("~", "^VEG", None),
        ("~*", "^VEG", "vegetables"),
        ("!~", "(t|s)$", "seafood"),
        ("!~*", "(T|S)$", "seafood"),
        ("!~*", "^.E", "fruit"),
        ("!~*", "[aeiOU]", None),
        ("RLIKE", "^veg", "vegetables"),
        ("RLIKE", "^VEG", None),
        ("RLIKE", "(?i)^VEG", "vegetables"),
        ("NOT RLIKE", "(t|s)$", "seafood"),
        ("NOT RLIKE", "(?i)(T|S)$", "seafood"),
        ("NOT RLIKE", "(?i)^.E", "fruit"),
        ("NOT RLIKE", "(?i)[aeiOU]", None),
        ("REGEXP", "^veg", "vegetables"),
        ("REGEXP", "^VEG", None),
        ("REGEXP", "(?i)^VEG", "vegetables"),
        ("NOT REGEXP", "(t|s)$", "seafood"),
        ("NOT REGEXP", "(?i)(T|S)$", "seafood"),
        ("NOT REGEXP", "(?i)^.E", "fruit"),
        ("NOT REGEXP", "(?i)[aeiOU]", None),
    ],
)
def test_regex_operators(
    foods_ipc_path: Path, op: str, pattern: str, expected: str | None
) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    with pl.SQLContext(foods=lf, eager=True) as ctx:
        out = ctx.execute(
            f"""
            SELECT DISTINCT category FROM foods
            WHERE category {op} '{pattern}'
            """
        )
        assert out.rows() == ([(expected,)] if expected else [])


def test_regex_operators_error() -> None:
    df = pl.LazyFrame({"sval": ["ABC", "abc", "000", "A0C", "a0c"]})
    with pl.SQLContext(df=df, eager=True) as ctx:
        with pytest.raises(
            SQLSyntaxError, match=r"invalid pattern for '~' operator: dyn .*12345"
        ):
            ctx.execute("SELECT * FROM df WHERE sval ~ 12345")
        with pytest.raises(
            SQLSyntaxError,
            match=r"""invalid pattern for '!~\*' operator: col\("abcde"\)""",
        ):
            ctx.execute("SELECT * FROM df WHERE sval !~* abcde")


@pytest.mark.parametrize(
    ("not_", "pattern", "flags", "expected"),
    [
        ("", "^veg", None, "vegetables"),
        ("", "^VEG", None, None),
        ("", "(?i)^VEG", None, "vegetables"),
        ("NOT", "(t|s)$", None, "seafood"),
        ("NOT", "T|S$", "i", "seafood"),
        ("NOT", "^.E", "i", "fruit"),
        ("NOT", "[aeiOU]", "i", None),
    ],
)
def test_regexp_like(
    foods_ipc_path: Path,
    not_: str,
    pattern: str,
    flags: str | None,
    expected: str | None,
) -> None:
    lf = pl.scan_ipc(foods_ipc_path)
    flags = "" if flags is None else f",'{flags}'"
    with pl.SQLContext(foods=lf, eager=True) as ctx:
        out = ctx.execute(
            f"""
            SELECT DISTINCT category FROM foods
            WHERE {not_} REGEXP_LIKE(category,'{pattern}'{flags})
            """
        )
        assert out.rows() == ([(expected,)] if expected else [])


def test_regexp_like_errors() -> None:
    with pl.SQLContext(df=pl.DataFrame({"scol": ["xyz"]})) as ctx:
        with pytest.raises(
            SQLSyntaxError,
            match="invalid/empty 'flags' for REGEXP_LIKE",
        ):
            ctx.execute(
                "SELECT * FROM df WHERE REGEXP_LIKE(scol,'[x-z]+','')"
            ).collect()

        with pytest.raises(
            SQLSyntaxError,
            match="invalid arguments for REGEXP_LIKE",
        ):
            ctx.execute("SELECT * FROM df WHERE REGEXP_LIKE(scol,999,999)").collect()

        with pytest.raises(
            SQLSyntaxError,
            match=r"REGEXP_LIKE expects 2-3 arguments \(found 1\)",
        ):
            ctx.execute("SELECT * FROM df WHERE REGEXP_LIKE(scol)").collect()


@pytest.mark.parametrize(
    "func",
    [
        r"REGEXP_REPLACE(s, 'b', 'X')",
        r"REGEXP_REPLACE(s, 'b', 'X', 'g')",
        r"REGEXP_REPLACE(s, 'B', 'X', 'gi')",
        r"REGEXP_REPLACE(s, '(\w+)-(\w+)', '\2-\1')",
        r"REGEXP_REPLACE(s, '^https?://(?:www\.)?([^/]+)/.*$', '\1')",
        r"REGEXP_REPLACE(s, 'b', '$1 \\', 'g')",
        r"REGEXP_REPLACE(s, '[ab]', '<\0>', 'g')",
        r"REGEXP_REPLACE(s, 'x', 'y')",
    ],
)
def test_regexp_replace(func: str) -> None:
    df = pl.DataFrame(
        {"s": ["abcb", "ab-cd", "http://www.a.com/x", "https://b.org/", None]}
    )
    assert_sql_matches(
        df,
        query=f"SELECT {func} AS r FROM self",
        compare_with="duckdb",
    )


def test_regexp_replace_whole_match_and_null() -> None:
    df = pl.DataFrame({"s": ["abcb", None]})
    out = df.sql(
        r"""
        SELECT
          REGEXP_REPLACE(s, 'b', '[\&]', 'g') AS whole,
          REGEXP_REPLACE(s, 'b', NULL) AS null_replacement,
          REGEXP_REPLACE(s, NULL, 'x') AS null_pattern,
          REGEXP_REPLACE(s, 'b', 'x', NULL) AS null_flags,
          REGEXP_REPLACE(NULL, 'b', 'x') AS null_value
        FROM self
        """
    )
    assert out.to_dict(as_series=False) == {
        "whole": ["a[b]c[b]", None],
        "null_replacement": [None, None],
        "null_pattern": [None, None],
        "null_flags": [None, None],
        "null_value": [None, None],
    }


def test_regexp_replace_errors() -> None:
    df = pl.DataFrame({"s": ["xyz"], "t": ["a"]})
    with pytest.raises(SQLSyntaxError, match="REGEXP_REPLACE expects 3-4 arguments"):
        df.sql("SELECT REGEXP_REPLACE(s, 'x') FROM self")
    with pytest.raises(SQLSyntaxError, match="string literal replacement"):
        df.sql("SELECT REGEXP_REPLACE(s, 'x', t) FROM self")
    with pytest.raises(SQLSyntaxError, match="invalid 'flags' for REGEXP_REPLACE"):
        df.sql("SELECT REGEXP_REPLACE(s, 'x', 'y', t) FROM self")
    with pytest.raises(SQLSyntaxError, match="string literal pattern"):
        df.sql("SELECT REGEXP_REPLACE(s, t, 'y') FROM self")
