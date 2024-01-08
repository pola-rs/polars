from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import ComputeError, InvalidOperationError


# TODO: Do not rely on I/O for these tests
@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_string_case() -> None:
    df = pl.DataFrame({"words": ["Test SOME words"]})

    with pl.SQLContext(frame=df) as ctx:
        res = ctx.execute(
            """
            SELECT
              words,
              INITCAP(words) as cap,
              UPPER(words) as upper,
              LOWER(words) as lower,
            FROM frame
            """
        ).collect()

        assert res.to_dict(as_series=False) == {
            "words": ["Test SOME words"],
            "cap": ["Test Some Words"],
            "upper": ["TEST SOME WORDS"],
            "lower": ["test some words"],
        }


def test_string_concat() -> None:
    lf = pl.LazyFrame(
        {
            "x": ["a", None, "c"],
            "y": ["d", "e", "f"],
            "z": [1, 2, 3],
        }
    )
    res = pl.SQLContext(data=lf).execute(
        """
        SELECT
          ("x" || "x" || "y")           AS c0,
          ("x" || "y" || "z")           AS c1,
          CONCAT(("x" || '-'), "y")     AS c2,
          CONCAT("x", "x", "y")         AS c3,
          CONCAT("x", "y", ("z" * 2))   AS c4,
          CONCAT_WS(':', "x", "y", "z") AS c5,
          CONCAT_WS('', "y", "z", '!')  AS c6
        FROM data
        """,
        eager=True,
    )
    assert res.to_dict(as_series=False) == {
        "c0": ["aad", None, "ccf"],
        "c1": ["ad1", None, "cf3"],
        "c2": ["a-d", None, "c-f"],
        "c3": ["aad", None, "ccf"],
        "c4": ["ad2", None, "cf6"],
        "c5": ["a:d:1", None, "c:f:3"],
        "c6": ["d1!", "e2!", "f3!"],
    }


@pytest.mark.parametrize(
    "invalid_concat", ["CONCAT()", "CONCAT_WS()", "CONCAT_WS(':')"]
)
def test_string_concat_errors(invalid_concat: str) -> None:
    lf = pl.LazyFrame({"x": ["a", "b", "c"]})
    with pytest.raises(InvalidOperationError, match="Invalid number of arguments"):
        pl.SQLContext(data=lf).execute(f"SELECT {invalid_concat} FROM data")


def test_string_left_right_reverse() -> None:
    df = pl.DataFrame({"txt": ["abcde", "abc", "a", None]})
    ctx = pl.SQLContext(df=df)
    res = ctx.execute(
        """
        SELECT
          LEFT(txt,2) AS "l",
          RIGHT(txt,2) AS "r",
          REVERSE(txt) AS "rev"
        FROM df
        """,
    ).collect()

    assert res.to_dict(as_series=False) == {
        "l": ["ab", "ab", "a", None],
        "r": ["de", "bc", "a", None],
        "rev": ["edcba", "cba", "a", None],
    }
    for func, invalid in (("LEFT", "'xyz'"), ("RIGHT", "-1")):
        with pytest.raises(
            InvalidOperationError,
            match=f"Invalid 'length' for {func.capitalize()}: {invalid}",
        ):
            ctx.execute(f"""SELECT {func}(txt,{invalid}) FROM df""").collect()


def test_string_lengths() -> None:
    df = pl.DataFrame({"words": ["Café", None, "東京", ""]})

    with pl.SQLContext(frame=df) as ctx:
        res = ctx.execute(
            """
            SELECT
              words,
              LENGTH(words) AS n_chrs1,
              CHAR_LENGTH(words) AS n_chrs2,
              CHARACTER_LENGTH(words) AS n_chrs3,
              OCTET_LENGTH(words) AS n_bytes,
              BIT_LENGTH(words) AS n_bits
            FROM frame
            """
        ).collect()

    assert res.to_dict(as_series=False) == {
        "words": ["Café", None, "東京", ""],
        "n_chrs1": [4, None, 2, 0],
        "n_chrs2": [4, None, 2, 0],
        "n_chrs3": [4, None, 2, 0],
        "n_bytes": [5, None, 6, 0],
        "n_bits": [40, None, 48, 0],
    }


@pytest.mark.parametrize(
    ("pattern", "like", "expected"),
    [
        ("a%", "LIKE", [1, 4]),
        ("a%", "ILIKE", [0, 1, 3, 4]),
        ("ab%", "LIKE", [1]),
        ("AB%", "ILIKE", [0, 1]),
        ("ab_", "LIKE", [1]),
        ("A__", "ILIKE", [0, 1]),
        ("_0%_", "LIKE", [2, 4]),
        ("%0", "LIKE", [2]),
        ("0%", "LIKE", [2]),
        ("__0%", "LIKE", [2, 3]),
        ("%*%", "ILIKE", [3]),
        ("____", "LIKE", [4]),
        ("a%C", "LIKE", []),
        ("a%C", "ILIKE", [0, 1, 3]),
        ("%C?", "ILIKE", [4]),
        ("a0c?", "LIKE", [4]),
        ("000", "LIKE", [2]),
        ("00", "LIKE", []),
    ],
)
def test_string_like(pattern: str, like: str, expected: list[int]) -> None:
    df = pl.DataFrame(
        {
            "idx": [0, 1, 2, 3, 4],
            "txt": ["ABC", "abc", "000", "A[0]*C", "a0c?"],
        }
    )
    with pl.SQLContext(df=df) as ctx:
        for not_ in ("", "NOT "):
            out = ctx.execute(
                f"""SELECT idx FROM df WHERE txt {not_}{like} '{pattern}'"""
            ).collect()

            res = out["idx"].to_list()
            if not_:
                expected = [i for i in df["idx"] if i not in expected]
            assert res == expected


def test_string_replace() -> None:
    df = pl.DataFrame({"words": ["Yemeni coffee is the best coffee", "", None]})
    with pl.SQLContext(df=df) as ctx:
        out = ctx.execute(
            """
            SELECT
              REPLACE(
                REPLACE(words, 'coffee', 'tea'),
                'Yemeni',
                'English breakfast'
              )
            FROM df
            """
        ).collect()

        res = out["words"].to_list()
        assert res == ["English breakfast tea is the best tea", "", None]

        with pytest.raises(InvalidOperationError, match="Invalid number of arguments"):
            ctx.execute("SELECT REPLACE(words,'coffee') FROM df")


def test_string_substr() -> None:
    df = pl.DataFrame({"scol": ["abcdefg", "abcde", "abc", None]})
    with pl.SQLContext(df=df) as ctx:
        res = ctx.execute(
            """
            SELECT
              -- note: sql is 1-indexed
              SUBSTR(scol,1) AS s1,
              SUBSTR(scol,2) AS s2,
              SUBSTR(scol,3) AS s3,
              SUBSTR(scol,1,5) AS s1_5,
              SUBSTR(scol,2,2) AS s2_2,
              SUBSTR(scol,3,1) AS s3_1,
            FROM df
            """
        ).collect()

    assert res.to_dict(as_series=False) == {
        "s1": ["abcdefg", "abcde", "abc", None],
        "s2": ["bcdefg", "bcde", "bc", None],
        "s3": ["cdefg", "cde", "c", None],
        "s1_5": ["abcde", "abcde", "abc", None],
        "s2_2": ["bc", "bc", "bc", None],
        "s3_1": ["c", "c", "c", None],
    }

    # negative indexes are expected to be invalid
    with pytest.raises(
        InvalidOperationError,
        match="Invalid 'start' for Substring: -1",
    ), pl.SQLContext(df=df) as ctx:
        ctx.execute("SELECT SUBSTR(scol,-1) FROM df")


def test_string_trim(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)
    out = pl.SQLContext(foods1=lf).execute(
        """
        SELECT DISTINCT TRIM(LEADING 'vmf' FROM category) as new_category
        FROM foods1
        ORDER BY new_category DESC
        """,
        eager=True,
    )
    assert out.to_dict(as_series=False) == {
        "new_category": ["seafood", "ruit", "egetables", "eat"]
    }
    with pytest.raises(
        ComputeError,
        match="unsupported TRIM",
    ):
        # currently unsupported (snowflake) trim syntax
        pl.SQLContext(foods=lf).execute(
            """
            SELECT DISTINCT TRIM('*^xxxx^*', '^*') as new_category FROM foods
            """,
        )
