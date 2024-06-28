from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError
from polars.testing import assert_frame_equal


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
    res = lf.sql(
        """
        SELECT
          ("x" || "x" || "y")           AS c0,
          ("x" || "y" || "z")           AS c1,
          CONCAT(("x" || '-'), "y")     AS c2,
          CONCAT("x", "x", "y")         AS c3,
          CONCAT("x", "y", ("z" * 2))   AS c4,
          CONCAT_WS(':', "x", "y", "z") AS c5,
          CONCAT_WS('', "y", "z", '!')  AS c6
        FROM self
        """,
    ).collect()

    assert res.to_dict(as_series=False) == {
        "c0": ["aad", None, "ccf"],
        "c1": ["ad1", None, "cf3"],
        "c2": ["a-d", "e", "c-f"],
        "c3": ["aad", "e", "ccf"],
        "c4": ["ad2", "e4", "cf6"],
        "c5": ["a:d:1", "e:2", "c:f:3"],
        "c6": ["d1!", "e2!", "f3!"],
    }


@pytest.mark.parametrize(
    "invalid_concat", ["CONCAT()", "CONCAT_WS()", "CONCAT_WS(':')"]
)
def test_string_concat_errors(invalid_concat: str) -> None:
    lf = pl.LazyFrame({"x": ["a", "b", "c"]})
    with pytest.raises(
        SQLSyntaxError,
        match=r"CONCAT.*expects at least \d argument[s]? \(found \d\)",
    ):
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
    for func, invalid_arg, invalid_err in (
        ("LEFT", "'xyz'", '"xyz"'),
        ("RIGHT", "6.66", "(dyn float: 6.66)"),
    ):
        with pytest.raises(
            SQLSyntaxError,
            match=rf"""invalid 'n_chars' for {func} \({invalid_err}\)""",
        ):
            ctx.execute(f"""SELECT {func}(txt,{invalid_arg}) FROM df""").collect()


def test_string_left_negative_expr() -> None:
    # negative values and expressions
    df = pl.DataFrame({"s": ["alphabet", "alphabet"], "n": [-6, 6]})
    with pl.SQLContext(df=df, eager=True) as sql:
        res = sql.execute(
            """
            SELECT
              LEFT("s",-50)      AS l0,  -- empty string
              LEFT("s",-3)       AS l1,  -- all but last three chars
              LEFT("s",SIGN(-1)) AS l2,  -- all but last char (expr => -1)
              LEFT("s",0)        AS l3,  -- empty string
              LEFT("s",NULL)     AS l4,  -- null
              LEFT("s",1)        AS l5,  -- first char
              LEFT("s",SIGN(1))  AS l6,  -- first char (expr => 1)
              LEFT("s",3)        AS l7,  -- first three chars
              LEFT("s",50)       AS l8,  -- entire string
              LEFT("s","n")      AS l9,  -- from other col
            FROM df
            """
        )
        assert res.to_dict(as_series=False) == {
            "l0": ["", ""],
            "l1": ["alpha", "alpha"],
            "l2": ["alphabe", "alphabe"],
            "l3": ["", ""],
            "l4": [None, None],
            "l5": ["a", "a"],
            "l6": ["a", "a"],
            "l7": ["alp", "alp"],
            "l8": ["alphabet", "alphabet"],
            "l9": ["al", "alphab"],
        }


def test_string_right_negative_expr() -> None:
    # negative values and expressions
    df = pl.DataFrame({"s": ["alphabet", "alphabet"], "n": [-6, 6]})
    with pl.SQLContext(df=df, eager=True) as sql:
        res = sql.execute(
            """
            SELECT
              RIGHT("s",-50)      AS l0,  -- empty string
              RIGHT("s",-3)       AS l1,  -- all but first three chars
              RIGHT("s",SIGN(-1)) AS l2,  -- all but first char (expr => -1)
              RIGHT("s",0)        AS l3,  -- empty string
              RIGHT("s",NULL)     AS l4,  -- null
              RIGHT("s",1)        AS l5,  -- last char
              RIGHT("s",SIGN(1))  AS l6,  -- last char (expr => 1)
              RIGHT("s",3)        AS l7,  -- last three chars
              RIGHT("s",50)       AS l8,  -- entire string
              RIGHT("s","n")      AS l9,  -- from other col
            FROM df
            """
        )
        assert res.to_dict(as_series=False) == {
            "l0": ["", ""],
            "l1": ["habet", "habet"],
            "l2": ["lphabet", "lphabet"],
            "l3": ["", ""],
            "l4": [None, None],
            "l5": ["t", "t"],
            "l6": ["t", "t"],
            "l7": ["bet", "bet"],
            "l8": ["alphabet", "alphabet"],
            "l9": ["et", "phabet"],
        }


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
        ("__0%", "~~", [2, 3]),
        ("%*%", "~~*", [3]),
        ("____", "~~", [4]),
        ("a%C", "~~", []),
        ("a%C", "~~*", [0, 1, 3]),
        ("%C?", "~~*", [4]),
        ("a0c?", "~~", [4]),
        ("000", "~~", [2]),
        ("00", "~~", []),
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
        for not_ in ("", ("NOT " if like.endswith("LIKE") else "!")):
            out = ctx.execute(
                f"SELECT idx FROM df WHERE txt {not_}{like} '{pattern}'"
            ).collect()

            res = out["idx"].to_list()
            if not_:
                expected = [i for i in df["idx"] if i not in expected]
            assert res == expected


def test_string_position() -> None:
    df = pl.Series(
        name="city",
        values=["Dubai", "Abu Dhabi", "Sharjah", "Al Ain", "Ajman", "Ras Al Khaimah"],
    ).to_frame()

    with pl.SQLContext(cities=df, eager=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              POSITION('a' IN city) AS a_lc1,
              POSITION('A' IN city) AS a_uc1,
              STRPOS(city,'a') AS a_lc2,
              STRPOS(city,'A') AS a_uc2,
            FROM cities
            """
        )
        expected_lc = [4, 7, 3, 0, 4, 2]
        expected_uc = [0, 1, 0, 1, 1, 5]

        assert res.to_dict(as_series=False) == {
            "a_lc1": expected_lc,
            "a_uc1": expected_uc,
            "a_lc2": expected_lc,
            "a_uc2": expected_uc,
        }

    df = pl.DataFrame({"txt": ["AbCdEXz", "XyzFDkE"]})
    with pl.SQLContext(txt=df) as ctx:
        res = ctx.execute(
            """
            SELECT
              txt,
              POSITION('E' IN txt) AS match_E,
              STRPOS(txt,'X') AS match_X
            FROM txt
            """,
            eager=True,
        )
        assert_frame_equal(
            res,
            pl.DataFrame(
                data={
                    "txt": ["AbCdEXz", "XyzFDkE"],
                    "match_E": [5, 7],
                    "match_X": [6, 1],
                },
                schema={
                    "txt": pl.String,
                    "match_E": pl.UInt32,
                    "match_X": pl.UInt32,
                },
            ),
        )


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

        with pytest.raises(
            SQLSyntaxError, match=r"REPLACE expects 3 arguments \(found 2\)"
        ):
            ctx.execute("SELECT REPLACE(words,'coffee') FROM df")


def test_string_substr() -> None:
    df = pl.DataFrame(
        {"scol": ["abcdefg", "abcde", "abc", None], "n": [-2, 3, 2, None]}
    )
    with pl.SQLContext(df=df) as ctx:
        res = ctx.execute(
            """
            SELECT
              -- note: sql is 1-indexed
              SUBSTR(scol,1)    AS s1,
              SUBSTR(scol,2)    AS s2,
              SUBSTR(scol,3)    AS s3,
              SUBSTR(scol,1,5)  AS s1_5,
              SUBSTR(scol,2,2)  AS s2_2,
              SUBSTR(scol,3,1)  AS s3_1,
              SUBSTR(scol,-3)   AS "s-3",
              SUBSTR(scol,-3,3) AS "s-3_3",
              SUBSTR(scol,-3,4) AS "s-3_4",
              SUBSTR(scol,-3,5) AS "s-3_5",
              SUBSTR(scol,-10,13) AS "s-10_13",
              SUBSTR(scol,"n",2) AS "s-n2",
              SUBSTR(scol,2,"n"+3) AS "s-2n3"
            FROM df
            """
        ).collect()

        with pytest.raises(
            SQLSyntaxError,
            match=r"SUBSTR does not support negative length \(-99\)",
        ):
            ctx.execute("SELECT SUBSTR(scol,2,-99) FROM df")

        with pytest.raises(
            SQLSyntaxError,
            match=r"SUBSTR expects 2-3 arguments \(found 1\)",
        ):
            pl.sql_expr("SUBSTR(s)")

    assert res.to_dict(as_series=False) == {
        "s1": ["abcdefg", "abcde", "abc", None],
        "s2": ["bcdefg", "bcde", "bc", None],
        "s3": ["cdefg", "cde", "c", None],
        "s1_5": ["abcde", "abcde", "abc", None],
        "s2_2": ["bc", "bc", "bc", None],
        "s3_1": ["c", "c", "c", None],
        "s-3": ["abcdefg", "abcde", "abc", None],
        "s-3_3": ["", "", "", None],
        "s-3_4": ["", "", "", None],
        "s-3_5": ["a", "a", "a", None],
        "s-10_13": ["ab", "ab", "ab", None],
        "s-n2": ["", "cd", "bc", None],
        "s-2n3": ["b", "bcde", "bc", None],
    }


def test_string_trim(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)
    out = lf.sql(
        """
        SELECT DISTINCT TRIM(LEADING 'vmf' FROM category) as new_category
        FROM self ORDER BY new_category DESC
        """
    ).collect()
    assert out.to_dict(as_series=False) == {
        "new_category": ["seafood", "ruit", "egetables", "eat"]
    }
    with pytest.raises(
        SQLSyntaxError,
        match="unsupported TRIM syntax",
    ):
        # currently unsupported (snowflake-style) trim syntax
        lf.sql("SELECT DISTINCT TRIM('*^xxxx^*', '^*') as new_category FROM self")
