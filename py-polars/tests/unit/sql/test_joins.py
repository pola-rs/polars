from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


@pytest.mark.parametrize(
    ("sql", "expected"),
    [
        (
            "SELECT * FROM tbl_a LEFT SEMI JOIN tbl_b USING (a,c)",
            pl.DataFrame({"a": [2], "b": [0], "c": ["y"]}),
        ),
        (
            "SELECT * FROM tbl_a SEMI JOIN tbl_b USING (a,c)",
            pl.DataFrame({"a": [2], "b": [0], "c": ["y"]}),
        ),
        (
            "SELECT * FROM tbl_a LEFT SEMI JOIN tbl_b USING (a)",
            pl.DataFrame({"a": [1, 2, 3], "b": [4, 0, 6], "c": ["w", "y", "z"]}),
        ),
        (
            "SELECT * FROM tbl_a LEFT ANTI JOIN tbl_b USING (a)",
            pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64, "c": pl.String}),
        ),
        (
            "SELECT * FROM tbl_a ANTI JOIN tbl_b USING (a)",
            pl.DataFrame(schema={"a": pl.Int64, "b": pl.Int64, "c": pl.String}),
        ),
        (
            "SELECT * FROM tbl_a LEFT SEMI JOIN tbl_b USING (b) LEFT SEMI JOIN tbl_c USING (c)",
            pl.DataFrame({"a": [1, 3], "b": [4, 6], "c": ["w", "z"]}),
        ),
        (
            "SELECT * FROM tbl_a LEFT ANTI JOIN tbl_b USING (b) LEFT SEMI JOIN tbl_c USING (c)",
            pl.DataFrame({"a": [2], "b": [0], "c": ["y"]}),
        ),
        (
            "SELECT * FROM tbl_a RIGHT ANTI JOIN tbl_b USING (b) LEFT SEMI JOIN tbl_c USING (c)",
            pl.DataFrame({"a": [2], "b": [5], "c": ["y"]}),
        ),
        (
            "SELECT * FROM tbl_a RIGHT SEMI JOIN tbl_b USING (b) RIGHT SEMI JOIN tbl_c USING (c)",
            pl.DataFrame({"c": ["z"], "d": [25.5]}),
        ),
        (
            "SELECT * FROM tbl_a RIGHT SEMI JOIN tbl_b USING (b) RIGHT ANTI JOIN tbl_c USING (c)",
            pl.DataFrame({"c": ["w", "y"], "d": [10.5, -50.0]}),
        ),
    ],
)
def test_join_anti_semi(sql: str, expected: pl.DataFrame) -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, 0, 6], "c": ["w", "y", "z"]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    ctx = pl.SQLContext(frames, eager=True)
    assert_frame_equal(expected, ctx.execute(sql))


def test_join_cross() -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, 0, 6], "c": ["w", "y", "z"]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
    }
    with pl.SQLContext(frames, eager=True) as ctx:
        out = ctx.execute(
            """
            SELECT *
            FROM tbl_a
            CROSS JOIN tbl_b
            ORDER BY a, b, c
            """
        )
        assert out.rows() == [
            (1, 4, "w", 3, 6, "x"),
            (1, 4, "w", 2, 5, "y"),
            (1, 4, "w", 1, 4, "z"),
            (2, 0, "y", 3, 6, "x"),
            (2, 0, "y", 2, 5, "y"),
            (2, 0, "y", 1, 4, "z"),
            (3, 6, "z", 3, 6, "x"),
            (3, 6, "z", 2, 5, "y"),
            (3, 6, "z", 1, 4, "z"),
        ]


def test_join_cross_11927() -> None:
    df1 = pl.DataFrame({"id": [1, 2, 3]})
    df2 = pl.DataFrame({"id": [3, 4, 5]})
    df3 = pl.DataFrame({"id": [4, 5, 6]})

    res = pl.sql("SELECT df1.id FROM df1 CROSS JOIN df2 WHERE df1.id = df2.id")
    assert_frame_equal(res.collect(), pl.DataFrame({"id": [3]}))

    res = pl.sql("SELECT * FROM df1 CROSS JOIN df3 WHERE df1.id = df3.id")
    assert res.collect().is_empty()


def test_cross_join_unnest_from_table() -> None:
    df = pl.DataFrame({"id": [1, 2], "items": [[100, 200], [300, 400, 500]]})
    assert_sql_matches(
        frames=df,
        query="""
            SELECT id, item
            FROM self CROSS JOIN UNNEST(items) AS item
            ORDER BY id DESC, item ASC
        """,
        compare_with="duckdb",
        expected={
            "id": [2, 2, 2, 1, 1],
            "item": [300, 400, 500, 100, 200],
        },
    )


def test_cross_join_unnest_from_cte() -> None:
    assert_sql_matches(
        {},
        query="""
            WITH data AS (
                SELECT 'xyz' AS id, [0,1,2] AS items
                UNION ALL
                SELECT 'abc', [3,4]
            )
            SELECT id, item
            FROM data CROSS JOIN UNNEST(items) AS item
            ORDER BY item
        """,
        compare_with="duckdb",
        expected={
            "id": ["xyz", "xyz", "xyz", "abc", "abc"],
            "item": [0, 1, 2, 3, 4],
        },
    )


@pytest.mark.parametrize(
    "join_clause",
    [
        "ON f1.category = f2.category",
        "ON f2.category = f1.category",
        "USING (category)",
    ],
)
def test_join_inner(foods_ipc_path: Path, join_clause: str) -> None:
    foods1 = pl.scan_ipc(foods_ipc_path)
    foods2 = foods1
    schema = foods1.collect_schema()

    out = pl.sql(
        f"""
        SELECT *
        FROM
          (SELECT * FROM foods1 WHERE fats_g != 0) f1
        INNER JOIN
          (SELECT * FROM foods2 WHERE fats_g = 0) f2
        {join_clause}
        ORDER BY ALL
        LIMIT 2
        """,
        eager=True,
    )
    expected = pl.DataFrame(
        {
            "category": ["fruit", "fruit"],
            "calories": [50, 50],
            "fats_g": [4.5, 4.5],
            "sugars_g": [0, 0],
            "category:f2": ["fruit", "fruit"],
            "calories:f2": [30, 30],
            "fats_g:f2": [0.0, 0.0],
            "sugars_g:f2": [3, 5],
        }
    )
    assert_frame_equal(expected, out, check_dtypes=False)


def test_join_inner_15663() -> None:
    df_a = pl.DataFrame({"LOCID": [1, 2, 3], "VALUE": [0.1, 0.2, 0.3]})
    df_b = pl.DataFrame({"LOCID": [1, 2, 3], "VALUE": [25.6, 53.4, 12.7]})
    df_expected = pl.DataFrame(
        {
            "LOCID": [1, 2, 3],
            "VALUE_A": [0.1, 0.2, 0.3],
            "VALUE_B": [25.6, 53.4, 12.7],
        }
    )
    with pl.SQLContext(register_globals=True, eager=True) as ctx:
        query = """
            SELECT
                a.LOCID,
                a.VALUE AS VALUE_A,
                b.VALUE AS VALUE_B
            FROM df_a AS a INNER JOIN df_b AS b USING (LOCID)
            ORDER BY LOCID
        """
        actual = ctx.execute(query)
        assert_frame_equal(df_expected, actual)


@pytest.mark.parametrize(
    ("join_clause", "expected_error"),
    [
        (
            """
            INNER JOIN tbl_b USING (a,b)
            INNER JOIN tbl_c USING (c)
            """,
            None,
        ),
        (
            """
            INNER JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
            INNER JOIN tbl_c ON tbl_b.c = tbl_c.c
            """,
            None,
        ),
        (
            """
            INNER JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
            INNER JOIN tbl_c ON tbl_a.c = tbl_c.c  --<< (no "c" in 'tbl_a')
            """,
            "no column named 'c' found in table 'tbl_a'",
        ),
    ],
)
def test_join_inner_multi(join_clause: str, expected_error: str | None) -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    with pl.SQLContext(frames) as ctx:
        assert ctx.tables() == ["tbl_a", "tbl_b", "tbl_c"]
        query = f"""
            SELECT tbl_a.a, tbl_a.b, tbl_b.c, tbl_c.d
            FROM tbl_a {join_clause}
            ORDER BY tbl_a.a DESC
        """
        try:
            out = ctx.execute(query)
            assert out.collect().rows() == [(1, 4, "z", 25.5)]

        except SQLInterfaceError as err:
            if not (expected_error and expected_error in str(err)):
                raise


@pytest.mark.parametrize(
    "join_clause",
    [
        """
        LEFT JOIN tbl_b USING (a,b)
        LEFT JOIN tbl_c USING (c)
        """,
        """
        LEFT JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
        LEFT JOIN tbl_c ON tbl_b.c = tbl_c.c
        """,
    ],
)
def test_join_left_multi(join_clause: str) -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    with pl.SQLContext(frames) as ctx:
        for select_cols in (
            "tbl_a.a, tbl_a.b, tbl_b.c, tbl_c.d",
            "tbl_a.a, tbl_a.b, tbl_b.c, d",
        ):
            out = ctx.execute(
                f"SELECT {select_cols} FROM tbl_a {join_clause} ORDER BY a DESC"
            )
            assert out.collect().rows() == [
                (3, 6, "x", None),
                (2, None, None, None),
                (1, 4, "z", 25.5),
            ]


def test_join_left_multi_nested() -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    with pl.SQLContext(frames) as ctx:
        out = ctx.execute(
            """
            SELECT tbl_x.a, tbl_x.b, tbl_x.c, tbl_c.d FROM (
                SELECT *
                FROM tbl_a
                LEFT JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
            ) tbl_x
            LEFT JOIN tbl_c ON tbl_x.c = tbl_c.c
            ORDER BY tbl_x.a ASC
            """
        ).collect()

        assert out.rows() == [
            (1, 4, "z", 25.5),
            (2, None, None, None),
            (3, 6, "x", None),
        ]


def test_join_misc_13618() -> None:
    import polars as pl

    df = pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [5, 4, 3, 2, 1],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        }
    )
    res = (
        pl.SQLContext(t=df, t1=df, eager=True)
        .execute(
            """
            SELECT t.A, t.fruits, t1.B, t1.cars
            FROM t
            JOIN t1 ON t.A = t1.B
            ORDER BY t.A DESC
            """
        )
        .to_dict(as_series=False)
    )
    assert res == {
        "A": [5, 4, 3, 2, 1],
        "fruits": ["banana", "apple", "apple", "banana", "banana"],
        "B": [5, 4, 3, 2, 1],
        "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
    }


def test_join_misc_16255() -> None:
    df1 = pl.read_csv(BytesIO(b"id,data\n1,open"))
    df2 = pl.read_csv(BytesIO(b"id,data\n1,closed"))
    res = pl.sql(
        """
        SELECT a.id, a.data AS d1, b.data AS d2
        FROM df1 AS a JOIN df2 AS b
        ON a.id = b.id
        """,
        eager=True,
    )
    assert res.rows() == [(1, "open", "closed")]


@pytest.mark.parametrize(
    "constraint", ["tbl.a != tbl.b", "tbl.a > tbl.b", "a >= b", "a < b", "b <= a"]
)
def test_non_equi_joins(constraint: str) -> None:
    # no support (yet) for non equi-joins in polars joins
    # TODO: integrate awareness of new IEJoin
    with (
        pytest.raises(
            SQLInterfaceError,
            match=r"only equi-join constraints \(combined with 'AND'\) are currently supported",
        ),
        pl.SQLContext({"tbl": pl.DataFrame({"a": [1, 2, 3], "b": [4, 3, 2]})}) as ctx,
    ):
        ctx.execute(
            f"""
            SELECT *
            FROM tbl
            LEFT JOIN tbl ON {constraint}  -- not an equi-join
            """
        )


def test_implicit_joins() -> None:
    # no support for this yet; ensure we catch it
    with (
        pytest.raises(
            SQLInterfaceError,
            match=r"not currently supported .* use explicit JOIN syntax instead",
        ),
        pl.SQLContext(
            {
                "tbl": pl.DataFrame(
                    {"a": [1, 2, 3], "b": [4, 3, 2], "c": ["x", "y", "z"]}
                )
            }
        ) as ctx,
    ):
        ctx.execute(
            """
            SELECT t1.*
            FROM tbl AS t1, tbl AS t2
            WHERE t1.a = t2.b
            """
        )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        # INNER joins
        (
            "SELECT df1.* FROM df1 INNER JOIN df2 USING (a)",
            {"a": [1, 3], "b": ["x", "z"], "c": [100, 300]},
        ),
        (
            "SELECT df2.* FROM df1 INNER JOIN df2 USING (a)",
            {"a": [1, 3], "b": ["qq", "pp"], "c": [400, 500]},
        ),
        (
            "SELECT df1.* FROM df2 INNER JOIN df1 USING (a)",
            {"a": [1, 3], "b": ["x", "z"], "c": [100, 300]},
        ),
        (
            "SELECT df2.* FROM df2 INNER JOIN df1 USING (a)",
            {"a": [1, 3], "b": ["qq", "pp"], "c": [400, 500]},
        ),
        # LEFT joins
        (
            "SELECT df1.* FROM df1 LEFT JOIN df2 USING (a)",
            {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [100, 200, 300]},
        ),
        (
            "SELECT df2.* FROM df1 LEFT JOIN df2 USING (a)",
            {"a": [1, 3, None], "b": ["qq", "pp", None], "c": [400, 500, None]},
        ),
        (
            "SELECT df1.* FROM df2 LEFT JOIN df1 USING (a)",
            {"a": [1, 3, None], "b": ["x", "z", None], "c": [100, 300, None]},
        ),
        (
            "SELECT df2.* FROM df2 LEFT JOIN df1 USING (a)",
            {"a": [1, 3, 4], "b": ["qq", "pp", "oo"], "c": [400, 500, 600]},
        ),
        # RIGHT joins
        (
            "SELECT df1.* FROM df1 RIGHT JOIN df2 USING (a)",
            {"a": [1, 3, None], "b": ["x", "z", None], "c": [100, 300, None]},
        ),
        (
            "SELECT df2.* FROM df1 RIGHT JOIN df2 USING (a)",
            {"a": [1, 3, 4], "b": ["qq", "pp", "oo"], "c": [400, 500, 600]},
        ),
        (
            "SELECT df1.* FROM df2 RIGHT JOIN df1 USING (a)",
            {"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [100, 200, 300]},
        ),
        (
            "SELECT df2.* FROM df2 RIGHT JOIN df1 USING (a)",
            {"a": [1, 3, None], "b": ["qq", "pp", None], "c": [400, 500, None]},
        ),
        # FULL joins
        (
            "SELECT df1.* FROM df1 FULL JOIN df2 USING (a)",
            {
                "a": [1, 2, 3, None],
                "b": ["x", "y", "z", None],
                "c": [100, 200, 300, None],
            },
        ),
        (
            "SELECT df2.* FROM df1 FULL JOIN df2 USING (a)",
            {
                "a": [1, 3, 4, None],
                "b": ["qq", "pp", "oo", None],
                "c": [400, 500, 600, None],
            },
        ),
        (
            "SELECT df1.* FROM df2 FULL JOIN df1 USING (a)",
            {
                "a": [1, 2, 3, None],
                "b": ["x", "y", "z", None],
                "c": [100, 200, 300, None],
            },
        ),
        (
            "SELECT df2.* FROM df2 FULL JOIN df1 USING (a)",
            {
                "a": [1, 3, 4, None],
                "b": ["qq", "pp", "oo", None],
                "c": [400, 500, 600, None],
            },
        ),
    ],
)
def test_wildcard_resolution_and_join_order(
    query: str, expected: dict[str, Any]
) -> None:
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [100, 200, 300]})
    df2 = pl.DataFrame({"a": [1, 3, 4], "b": ["qq", "pp", "oo"], "c": [400, 500, 600]})

    res = pl.sql(query).collect()
    assert_frame_equal(
        res,
        pl.DataFrame(expected),
        check_row_order=False,
    )


def test_natural_joins_01() -> None:
    df1 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 4],
            "FirstName": ["Jernau Morat", "Cheradenine", "Byr", "Diziet"],
            "LastName": ["Gurgeh", "Zakalwe", "Genar-Hofoen", "Sma"],
        }
    )
    df2 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 5],
            "Role": ["Protagonist", "Protagonist", "Protagonist", "Antagonist"],
            "Book": [
                "Player of Games",
                "Use of Weapons",
                "Excession",
                "Consider Phlebas",
            ],
        }
    )
    df3 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 4],
            "Affiliation": ["Culture", "Culture", "Culture", "Shellworld"],
            "Species": ["Pan-human", "Human", "Human", "Oct"],
        }
    )
    df4 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 6],
            "Ship": [
                "Limiting Factor",
                "Xenophobe",
                "Grey Area",
                "Falling Outside The Normal Moral Constraints",
            ],
            "Drone": ["Flere-Imsaho", "Skaffen-Amtiskaw", "Eccentric", "Psychopath"],
        }
    )
    with pl.SQLContext(
        {"df1": df1, "df2": df2, "df3": df3, "df4": df4}, eager=True
    ) as ctx:
        res = ctx.execute(
            """
            SELECT *
            FROM df1
            NATURAL LEFT JOIN df2
            NATURAL INNER JOIN df3
            NATURAL LEFT JOIN df4
            ORDER BY ALL
            """
        )
        assert res.rows(named=True) == [
            {
                "CharacterID": 1,
                "FirstName": "Jernau Morat",
                "LastName": "Gurgeh",
                "Role": "Protagonist",
                "Book": "Player of Games",
                "Affiliation": "Culture",
                "Species": "Pan-human",
                "Ship": "Limiting Factor",
                "Drone": "Flere-Imsaho",
            },
            {
                "CharacterID": 2,
                "FirstName": "Cheradenine",
                "LastName": "Zakalwe",
                "Role": "Protagonist",
                "Book": "Use of Weapons",
                "Affiliation": "Culture",
                "Species": "Human",
                "Ship": "Xenophobe",
                "Drone": "Skaffen-Amtiskaw",
            },
            {
                "CharacterID": 3,
                "FirstName": "Byr",
                "LastName": "Genar-Hofoen",
                "Role": "Protagonist",
                "Book": "Excession",
                "Affiliation": "Culture",
                "Species": "Human",
                "Ship": "Grey Area",
                "Drone": "Eccentric",
            },
            {
                "CharacterID": 4,
                "FirstName": "Diziet",
                "LastName": "Sma",
                "Role": None,
                "Book": None,
                "Affiliation": "Shellworld",
                "Species": "Oct",
                "Ship": None,
                "Drone": None,
            },
        ]

    # misc errors
    with pytest.raises(SQLSyntaxError, match=r"did you mean COLUMNS\(\*\)\?"):
        pl.sql("SELECT * FROM df1 NATURAL JOIN df2 WHERE COLUMNS('*') >= 5")

    with pytest.raises(SQLSyntaxError, match=r"COLUMNS expects a regex"):
        pl.sql("SELECT COLUMNS(1234) FROM df1 NATURAL JOIN df2")


@pytest.mark.parametrize(
    ("cols_constraint", "expect_data"),
    [
        (">= 5", [(8, 8, 6)]),
        ("< 7", [(5, 4, 4)]),
        ("< 8", [(5, 4, 4), (7, 4, 4), (0, 7, 2)]),
        ("!= 4", [(8, 8, 6), (2, 8, 6), (0, 7, 2)]),
    ],
)
def test_natural_joins_02(cols_constraint: str, expect_data: list[tuple[int]]) -> None:
    df1 = pl.DataFrame(
        {
            "x": [1, 5, 3, 8, 6, 7, 4, 0, 2],
            "y": [3, 4, 6, 8, 3, 4, 1, 7, 8],
        }
    )
    df2 = pl.DataFrame(
        {
            "y": [0, 4, 0, 8, 0, 4, 0, 7, None],
            "z": [9, 8, 7, 6, 5, 4, 3, 2, 1],
        },
    )
    actual = pl.sql(
        f"""
        SELECT *
        FROM df1 NATURAL JOIN df2
        WHERE COLUMNS(*) {cols_constraint}
        """
    ).collect()

    df_expected = pl.DataFrame(expect_data, schema=actual.columns, orient="row")
    assert_frame_equal(actual, df_expected, check_row_order=False)


@pytest.mark.parametrize(
    "join_clause",
    [
        """
        df2 JOIN df3 ON
        df2.CharacterID = df3.CharacterID
        """,
        """
        df2 INNER JOIN (
          df3 JOIN df4 ON df3.CharacterID = df4.CharacterID
        ) AS r0 ON df2.CharacterID = df3.CharacterID
        """,
    ],
)
def test_nested_join(join_clause: str) -> None:
    df1 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 4],
            "FirstName": ["Jernau Morat", "Cheradenine", "Byr", "Diziet"],
            "LastName": ["Gurgeh", "Zakalwe", "Genar-Hofoen", "Sma"],
        }
    )
    df2 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 5],
            "Role": ["Protagonist", "Protagonist", "Protagonist", "Antagonist"],
            "Book": [
                "Player of Games",
                "Use of Weapons",
                "Excession",
                "Consider Phlebas",
            ],
        }
    )
    df3 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 5, 6],
            "Affiliation": ["Culture", "Culture", "Culture", "Shellworld"],
            "Species": ["Pan-human", "Human", "Human", "Oct"],
        }
    )
    df4 = pl.DataFrame(
        {
            "CharacterID": [1, 2, 3, 6],
            "Ship": [
                "Limiting Factor",
                "Xenophobe",
                "Grey Area",
                "Falling Outside The Normal Moral Constraints",
            ],
            "Drone": ["Flere-Imsaho", "Skaffen-Amtiskaw", "Eccentric", "Psychopath"],
        }
    )

    with pl.SQLContext(
        {"df1": df1, "df2": df2, "df3": df3, "df4": df4}, eager=True
    ) as ctx:
        res = ctx.execute(
            f"""
            SELECT df1.CharacterID, df1.FirstName, df2.Role, df3.Species
            FROM df1
            INNER JOIN ({join_clause}) AS r99
            ON df1.CharacterID = df2.CharacterID
            ORDER BY ALL
            """
        )
        assert res.rows(named=True) == [
            {
                "CharacterID": 1,
                "FirstName": "Jernau Morat",
                "Role": "Protagonist",
                "Species": "Pan-human",
            },
            {
                "CharacterID": 2,
                "FirstName": "Cheradenine",
                "Role": "Protagonist",
                "Species": "Human",
            },
        ]


def test_miscellaneous_cte_join_aliasing() -> None:
    ctx = pl.SQLContext()
    res = ctx.execute(
        """
        WITH t AS (SELECT a FROM (VALUES(1),(2)) tbl(a))
        SELECT * FROM t CROSS JOIN t
        """,
        eager=True,
    )
    assert sorted(res.rows()) == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]


def test_nested_joins_17381() -> None:
    df = pl.DataFrame({"id": ["one", "two"]})

    ctx = pl.SQLContext({"a": df})
    res = ctx.execute(
        """
        -- the interaction of the (unused) CTE and the nested subquery resulted
        -- in arena mutation/cleanup that wasn't accounted for, affecting state
        WITH c AS (SELECT a.id FROM a)
        SELECT *
        FROM a
        WHERE id IN (
            SELECT a2.id
            FROM a
            INNER JOIN a AS a2 ON a.id = a2.id
        )
        """,
        eager=True,
    )
    assert set(res["id"]) == {"one", "two"}


def test_unnamed_nested_join_relation() -> None:
    df = pl.DataFrame({"a": 1})

    with (
        pl.SQLContext({"left": df, "right": df}) as ctx,
        pytest.raises(SQLInterfaceError, match="cannot JOIN on unnamed relation"),
    ):
        ctx.execute(
            """
            SELECT *
            FROM left
            JOIN (right JOIN right ON right.a = right.a)
            ON left.a = right.a
            """
        )


def test_nulls_equal_19624() -> None:
    df1 = pl.DataFrame({"a": [1, 2, None, None]})
    df2 = pl.DataFrame({"a": [1, 1, 2, 2, None], "b": [0, 1, 2, 3, 4]})

    # left join
    res_df = df1.join(df2, how="left", on="a", nulls_equal=False, validate="1:m")
    expected_df = pl.DataFrame(
        {"a": [1, 1, 2, 2, None, None], "b": [0, 1, 2, 3, None, None]}
    )
    assert_frame_equal(res_df, expected_df)
    res_df = df2.join(df1, how="left", on="a", nulls_equal=False, validate="m:1")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2, None], "b": [0, 1, 2, 3, 4]})
    assert_frame_equal(res_df, expected_df)

    # inner join
    res_df = df1.join(df2, how="inner", on="a", nulls_equal=False, validate="1:m")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [0, 1, 2, 3]})
    assert_frame_equal(res_df, expected_df)
    res_df = df2.join(df1, how="inner", on="a", nulls_equal=False, validate="m:1")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [0, 1, 2, 3]})
    assert_frame_equal(res_df, expected_df)


def test_join_on_literal_string_comparison() -> None:
    df1 = pl.DataFrame(
        {
            "name": ["alice", "bob", "adam", "charlie"],
            "role": ["admin", "user", "admin", "user"],
        }
    )
    df2 = pl.DataFrame(
        {
            "name": ["alice", "bob", "charlie", "adam"],
            "dept": ["IT", "HR", "IT", "SEC"],
        }
    )
    query = """
        SELECT df1.name, df1.role, df2.dept
        FROM df1
        INNER JOIN df2 ON df1.name = df2.name AND df1.role = 'admin'
        ORDER BY df1.name
    """
    df_expected = pl.DataFrame(
        data=[("adam", "admin", "SEC"), ("alice", "admin", "IT")],
        schema={"name": str, "role": str, "dept": str},
        orient="row",
    )
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("expression", "expected_length"),
    [
        ("LOWER(df1.text) = df2.text", 2),  # case conversion
        ("SUBSTR(df1.code, 1, 2) = SUBSTR(df2.code, 1, 2)", 3),  # first letter match
        ("LENGTH(df1.text) = LENGTH(df2.text)", 5),  # cartesian on matching lengths
    ],
)
def test_join_on_expression_conditions(expression: str, expected_length: int) -> None:
    df1 = pl.DataFrame(
        {
            "text": ["HELLO", "WORLD", "FOO"],
            "code": ["ABC", "DEF", "GHI"],
        }
    )
    df2 = pl.DataFrame(
        {
            "text": ["hello", "world", "bar"],
            "code": ["ABX", "DEY", "GHZ"],
        }
    )
    query = f"""
        SELECT df1.text AS text1, df2.text AS text2
        FROM df1
        INNER JOIN df2 ON {expression}
        ORDER BY text1
    """
    res = pl.sql(query, eager=True)
    assert len(res) == expected_length


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "select_cols", "expected", "schema"),
    [
        (
            pl.DataFrame(
                {
                    "category": ["fruit", "fruit", "vegetable"],
                    "name": ["apple", "banana", "carrot"],
                    "code": [1, 2, 3],
                }
            ),
            pl.DataFrame(
                {
                    "category": ["fruit", "fruit", "vegetable"],
                    "type": ["sweet", "tropical", "root"],
                    "code_doubled": [2, 4, 6],
                }
            ),
            "df1.category = df2.category AND (df1.code * 2) = df2.code_doubled",
            "df1.name, df1.code, df2.type",
            [("apple", 1, "sweet"), ("banana", 2, "tropical"), ("carrot", 3, "root")],
            ["name", "code", "type"],
        ),
        (
            pl.DataFrame({"id": [1, 2, 3], "name": ["ALICE", "BOB", "CHARLIE"]}),
            pl.DataFrame({"id": [1, 2, 3], "match": ["alice", "bob", "charlie"]}),
            "df1.id = df2.id AND LOWER(df1.name) = df2.match",
            "df1.id, df1.name, df2.match",
            [(1, "ALICE", "alice"), (2, "BOB", "bob"), (3, "CHARLIE", "charlie")],
            ["id", "name", "match"],
        ),
        (
            pl.DataFrame({"x": [2, 4, 6], "y": [1, 2, 3]}),
            pl.DataFrame({"a": [4, 8, 12], "b": [1, 2, 3]}),
            "df1.x * 2 = df2.a AND df1.y = df2.b",
            "df1.x, df1.y, df2.a",
            [(2, 1, 4), (4, 2, 8), (6, 3, 12)],
            ["x", "y", "a"],
        ),
    ],
)
def test_join_on_mixed_expression_conditions(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    select_cols: str,
    expected: list[tuple[Any, ...]],
    schema: list[str],
) -> None:
    query = f"""
        SELECT {select_cols}
        FROM df1
        INNER JOIN df2 ON {join_constraint}
        ORDER BY ALL
    """
    df_expected = pl.DataFrame(expected, schema=schema, orient="row")
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "expected"),
    [
        (
            pl.DataFrame({"text": ["  Hello  ", "  World  ", "  Test  "]}),
            pl.DataFrame({"text": ["hello", "world", "other"]}),
            "LOWER(TRIM(df1.text)) = df2.text",
            [("  Hello  ", "hello"), ("  World  ", "world")],
        ),
        (
            pl.DataFrame({"code": ["PREFIX_A", "SECOND_B", "OTHERS_C"]}),
            pl.DataFrame({"code": ["prefix", "second", "others"]}),
            "LOWER(SUBSTR(df1.code,1,6)) = df2.code",
            [("OTHERS_C", "others"), ("PREFIX_A", "prefix"), ("SECOND_B", "second")],
        ),
        (
            pl.DataFrame({"name": ["abc", "abcde", "x"]}),
            pl.DataFrame({"len": [3, 5, 1]}),
            "LENGTH(df1.name) = df2.len",
            [("x", 1), ("abc", 3), ("abcde", 5)],
        ),
    ],
)
def test_join_on_nested_function_expressions(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    expected: list[tuple[Any, ...]],
) -> None:
    col1 = df1.columns[0]
    col2 = df2.columns[0]

    query = f"""
        SELECT df1.{col1} AS col1, df2.{col2} AS col2
        FROM df1
        INNER JOIN df2 ON {join_constraint}
        ORDER BY df2.{col2}
    """
    df_expected = pl.DataFrame(expected, schema=["col1", "col2"], orient="row")
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "select_cols", "expected", "schema"),
    [
        (
            pl.DataFrame(
                {"id": [1, 2, 3], "category": ["A", "B", "A"], "multiplier": [2, 3, 4]}
            ),
            pl.DataFrame(
                {"id": [1, 2, 3], "base": [5, 15, 20], "category": ["A", "B", "C"]}
            ),
            "df1.id = df2.id AND df1.multiplier * 5 = df2.base AND df1.category = 'A'",
            "df1.id, df1.multiplier, df2.base",
            [(3, 4, 20)],
            ["id", "multiplier", "base"],
        ),
        (
            pl.DataFrame({"id": [1, 2, 3], "value": [10, 20, 30]}),
            pl.DataFrame({"id": [1, 2, 3], "target": [20, 40, 60]}),
            "df1.id = df2.id AND (df1.value * 2) = df2.target AND df1.id = 2",
            "df1.id, df1.value, df2.target",
            [(2, 20, 40)],
            ["id", "value", "target"],
        ),
        (
            pl.DataFrame(
                {
                    "x": [1, 2, 3],
                    "type": ["A", "B", "A"],
                    "status": ["active", "inactive", "active"],
                }
            ),
            pl.DataFrame({"x": [1, 2, 3], "data": ["foo", "bar", "baz"]}),
            "df1.x = df2.x AND df1.type = 'A' AND df1.status = 'active'",
            "df1.x, df2.data",
            [(1, "foo"), (3, "baz")],
            ["x", "data"],
        ),
    ],
)
def test_join_on_expression_with_literals(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    select_cols: str,
    expected: list[tuple[Any, ...]],
    schema: list[str],
) -> None:
    query = f"""
        SELECT {select_cols}
        FROM df1
        INNER JOIN df2 ON {join_constraint}
        ORDER BY ALL
    """
    df_expected = pl.DataFrame(
        expected,
        schema=schema,
        orient="row",
    )
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "reversed_join_constraint", "expected", "schema"),
    [
        (
            pl.DataFrame({"id": [1, 2, 3], "val": ["a", "b", "c"]}),
            pl.DataFrame({"id": [2, 3, 4], "val": ["x", "y", "z"]}),
            "df1.id = df2.id",
            "df2.id = df1.id",
            [(2, "b", "x"), (3, "c", "y")],
            ["id", "val1", "val2"],
        ),
        (
            pl.DataFrame({"x": [1, 2, 3]}),
            pl.DataFrame({"y": [2, 4, 6]}),
            "df1.x * 2 = df2.y",
            "df2.y = (df1.x * 2)",
            [(1, 2), (2, 4), (3, 6)],
            ["x", "y"],
        ),
        (
            pl.DataFrame({"a": [5, 10, 15]}),
            pl.DataFrame({"b": [10, 20, 30]}),
            "(df1.a + df1.a) = df2.b",
            "df2.b = (df1.a + df1.a)",
            [(5, 10), (10, 20), (15, 30)],
            ["a", "b"],
        ),
    ],
)
def test_join_on_reversed_constraint_order(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    reversed_join_constraint: str,
    expected: list[tuple[Any, ...]],
    schema: list[str],
) -> None:
    select_cols = (
        "df1.id, df1.val AS val1, df2.val AS val2"
        if len(schema) == 3
        else ", ".join(f"df{i + 1}.{col}" for i, col in enumerate(schema))
    )
    df_expected = pl.DataFrame(
        expected,
        schema=schema,
        orient="row",
    )
    for constraint in (join_constraint, reversed_join_constraint):
        res = pl.sql(
            query=f"""
                SELECT {select_cols}
                FROM df1
                INNER JOIN df2 ON {constraint}
                ORDER BY ALL
            """,
            eager=True,
        )
        assert_frame_equal(res, df_expected)


@pytest.mark.parametrize(
    ("df1", "df2", "join_constraint", "expected", "schema"),
    [
        (
            pl.DataFrame({"a": [1, 2, 3]}),
            pl.DataFrame({"b": [2, 4, 6]}),
            "a * 2 = b",
            [(1, 2), (2, 4), (3, 6)],
            ["a", "b"],
        ),
        (
            pl.DataFrame({"x": [5, 10, 15], "y": [3, 5, 7]}),
            pl.DataFrame({"sum": [8, 15, 22]}),
            "x + y = sum",
            [(5, 3, 8), (10, 5, 15), (15, 7, 22)],
            ["x", "y", "sum"],
        ),
        (
            pl.DataFrame({"name": ["abc", "hello", "test"]}),
            pl.DataFrame({"len": [3, 5, 4]}),
            "LENGTH(name) = len",
            [("abc", 3), ("hello", 5), ("test", 4)],
            ["name", "len"],
        ),
    ],
)
def test_join_on_unqualified_expressions(
    df1: pl.DataFrame,
    df2: pl.DataFrame,
    join_constraint: str,
    expected: list[tuple[Any, ...]],
    schema: list[str],
) -> None:
    df1_cols = ", ".join(f"df1.{col}" for col in df1.columns)
    df2_cols = ", ".join(f"df2.{col}" for col in df2.columns)

    query = f"""
        SELECT {df1_cols}, {df2_cols}
        FROM df1
        INNER JOIN df2 ON {join_constraint}
        ORDER BY ALL
    """
    df_expected = pl.DataFrame(
        expected,
        schema=schema,
        orient="row",
    )
    res = pl.sql(query, eager=True)
    assert_frame_equal(res, df_expected)


def test_multiway_join_chain_with_aliased_cols() -> None:
    # tracking/resolving constraints for 3-way (or more) joins can be... "fun" :)
    # ref: https://github.com/pola-rs/polars/issues/25126

    df1 = pl.DataFrame({"a": [111, 222], "x1": ["df1", "df1"]})
    df2 = pl.DataFrame({"a": [333, 111], "b": [444, 222], "x2": ["df2", "df2"]})
    df3 = pl.DataFrame({"a": [222, 111], "x3": ["df3", "df3"]})

    for query, expected_cols, expected_row in (
        (
            # three-way join where "a" exists in all three frames (df1, df2, df3)
            """
            SELECT * FROM df3
            INNER JOIN df2 ON df2.b = df3.a
            INNER JOIN df1 ON df1.a = df2.a
            """,
            ["a", "x3", "a:df2", "b", "x2", "a:df1", "x1"],
            (222, "df3", 111, 222, "df2", 111, "df1"),
        ),
        (
            # almost the same, but the final constraint on "a" refers back to df1
            """
            SELECT * FROM df3
            INNER JOIN df2 ON df2.b = df3.a
            INNER JOIN df1 ON df1.a = df3.a
            """,
            ["a", "x3", "a:df2", "b", "x2", "a:df1", "x1"],
            (222, "df3", 111, 222, "df2", 222, "df1"),
        ),
    ):
        res = pl.sql(query, eager=True)

        assert res.height == 1
        assert res.columns == expected_cols
        assert res.row(0) == expected_row


@pytest.mark.parametrize(
    ("join_condition", "expected_error"),
    [
        (
            "(df1.id + df2.val) = df2.id",
            r"unsupported join condition: left side references both 'df1' and 'df2'",
        ),
        (
            "df1.id = (df2.id + df1.val)",
            r"unsupported join condition: right side references both 'df1' and 'df2'",
        ),
    ],
)
def test_unsupported_join_conditions(join_condition: str, expected_error: str) -> None:
    # note: this is technically valid (if unusual) SQL, but we don't support it
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    df2 = pl.DataFrame({"id": [2, 3, 4], "val": [20, 30, 40]})

    with pytest.raises(SQLInterfaceError, match=expected_error):
        pl.sql(f"SELECT * FROM df1 INNER JOIN df2 ON {join_condition}")


def test_ambiguous_column_detection_in_joins() -> None:
    # unqualified column references that exist in multiple tables should raise
    # an error (with a helpful suggestion about qualifying the reference)
    with pytest.raises(
        SQLInterfaceError,
        match=r'ambiguous reference to column "k" \(use one of: a\.k, c\.k\)',
    ):
        pl.sql(
            query="""
                WITH
                  a AS (SELECT 0 AS k),
                  c AS (SELECT 0 AS k)
                SELECT k FROM a JOIN c ON a.k = c.k
            """,
            eager=True,
        )


def test_duplicate_column_detection_via_wildcard() -> None:
    # selecting a column explicitly that is already included in a qualified
    # wildcard from the same table should raise a duplicate column error
    a = pl.DataFrame({"id": [1, 2], "x": [10, 20]})
    b = pl.DataFrame({"id": [1, 2], "y": [30, 40]})

    with pytest.raises(
        SQLInterfaceError,
        match=r"column 'id' is duplicated in the SELECT",
    ):
        pl.sql("SELECT a.*, a.id FROM a JOIN b ON a.id = b.id", eager=True)


def test_qualified_wildcard_multiway_join() -> None:
    df1 = pl.DataFrame({"id": [1, 2], "a": ["x", "y"]})
    df2 = pl.DataFrame({"id": [1, 2], "b": ["p", "q"]})
    df3 = pl.DataFrame({"id": [1, 2], "c": ["m", "n"]})

    res = pl.sql("""
        SELECT df1.*, df2.*, df3.*
        FROM df1
        INNER JOIN df2 ON df1.id = df2.id
        INNER JOIN df3 ON df1.id = df3.id
        ORDER BY id
    """).collect()
    expected = pl.DataFrame(
        {
            "id": [1, 2],
            "a": ["x", "y"],
            "id:df2": [1, 2],
            "b": ["p", "q"],
            "id:df3": [1, 2],
            "c": ["m", "n"],
        }
    )
    assert_frame_equal(res, expected)


def test_qualified_wildcard_self_join() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "parent": [None, 1, 1],
            "name": ["root", "child1", "child2"],
        }
    )
    res = pl.sql("""
        SELECT child.*, parent.*
        FROM df AS child
        LEFT JOIN df AS parent ON child.parent = parent.id
        ORDER BY id
    """).collect()

    expected = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "parent": [None, 1, 1],
            "name": ["root", "child1", "child2"],
            "id:parent": [None, 1, 1],
            "parent:parent": [None, None, None],
            "name:parent": [None, "root", "root"],
        },
        schema_overrides={"parent:parent": pl.Int64},
    )
    assert_frame_equal(res, expected)


@pytest.mark.parametrize(
    ("join_type", "result"),
    [
        (
            "INNER",
            {"k": [1], "v": ["a"], "k:df2": [1], "v:df2": ["x"]},
        ),
        (
            "LEFT",
            {"k": [1, 2], "v": ["a", "b"], "k:df2": [1, None], "v:df2": ["x", None]},
        ),
        (
            "RIGHT",
            {"k": [1, None], "v": ["a", None], "k:df2": [1, 3], "v:df2": ["x", "y"]},
        ),
    ],
)
def test_qualified_wildcard_join_types(join_type: str, result: dict[str, Any]) -> None:
    df1 = pl.DataFrame({"k": [1, 2], "v": ["a", "b"]})
    df2 = pl.DataFrame({"k": [1, 3], "v": ["x", "y"]})

    actual = pl.sql(
        query=f"""
        SELECT df1.*, df2.*
        FROM df1 {join_type} JOIN df2 ON df1.k = df2.k
        """,
        eager=True,
    )
    expected = pl.DataFrame(result)
    assert_frame_equal(
        left=expected,
        right=actual,
        check_row_order=False,
    )


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        (  # specific column conflicts with wildcard
            "SELECT a.id, b.* FROM a JOIN b ON a.id = b.id",
            {"id": [1, 2], "id:b": [1, 2], "y": [30, 40]},
        ),
        (  # specific column doesn't conflict with wildcard
            "SELECT b.y, a.* FROM a JOIN b ON a.id = b.id",
            {"y": [30, 40], "id": [1, 2], "x": [10, 20]},
        ),
        (  # single-table wildcard (no conflict, uses original names)
            "SELECT b.* FROM a JOIN b ON a.id = b.id",
            {"id": [1, 2], "y": [30, 40]},
        ),
        (  # table aliases (disambiguation should use the alias)
            "SELECT t1.*, t2.* FROM a AS t1 JOIN b AS t2 ON t1.id = t2.id",
            {"id": [1, 2], "x": [10, 20], "id:t2": [1, 2], "y": [30, 40]},
        ),
        (  # no column overlap (expect no disambiguation)
            "SELECT a.*, c.* FROM a JOIN c ON a.id = c.k",
            {"id": [1, 2], "x": [10, 20], "k": [1, 2], "z": [50, 60]},
        ),
        (  # reverse wildcard order (disambiguation follows *table* order)
            "SELECT b.*, a.* FROM a JOIN b ON a.id = b.id",
            {"id:b": [1, 2], "y": [30, 40], "id": [1, 2], "x": [10, 20]},
        ),
    ],
)
def test_qualified_wildcard_combinations(query: str, expected: dict[str, Any]) -> None:
    a = pl.DataFrame({"id": [1, 2], "x": [10, 20]})
    b = pl.DataFrame({"id": [1, 2], "y": [30, 40]})
    c = pl.DataFrame({"k": [1, 2], "z": [50, 60]})

    assert_frame_equal(
        left=pl.DataFrame(expected),
        right=pl.sql(query).collect(),
        check_row_order=False,
    )
