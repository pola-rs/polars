from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal


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
    df1 = pl.DataFrame({"id": [1, 2, 3]})  # noqa: F841
    df2 = pl.DataFrame({"id": [3, 4, 5]})  # noqa: F841
    df3 = pl.DataFrame({"id": [4, 5, 6]})  # noqa: F841

    res = pl.sql("SELECT df1.id FROM df1 CROSS JOIN df2 WHERE df1.id = df2.id")
    assert_frame_equal(res.collect(), pl.DataFrame({"id": [3]}))

    res = pl.sql("SELECT * FROM df1 CROSS JOIN df3 WHERE df1.id = df3.id")
    assert res.collect().is_empty()


@pytest.mark.parametrize(
    "join_clause",
    [
        "ON foods1.category = foods2.category",
        "ON foods2.category = foods1.category",
        "USING (category)",
    ],
)
def test_join_inner(foods_ipc_path: Path, join_clause: str) -> None:
    foods1 = pl.scan_ipc(foods_ipc_path)
    foods2 = foods1  # noqa: F841
    schema = foods1.collect_schema()

    sort_clause = ", ".join(f'{c} ASC, "{c}:foods2" DESC' for c in schema)
    out = pl.sql(
        f"""
        SELECT *
        FROM foods1
        INNER JOIN foods2 {join_clause}
        ORDER BY {sort_clause}
        LIMIT 2
        """,
        eager=True,
    )

    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "category": ["fruit", "fruit"],
                "calories": [30, 30],
                "fats_g": [0.0, 0.0],
                "sugars_g": [3, 5],
                "category:foods2": ["fruit", "fruit"],
                "calories:foods2": [130, 130],
                "fats_g:foods2": [0.0, 0.0],
                "sugars_g:foods2": [25, 25],
            }
        ),
        check_dtypes=False,
    )


@pytest.mark.parametrize(
    "join_clause",
    [
        """
        INNER JOIN tbl_b USING (a,b)
        INNER JOIN tbl_c USING (c)
        """,
        """
        INNER JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
        INNER JOIN tbl_c ON tbl_a.c = tbl_c.c
        """,
    ],
)
def test_join_inner_multi(join_clause: str) -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    with pl.SQLContext(frames) as ctx:
        assert ctx.tables() == ["tbl_a", "tbl_b", "tbl_c"]
        for select_cols in ("a, b, c, d", "tbl_a.a, tbl_a.b, tbl_b.c, tbl_c.d"):
            out = ctx.execute(
                f"SELECT {select_cols} FROM tbl_a {join_clause} ORDER BY a DESC"
            )
            assert out.collect().rows() == [(1, 4, "z", 25.5)]


def test_join_inner_15663() -> None:
    df_a = pl.DataFrame({"LOCID": [1, 2, 3], "VALUE": [0.1, 0.2, 0.3]})  # noqa: F841
    df_b = pl.DataFrame({"LOCID": [1, 2, 3], "VALUE": [25.6, 53.4, 12.7]})  # noqa: F841
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
        FROM df_a AS a
        INNER JOIN df_b AS b
        USING (LOCID)
        ORDER BY LOCID
        """
        actual = ctx.execute(query)
        assert_frame_equal(df_expected, actual)


@pytest.mark.parametrize(
    "join_clause",
    [
        """
        LEFT JOIN tbl_b USING (a,b)
        LEFT JOIN tbl_c USING (c)
        """,
        """
        LEFT JOIN tbl_b ON tbl_a.a = tbl_b.a AND tbl_a.b = tbl_b.b
        LEFT JOIN tbl_c ON tbl_a.c = tbl_c.c
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
        for select_cols in ("a, b, c, d", "tbl_a.a, tbl_a.b, tbl_b.c, tbl_c.d"):
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
        for select_cols in ("a, b, c, d", "tbl_x.a, tbl_x.b, tbl_x.c, tbl_c.d"):
            out = ctx.execute(
                f"""
                SELECT {select_cols} FROM (SELECT *
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
    df1 = pl.read_csv(BytesIO(b"id,data\n1,open"))  # noqa: F841
    df2 = pl.read_csv(BytesIO(b"id,data\n1,closed"))  # noqa: F841
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
    df1 = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [100, 200, 300]})  # noqa: F841
    df2 = pl.DataFrame({"a": [1, 3, 4], "b": ["qq", "pp", "oo"], "c": [400, 500, 600]})  # noqa: F841

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
        # note: use of 'COLUMNS' is a neat way to drop
        # all non-coalesced "<name>:<suffix>" cols
        res = ctx.execute(
            """
            SELECT COLUMNS('^[^:]*$')
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
    df1 = pl.DataFrame(  # noqa: F841
        {
            "x": [1, 5, 3, 8, 6, 7, 4, 0, 2],
            "y": [3, 4, 6, 8, 3, 4, 1, 7, 8],
        }
    )
    df2 = pl.DataFrame(  # noqa: F841
        {
            "y": [0, 4, 0, 8, 0, 4, 0, 7, None],
            "z": [9, 8, 7, 6, 5, 4, 3, 2, 1],
        },
    )
    actual = pl.sql(
        f"""
        SELECT * EXCLUDE "y:df2"
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
            SELECT id
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
        pytest.raises(SQLInterfaceError, match="cannot join on unnamed relation"),
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
    result_df = df1.join(df2, how="left", on="a", nulls_equal=False, validate="1:m")
    expected_df = pl.DataFrame(
        {"a": [1, 1, 2, 2, None, None], "b": [0, 1, 2, 3, None, None]}
    )
    assert_frame_equal(result_df, expected_df)
    result_df = df2.join(df1, how="left", on="a", nulls_equal=False, validate="m:1")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2, None], "b": [0, 1, 2, 3, 4]})
    assert_frame_equal(result_df, expected_df)

    # inner join
    result_df = df1.join(df2, how="inner", on="a", nulls_equal=False, validate="1:m")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [0, 1, 2, 3]})
    assert_frame_equal(result_df, expected_df)
    result_df = df2.join(df1, how="inner", on="a", nulls_equal=False, validate="m:1")
    expected_df = pl.DataFrame({"a": [1, 1, 2, 2], "b": [0, 1, 2, 3]})
    assert_frame_equal(result_df, expected_df)


def test_join_on_literal_string_comparison() -> None:
    df1 = pl.DataFrame(  # noqa: F841
        {
            "name": ["alice", "bob", "adam", "charlie"],
            "role": ["admin", "user", "admin", "user"],
        }
    )
    df2 = pl.DataFrame(  # noqa: F841
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
    df1 = pl.DataFrame(  # noqa: F841
        {
            "text": ["HELLO", "WORLD", "FOO"],
            "code": ["ABC", "DEF", "GHI"],
        }
    )
    df2 = pl.DataFrame(  # noqa: F841
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
    df1 = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})  # noqa: F841
    df2 = pl.DataFrame({"id": [2, 3, 4], "val": [20, 30, 40]})  # noqa: F841

    with pytest.raises(SQLInterfaceError, match=expected_error):
        pl.sql(f"SELECT * FROM df1 INNER JOIN df2 ON {join_condition}")
