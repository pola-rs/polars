from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal


@pytest.fixture()
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
            "SELECT * FROM tbl_a LEFT SEMI JOIN tbl_b USING (a)",
            pl.DataFrame({"a": [1, 2, 3], "b": [4, 0, 6], "c": ["w", "y", "z"]}),
        ),
        (
            "SELECT * FROM tbl_a LEFT ANTI JOIN tbl_b USING (a)",
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

    out = pl.sql(
        f"""
        SELECT *
        FROM foods1
        INNER JOIN foods2 {join_clause}
        LIMIT 2
        """,
        eager=True,
    )

    assert out.to_dict(as_series=False) == {
        "category": ["vegetables", "vegetables"],
        "calories": [45, 20],
        "fats_g": [0.5, 0.0],
        "sugars_g": [2, 2],
        "category:foods2": ["vegetables", "vegetables"],
        "calories:foods2": [45, 45],
        "fats_g:foods2": [0.5, 0.5],
        "sugars_g:foods2": [2, 2],
    }


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
    expected = pl.DataFrame(
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
        assert_frame_equal(expected, actual)


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
    with pytest.raises(
        SQLInterfaceError,
        match=r"only equi-join constraints are supported",
    ), pl.SQLContext({"tbl": pl.DataFrame({"a": [1, 2, 3], "b": [4, 3, 2]})}) as ctx:
        ctx.execute(
            f"""
            SELECT *
            FROM tbl
            LEFT JOIN tbl ON {constraint}  -- not an equi-join
            """
        )


def test_implicit_joins() -> None:
    # no support for this yet; ensure we catch it
    with pytest.raises(
        SQLInterfaceError,
        match=r"not currently supported .* use explicit JOIN syntax instead",
    ), pl.SQLContext(
        {"tbl": pl.DataFrame({"a": [1, 2, 3], "b": [4, 3, 2], "c": ["x", "y", "z"]})}
    ) as ctx:
        ctx.execute(
            """
            SELECT t1.*
            FROM tbl AS t1, tbl AS t2
            WHERE t1.a = t2.b
            """
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

    expected = pl.DataFrame(expect_data, schema=actual.columns, orient="row")
    assert_frame_equal(actual, expected, check_row_order=False)


@pytest.mark.parametrize(
    "join_clause",
    [
        "df2 INNER JOIN df3 ON df2.CharacterID=df3.CharacterID",
        "df2 INNER JOIN (df3 INNDER JOIN df4 ON df3.CharacterID=df4.CharacterID) ON df2.CharacterID=df3.CharacterID",
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
            INNER JOIN ({join_clause})
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
