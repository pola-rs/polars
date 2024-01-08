from __future__ import annotations

from pathlib import Path

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
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
    ctx = pl.SQLContext(frames, eager_execution=True)
    assert_frame_equal(expected, ctx.execute(sql))


@pytest.mark.parametrize(
    "join_clause",
    [
        "ON foods1.category = foods2.category",
        "ON foods2.category = foods1.category",
        "USING (category)",
    ],
)
def test_join_inner(foods_ipc_path: Path, join_clause: str) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    ctx = pl.SQLContext()
    ctx.register_many(foods1=lf, foods2=lf)

    out = ctx.execute(
        f"""
        SELECT *
        FROM foods1
        INNER JOIN foods2 {join_clause}
        LIMIT 2
        """
    )
    assert out.collect().to_dict(as_series=False) == {
        "category": ["vegetables", "vegetables"],
        "calories": [45, 20],
        "fats_g": [0.5, 0.0],
        "sugars_g": [2, 2],
        "calories_right": [45, 45],
        "fats_g_right": [0.5, 0.5],
        "sugars_g_right": [2, 2],
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


@pytest.mark.parametrize(
    "constraint", ["tbl.a != tbl.b", "tbl.a > tbl.b", "a >= b", "a < b", "b <= a"]
)
def test_non_equi_joins(constraint: str) -> None:
    # no support (yet) for non equi-joins in polars joins
    with pytest.raises(
        InvalidOperationError,
        match=r"SQL interface \(currently\) only supports basic equi-join constraints",
    ), pl.SQLContext({"tbl": pl.DataFrame({"a": [1, 2, 3], "b": [4, 3, 2]})}) as ctx:
        ctx.execute(
            f"""
            SELECT *
            FROM tbl
            LEFT JOIN tbl ON {constraint}  -- not an equi-join
            """
        )
