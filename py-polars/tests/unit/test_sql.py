from __future__ import annotations

import os
import warnings
from pathlib import Path

import pytest

import polars as pl
import polars.selectors as cs
from polars.testing import assert_frame_equal, assert_series_equal


# TODO: Do not rely on I/O for these tests
@pytest.fixture()
def foods_ipc_path() -> str:
    return str(Path(os.path.dirname(__file__)) / "io" / "files" / "foods1.ipc")


def test_sql_distinct() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    c = pl.SQLContext(register_globals=True, eager_execution=True)
    res1 = c.execute("SELECT DISTINCT a FROM df ORDER BY a DESC")
    assert_frame_equal(
        left=df.select("a").unique().sort(by="a", descending=True),
        right=res1,
    )

    res2 = c.execute(
        """
        SELECT DISTINCT
          a*2 AS two_a,
          b/2 AS half_b
        FROM df
        ORDER BY two_a ASC, half_b DESC
        """,
    )
    assert res2.to_dict(False) == {
        "two_a": [2, 2, 4, 6],
        "half_b": [1, 0, 2, 3],
    }

    # test unregistration
    c.unregister("df")
    with pytest.raises(pl.ComputeError, match=".*'df'.*not found"):
        c.execute("SELECT * FROM df")


def test_sql_div() -> None:
    df = pl.LazyFrame(
        {
            "a": [10.0, 20.0, 30.0, 40.0, 50.0],
            "b": [-100.5, 7.0, 2.5, None, -3.14],
        }
    )
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              a / b AS a_div_b,
              a // b AS a_floordiv_b
            FROM df
            """
        )

    assert_frame_equal(
        pl.DataFrame(
            [
                [-0.0995024875621891, 2.85714285714286, 12.0, None, -15.92356687898089],
                [-1, 2, 12, None, -16],
            ],
            schema=["a_div_b", "a_floordiv_b"],
        ),
        res,
    )


def test_sql_equal_not_equal() -> None:
    # validate null-aware/unaware equality comparisons
    df = pl.DataFrame({"a": [1, None, 3, 6, 5], "b": [1, None, 3, 4, None]})

    with pl.SQLContext(frame_data=df) as ctx:
        out = ctx.execute(
            """
            SELECT
              -- not null-aware
              (a = b)  as "1_eq_unaware",
              (a <> b) as "2_neq_unaware",
              (a != b) as "3_neq_unaware",
              -- null-aware
              (a <=> b) as "4_eq_aware",
              (a IS NOT DISTINCT FROM b) as "5_eq_aware",
              (a IS DISTINCT FROM b) as "6_neq_aware",
            FROM frame_data
            """
        ).collect()

    assert out.select(cs.contains("_aware").null_count().sum()).row(0) == (0, 0, 0)
    assert out.select(cs.contains("_unaware").null_count().sum()).row(0) == (2, 2, 2)

    assert out.to_dict(False) == {
        "1_eq_unaware": [True, None, True, False, None],
        "2_neq_unaware": [False, None, False, True, None],
        "3_neq_unaware": [False, None, False, True, None],
        "4_eq_aware": [True, True, True, False, False],
        "5_eq_aware": [True, True, True, False, False],
        "6_neq_aware": [False, False, False, True, True],
    }


def test_sql_trig() -> None:
    df = pl.DataFrame(
        {
            "a": [-4, -3, -2, -1.00001, 0, 1.00001, 2, 3, 4],
        }
    )

    c = pl.SQLContext(df=df)
    res = c.execute(
        """
        SELECT
        asin(1.0)/a as "pi values",
        cos(asin(1.0)/a) AS "cos",
        cot(asin(1.0)/a) AS "cot",
        sin(asin(1.0)/a) AS "sin",
        tan(asin(1.0)/a) AS "tan",

        cosd(asind(1.0)/a) AS "cosd",
        cotd(asind(1.0)/a) AS "cotd",
        sind(asind(1.0)/a) AS "sind",
        tand(asind(1.0)/a) AS "tand",

        1.0/a as "inverse pi values",
        acos(1.0/a) AS "acos",
        asin(1.0/a) AS "asin",
        atan(1.0/a) AS "atan",

        acosd(1.0/a) AS "acosd",
        asind(1.0/a) AS "asind",
        atand(1.0/a) AS "atand"
        FROM df
        """,
        eager=True,
    )

    df_result = pl.DataFrame(
        {
            "pi values": [
                -0.392699,
                -0.523599,
                -0.785398,
                -1.570781,
                float("inf"),
                1.570781,
                0.785398,
                0.523599,
                0.392699,
            ],
            "cos": [
                0.92388,
                0.866025,
                0.707107,
                0.000016,
                float("NaN"),
                0.000016,
                0.707107,
                0.866025,
                0.92388,
            ],
            "cot": [
                -2.414214,
                -1.732051,
                -1.0,
                -0.000016,
                float("NaN"),
                0.000016,
                1.0,
                1.732051,
                2.414214,
            ],
            "sin": [
                -0.382683,
                -0.5,
                -0.707107,
                -1.0,
                float("NaN"),
                1,
                0.707107,
                0.5,
                0.382683,
            ],
            "tan": [
                -0.414214,
                -0.57735,
                -1,
                -63662.613851,
                float("NaN"),
                63662.613851,
                1,
                0.57735,
                0.414214,
            ],
            "cosd": [
                0.92388,
                0.866025,
                0.707107,
                0.000016,
                float("NaN"),
                0.000016,
                0.707107,
                0.866025,
                0.92388,
            ],
            "cotd": [
                -2.414214,
                -1.732051,
                -1.0,
                -0.000016,
                float("NaN"),
                0.000016,
                1.0,
                1.732051,
                2.414214,
            ],
            "sind": [
                -0.382683,
                -0.5,
                -0.707107,
                -1.0,
                float("NaN"),
                1,
                0.707107,
                0.5,
                0.382683,
            ],
            "tand": [
                -0.414214,
                -0.57735,
                -1,
                -63662.613851,
                float("NaN"),
                63662.613851,
                1,
                0.57735,
                0.414214,
            ],
            "inverse pi values": [
                -0.25,
                -0.333333,
                -0.5,
                -0.99999,
                float("inf"),
                0.99999,
                0.5,
                0.333333,
                0.25,
            ],
            "acos": [
                1.823477,
                1.910633,
                2.094395,
                3.137121,
                float("NaN"),
                0.004472,
                1.047198,
                1.230959,
                1.318116,
            ],
            "asin": [
                -0.25268,
                -0.339837,
                -0.523599,
                -1.566324,
                float("NaN"),
                1.566324,
                0.523599,
                0.339837,
                0.25268,
            ],
            "atan": [
                -0.244979,
                -0.321751,
                -0.463648,
                -0.785393,
                1.570796,
                0.785393,
                0.463648,
                0.321751,
                0.244979,
            ],
            "acosd": [
                104.477512,
                109.471221,
                120.0,
                179.743767,
                float("NaN"),
                0.256233,
                60.0,
                70.528779,
                75.522488,
            ],
            "asind": [
                -14.477512,
                -19.471221,
                -30.0,
                -89.743767,
                float("NaN"),
                89.743767,
                30.0,
                19.471221,
                14.477512,
            ],
            "atand": [
                -14.036243,
                -18.434949,
                -26.565051,
                -44.999714,
                90.0,
                44.999714,
                26.565051,
                18.434949,
                14.036243,
            ],
        }
    )

    assert_frame_equal(left=df_result, right=res, atol=1e-5)


def test_sql_groupby(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext(eager_execution=True)
    c.register("foods", lf)

    out = c.execute(
        """
        SELECT
            category,
            count(category) as n,
            max(calories),
            min(fats_g)
        FROM foods
        GROUP BY category
        HAVING n > 5
        ORDER BY n, category DESC
        """
    )
    assert out.to_dict(False) == {
        "category": ["vegetables", "fruit", "seafood"],
        "n": [7, 7, 8],
        "calories": [45, 130, 200],
        "fats_g": [0.0, 0.0, 1.5],
    }

    lf = pl.LazyFrame(
        {
            "grp": ["a", "b", "c", "c", "b"],
            "att": ["x", "y", "x", "y", "y"],
        }
    )
    assert c.tables() == ["foods"]

    c.register("test", lf)
    assert c.tables() == ["foods", "test"]

    out = c.execute(
        """
        SELECT
            grp,
            COUNT(DISTINCT att) AS n_dist_attr
        FROM test
        GROUP BY grp
        HAVING n_dist_attr > 1
        """
    )
    assert out.to_dict(False) == {"grp": ["c"], "n_dist_attr": [2]}


def test_sql_limit_offset() -> None:
    n_values = 11
    lf = pl.LazyFrame({"a": range(n_values), "b": reversed(range(n_values))})
    c = pl.SQLContext(tbl=lf)

    assert c.execute("SELECT * FROM tbl LIMIT 3 OFFSET 4", eager=True).rows() == [
        (4, 6),
        (5, 5),
        (6, 4),
    ]
    for offset, limit in [(0, 3), (1, n_values), (2, 3), (5, 3), (8, 5), (n_values, 1)]:
        out = c.execute(f"SELECT * FROM tbl LIMIT {limit} OFFSET {offset}", eager=True)
        assert_frame_equal(out, lf.slice(offset, limit).collect())
        assert len(out) == min(limit, n_values - offset)


def test_sql_join_inner(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext()
    c.register_many(foods1=lf, foods2=lf)

    for join_clause in (
        "ON foods1.category = foods2.category",
        "USING (category)",
    ):
        out = c.execute(
            f"""
            SELECT *
            FROM foods1
            INNER JOIN foods2 {join_clause}
            LIMIT 2
            """
        )
        assert out.collect().to_dict(False) == {
            "category": ["vegetables", "vegetables"],
            "calories": [45, 20],
            "fats_g": [0.5, 0.0],
            "sugars_g": [2, 2],
            "calories_right": [45, 45],
            "fats_g_right": [0.5, 0.5],
            "sugars_g_right": [2, 2],
        }


def test_sql_join_left() -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    c = pl.SQLContext(frames)
    out = c.execute(
        """
        SELECT a, b, c, d
        FROM tbl_a
        LEFT JOIN tbl_b USING (a,b)
        LEFT JOIN tbl_c USING (c)
        ORDER BY c DESC
        """
    )
    assert out.collect().rows() == [
        (1, 4, "z", 25.5),
        (2, None, "y", -50.0),
        (3, 6, "x", None),
    ]
    assert c.tables() == ["tbl_a", "tbl_b", "tbl_c"]


def test_sql_is_between(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    c = pl.SQLContext(foods1=lf, eager_execution=True)
    out = c.execute(
        """
        SELECT *
        FROM foods1
        WHERE foods1.calories BETWEEN 22 AND 30
        ORDER BY "calories" DESC, "sugars_g" DESC
    """
    )
    assert out.rows() == [
        ("fruit", 30, 0.0, 5),
        ("vegetables", 30, 0.0, 5),
        ("fruit", 30, 0.0, 3),
        ("vegetables", 25, 0.0, 4),
        ("vegetables", 25, 0.0, 3),
        ("vegetables", 25, 0.0, 2),
        ("vegetables", 22, 0.0, 3),
    ]

    out = c.execute(
        """
        SELECT *
        FROM foods1
        WHERE calories NOT BETWEEN 22 AND 30
        ORDER BY "calories" ASC
        """
    )
    assert not any((22 <= cal <= 30) for cal in out["calories"])


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
    ],
)
def test_sql_regex(
    foods_ipc_path: Path, op: str, pattern: str, expected: str | None
) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    with pl.SQLContext(foods=lf, eager_execution=True) as ctx:
        out = ctx.execute(
            f"""
            SELECT DISTINCT category FROM foods
            WHERE category {op} '{pattern}'
            """
        )
        assert out.rows() == ([(expected,)] if expected else [])


def test_sql_regex_error() -> None:
    df = pl.LazyFrame({"sval": ["ABC", "abc", "000", "A0C", "a0c"]})
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        with pytest.raises(
            pl.ComputeError, match="Invalid pattern for '~' operator: 12345"
        ):
            ctx.execute("SELECT * FROM df WHERE sval ~ 12345")
        with pytest.raises(
            pl.ComputeError,
            match=r"""Invalid pattern for '!~\*' operator: col\("abcde"\)""",
        ):
            ctx.execute("SELECT * FROM df WHERE sval !~* abcde")


@pytest.mark.parametrize(
    ("decimals", "expected"),
    [
        (0, [-8192.0, -4.0, -2.0, 2.0, 4.0, 8193.0]),
        (1, [-8192.5, -4.0, -1.5, 2.5, 3.6, 8192.5]),
        (2, [-8192.5, -3.96, -1.54, 2.46, 3.6, 8192.5]),
        (3, [-8192.499, -3.955, -1.543, 2.457, 3.599, 8192.5]),
        (4, [-8192.499, -3.955, -1.5432, 2.4568, 3.599, 8192.5001]),
    ],
)
def test_sql_round_ndigits(decimals: int, expected: list[float]) -> None:
    df = pl.DataFrame(
        {"n": [-8192.499, -3.9550, -1.54321, 2.45678, 3.59901, 8192.5001]},
    )
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        if decimals == 0:
            out = ctx.execute("SELECT ROUND(n) AS n FROM df")
            assert_series_equal(out["n"], pl.Series("n", values=expected))

        out = ctx.execute(f"""SELECT ROUND("n",{decimals}) AS n FROM df""")
        assert_series_equal(out["n"], pl.Series("n", values=expected))


def test_sql_round_ndigits_errors() -> None:
    df = pl.DataFrame({"n": [99.999]})
    with pl.SQLContext(df=df, eager_execution=True) as ctx, pytest.raises(
        pl.InvalidOperationError, match="Invalid 'decimals' for Round: -1"
    ):
        ctx.execute("SELECT ROUND(n,-1) AS n FROM df")


def test_sql_trim(foods_ipc_path: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)

        out = pl.SQLContext(foods1=pl.scan_ipc(foods_ipc_path)).query(  # type: ignore[attr-defined]
            """
            SELECT DISTINCT TRIM(LEADING 'vmf' FROM category)
            FROM foods1
            ORDER BY category DESC
            """
        )
        assert out.to_dict(False) == {
            "category": ["seafood", "ruit", "egetables", "eat"]
        }


def test_register_context() -> None:
    # use as context manager unregisters tables created within each scope
    # on exit from that scope; arbitrary levels of nesting are supported.
    with pl.SQLContext() as ctx:
        _lf1 = pl.LazyFrame({"a": [1, 2, 3], "b": ["m", "n", "o"]})
        _lf2 = pl.LazyFrame({"a": [2, 3, 4], "c": ["p", "q", "r"]})
        ctx.register_globals()
        assert ctx.tables() == ["_lf1", "_lf2"]

        with ctx:
            _lf3 = pl.LazyFrame({"a": [3, 4, 5], "b": ["s", "t", "u"]})
            _lf4 = pl.LazyFrame({"a": [4, 5, 6], "c": ["v", "w", "x"]})
            ctx.register_globals(n=2)
            assert ctx.tables() == ["_lf1", "_lf2", "_lf3", "_lf4"]

        assert ctx.tables() == ["_lf1", "_lf2"]

    assert ctx.tables() == []


def test_sql_expr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]})
    sql_expr = pl.sql_expr("MIN(a)")
    expected = pl.DataFrame({"a": [1]})
    assert df.select(sql_expr).frame_equal(expected)
