from __future__ import annotations

import datetime
import math
from pathlib import Path

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import ComputeError, InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


# TODO: Do not rely on I/O for these tests
@pytest.fixture()
def foods_ipc_path() -> Path:
    return Path(__file__).parent.parent / "io" / "files" / "foods1.ipc"


def test_sql_cast() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": [1.1, 2.2, 3.3, 4.4, 5.5],
            "c": ["a", "b", "c", "d", "e"],
            "d": [True, False, True, False, True],
        }
    )
    # test various dtype casts, using standard ("CAST <col> AS <dtype>")
    # and postgres-specific ("<col>::<dtype>") cast syntax
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        res = ctx.execute(
            """
            SELECT
              -- float
              CAST(a AS DOUBLE PRECISION) AS a_f64,
              a::real AS a_f32,
              -- integer
              CAST(b AS TINYINT) AS b_i8,
              CAST(b AS SMALLINT) AS b_i16,
              b::bigint AS b_i64,
              d::tinyint AS d_i8,
              -- string/binary
              CAST(a AS CHAR) AS a_char,
              CAST(b AS VARCHAR) AS b_varchar,
              c::blob AS c_blob,
              c::VARBINARY AS c_varbinary,
              CAST(d AS CHARACTER VARYING) AS d_charvar,
            FROM df
            """
        )
    assert res.schema == {
        "a_f64": pl.Float64,
        "a_f32": pl.Float32,
        "b_i8": pl.Int8,
        "b_i16": pl.Int16,
        "b_i64": pl.Int64,
        "d_i8": pl.Int8,
        "a_char": pl.String,
        "b_varchar": pl.String,
        "c_blob": pl.Binary,
        "c_varbinary": pl.Binary,
        "d_charvar": pl.String,
    }
    assert res.rows() == [
        (1.0, 1.0, 1, 1, 1, 1, "1", "1.1", b"a", b"a", "true"),
        (2.0, 2.0, 2, 2, 2, 0, "2", "2.2", b"b", b"b", "false"),
        (3.0, 3.0, 3, 3, 3, 1, "3", "3.3", b"c", b"c", "true"),
        (4.0, 4.0, 4, 4, 4, 0, "4", "4.4", b"d", b"d", "false"),
        (5.0, 5.0, 5, 5, 5, 1, "5", "5.5", b"e", b"e", "true"),
    ]

    with pytest.raises(ComputeError, match="unsupported use of FORMAT in CAST"):
        pl.SQLContext(df=df, eager_execution=True).execute(
            "SELECT CAST(a AS STRING FORMAT 'HEX') FROM df"
        )


def test_sql_any_all() -> None:
    df = pl.DataFrame(
        {
            "x": [-1, 0, 1, 2, 3, 4],
            "y": [1, 0, 0, 1, 2, 3],
        }
    )

    sql = pl.SQLContext(df=df)

    res = sql.execute(
        """
        SELECT
        x >= ALL(df.y) as 'All Geq',
        x > ALL(df.y) as 'All G',
        x < ALL(df.y) as 'All L',
        x <= ALL(df.y) as 'All Leq',
        x >= ANY(df.y) as 'Any Geq',
        x > ANY(df.y) as 'Any G',
        x < ANY(df.y) as 'Any L',
        x <= ANY(df.y) as 'Any Leq',
        x == ANY(df.y) as 'Any eq',
        x != ANY(df.y) as 'Any Neq',
        FROM df
        """,
        eager=True,
    )

    assert res.to_dict(as_series=False) == {
        "All Geq": [0, 0, 0, 0, 1, 1],
        "All G": [0, 0, 0, 0, 0, 1],
        "All L": [1, 0, 0, 0, 0, 0],
        "All Leq": [1, 1, 0, 0, 0, 0],
        "Any Geq": [0, 1, 1, 1, 1, 1],
        "Any G": [0, 0, 1, 1, 1, 1],
        "Any L": [1, 1, 1, 1, 0, 0],
        "Any Leq": [1, 1, 1, 1, 1, 0],
        "Any eq": [0, 1, 1, 1, 1, 0],
        "Any Neq": [1, 0, 0, 0, 0, 1],
    }


def test_sql_distinct() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 1, 2, 2, 3],
            "b": [1, 2, 3, 4, 5, 6],
        }
    )
    ctx = pl.SQLContext(register_globals=True, eager_execution=True)
    res1 = ctx.execute("SELECT DISTINCT a FROM df ORDER BY a DESC")
    assert_frame_equal(
        left=df.select("a").unique().sort(by="a", descending=True),
        right=res1,
    )

    res2 = ctx.execute(
        """
        SELECT DISTINCT
          a*2 AS two_a,
          b/2 AS half_b
        FROM df
        ORDER BY two_a ASC, half_b DESC
        """,
    )
    assert res2.to_dict(as_series=False) == {
        "two_a": [2, 2, 4, 6],
        "half_b": [1, 0, 2, 3],
    }

    # test unregistration
    ctx.unregister("df")
    with pytest.raises(ComputeError, match=".*'df'.*not found"):
        ctx.execute("SELECT * FROM df")


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

    assert out.to_dict(as_series=False) == {
        "1_eq_unaware": [True, None, True, False, None],
        "2_neq_unaware": [False, None, False, True, None],
        "3_neq_unaware": [False, None, False, True, None],
        "4_eq_aware": [True, True, True, False, False],
        "5_eq_aware": [True, True, True, False, False],
        "6_neq_aware": [False, False, False, True, True],
    }


def test_sql_arctan2() -> None:
    twoRootTwo = math.sqrt(2) / 2.0
    df = pl.DataFrame(
        {
            "y": [twoRootTwo, -twoRootTwo, twoRootTwo, -twoRootTwo],
            "x": [twoRootTwo, twoRootTwo, -twoRootTwo, -twoRootTwo],
        }
    )

    sql = pl.SQLContext(df=df)
    res = sql.execute(
        """
        SELECT
        ATAN2D(y,x) as "atan2d",
        ATAN2(y,x) as "atan2"
        FROM df
        """,
        eager=True,
    )

    df_result = pl.DataFrame({"atan2d": [45.0, -45.0, 135.0, -135.0]})
    df_result = df_result.with_columns(pl.col("atan2d").cast(pl.Float64))
    df_result = df_result.with_columns(pl.col("atan2d").radians().alias("atan2"))

    assert_frame_equal(df_result, res)


def test_sql_trig() -> None:
    df = pl.DataFrame(
        {
            "a": [-4, -3, -2, -1.00001, 0, 1.00001, 2, 3, 4],
        }
    )

    ctx = pl.SQLContext(df=df)
    res = ctx.execute(
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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
                float("nan"),
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


def test_sql_group_by(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    ctx = pl.SQLContext(eager_execution=True)
    ctx.register("foods", lf)

    out = ctx.execute(
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
    assert out.to_dict(as_series=False) == {
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
    assert ctx.tables() == ["foods"]

    ctx.register("test", lf)
    assert ctx.tables() == ["foods", "test"]

    out = ctx.execute(
        """
        SELECT
            grp,
            COUNT(DISTINCT att) AS n_dist_attr
        FROM test
        GROUP BY grp
        HAVING n_dist_attr > 1
        """
    )
    assert out.to_dict(as_series=False) == {"grp": ["c"], "n_dist_attr": [2]}


def test_sql_left() -> None:
    df = pl.DataFrame({"scol": ["abcde", "abc", "a", None]})
    ctx = pl.SQLContext(df=df)
    res = ctx.execute(
        'SELECT scol, LEFT(scol,2) AS "scol:left2" FROM df',
    ).collect()

    assert res.to_dict(as_series=False) == {
        "scol": ["abcde", "abc", "a", None],
        "scol:left2": ["ab", "ab", "a", None],
    }
    with pytest.raises(
        InvalidOperationError,
        match="Invalid 'length' for Left: 'xyz'",
    ):
        ctx.execute(
            """SELECT scol, LEFT(scol,'xyz') AS "scol:left2" FROM df"""
        ).collect()


def test_sql_limit_offset() -> None:
    n_values = 11
    lf = pl.LazyFrame({"a": range(n_values), "b": reversed(range(n_values))})
    ctx = pl.SQLContext(tbl=lf)

    assert ctx.execute("SELECT * FROM tbl LIMIT 3 OFFSET 4", eager=True).rows() == [
        (4, 6),
        (5, 5),
        (6, 4),
    ]
    for offset, limit in [(0, 3), (1, n_values), (2, 3), (5, 3), (8, 5), (n_values, 1)]:
        out = ctx.execute(
            f"SELECT * FROM tbl LIMIT {limit} OFFSET {offset}", eager=True
        )
        assert_frame_equal(out, lf.slice(offset, limit).collect())
        assert len(out) == min(limit, n_values - offset)


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
def test_sql_join_anti_semi(sql: str, expected: pl.DataFrame) -> None:
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
def test_sql_join_inner(foods_ipc_path: Path, join_clause: str) -> None:
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


def test_sql_join_left() -> None:
    frames = {
        "tbl_a": pl.DataFrame({"a": [1, 2, 3], "b": [4, None, 6]}),
        "tbl_b": pl.DataFrame({"a": [3, 2, 1], "b": [6, 5, 4], "c": ["x", "y", "z"]}),
        "tbl_c": pl.DataFrame({"c": ["w", "y", "z"], "d": [10.5, -50.0, 25.5]}),
    }
    ctx = pl.SQLContext(frames)
    out = ctx.execute(
        """
        SELECT a, b, c, d
        FROM tbl_a
        LEFT JOIN tbl_b USING (a,b)
        LEFT JOIN tbl_c USING (c)
        ORDER BY a DESC
        """
    )
    assert out.collect().rows() == [
        (3, 6, "x", None),
        (2, None, None, None),
        (1, 4, "z", 25.5),
    ]
    assert ctx.tables() == ["tbl_a", "tbl_b", "tbl_c"]


@pytest.mark.parametrize(
    "constraint", ["tbl.a != tbl.b", "tbl.a > tbl.b", "a >= b", "a < b", "b <= a"]
)
def test_sql_non_equi_joins(constraint: str) -> None:
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


def test_sql_stddev_variance() -> None:
    df = pl.DataFrame(
        {
            "v1": [-1.0, 0.0, 1.0],
            "v2": [5.5, 0.0, 3.0],
            "v3": [-10, None, 10],
            "v4": [-100, 0.0, -50.0],
        }
    )
    with pl.SQLContext(df=df) as ctx:
        # note: we support all common aliases for std/var
        out = ctx.execute(
            """
            SELECT
              STDEV(v1) AS "v1_std",
              STDDEV(v2) AS "v2_std",
              STDEV_SAMP(v3) AS "v3_std",
              STDDEV_SAMP(v4) AS "v4_std",
              VAR(v1) AS "v1_var",
              VARIANCE(v2) AS "v2_var",
              VARIANCE(v3) AS "v3_var",
              VAR_SAMP(v4) AS "v4_var"
            FROM df
            """
        ).collect()

        assert_frame_equal(
            out,
            pl.DataFrame(
                {
                    "v1_std": [1.0],
                    "v2_std": [2.7537852736431],
                    "v3_std": [14.142135623731],
                    "v4_std": [50.0],
                    "v1_var": [1.0],
                    "v2_var": [7.5833333333333],
                    "v3_var": [200.0],
                    "v4_var": [2500.0],
                }
            ),
        )


def test_sql_is_between(foods_ipc_path: Path) -> None:
    lf = pl.scan_ipc(foods_ipc_path)

    ctx = pl.SQLContext(foods1=lf, eager_execution=True)
    out = ctx.execute(
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
    out = ctx.execute(
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
def test_sql_regex_operators(
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


def test_sql_regex_operators_error() -> None:
    df = pl.LazyFrame({"sval": ["ABC", "abc", "000", "A0C", "a0c"]})
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        with pytest.raises(
            ComputeError, match="Invalid pattern for '~' operator: 12345"
        ):
            ctx.execute("SELECT * FROM df WHERE sval ~ 12345")
        with pytest.raises(
            ComputeError,
            match=r"""Invalid pattern for '!~\*' operator: col\("abcde"\)""",
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
def test_sql_regexp_like(
    foods_ipc_path: Path,
    not_: str,
    pattern: str,
    flags: str | None,
    expected: str | None,
) -> None:
    lf = pl.scan_ipc(foods_ipc_path)
    flags = "" if flags is None else f",'{flags}'"
    with pl.SQLContext(foods=lf, eager_execution=True) as ctx:
        out = ctx.execute(
            f"""
            SELECT DISTINCT category FROM foods
            WHERE {not_} REGEXP_LIKE(category,'{pattern}'{flags})
            """
        )
        assert out.rows() == ([(expected,)] if expected else [])


def test_sql_regexp_like_errors() -> None:
    with pl.SQLContext(df=pl.DataFrame({"scol": ["xyz"]})) as ctx:
        with pytest.raises(
            InvalidOperationError,
            match="Invalid/empty 'flags' for RegexpLike",
        ):
            ctx.execute("SELECT * FROM df WHERE REGEXP_LIKE(scol,'[x-z]+','')")

        with pytest.raises(
            InvalidOperationError,
            match="Invalid arguments for RegexpLike",
        ):
            ctx.execute("SELECT * FROM df WHERE REGEXP_LIKE(scol,999,999)")

        with pytest.raises(
            InvalidOperationError,
            match="Invalid number of arguments for RegexpLike",
        ):
            ctx.execute("SELECT * FROM df WHERE REGEXP_LIKE(scol)")


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

        out = ctx.execute(f'SELECT ROUND("n",{decimals}) AS n FROM df')
        assert_series_equal(out["n"], pl.Series("n", values=expected))


def test_sql_round_ndigits_errors() -> None:
    df = pl.DataFrame({"n": [99.999]})
    with pl.SQLContext(df=df, eager_execution=True) as ctx, pytest.raises(
        InvalidOperationError, match="Invalid 'decimals' for Round: -1"
    ):
        ctx.execute("SELECT ROUND(n,-1) AS n FROM df")


def test_sql_string_case() -> None:
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


def test_sql_string_lengths() -> None:
    df = pl.DataFrame({"words": ["Café", None, "東京"]})

    with pl.SQLContext(frame=df) as ctx:
        res = ctx.execute(
            """
            SELECT
              words,
              LENGTH(words) AS n_chars,
              OCTET_LENGTH(words) AS n_bytes
            FROM frame
            """
        ).collect()

    assert res.to_dict(as_series=False) == {
        "words": ["Café", None, "東京"],
        "n_chars": [4, None, 2],
        "n_bytes": [5, None, 6],
    }


def test_sql_substr() -> None:
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


def test_sql_trim(foods_ipc_path: Path) -> None:
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


@pytest.mark.parametrize(
    ("cols1", "cols2", "union_subtype", "expected"),
    [
        (
            ["*"],
            ["*"],
            "",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        (
            ["*"],
            ["frame2.*"],
            "ALL",
            [(1, "zz"), (2, "yy"), (2, "yy"), (3, "xx")],
        ),
        (
            ["frame1.*"],
            ["c1", "c2"],
            "DISTINCT",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        (
            ["*"],
            ["c2", "c1"],
            "ALL BY NAME",
            [(1, "zz"), (2, "yy"), (2, "yy"), (3, "xx")],
        ),
        (
            ["c1", "c2"],
            ["c2", "c1"],
            "BY NAME",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
        pytest.param(
            ["c1", "c2"],
            ["c2", "c1"],
            "DISTINCT BY NAME",
            [(1, "zz"), (2, "yy"), (3, "xx")],
        ),
    ],
)
def test_sql_union(
    cols1: list[str],
    cols2: list[str],
    union_subtype: str,
    expected: list[tuple[int, str]],
) -> None:
    with pl.SQLContext(
        frame1=pl.DataFrame({"c1": [1, 2], "c2": ["zz", "yy"]}),
        frame2=pl.DataFrame({"c1": [2, 3], "c2": ["yy", "xx"]}),
        eager_execution=True,
    ) as ctx:
        query = f"""
            SELECT {', '.join(cols1)} FROM frame1
            UNION {union_subtype}
            SELECT {', '.join(cols2)} FROM frame2
        """
        assert sorted(ctx.execute(query).rows()) == expected


def test_sql_nullif_coalesce(foods_ipc_path: Path) -> None:
    nums = pl.LazyFrame(
        {
            "x": [1, None, 2, 3, None, 4],
            "y": [5, 4, None, 3, None, 2],
            "z": [3, 4, None, 3, None, None],
        }
    )

    res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        COALESCE(x,y,z) as "coal",
        NULLIF(x,y) as "nullif x_y",
        NULLIF(y,z) as "nullif y_z",
        COALESCE(x, NULLIF(y,z)) as "both"
        FROM df
        """,
        eager=True,
    )

    assert res.to_dict(as_series=False) == {
        "coal": [1, 4, 2, 3, None, 4],
        "nullif x_y": [1, None, 2, None, None, 4],
        "nullif y_z": [5, None, None, None, None, 2],
        "both": [1, None, 2, 3, None, 4],
    }


def test_sql_order_by(foods_ipc_path: Path) -> None:
    foods = pl.scan_ipc(foods_ipc_path)
    nums = pl.LazyFrame(
        {
            "x": [1, 2, 3],
            "y": [4, 3, 2],
        }
    )

    order_by_distinct_res = pl.SQLContext(foods1=foods).execute(
        """
        SELECT DISTINCT category
        FROM foods1
        ORDER BY category DESC
        """,
        eager=True,
    )
    assert order_by_distinct_res.to_dict(as_series=False) == {
        "category": ["vegetables", "seafood", "meat", "fruit"]
    }

    order_by_group_by_res = pl.SQLContext(foods1=foods).execute(
        """
        SELECT category
        FROM foods1
        GROUP BY category
        ORDER BY category DESC
        """,
        eager=True,
    )
    assert order_by_group_by_res.to_dict(as_series=False) == {
        "category": ["vegetables", "seafood", "meat", "fruit"]
    }

    order_by_constructed_group_by_res = pl.SQLContext(foods1=foods).execute(
        """
        SELECT category, SUM(calories) as summed_calories
        FROM foods1
        GROUP BY category
        ORDER BY summed_calories DESC
        """,
        eager=True,
    )
    assert order_by_constructed_group_by_res.to_dict(as_series=False) == {
        "category": ["seafood", "meat", "fruit", "vegetables"],
        "summed_calories": [1250, 540, 410, 192],
    }

    order_by_unselected_res = pl.SQLContext(foods1=foods).execute(
        """
        SELECT SUM(calories) as summed_calories
        FROM foods1
        GROUP BY category
        ORDER BY summed_calories DESC
        """,
        eager=True,
    )
    assert order_by_unselected_res.to_dict(as_series=False) == {
        "summed_calories": [1250, 540, 410, 192],
    }

    order_by_unselected_nums_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        df.x,
        df.y as y_alias
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_unselected_nums_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
        "y_alias": [2, 3, 4],
    }

    order_by_wildcard_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        *,
        df.y as y_alias
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_wildcard_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
        "y": [2, 3, 4],
        "y_alias": [2, 3, 4],
    }

    order_by_qualified_wildcard_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        df.*
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_qualified_wildcard_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
        "y": [2, 3, 4],
    }

    order_by_exclude_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        * EXCLUDE y
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_exclude_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
    }

    order_by_qualified_exclude_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        df.* EXCLUDE y
        FROM df
        ORDER BY y
        """,
        eager=True,
    )
    assert order_by_qualified_exclude_res.to_dict(as_series=False) == {
        "x": [3, 2, 1],
    }

    order_by_expression_res = pl.SQLContext(df=nums).execute(
        """
        SELECT
        x % y as modded
        FROM df
        ORDER BY x % y
        """,
        eager=True,
    )
    assert order_by_expression_res.to_dict(as_series=False) == {
        "modded": [1, 1, 2],
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
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["xyz", "abcde", None]})
    sql_exprs = pl.sql_expr(
        [
            "MIN(a)",
            "POWER(a,a) AS aa",
            "SUBSTR(b,2,2) AS b2",
        ]
    )
    result = df.select(*sql_exprs)
    expected = pl.DataFrame(
        {"a": [1, 1, 1], "aa": [1.0, 4.0, 27.0], "b2": ["yz", "bc", None]}
    )
    assert_frame_equal(result, expected)

    # expect expressions that can't reasonably be parsed as expressions to raise
    # (for example: those that explicitly reference tables and/or use wildcards)
    with pytest.raises(
        InvalidOperationError, match=r"Unable to parse 'xyz\.\*' as Expr"
    ):
        pl.sql_expr("xyz.*")


@pytest.mark.parametrize("match_float", [False, True])
def test_sql_unary_ops_8890(match_float: bool) -> None:
    with pl.SQLContext(
        df=pl.DataFrame({"a": [-2, -1, 1, 2], "b": ["w", "x", "y", "z"]}),
    ) as ctx:
        in_values = "(-3.0, -1.0, +2.0, +4.0)" if match_float else "(-3, -1, +2, +4)"
        res = ctx.execute(
            f"""
            SELECT *, -(3) as c, (+4) as d
            FROM df WHERE a IN {in_values}
            """
        )
        assert res.collect().to_dict(as_series=False) == {
            "a": [-1, 2],
            "b": ["x", "z"],
            "c": [-3, -3],
            "d": [4, 4],
        }


def test_sql_in_no_ops_11946() -> None:
    df = pl.LazyFrame(
        [
            {"i1": 1},
            {"i1": 2},
            {"i1": 3},
        ]
    )

    ctx = pl.SQLContext(frame_data=df, eager_execution=False)

    out = ctx.execute(
        "SELECT * FROM frame_data WHERE i1 in (1, 3)", eager=False
    ).collect()
    assert out.to_dict(as_series=False) == {"i1": [1, 3]}


def test_sql_date() -> None:
    df = pl.DataFrame(
        {
            "date": [
                datetime.date(2021, 3, 15),
                datetime.date(2021, 3, 28),
                datetime.date(2021, 4, 4),
            ],
            "version": ["0.0.1", "0.7.3", "0.7.4"],
        }
    )

    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        result = ctx.execute("SELECT date < DATE('2021-03-20') from df")

    expected = pl.DataFrame({"date": [True, False, False]})
    assert_frame_equal(result, expected)

    result = pl.select(pl.sql_expr("""CAST(DATE('2023-03', '%Y-%m') as STRING)"""))
    expected = pl.DataFrame({"literal": ["2023-03-01"]})
    assert_frame_equal(result, expected)
