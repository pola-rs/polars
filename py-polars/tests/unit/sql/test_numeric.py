from __future__ import annotations

import re
from decimal import Decimal as D
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal, assert_series_equal
from tests.unit.sql import assert_sql_matches

if TYPE_CHECKING:
    from polars._typing import PolarsDataType


def test_div() -> None:
    res = pl.sql(
        """
        SELECT label, DIV(a, b) AS a_div_b, DIV(tbl.b, tbl.a) AS b_div_a
        FROM (
          VALUES
            ('a', 20.5, 6),
            ('b', NULL, 12),
            ('c', 10.0, 24),
            ('d', 5.0, NULL),
            ('e', 2.5, 5)
        ) AS tbl(label, a, b)
        """
    ).collect()

    assert res.to_dict(as_series=False) == {
        "label": ["a", "b", "c", "d", "e"],
        "a_div_b": [3, None, 0, None, 0],
        "b_div_a": [0, None, 2, None, 2],
    }


def test_erf_erfc() -> None:
    df = pl.DataFrame({"a": [-1.0, 0.0, 0.5, None]})
    res = df.sql("SELECT ERF(a) AS erf_a, ERFC(a) AS erfc_a FROM self")
    expected = df.select(
        erf_a=pl.col("a").erf(),
        erfc_a=pl.col("a").erfc(),
    )
    assert_frame_equal(res, expected)


def test_modulo() -> None:
    df = pl.DataFrame(
        {
            "a": [1.5, None, 3.0, 13 / 3, 5.0],
            "b": [6, 7, 8, 9, 10],
            "c": [11, 12, 13, 14, 15],
            "d": [16.5, 17.0, 18.5, None, 20.0],
        }
    )
    out = df.sql(
        """
        SELECT
          ROW_NUMBER() AS idx,
          a % 2 AS a2,
          b % 3 AS b3,
          MOD(c, 4) AS c4,
          MOD(d, 5.5) AS d55
        FROM self
        """
    )
    assert_frame_equal(
        out,
        pl.DataFrame(
            {
                "idx": [1, 2, 3, 4, 5],
                "a2": [1.5, None, 1.0, 1 / 3, 1.0],
                "b3": [0, 1, 2, 0, 1],
                "c4": [3, 0, 1, 2, 3],
                "d55": [0.0, 0.5, 2.0, None, 3.5],
            },
            schema_overrides={"idx": pl.UInt32},
        ),
    )


@pytest.mark.parametrize(
    ("value", "sqltype", "prec_scale", "expected_value", "expected_dtype"),
    [
        (64.5, "numeric", "(3,1)", D("64.5"), pl.Decimal(3, 1)),
        (512.5, "decimal", "(4,1)", D("512.5"), pl.Decimal(4, 1)),
        (512.5, "numeric", "(4,0)", D("512"), pl.Decimal(4, 0)),
        (-1024.75, "decimal", "(10,0)", D("-1025"), pl.Decimal(10, 0)),
        (-1024.75, "numeric", "(10)", D("-1025"), pl.Decimal(10, 0)),
        (-1024.75, "dec", "", D("-1024.75"), pl.Decimal(38, 9)),
    ],
)
def test_numeric_decimal_type(
    value: float,
    sqltype: str,
    prec_scale: str,
    expected_value: D,
    expected_dtype: PolarsDataType,
) -> None:
    df = pl.DataFrame({"n": [value]})
    with pl.SQLContext(df=df) as ctx:
        result = ctx.execute(
            f"""
            SELECT n::{sqltype}{prec_scale} AS "dec" FROM df
            """
        )
    expected = pl.LazyFrame(
        data={"dec": [expected_value]},
        schema={"dec": expected_dtype},
    )
    assert_frame_equal(result, expected)


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
def test_round_ndigits(decimals: int, expected: list[float]) -> None:
    df = pl.DataFrame(
        {"n": [-8192.499, -3.9550, -1.54321, 2.45678, 3.59901, 8192.5001]},
    )
    with pl.SQLContext(df=df, eager=True) as ctx:
        if decimals == 0:
            out = ctx.execute("SELECT ROUND(n) AS n FROM df")
            assert_series_equal(out["n"], pl.Series("n", values=expected))

        out = ctx.execute(f'SELECT ROUND("n",{decimals}) AS n FROM df')
        assert_series_equal(out["n"], pl.Series("n", values=expected))


@pytest.mark.parametrize(
    ("decimals", "expected"),
    [
        (0, [-8192.0, -3.0, -1.0, 2.0, 3.0, 8192.0]),
        (1, [-8192.4, -3.9, -1.5, 2.4, 3.5, 8192.5]),
        (2, [-8192.49, -3.95, -1.54, 2.45, 3.59, 8192.50]),
        (3, [-8192.499, -3.955, -1.543, 2.456, 3.599, 8192.5001]),
    ],
)
def test_truncate_ndigits(decimals: int, expected: list[float]) -> None:
    df = pl.DataFrame(
        {"n": [-8192.499, -3.9550, -1.54321, 2.45678, 3.59901, 8192.5001]},
    )
    with pl.SQLContext(df=df, eager=True) as ctx:
        if decimals == 0:
            out = ctx.execute("SELECT TRUNCATE(n) AS n FROM df")
            assert_series_equal(out["n"], pl.Series("n", values=expected))

        out = ctx.execute(f'SELECT TRUNCATE("n",{decimals}) AS n FROM df')
        assert_series_equal(out["n"], pl.Series("n", values=expected))

        out = ctx.execute(f'SELECT TRUNC("n",{decimals}) AS n FROM df')
        assert_series_equal(out["n"], pl.Series("n", values=expected))


def test_round_ndigits_errors() -> None:
    df = pl.DataFrame({"n": [99.999]})
    with pl.SQLContext(df=df, eager=True) as ctx:
        with pytest.raises(
            SQLSyntaxError, match=r"invalid value for ROUND decimals \('!!'\)"
        ):
            ctx.execute("SELECT ROUND(n,'!!') AS n FROM df")

        with pytest.raises(
            SQLInterfaceError, match=r"ROUND .* negative decimals value \(-1\)"
        ):
            ctx.execute("SELECT ROUND(n,-1) AS n FROM df")

        with pytest.raises(
            SQLSyntaxError, match=r"ROUND expects 1-2 arguments \(found 4\)"
        ):
            ctx.execute("SELECT ROUND(1.2345,6,7,8) AS n FROM df")


def test_stddev_variance() -> None:
    df = pl.DataFrame(
        {
            "v1": [-1.0, 0.0, 1.0],
            "v2": [5.5, 0.0, 3.0],
            "v3": [-10, None, 10],
            "v4": [-100.0, 0.0, -50.0],
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


def test_decimal_literal_arithmetic_is_exact() -> None:
    df = pl.DataFrame(
        {
            "disc": [D("0.04"), D("0.05"), D("0.06"), D("0.07"), D("0.08")],
            "qty": [1, 2, 3, 4, 5],
        },
        schema={"disc": pl.Decimal(15, 2), "qty": pl.Int64},
    )
    assert_sql_matches(
        df,
        query="""
            SELECT disc, qty
            FROM self
            WHERE disc BETWEEN .06 - 0.01 AND .06 + 0.01
              AND disc <= (0.03 + .01) * 2 - -0.01
            ORDER BY disc
        """,
        expected={"disc": [D("0.05"), D("0.06"), D("0.07")], "qty": [2, 3, 4]},
        compare_with="duckdb",
    )
    res = df.sql("SELECT 0.1 + 0.2 AS x, 1 + 2 AS y, 2 * 1.5 AS z FROM self LIMIT 1")
    assert res.row(0) == (0.3, 3, 3.0)
    assert res.schema == {"x": pl.Float64, "y": pl.Int32, "z": pl.Float64}


@pytest.mark.parametrize(
    "expr",
    [
        "0.1 + 0.2",
        "-.5 + .5",
        "-(0.5) * 2",
        "+1.5 + 1",
        "-(-1.5)",
        "1.5 - 3",
        "3 * 0.1",
        "1.10 * 1.10",
        "(1.5 + 0.5) * (2 - 0.5)",
        "1.5 + 2 * 0.25",
        "0.00000000000000000000000000000000000001 * 0.00000000000000000000000000000000000001",
        "12345678901234567890.123456789 + 0.000000001",
        "99999999999999999999999999999999999999.9 + 0.1",
    ],
)
def test_literal_arithmetic_folds_exactly(expr: str) -> None:
    # literal-only `+`/`-`/`*` is computed exactly, then converted to Float64 once;
    # the reference evaluates the same expression with Python's exact Decimal
    decimal_expr = re.sub(r"\d*\.\d+|\d+", lambda m: f"D('{m.group()}')", expr)
    expected = float(eval(decimal_expr))
    res = pl.sql(f"SELECT {expr} AS x", eager=True)
    assert res.schema == {"x": pl.Float64}
    assert res.item() == expected


@pytest.mark.parametrize(
    ("expr", "expected", "dtype"),
    [
        # integer-only arithmetic stays on the ordinary path
        ("1 + 2", 3, pl.Int32),
        # eligible children fold even when the parent cannot
        ("0.1 + 0.2 + a", 1.3, pl.Float64),
        ("0.1 + 0.2 = 0.3", True, pl.Boolean),
        # division and casts are left to the engine
        ("0.5 / 0.25", 2.0, pl.Float64),
        ("1.5 + 0.5 / 2", 1.75, pl.Float64),
        ("1.5 + CAST(1 AS FLOAT)", 2.5, pl.Float64),
        ("0.0 - 0.0", 0.0, pl.Float64),
        # beyond the exact domain: falls back to float arithmetic
        (
            "170141183460469231731687303715884105727.0 + 1.0",
            1.7014118346046923e38,
            pl.Float64,
        ),
    ],
)
def test_literal_arithmetic_fallback(
    expr: str, expected: object, dtype: pl.DataType
) -> None:
    res = pl.DataFrame({"a": [1]}).sql(f"SELECT {expr} AS x FROM self")
    assert res.schema == {"x": dtype}
    assert res.item() == expected


def test_literal_scientific_notation_arithmetic() -> None:
    result = pl.sql("SELECT 1e2 + 0.5 AS x", eager=True)
    assert_frame_equal(result, pl.DataFrame({"x": [100.5]}))


def test_int_div_true_division() -> None:
    df = pl.DataFrame({"num": [1], "denum": [3]})
    with pl.SQLContext(df=df, eager=True) as ctx:
        result = ctx.execute("SELECT num / denum AS div FROM df")
        assert_frame_equal(
            result,
            pl.DataFrame({"div": [1 / 3]}),
        )


def test_decimal_mul_div_result_scale() -> None:
    df = pl.DataFrame(
        {
            "a": [D("0.05")],
            "b": [D("0.05")],
            "c": [D("0.01")],
            "big": [D(10**35)],
            "i": [2],
        },
        schema={
            "a": pl.Decimal(15, 2),
            "b": pl.Decimal(15, 2),
            "c": pl.Decimal(38, 2),
            "big": pl.Decimal(38, 2),
            "i": pl.Int64,
        },
    )
    res = df.sql(
        """
        SELECT
          a * b AS mul, c * big AS mul_small_big, big * c AS mul_big_small,
          a * i AS mul_int, a * 3 AS mul_lit,
          big / big AS div_big, a / b AS div, a / i AS div_int, i / a AS int_div
        FROM self
        """
    )
    assert res.schema == pl.Schema(
        {
            "mul": pl.Decimal(38, 4),
            "mul_small_big": pl.Decimal(38, 4),
            "mul_big_small": pl.Decimal(38, 4),
            "mul_int": pl.Decimal(38, 2),
            "mul_lit": pl.Decimal(38, 2),
            "div_big": pl.Decimal(38, 8),
            "div": pl.Decimal(38, 8),
            "div_int": pl.Decimal(38, 8),
            "int_div": pl.Decimal(38, 6),
        }
    )
    assert res.row(0) == (
        D("0.0025"),
        D(10**33),
        D(10**33),
        D("0.10"),
        D("0.15"),
        D("1.00000000"),
        D("1.00000000"),
        D("0.02500000"),
        D("40.000000"),
    )


@pytest.mark.parametrize(
    ("expr", "expected", "dtype"),
    [
        # scale max(s1, min(s1 + 6, 12)); Postgres gives 2.5000000000000000, since
        # its division scale is value-dependent
        ("CAST(5.0 AS DECIMAL(2,1)) / 2", D("2.5000000"), pl.Decimal(38, 7)),
        ("CAST(2.0 AS DECIMAL(2,1)) / 3", D("0.6666667"), pl.Decimal(38, 7)),
        ("CAST(-2.0 AS DECIMAL(2,1)) / 3", D("-0.6666667"), pl.Decimal(38, 7)),
        ("CAST(2 AS DECIMAL(5,0)) / 3", D("0.666667"), pl.Decimal(38, 6)),
        # capped at 12
        ("CAST(2 AS DECIMAL(10,7)) / 3", D("0.666666666667"), pl.Decimal(38, 12)),
        # a dividend scale above 12 is kept
        (
            "CAST(2 AS DECIMAL(20,14)) / 3",
            D("0.66666666666667"),
            pl.Decimal(38, 14),
        ),
        # half-even: half-up would give ±0.000000000003
        (
            "CAST(0.000000000005 AS DECIMAL(38,12)) / 2",
            D("0.000000000002"),
            pl.Decimal(38, 12),
        ),
        (
            "CAST(-0.000000000005 AS DECIMAL(38,12)) / 2",
            D("-0.000000000002"),
            pl.Decimal(38, 12),
        ),
        (
            "CAST(0.000000000015 AS DECIMAL(38,12)) / 2",
            D("0.000000000008"),
            pl.Decimal(38, 12),
        ),
    ],
)
def test_decimal_div_scale_and_rounding(
    expr: str, expected: D, dtype: pl.DataType
) -> None:
    res = pl.sql(f"SELECT {expr} AS x", eager=True)
    assert res.schema == pl.Schema({"x": dtype})
    assert res.item() == expected


def test_decimal_mul_div_out_of_range() -> None:
    df = pl.DataFrame(
        {"a": [D("1")], "big": [D(10**33)]},
        schema={"a": pl.Decimal(38, 20), "big": pl.Decimal(38, 0)},
    )
    match = "multiplication result scale 40 exceeds 38"
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        df.sql("SELECT a * a FROM self")
    with pytest.raises(pl.exceptions.InvalidOperationError, match=match):
        df.select(pl.sql_expr("a * a"))
    # 10^33 at scale 6 loses leading digits
    with pytest.raises(pl.exceptions.ComputeError, match="overflow in decimal"):
        df.sql("SELECT big / 1 FROM self")


def test_decimal_mul_div_float_and_int_unchanged() -> None:
    df = pl.DataFrame(
        {"a": [D("1.50")], "f": [2.0], "i": [3], "j": [2]},
        schema={"a": pl.Decimal(10, 2), "f": pl.Float64, "i": pl.Int64, "j": pl.Int64},
    )
    res = df.sql(
        "SELECT a * f AS af, a / f AS adf, i / j AS ij, i * j AS imj FROM self"
    )
    assert res.schema == pl.Schema(
        {"af": pl.Float64, "adf": pl.Float64, "ij": pl.Float64, "imj": pl.Int64}
    )
    assert res.row(0) == (3.0, 0.75, 1.5, 6)


def test_decimal_div_scale_ignores_dividend_precision() -> None:
    # `a * 1` and `a + 0` widen the dividend to precision 38 but keep its scale,
    # so the division scale, and the result, must not change
    df = pl.DataFrame(
        {"a": [D("1.00"), D("0.05")], "b": [D("3.01"), D("0.03")]},
        schema={"a": pl.Decimal(7, 2), "b": pl.Decimal(7, 2)},
    )
    dividends = df.sql("SELECT a, a * 1 AS a_mul, a + 0 AS a_add FROM self")
    assert dividends.schema == pl.Schema(
        {"a": pl.Decimal(7, 2), "a_mul": pl.Decimal(38, 2), "a_add": pl.Decimal(38, 2)}
    )
    res = df.sql("SELECT a / b AS x, (a * 1) / b AS y, (a + 0) / b AS z FROM self")
    assert res.schema == pl.Schema(dict.fromkeys("xyz", pl.Decimal(38, 8)))
    assert res.rows() == [(D("0.33222591"),) * 3, (D("1.66666667"),) * 3]


def test_decimal_ratio_comparison_uses_division_scale() -> None:
    # Both ratios are 0.33 at scale 2, but 0.33333333 < 0.33557047 at scale 8.
    df = pl.DataFrame(
        {
            "g": [1, 1, 2],
            "x": [D("0.50"), D("0.50"), D("1.00")],
            "y": [D("1.50"), D("1.50"), D("2.00")],
            "z": [D("0.50"), D("0.50"), D("1.00")],
            "w": [D("1.49"), D("1.49"), D("2.00")],
        },
        schema={
            "g": pl.Int64,
            "x": pl.Decimal(7, 2),
            "y": pl.Decimal(7, 2),
            "z": pl.Decimal(7, 2),
            "w": pl.Decimal(7, 2),
        },
    )
    res = df.sql(
        """
        SELECT g, SUM(x) / SUM(y) AS xy, SUM(z) / SUM(w) AS zw,
               SUM(z) / SUM(w) > SUM(x) / SUM(y) AS gt
        FROM self GROUP BY g ORDER BY g
        """
    )
    assert res.schema == pl.Schema(
        {
            "g": pl.Int64(),
            "xy": pl.Decimal(38, 8),
            "zw": pl.Decimal(38, 8),
            "gt": pl.Boolean(),
        }
    )
    assert res.to_dict(as_series=False) == {
        "g": [1, 2],
        "xy": [D("0.33333333"), D("0.50000000")],
        "zw": [D("0.33557047"), D("0.50000000")],
        "gt": [True, False],
    }

    res = df.sql(
        "SELECT g FROM self GROUP BY g HAVING SUM(z) / SUM(w) > SUM(x) / SUM(y)"
    )
    assert res["g"].to_list() == [1]

    res = df.sql(
        """
        WITH s AS (
          SELECT g, SUM(x) AS sx, SUM(y) AS sy, SUM(z) AS sz, SUM(w) AS sw
          FROM self GROUP BY g
        )
        SELECT g FROM s WHERE sz / sw > sx / sy
        """
    )
    assert res["g"].to_list() == [1]


@pytest.mark.parametrize("expr", ["a * b", "a / b", "a / i", "i / a", "a * f", "i / i"])
def test_decimal_mul_div_same_for_every_sql_entry_point(expr: str) -> None:
    # the result scale depends on the operand types, which `sql_expr` only learns when
    # the expression is planned against a frame
    df = pl.DataFrame(
        {"a": [D("0.05")], "b": [D("0.07")], "i": [3], "f": [2.0]},
        schema={
            "a": pl.Decimal(7, 2),
            "b": pl.Decimal(7, 2),
            "i": pl.Int64,
            "f": pl.Float64,
        },
    )
    expected = df.sql(f"SELECT {expr} AS x FROM self")
    assert_frame_equal(df.select(pl.sql_expr(f"{expr} AS x")), expected)
    assert_frame_equal(
        df.lazy().select(pl.sql_expr(expr).alias("x")).collect(), expected
    )
    assert (
        pl.SQLContext(t=df)
        .execute(f"SELECT {expr} AS x FROM t", eager=True)
        .equals(expected)
    )
