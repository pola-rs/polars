from __future__ import annotations

from decimal import Decimal as D
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import SQLInterfaceError, SQLSyntaxError
from polars.testing import assert_frame_equal, assert_series_equal

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
                "a2": [1.5, None, 1.0, 1 / 3, 1.0],
                "b3": [0, 1, 2, 0, 1],
                "c4": [3, 0, 1, 2, 3],
                "d55": [0.0, 0.5, 2.0, None, 3.5],
            }
        ),
    )


@pytest.mark.parametrize(
    ("value", "sqltype", "prec_scale", "expected_value", "expected_dtype"),
    [
        (64.5, "numeric", "(3,1)", D("64.5"), pl.Decimal(3, 1)),
        (512.5, "decimal", "(4,1)", D("512.5"), pl.Decimal(4, 1)),
        (512.5, "numeric", "(4,0)", D("512"), pl.Decimal(4, 0)),
        (-1024.75, "decimal", "(10,0)", D("-1024"), pl.Decimal(10, 0)),
        (-1024.75, "numeric", "(10)", D("-1024"), pl.Decimal(10, 0)),
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
