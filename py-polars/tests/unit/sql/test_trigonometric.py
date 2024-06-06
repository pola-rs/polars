from __future__ import annotations

import math

import polars as pl
from polars.testing import assert_frame_equal


def test_arctan2() -> None:
    twoRootTwo = math.sqrt(2) / 2.0
    df = pl.DataFrame(  # noqa: F841
        {
            "y": [twoRootTwo, -twoRootTwo, twoRootTwo, -twoRootTwo],
            "x": [twoRootTwo, twoRootTwo, -twoRootTwo, -twoRootTwo],
        }
    )
    res = pl.sql(
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


def test_trig() -> None:
    df = pl.DataFrame(
        {
            "a": [-4.0, -3.0, -2.0, -1.00001, 0.0, 1.00001, 2.0, 3.0, 4.0],
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
