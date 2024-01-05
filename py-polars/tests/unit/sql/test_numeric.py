from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import InvalidOperationError
from polars.testing import assert_frame_equal, assert_series_equal


def test_stddev_variance() -> None:
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
    with pl.SQLContext(df=df, eager_execution=True) as ctx:
        if decimals == 0:
            out = ctx.execute("SELECT ROUND(n) AS n FROM df")
            assert_series_equal(out["n"], pl.Series("n", values=expected))

        out = ctx.execute(f'SELECT ROUND("n",{decimals}) AS n FROM df')
        assert_series_equal(out["n"], pl.Series("n", values=expected))


def test_round_ndigits_errors() -> None:
    df = pl.DataFrame({"n": [99.999]})
    with pl.SQLContext(df=df, eager_execution=True) as ctx, pytest.raises(
        InvalidOperationError, match="Invalid 'decimals' for Round: -1"
    ):
        ctx.execute("SELECT ROUND(n,-1) AS n FROM df")
