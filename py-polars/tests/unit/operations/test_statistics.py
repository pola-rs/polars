from datetime import timedelta
from typing import cast

import numpy as np
from numpy import inf

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


def test_corr() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 2, 4],
            "b": [-1, 23, 8],
        }
    )
    result = df.corr()
    expected = pl.DataFrame(
        {
            "a": [1.0, 0.18898223650461357],
            "b": [0.1889822365046136, 1.0],
        }
    )
    assert_frame_equal(result, expected)


def test_cut() -> None:
    a = pl.Series("a", [v / 10 for v in range(-30, 30, 5)])
    out = cast(pl.DataFrame, a.cut(bins=[-1, 1], series=False))

    assert out.shape == (12, 3)
    assert out.filter(pl.col("break_point") < 1e9).to_dict(False) == {
        "a": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        "break_point": [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        "category": [
            "(-inf, -1]",
            "(-inf, -1]",
            "(-inf, -1]",
            "(-inf, -1]",
            "(-inf, -1]",
            "(-1, 1]",
            "(-1, 1]",
            "(-1, 1]",
            "(-1, 1]",
        ],
    }

    # test cut on integers #4939
    inf = float("inf")
    df = pl.DataFrame({"a": list(range(5))})
    ser = df.select("a").to_series()
    assert cast(pl.DataFrame, ser.cut(bins=[-1, 1], series=False)).rows() == [
        (0.0, 1.0, "(-1, 1]"),
        (1.0, 1.0, "(-1, 1]"),
        (2.0, inf, "(1, inf]"),
        (3.0, inf, "(1, inf]"),
        (4.0, inf, "(1, inf]"),
    ]

    expected_df = pl.DataFrame(
        {
            "a": [5.0, 8.0, 9.0, 5.0, 0.0, 0.0, 1.0, 7.0, 6.0, 9.0],
            "break_point": [inf, inf, inf, inf, 1.0, 1.0, 1.0, inf, inf, inf],
            "category": [
                "(1, inf]",
                "(1, inf]",
                "(1, inf]",
                "(1, inf]",
                "(-1, 1]",
                "(-1, 1]",
                "(-1, 1]",
                "(1, inf]",
                "(1, inf]",
                "(1, inf]",
            ],
        }
    )
    np.random.seed(1)
    a = pl.Series("a", np.random.randint(0, 10, 10))
    out = cast(pl.DataFrame, a.cut(bins=[-1, 1], series=False))
    out_s = cast(pl.Series, a.cut(bins=[-1, 1], series=True))
    assert out["a"].cast(int).series_equal(a)
    # Compare strings and categoricals without a hassle
    assert_frame_equal(expected_df, out, check_dtype=False)
    # It formats differently
    assert_series_equal(
        pl.Series(["(1, inf]"] * 4 + ["(-1, 1]"] * 3 + ["(1, inf]"] * 3),
        out_s,
        check_dtype=False,
        check_names=False,
    )


def test_qcut() -> None:
    input = pl.Series("a", range(-5, 3))
    exp = pl.DataFrame(
        {
            "a": [-5.0, -4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0],
            "break_point": [-5.0, -3.25, 0.25, 0.25, 0.25, 0.25, inf, inf],
            "category": ["(-inf, -5]", "(-5, -3.25]"]
            + ["(-3.25, 0.25]"] * 4
            + ["(0.25, inf]"] * 2,
        }
    )
    out = cast(pl.DataFrame, input.qcut([0.0, 0.25, 0.75]))
    out_s = cast(pl.Series, input.qcut([0.0, 0.25, 0.75], series=True))
    assert_frame_equal(out, exp, check_dtype=False)
    assert_series_equal(
        exp["category"],
        out_s,
        check_dtype=False,
        check_names=False,
    )


def test_hist() -> None:
    a = pl.Series("a", [1, 3, 8, 8, 2, 1, 3])
    assert (
        str(a.hist(bin_count=4).to_dict(False))
        == "{'break_point': [0.0, 2.25, 4.5, 6.75, inf], 'category': ['(-inf, 0.0]', '(0.0, 2.25]', '(2.25, 4.5]', '(4.5, 6.75]', '(6.75, inf]'], 'a_count': [0, 3, 2, 0, 2]}"
    )


def test_cut_null_values() -> None:
    s = pl.Series([-1.0, None, 1.0, 2.0, None, 8.0, 4.0])
    exp = pl.DataFrame(
        {
            "": [-1.0, None, 1.0, 2.0, None, 8.0, 4.0],
            "break_point": [
                0.5999999999999996,
                None,
                1.2000000000000002,
                inf,
                None,
                inf,
                inf,
            ],
            "category": [
                "(-inf, 0.5999999999999996]",
                None,
                "(0.5999999999999996, 1.2000000000000002]",
                "(1.2000000000000002, inf]",
                None,
                "(1.2000000000000002, inf]",
                "(1.2000000000000002, inf]",
            ],
        }
    )
    assert_frame_equal(
        cast(pl.DataFrame, s.qcut([0.2, 0.3], series=False)),
        exp,
        check_dtype=False,
    )
    assert_series_equal(
        cast(pl.Series, s.qcut([0.2, 0.3], series=True)),
        exp.get_column("category"),
        check_dtype=False,
        check_names=False,
    )


def test_median_quantile_duration() -> None:
    df = pl.DataFrame({"A": [timedelta(days=0), timedelta(days=1)]})
    assert df.select(pl.col("A").median()).to_dict(False) == {
        "A": [timedelta(seconds=43200)]
    }
    assert df.select(pl.col("A").quantile(0.5, interpolation="linear")).to_dict(
        False
    ) == {"A": [timedelta(seconds=43200)]}


def test_correlation_cast_supertype() -> None:
    df = pl.DataFrame({"a": [1, 8, 3], "b": [4.0, 5.0, 2.0]})
    df = df.with_columns(pl.col("b"))
    assert df.select(pl.corr("a", "b")).to_dict(False) == {"a": [0.5447047794019223]}


def test_cut_over() -> None:
    with pl.StringCache():
        x = pl.Series(range(20))
    r = pl.Series(
        [pl.repeat("a", 10, eager=True), pl.repeat("b", 10, eager=True)]
    ).explode()
    df = pl.DataFrame({"x": x, "g": r})

    out = df.with_columns(pl.col("x").qcut([0.5]).over("g")).to_dict(False)
    assert out == {
        "x": [
            "(-inf, 4.5]",
            "(-inf, 4.5]",
            "(-inf, 4.5]",
            "(-inf, 4.5]",
            "(-inf, 4.5]",
            "(4.5, inf]",
            "(4.5, inf]",
            "(4.5, inf]",
            "(4.5, inf]",
            "(4.5, inf]",
            "(-inf, 14.5]",
            "(-inf, 14.5]",
            "(-inf, 14.5]",
            "(-inf, 14.5]",
            "(-inf, 14.5]",
            "(14.5, inf]",
            "(14.5, inf]",
            "(14.5, inf]",
            "(14.5, inf]",
            "(14.5, inf]",
        ],
        "g": [
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "a",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
            "b",
        ],
    }
