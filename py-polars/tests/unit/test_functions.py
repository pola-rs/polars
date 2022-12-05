from __future__ import annotations

import numpy as np
import pytest

import polars as pl


def test_date_datetime() -> None:
    df = pl.DataFrame(
        {
            "year": [2001, 2002, 2003],
            "month": [1, 2, 3],
            "day": [1, 2, 3],
            "hour": [23, 12, 8],
        }
    )
    out = df.select(
        [
            pl.all(),
            pl.datetime("year", "month", "day", "hour").dt.hour().cast(int).alias("h2"),
            pl.date("year", "month", "day").dt.day().cast(int).alias("date"),
        ]
    )
    assert out["date"].series_equal(df["day"].rename("date"))
    assert out["h2"].series_equal(df["hour"].rename("h2"))


def test_diag_concat() -> None:
    a = pl.DataFrame({"a": [1, 2]})
    b = pl.DataFrame({"b": ["a", "b"], "c": [1, 2]})
    c = pl.DataFrame({"a": [5, 7], "c": [1, 2], "d": [1, 2]})

    for out in [
        pl.concat([a, b, c], how="diagonal"),
        pl.concat([a.lazy(), b.lazy(), c.lazy()], how="diagonal").collect(),
    ]:
        expected = pl.DataFrame(
            {
                "a": [1, 2, None, None, 5, 7],
                "b": [None, None, "a", "b", None, None],
                "c": [None, None, 1, 2, 1, 2],
                "d": [None, None, None, None, 1, 2],
            }
        )

        assert out.frame_equal(expected, null_equal=True)


def test_concat_horizontal() -> None:
    a = pl.DataFrame({"a": ["a", "b"], "b": [1, 2]})
    b = pl.DataFrame({"c": [5, 7, 8, 9], "d": [1, 2, 1, 2], "e": [1, 2, 1, 2]})

    out = pl.concat([a, b], how="horizontal")
    expected = pl.DataFrame(
        {
            "a": ["a", "b", None, None],
            "b": [1, 2, None, None],
            "c": [5, 7, 8, 9],
            "d": [1, 2, 1, 2],
            "e": [1, 2, 1, 2],
        }
    )
    assert out.frame_equal(expected)


def test_all_any_horizontally() -> None:
    df = pl.DataFrame(
        [
            [False, False, True],
            [False, False, True],
            [True, False, False],
            [False, None, True],
            [None, None, False],
        ],
        columns=["var1", "var2", "var3"],
    )
    expected = pl.DataFrame(
        {
            "any": [True, True, False, True, None],
            "all": [False, False, False, None, False],
        }
    )
    assert df.select(
        [
            pl.any([pl.col("var2"), pl.col("var3")]),
            pl.all([pl.col("var2"), pl.col("var3")]),
        ]
    ).frame_equal(expected)


def test_cut() -> None:
    a = pl.Series("a", [v / 10 for v in range(-30, 30, 5)])
    out = pl.cut(a, bins=[-1, 1])

    assert out.shape == (12, 3)
    assert out.filter(pl.col("break_point") < 1e9).to_dict(False) == {
        "a": [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0],
        "break_point": [-1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        "category": [
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-inf, -1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
            "(-1.0, 1.0]",
        ],
    }

    # test cut on integers #4939
    inf = float("inf")
    df = pl.DataFrame({"a": list(range(5))})
    ser = df.select("a").to_series()
    assert pl.cut(ser, bins=[-1, 1]).rows() == [
        (0.0, 1.0, "(-1.0, 1.0]"),
        (1.0, 1.0, "(-1.0, 1.0]"),
        (2.0, inf, "(1.0, inf]"),
        (3.0, inf, "(1.0, inf]"),
        (4.0, inf, "(1.0, inf]"),
    ]


def test_null_handling_correlation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, None, 4], "b": [1, 2, 3, 10, 4]})

    out = df.select(
        [
            pl.pearson_corr("a", "b").alias("pearson"),
            pl.spearman_rank_corr("a", "b").alias("spearman"),
        ]
    )
    assert out["pearson"][0] == pytest.approx(1.0)
    assert out["spearman"][0] == pytest.approx(1.0)

    # see #4930
    df1 = pl.DataFrame({"a": [None, 1, 2], "b": [None, 2, 1]})
    df2 = pl.DataFrame({"a": [np.nan, 1, 2], "b": [np.nan, 2, 1]})

    assert np.isclose(df1.select(pl.spearman_rank_corr("a", "b"))[0, 0], -1.0)
    assert (
        str(df2.select(pl.spearman_rank_corr("a", "b", propagate_nans=True))[0, 0])
        == "nan"
    )


def test_align_frames() -> None:
    import numpy as np
    import pandas as pd

    # setup some test frames
    df1 = pd.DataFrame(
        {
            "date": pd.date_range(start="2019-01-02", periods=9),
            "a": np.array([0, 1, 2, np.nan, 4, 5, 6, 7, 8], dtype=np.float64),
            "b": np.arange(9, 18, dtype=np.float64),
        }
    ).set_index("date")

    df2 = pd.DataFrame(
        {
            "date": pd.date_range(start="2019-01-04", periods=7),
            "a": np.arange(9, 16, dtype=np.float64),
            "b": np.arange(10, 17, dtype=np.float64),
        }
    ).set_index("date")

    # calculate dot-product in pandas
    pd_dot = (df1 * df2).sum(axis="columns").to_frame("dot").reset_index()

    # use "align_frames" to calculate dot-product from disjoint rows. pandas uses an
    # index to automatically infer the correct frame-alignment for the calculation;
    # we need to do it explicitly (which also makes it clearer what is happening)
    pf1, pf2 = pl.align_frames(
        pl.from_pandas(df1.reset_index()),
        pl.from_pandas(df2.reset_index()),
        on="date",
    )
    pl_dot = (
        (pf1[["a", "b"]] * pf2[["a", "b"]])
        .fill_null(0)
        .select(pl.sum(pl.col("*")).alias("dot"))
        .insert_at_idx(0, pf1["date"])
    )
    # confirm we match the same operation in pandas
    assert pl_dot.frame_equal(pl.from_pandas(pd_dot))
    pd.testing.assert_frame_equal(pd_dot, pl_dot.to_pandas())

    # (also: confirm alignment function works with lazyframes)
    lf1, lf2 = pl.align_frames(
        pl.from_pandas(df1.reset_index()).lazy(),
        pl.from_pandas(df2.reset_index()).lazy(),
        on="date",
    )
    assert isinstance(lf1, pl.LazyFrame)
    assert lf1.collect().frame_equal(pf1)
    assert lf2.collect().frame_equal(pf2)

    # misc
    assert [] == pl.align_frames(on="date")

    # expected error condition
    with pytest.raises(TypeError):
        pl.align_frames(  # type: ignore[call-overload]
            pl.from_pandas(df1.reset_index()).lazy(),
            pl.from_pandas(df2.reset_index()),
            on="date",
        )

    # reverse
    pf1, pf2 = pl.align_frames(
        pl.DataFrame([[3, 5, 6], [5, 8, 9]], orient="row"),
        pl.DataFrame([[2, 5, 6], [3, 8, 9], [4, 2, 0]], orient="row"),
        on="column_0",
        reverse=True,
    )
    assert pf1.rows() == [(5, 8, 9), (4, None, None), (3, 5, 6), (2, None, None)]
    assert pf2.rows() == [(5, None, None), (4, 2, 0), (3, 8, 9), (2, 5, 6)]


def test_nan_aggregations() -> None:
    df = pl.DataFrame({"a": [1.0, float("nan"), 2.0, 3.0], "b": [1, 1, 1, 1]})

    aggs = [
        pl.col("a").max().alias("max"),
        pl.col("a").min().alias("min"),
        pl.col("a").nan_max().alias("nan_max"),
        pl.col("a").nan_min().alias("nan_min"),
    ]

    assert (
        str(df.select(aggs).to_dict(False))
        == "{'max': [3.0], 'min': [1.0], 'nan_max': [nan], 'nan_min': [nan]}"
    )
    assert (
        str(df.groupby("b").agg(aggs).to_dict(False))
        == "{'b': [1], 'max': [3.0], 'min': [2.0], 'nan_max': [nan], 'nan_min': [nan]}"
    )


def test_coalesce() -> None:
    df = pl.DataFrame(
        {
            "a": [None, None, None, None],
            "b": [1, 2, None, None],
            "c": [1, None, 3, None],
        }
    )
    assert df.select(pl.coalesce(["a", "b", "c", 10])).to_dict(False) == {
        "a": [1.0, 2.0, 3.0, 10.0]
    }
    assert df.select(pl.coalesce(pl.col(["a", "b", "c"]))).to_dict(False) == {
        "a": [1.0, 2.0, 3.0, None]
    }


def test_ones_zeros() -> None:
    ones = pl.ones(5)
    assert ones.dtype == pl.Float64
    assert ones.to_list() == [1.0, 1.0, 1.0, 1.0, 1.0]

    ones = pl.ones(3, dtype=pl.UInt8)
    assert ones.dtype == pl.UInt8
    assert ones.to_list() == [1, 1, 1]

    zeros = pl.zeros(5)
    assert zeros.dtype == pl.Float64
    assert zeros.to_list() == [0.0, 0.0, 0.0, 0.0, 0.0]

    zeros = pl.zeros(3, dtype=pl.UInt8)
    assert zeros.dtype == pl.UInt8
    assert zeros.to_list() == [0, 0, 0]
