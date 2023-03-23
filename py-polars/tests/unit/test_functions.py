from __future__ import annotations

import typing

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal


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
    assert_series_equal(out["date"], df["day"].rename("date"))
    assert_series_equal(out["h2"], df["hour"].rename("h2"))


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

        assert_frame_equal(out, expected)


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
    assert_frame_equal(out, expected)


def test_all_any_horizontally() -> None:
    df = pl.DataFrame(
        [
            [False, False, True],
            [False, False, True],
            [True, False, False],
            [False, None, True],
            [None, None, False],
        ],
        schema=["var1", "var2", "var3"],
    )
    expected = pl.DataFrame(
        {
            "any": [True, True, False, True, None],
            "all": [False, False, False, None, False],
        }
    )
    result = df.select(
        [
            pl.any([pl.col("var2"), pl.col("var3")]),
            pl.all([pl.col("var2"), pl.col("var3")]),
        ]
    )
    assert_frame_equal(result, expected)


def test_cut_deprecated() -> None:
    with pytest.deprecated_call():
        a = pl.Series("a", [v / 10 for v in range(-30, 30, 5)])
        pl.cut(a, bins=[-1, 1])


def test_null_handling_correlation() -> None:
    df = pl.DataFrame({"a": [1, 2, 3, None, 4], "b": [1, 2, 3, 10, 4]})

    out = df.select(
        [
            pl.corr("a", "b").alias("pearson"),
            pl.corr("a", "b", method="spearman").alias("spearman"),
        ]
    )
    assert out["pearson"][0] == pytest.approx(1.0)
    assert out["spearman"][0] == pytest.approx(1.0)

    # see #4930
    df1 = pl.DataFrame({"a": [None, 1, 2], "b": [None, 2, 1]})
    df2 = pl.DataFrame({"a": [np.nan, 1, 2], "b": [np.nan, 2, 1]})

    assert np.isclose(df1.select(pl.corr("a", "b", method="spearman")).item(), -1.0)
    assert (
        str(
            df2.select(pl.corr("a", "b", method="spearman", propagate_nans=True)).item()
        )
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
    assert_frame_equal(pl_dot, pl.from_pandas(pd_dot))
    pd.testing.assert_frame_equal(pd_dot, pl_dot.to_pandas())

    # (also: confirm alignment function works with lazyframes)
    lf1, lf2 = pl.align_frames(
        pl.from_pandas(df1.reset_index()).lazy(),
        pl.from_pandas(df2.reset_index()).lazy(),
        on="date",
    )
    assert isinstance(lf1, pl.LazyFrame)
    assert_frame_equal(lf1.collect(), pf1)
    assert_frame_equal(lf2.collect(), pf2)

    # misc
    assert [] == pl.align_frames(on="date")

    # expected error condition
    with pytest.raises(TypeError):
        pl.align_frames(  # type: ignore[call-overload]
            pl.from_pandas(df1.reset_index()).lazy(),
            pl.from_pandas(df2.reset_index()),
            on="date",
        )

    # descending
    pf1, pf2 = pl.align_frames(
        pl.DataFrame([[3, 5, 6], [5, 8, 9]], orient="row"),
        pl.DataFrame([[2, 5, 6], [3, 8, 9], [4, 2, 0]], orient="row"),
        on="column_0",
        descending=True,
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
            "a": [1, None, None, None],
            "b": [1, 2, None, None],
            "c": [5, None, 3, None],
        }
    )

    # List inputs
    expected = pl.Series("d", [1, 2, 3, 10]).to_frame()
    result = df.select(pl.coalesce(["a", "b", "c", 10]).alias("d"))
    assert_frame_equal(expected, result)

    # Positional inputs
    expected = pl.Series("d", [1.0, 2.0, 3.0, 10.0]).to_frame()
    result = df.select(pl.coalesce(pl.col(["a", "b", "c"]), 10.0).alias("d"))
    assert_frame_equal(result, expected)


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


def test_overflow_diff() -> None:
    df = pl.DataFrame(
        {
            "a": [20, 10, 30],
        }
    )
    assert df.select(pl.col("a").cast(pl.UInt64).diff()).to_dict(False) == {
        "a": [None, -10, 20]
    }


@typing.no_type_check
def test_fill_null_unknown_output_type() -> None:
    df = pl.DataFrame(
        {
            "a": [
                None,
                2,
                3,
                4,
                5,
            ]
        }
    )
    assert df.with_columns(
        np.exp(pl.col("a")).fill_null(pl.lit(1, pl.Float64))
    ).to_dict(False) == {
        "a": [
            1.0,
            7.38905609893065,
            20.085536923187668,
            54.598150033144236,
            148.4131591025766,
        ]
    }


def test_repeat() -> None:
    s = pl.select(pl.repeat(2**31 - 1, 3)).to_series()
    assert s.dtype == pl.Int32
    assert s.len() == 3
    assert s.to_list() == [2**31 - 1] * 3
    s = pl.select(pl.repeat(-(2**31), 4)).to_series()
    assert s.dtype == pl.Int32
    assert s.len() == 4
    assert s.to_list() == [-(2**31)] * 4
    s = pl.select(pl.repeat(2**31, 5)).to_series()
    assert s.dtype == pl.Int64
    assert s.len() == 5
    assert s.to_list() == [2**31] * 5
    s = pl.select(pl.repeat(-(2**31) - 1, 3)).to_series()
    assert s.dtype == pl.Int64
    assert s.len() == 3
    assert s.to_list() == [-(2**31) - 1] * 3
    s = pl.select(pl.repeat("foo", 2)).to_series()
    assert s.dtype == pl.Utf8
    assert s.len() == 2
    assert s.to_list() == ["foo"] * 2
    s = pl.select(pl.repeat(1.0, 5)).to_series()
    assert s.dtype == pl.Float64
    assert s.len() == 5
    assert s.to_list() == [1.0] * 5
    s = pl.select(pl.repeat(True, 4)).to_series()
    assert s.dtype == pl.Boolean
    assert s.len() == 4
    assert s.to_list() == [True] * 4
    s = pl.select(pl.repeat(None, 7)).to_series()
    assert s.dtype == pl.Null
    assert s.len() == 7
    assert s.to_list() == [None] * 7
    s = pl.select(pl.repeat(0, 0)).to_series()
    assert s.dtype == pl.Int32
    assert s.len() == 0


def test_min() -> None:
    s = pl.Series([1, 2, 3])
    assert pl.min(s) == 1

    df = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    assert df.select(pl.min("a")).item() == 1

    result = df.select(pl.min(["a", "b"]))
    assert_frame_equal(result, pl.DataFrame({"min": [1, 2]}))

    result = df.select(pl.min("a", 3))
    assert_frame_equal(result, pl.DataFrame({"min": [1, 3]}))


def test_max() -> None:
    s = pl.Series([1, 2, 3])
    assert pl.max(s) == 3

    df = pl.DataFrame({"a": [1, 4], "b": [3, 2]})
    assert df.select(pl.max("a")).item() == 4

    result = df.select(pl.max(["a", "b"]))
    assert_frame_equal(result, pl.DataFrame({"max": [3, 4]}))

    result = df.select(pl.max("a", 3))
    assert_frame_equal(result, pl.DataFrame({"max": [3, 4]}))
