from __future__ import annotations

from datetime import timedelta
from typing import cast

import numpy as np
import pytest

import polars as pl
from polars import StringCache
from polars.testing import assert_frame_equal


def test_corr() -> None:
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = df.corr()
    expected = pl.DataFrame({"a": [1.0]})
    assert_frame_equal(result, expected)

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


def test_corr_nan() -> None:
    df = pl.DataFrame({"a": [1.0, 1.0], "b": [1.0, 2.0]})
    assert str(df.select(pl.corr("a", "b", ddof=1))[0, 0]) == "nan"


@StringCache()
def test_hist() -> None:
    s = pl.Series("a", [1, 3, 8, 8, 2, 1, 3])
    out = s.hist(bin_count=4)
    expected = pl.DataFrame(
        {
            "breakpoint": pl.Series([2.75, 4.5, 6.25, 8.0], dtype=pl.Float64),
            "category": pl.Series(
                ["(0.993, 2.75]", "(2.75, 4.5]", "(4.5, 6.25]", "(6.25, 8.0]"],
                dtype=pl.Categorical,
            ),
            "count": pl.Series([3, 2, 0, 2], dtype=pl.get_index_type()),
        }
    )
    assert_frame_equal(out, expected, categorical_as_str=True)


@pytest.mark.parametrize("values", [[], [None]])
def test_hist_empty_or_all_null(values: list[None]) -> None:
    ser = pl.Series(values, dtype=pl.Float64)
    assert (
        str(ser.hist().to_dict(as_series=False))
        == "{'breakpoint': [inf], 'category': ['(-inf, inf]'], 'count': [0]}"
    )


@pytest.mark.parametrize("n", [3, 10, 25])
def test_hist_rand(n: int) -> None:
    a = pl.Series(np.random.randint(0, 100, n))
    out = a.hist(bin_count=10)

    bp = out["breakpoint"]
    count = out["count"]
    for i in range(out.height):
        if i == 0:
            lower = float("-inf")
        else:
            lower = bp[i - 1]
        upper = bp[i]

        assert ((a <= upper) & (a > lower)).sum() == count[i]


def test_median_quantile_duration() -> None:
    df = pl.DataFrame({"A": [timedelta(days=0), timedelta(days=1)]})

    result = df.select(pl.col("A").median())
    expected = pl.DataFrame({"A": [timedelta(seconds=43200)]})
    assert_frame_equal(result, expected)

    result = df.select(pl.col("A").quantile(0.5, interpolation="linear"))
    expected = pl.DataFrame({"A": [timedelta(seconds=43200)]})
    assert_frame_equal(result, expected)


def test_correlation_cast_supertype() -> None:
    df = pl.DataFrame({"a": [1, 8, 3], "b": [4.0, 5.0, 2.0]})
    df = df.with_columns(pl.col("b"))
    assert df.select(pl.corr("a", "b")).to_dict(as_series=False) == {
        "a": [0.5447047794019219]
    }


def test_cov_corr_f32_type() -> None:
    df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]}).select(
        pl.all().cast(pl.Float32)
    )
    assert df.select(pl.cov("a", "b")).dtypes == [pl.Float32]
    assert df.select(pl.corr("a", "b")).dtypes == [pl.Float32]


def test_cov(fruits_cars: pl.DataFrame) -> None:
    ldf = fruits_cars.lazy()
    cov_a_b = pl.cov(pl.col("A"), pl.col("B"))
    cov_ab = pl.cov("A", "B")
    assert cast(float, ldf.select(cov_a_b).collect().item()) == -2.5
    assert cast(float, ldf.select(cov_ab).collect().item()) == -2.5


def test_std(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().std().collect()["A"][0] == pytest.approx(
        1.5811388300841898
    )


def test_var(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().var().collect()["A"][0] == pytest.approx(2.5)


def test_max(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().max().collect()["A"][0] == 5
    assert fruits_cars.select(pl.col("A").max())["A"][0] == 5


def test_min(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().min().collect()["A"][0] == 1
    assert fruits_cars.select(pl.col("A").min())["A"][0] == 1


def test_median(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().median().collect()["A"][0] == 3
    assert fruits_cars.select(pl.col("A").median())["A"][0] == 3


def test_quantile(fruits_cars: pl.DataFrame) -> None:
    assert fruits_cars.lazy().quantile(0.25, "nearest").collect()["A"][0] == 2
    assert fruits_cars.select(pl.col("A").quantile(0.25, "nearest"))["A"][0] == 2

    assert fruits_cars.lazy().quantile(0.24, "lower").collect()["A"][0] == 1
    assert fruits_cars.select(pl.col("A").quantile(0.24, "lower"))["A"][0] == 1

    assert fruits_cars.lazy().quantile(0.26, "higher").collect()["A"][0] == 3
    assert fruits_cars.select(pl.col("A").quantile(0.26, "higher"))["A"][0] == 3

    assert fruits_cars.lazy().quantile(0.24, "midpoint").collect()["A"][0] == 1.5
    assert fruits_cars.select(pl.col("A").quantile(0.24, "midpoint"))["A"][0] == 1.5

    assert fruits_cars.lazy().quantile(0.24, "linear").collect()["A"][0] == 1.96
    assert fruits_cars.select(pl.col("A").quantile(0.24, "linear"))["A"][0] == 1.96


def test_count() -> None:
    lf = pl.LazyFrame(
        {
            "nulls": [None, None, None],
            "one_null_str": ["one", None, "three"],
            "one_null_float": [1.0, 2.0, None],
            "no_nulls_int": [1, 2, 3],
        }
    )
    df = lf.collect()

    lf_result = lf.count()
    df_result = df.count()

    expected = pl.LazyFrame(
        {
            "nulls": [0],
            "one_null_str": [2],
            "one_null_float": [2],
            "no_nulls_int": [3],
        },
    ).cast(pl.UInt32)
    assert_frame_equal(lf_result, expected)
    assert_frame_equal(df_result, expected.collect())


def test_kurtosis_same_vals() -> None:
    df = pl.DataFrame({"a": [1.0042855193121334] * 11})
    assert_frame_equal(
        df.select(pl.col("a").kurtosis()), pl.select(a=pl.lit(float("nan")))
    )
