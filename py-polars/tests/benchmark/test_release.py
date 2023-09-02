"""
Various benchmark tests.

Tests in this module will be run in the CI using a release build of Polars.

To run these tests: pytest -m benchmark
"""
import time
from pathlib import Path
from typing import cast

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal

# Mark all tests in this module as benchmark tests
pytestmark = pytest.mark.benchmark()


@pytest.mark.skipif(
    not (Path(__file__).parent / "G1_1e7_1e2_5_0.csv").is_file(),
    reason="Dataset must be generated before running this test.",
)
def test_read_scan_large_csv() -> None:
    filename = "G1_1e7_1e2_5_0.csv"
    path = Path(__file__).parent / filename

    predicate = pl.col("v2") < 5

    shape_eager = pl.read_csv(path).filter(predicate).shape
    shape_lazy = (pl.scan_csv(path).filter(predicate)).collect().shape

    assert shape_lazy == shape_eager


def test_sort_nan_1942() -> None:
    # https://github.com/pola-rs/polars/issues/1942
    t0 = time.time()
    pl.repeat(float("nan"), 2 << 12, eager=True).sort()
    assert (time.time() - t0) < 1


def test_mean_overflow() -> None:
    np.random.seed(1)
    expected = 769.5607652

    df = pl.DataFrame(np.random.randint(500, 1040, 5000000), schema=["value"])

    result = df.with_columns(pl.mean("value"))[0, 0]
    assert np.isclose(result, expected)

    result = df.with_columns(pl.col("value").cast(pl.Int32)).with_columns(
        pl.mean("value")
    )[0, 0]
    assert np.isclose(result, expected)

    result = df.with_columns(pl.col("value").cast(pl.Int32)).get_column("value").mean()
    assert np.isclose(result, expected)


def test_min_max_2850() -> None:
    # https://github.com/pola-rs/polars/issues/2850
    df = pl.DataFrame(
        {
            "id": [
                130352432,
                130352277,
                130352611,
                130352833,
                130352305,
                130352258,
                130352764,
                130352475,
                130352368,
                130352346,
            ]
        }
    )

    minimum = 130352258
    maximum = 130352833.0

    for _ in range(10):
        permuted = df.sample(fraction=1.0, seed=0)
        computed = permuted.select(
            [pl.col("id").min().alias("min"), pl.col("id").max().alias("max")]
        )
        assert cast(int, computed[0, "min"]) == minimum
        assert cast(float, computed[0, "max"]) == maximum


def test_windows_not_cached() -> None:
    ldf = (
        pl.DataFrame(
            [
                pl.Series("key", ["a", "a", "b", "b"]),
                pl.Series("val", [2, 2, 1, 3]),
            ]
        )
        .lazy()
        .filter(
            (pl.col("key").cumcount().over("key") == 0)
            | (pl.col("val").shift(1).over("key").is_not_null())
            | (pl.col("val") != pl.col("val").shift(1).over("key"))
        )
    )
    # this might fail if they are cached
    for _ in range(1000):
        ldf.collect()


def test_cross_join() -> None:
    # triggers > 100 rows implementation
    # https://github.com/pola-rs/polars/blob/5f5acb2a523ce01bc710768b396762b8e69a9e07/polars/polars-core/src/frame/cross_join.rs#L34
    df1 = pl.DataFrame({"col1": ["a"], "col2": ["d"]})
    df2 = pl.DataFrame({"frame2": pl.arange(0, 100, eager=True)})
    out = df2.join(df1, how="cross")
    df2 = pl.DataFrame({"frame2": pl.arange(0, 101, eager=True)})
    assert_frame_equal(df2.join(df1, how="cross").slice(0, 100), out)


def test_cross_join_slice_pushdown() -> None:
    # this will likely go out of memory if we did not pushdown the slice
    df = (
        pl.Series("x", pl.arange(0, 2**16 - 1, eager=True, dtype=pl.UInt16) % 2**15)
    ).to_frame()

    result = df.lazy().join(df.lazy(), how="cross", suffix="_").slice(-5, 10).collect()
    expected = pl.DataFrame(
        {
            "x": [32766, 32766, 32766, 32766, 32766],
            "x_": [32762, 32763, 32764, 32765, 32766],
        },
        schema={"x": pl.UInt16, "x_": pl.UInt16},
    )
    assert_frame_equal(result, expected)

    result = df.lazy().join(df.lazy(), how="cross", suffix="_").slice(2, 10).collect()
    expected = pl.DataFrame(
        {
            "x": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            "x_": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        },
        schema={"x": pl.UInt16, "x_": pl.UInt16},
    )


def test_max_statistic_parquet_writer() -> None:
    # this hits the maximal page size
    # so the row group will be split into multiple pages
    # the page statistics need to be correctly reduced
    # for this query to make sense
    n = 150_000

    # int64 is important to hit the page size
    df = pl.int_range(0, n, eager=True, dtype=pl.Int64).to_frame()
    f = "/tmp/tmp.parquet"
    df.write_parquet(f, statistics=True, use_pyarrow=False, row_group_size=n)
    result = pl.scan_parquet(f).filter(pl.col("int") > n - 3).collect()
    expected = pl.DataFrame({"int": [149998, 149999]})
    assert_frame_equal(result, expected)


def test_boolean_min_max_agg() -> None:
    np.random.seed(0)
    idx = np.random.randint(0, 500, 1000)
    c = np.random.randint(0, 500, 1000) > 250

    df = pl.DataFrame({"idx": idx, "c": c})
    aggs = [pl.col("c").min().alias("c_min"), pl.col("c").max().alias("c_max")]
    assert df.group_by("idx").agg(aggs).sum().to_dict(False) == {
        "idx": [107583],
        "c_min": [120],
        "c_max": [321],
    }

    nulls = np.random.randint(0, 500, 1000) < 100
    assert df.with_columns(
        c=pl.when(pl.lit(nulls)).then(None).otherwise(pl.col("c"))
    ).group_by("idx").agg(aggs).sum().to_dict(False) == {
        "idx": [107583],
        "c_min": [133],
        "c_max": [276],
    }


def test_categorical_vs_str_group_by() -> None:
    # this triggers the perfect hash table
    s = pl.Series("a", np.random.randint(0, 50, 100))
    s_with_nulls = pl.select(
        pl.when(s < 3).then(None).otherwise(s).alias("a")
    ).to_series()

    for s_ in [s, s_with_nulls]:
        s_ = s_.cast(str)
        cat_out = (
            s_.cast(pl.Categorical)
            .to_frame("a")
            .group_by("a")
            .agg(pl.first().alias("first"))
        )

        str_out = s_.to_frame("a").group_by("a").agg(pl.first().alias("first"))
        cat_out.with_columns(pl.col("a").cast(str))
        assert_frame_equal(
            cat_out.with_columns(
                pl.col("a").cast(str), pl.col("first").cast(pl.List(str))
            ).sort("a"),
            str_out.sort("a"),
        )
