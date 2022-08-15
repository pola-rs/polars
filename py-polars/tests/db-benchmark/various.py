# may contain many things that seemed to go wrong at scale

import time
from typing import cast

import numpy as np

import polars as pl

# https://github.com/pola-rs/polars/issues/1942
t0 = time.time()
pl.repeat(float("nan"), 2 << 12).sort()
assert (time.time() - t0) < 1

# test mean overflow issues
np.random.seed(1)
mean = 769.5607652
df = pl.DataFrame(np.random.randint(500, 1040, 5000000), columns=["value"])
assert np.isclose(df.with_column(pl.mean("value"))[0, 0], mean)
assert np.isclose(
    df.with_column(pl.col("value").cast(pl.Int32)).with_column(pl.mean("value"))[0, 0],
    mean,
)
assert np.isclose(
    df.with_column(pl.col("value").cast(pl.Int32)).get_column("value").mean(), mean
)

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
    permuted = df.sample(frac=1.0, seed=0)
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
    assert df2.join(df1, how="cross").slice(0, 100).frame_equal(out)


def test_cross_join_slice_pushdown() -> None:
    # this will likely go out of memory if we did not pushdown the slice
    df = pl.DataFrame(
        [
            pl.Series("x", pl.arange(0, 2**16 - 1, eager=True) % 2**15).cast(
                pl.UInt16
            )
        ]
    )

    assert df.lazy().join(df.lazy(), how="cross", suffix="_").slice(
        -5, 10
    ).collect().to_dict(False) == {
        "x": [32766, 32766, 32766, 32766, 32766],
        "x_": [32762, 32763, 32764, 32765, 32766],
    }

    assert df.lazy().join(df.lazy(), how="cross", suffix="_").slice(
        2, 10
    ).collect().to_dict(False) == {
        "x": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        "x_": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }


if __name__ == "__main__":
    test_windows_not_cached()
    test_cross_join()
