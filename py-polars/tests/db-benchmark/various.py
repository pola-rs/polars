# may contain many things that seemed to go wrong at scale

import time

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
    assert computed[0, "min"] == minimum
    assert computed[0, "max"] == maximum


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
    for i in range(1000):
        ldf.collect()


if __name__ == "__main__":
    test_windows_not_cached()
