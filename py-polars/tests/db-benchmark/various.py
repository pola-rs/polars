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
