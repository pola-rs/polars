# may contain many things that seemed to go wrong at scale

import time

import polars as pl

# https://github.com/pola-rs/polars/issues/1942
t0 = time.time()
pl.repeat(float("nan"), 2 << 12).sort()
assert (time.time() - t0) < 1
