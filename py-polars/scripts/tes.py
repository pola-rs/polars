import time

import numpy as np

import polars as pl

x = np.random.random(1_000_000)


def signif(x, p):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10 ** (p - 1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


t1 = time.time()
s = signif(x, 2)
np_time = time.time() - t1
print(np_time)

s = pl.Series(x)
t1 = time.time()
s.round_sig_figs(2)
pl_time = time.time() - t1
print(pl_time)

print((pl_time - np_time) / np_time)
