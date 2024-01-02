import polars as pl
import numba as nb

df = pl.DataFrame({"a": [10, 9, 8, 7]})


@nb.guvectorize([(nb.int64[:], nb.int64, nb.int64[:])], "(n),()->(n)")
def cum_sum_reset(x, y, res):
    res[0] = x[0]
    for i in range(1, x.shape[0]):
        res[i] = x[i] + res[i - 1]
        if res[i] >= y:
            res[i] = x[i]


out = df.select(cum_sum_reset(pl.all(), 5))
print(out)
