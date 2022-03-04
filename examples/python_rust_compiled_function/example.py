import polars as pl
from my_polars_functions import hamming_distance

a = pl.Series("a", ["foo", "bar"])
b = pl.Series("b", ["fooy", "ham"])

dist = hamming_distance(a, b)
expected = pl.Series("", [None, 2], dtype=pl.UInt32)

print(hamming_distance(a, b))
assert dist.series_equal(expected, null_equal=True)

