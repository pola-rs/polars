import polars as pl
from io_plugin import new_bernoulli, new_uniform, scan_random


lf = scan_random(
    [
        new_bernoulli("b0.5", p=0.5, seed=1),
        new_bernoulli("b0.1", p=0.1, seed=2),
        new_uniform("uniform", low=10, high=100, dtype=pl.Int32, seed=2),
    ]
)

# predicate pushdown
assert lf.filter(pl.col("b0.5")).collect()["b0.5"].all()

# slice pushdown
assert lf.head(100).collect().shape == (100, 3)

# projection pushdown
out = lf.select("uniform", "b0.1").collect()
assert out.shape == (1000, 2)
assert out.columns == ["uniform", "b0.1"]
