import polars as pl

path = "G1_1e7_1e2_5_0.csv"
predicate = pl.col("v2") < 5

shape_eager = pl.read_csv(path).filter(predicate).shape

shape_lazy = (pl.scan_csv(path).filter(predicate)).collect().shape
assert shape_lazy == shape_eager
