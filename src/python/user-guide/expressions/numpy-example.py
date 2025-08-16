import polars as pl
import numpy as np

df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})

out = df.select(np.log(pl.all()).name.suffix("_log"))
print(out)
