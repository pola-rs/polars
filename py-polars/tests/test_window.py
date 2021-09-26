import numpy as np

import polars as pl


def test_std():
    for dtype in [pl.Float32, pl.Float64]:
        df = pl.DataFrame(
            [
                pl.Series("groups", ["a", "a", "b", "b"]),
                pl.Series("values", [1.0, 2.0, 3.0, 4.0], dtype=dtype),
            ]
        )

        out = df.select(pl.col("values").std().over("groups"))
        assert np.isclose(out["values"][0], 0.7071067690849304)

        out = df.select(pl.col("values").var().over("groups"))
        assert np.isclose(out["values"][0], 0.5)
