import numpy as np
import pytest

import polars as pl


@pytest.mark.parametrize("dtype", [pl.Float32, pl.Float64, pl.Int32])
def test_std(dtype):
    if dtype == pl.Int32:
        values = [1, 2, 3, 4]
    else:
        values = [1.0, 2.0, 3.0, 4.0]
    df = pl.DataFrame(
        [
            pl.Series("groups", ["a", "a", "b", "b"]),
            pl.Series("values", values, dtype=dtype),
        ]
    )

    out = df.select(pl.col("values").std().over("groups"))
    assert np.isclose(out["values"][0], 0.7071067690849304)

    out = df.select(pl.col("values").var().over("groups"))
    assert np.isclose(out["values"][0], 0.5)
    out = df.select(pl.col("values").mean().over("groups"))
    assert np.isclose(out["values"][0], 1.5)
