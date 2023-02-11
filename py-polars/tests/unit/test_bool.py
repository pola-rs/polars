import numpy as np
import pytest

import polars as pl


@pytest.mark.slow()
def test_bool_arg_min_max() -> None:
    # masks that ensures we take more than u64 chunks
    # and slicing and dicing to ensure the offsets work
    for _ in range(100):
        offset = np.random.randint(0, 100)
        sample = np.random.rand(1000)
        a = sample > 0.99
        idx = a[offset:].argmax()
        assert idx == pl.Series(a)[offset:].arg_max()
        idx = a[offset:].argmin()
        assert idx == pl.Series(a)[offset:].arg_min()

        a = sample > 0.01
        idx = a[offset:].argmax()
        assert idx == pl.Series(a)[offset:].arg_max()
        idx = a[offset:].argmin()
        assert idx == pl.Series(a)[offset:].arg_min()
