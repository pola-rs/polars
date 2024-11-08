import time

import pytest

import polars as pl


# TODO: this is slow in streaming
@pytest.mark.may_fail_auto_streaming
def test_with_columns_quadratic_19503() -> None:
    num_columns = 2000
    data1 = {f"col_{i}": [0] for i in range(num_columns)}
    df1 = pl.DataFrame(data1)

    data2 = {f"feature_{i}": [0] for i in range(num_columns)}
    df2 = pl.DataFrame(data2)

    t0 = time.time()
    df1.with_columns(df2)
    t1 = time.time()
    assert t1 - t0 < 0.2
