import pytest

import polars as pl


@pytest.mark.slow()
def test_concat_expressions_stack_overflow() -> None:
    n = 10000
    e = pl.concat([pl.lit(x) for x in range(0, n)])

    df = pl.select(e)
    assert df.shape == (n, 1)


@pytest.mark.slow()
def test_concat_lf_stack_overflow() -> None:
    n = 1000
    bar = pl.DataFrame({"a": 0}).lazy()

    for i in range(n):
        bar = pl.concat([bar, pl.DataFrame({"a": i}).lazy()])
    assert bar.collect().shape == (1001, 1)
