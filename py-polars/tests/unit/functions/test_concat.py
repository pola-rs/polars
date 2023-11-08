import pytest

import polars as pl
from polars.testing import assert_frame_equal


@pytest.mark.slow()
def test_concat_expressions_stack_overflow() -> None:
    n = 10000
    e = pl.concat([pl.lit(x) for x in range(n)])

    df = pl.select(e)
    assert df.shape == (n, 1)


@pytest.mark.slow()
def test_concat_lf_stack_overflow() -> None:
    n = 1000
    bar = pl.DataFrame({"a": 0}).lazy()

    for i in range(n):
        bar = pl.concat([bar, pl.DataFrame({"a": i}).lazy()])
    assert bar.collect().shape == (1001, 1)


def test_empty_df_concat_str_11701() -> None:
    df = pl.DataFrame({"a": []})
    out = df.select(pl.concat_str([pl.col("a").cast(pl.Utf8), pl.lit("x")]))
    assert_frame_equal(out, pl.DataFrame({"a": []}, schema={"a": pl.Utf8}))
