from time import perf_counter

import pytest

import polars as pl
import polars.selectors as cs


# TODO: this is slow in streaming
@pytest.mark.may_fail_auto_streaming
@pytest.mark.slow
def test_with_columns_quadratic_19503() -> None:
    num_columns = 10_000
    data1 = {f"col_{i}": [0] for i in range(num_columns)}
    df1 = pl.DataFrame(data1)

    data2 = {f"feature_{i}": [0] for i in range(num_columns)}
    df2 = pl.DataFrame(data2)

    df3 = df2.select(cs.by_index(range(num_columns // 1000)))

    times = []  # [slow, fast]

    class _:
        t = perf_counter()
        df1.with_columns(df2)
        times.append(perf_counter() - t)

    class _:  # type: ignore[no-redef]
        t = perf_counter()
        df1.with_columns(df3)
        times.append(perf_counter() - t)

    # Assert the relative rather than exact runtime to avoid flakiness in CI
    # From local testing, the fixed version was roughly 20x, while the quadratic
    # version was roughly 200x.

    factor = times[0] / times[1]

    if factor > 30:
        raise AssertionError(factor)
