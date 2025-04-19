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

    times = []  # [slow, fast]

    class _:
        rhs = df2
        t = perf_counter()
        df1.with_columns(rhs)
        times.append(perf_counter() - t)

    class _:  # type: ignore[no-redef]
        rhs = df2.select(cs.by_index(range(num_columns // 1_000)))
        t = perf_counter()
        df1.with_columns(rhs)
        times.append(perf_counter() - t)

    ratio = times[0] / times[1]

    # Assert the relative rather than exact runtime to avoid flakiness in CI
    # We pick a threshold just low enough to pass CI without any false
    # negatives.
    #                    1.12.0 | 1.14.0
    #   M3 Pro 11-core |   200x |    20x
    #  EC2 c7i.4xlarge |   150x |    13x
    # GitHub CI runner |        |    50x
    if ratio > 100:
        raise AssertionError(ratio)
