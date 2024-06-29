import pytest

import polars as pl

nan = float("nan")


def test_fill_nan_deprecated() -> None:
    with pytest.deprecated_call():
        pl.Series([1.0, 2.0, nan]).fill_nan(0)
    with pytest.deprecated_call():
        pl.col("a").fill_nan(0)
    with pytest.deprecated_call():
        pl.DataFrame({"a": [1.0, 2.0, nan]}).fill_nan(0)
    with pytest.deprecated_call():
        pl.LazyFrame({"a": [1.0, 2.0, nan]}).fill_nan(0)
