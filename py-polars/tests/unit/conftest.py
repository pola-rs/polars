from __future__ import annotations

import gc
import random
import string
import tracemalloc
from typing import Any, Generator, List, cast

import numpy as np
import pytest
import pyarrow as pa

import polars as pl


@pytest.fixture()
def df() -> pl.DataFrame:
    df = pl.DataFrame(
        {
            "bools": [False, True, False],
            "bools_nulls": [None, True, False],
            "int": [1, 2, 3],
            "int_nulls": [1, None, 3],
            "floats": [1.0, 2.0, 3.0],
            "floats_nulls": [1.0, None, 3.0],
            "strings": ["foo", "bar", "ham"],
            "strings_nulls": ["foo", None, "ham"],
            "date": [1324, 123, 1234],
            "datetime": [13241324, 12341256, 12341234],
            "time": [13241324, 12341256, 12341234],
            "list_str": [["a", "b", None], ["a"], []],
            "list_bool": [[True, False, None], [None], []],
            "list_int": [[1, None, 3], [None], []],
            "list_flt": [[1.0, None, 3.0], [None], []],
        }
    )
    return df.with_columns(
        [
            pl.col("date").cast(pl.Date),
            pl.col("datetime").cast(pl.Datetime),
            pl.col("strings").cast(pl.Categorical).alias("cat"),
            pl.col("strings").cast(pl.Enum(["foo", "ham", "bar"])).alias("enum"),
            pl.col("time").cast(pl.Time),
        ]
    )


@pytest.fixture()
def df_no_lists(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.all().exclude(["list_str", "list_int", "list_bool", "list_int", "list_flt"])
    )


@pytest.fixture()
def fruits_cars() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "fruits": ["banana", "banana", "apple", "apple", "banana"],
            "B": [5, 4, 3, 2, 1],
            "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
        },
        schema_overrides={"A": pl.Int64, "B": pl.Int64},
    )


@pytest.fixture()
def str_ints_df() -> pl.DataFrame:
    n = 1000

    strs = pl.Series("strs", random.choices(string.ascii_lowercase, k=n))
    strs = pl.select(
        pl.when(strs == "a")
        .then(pl.lit(""))
        .when(strs == "b")
        .then(None)
        .otherwise(strs)
        .alias("strs")
    ).to_series()

    vals = pl.Series("vals", np.random.rand(n))

    return pl.DataFrame([vals, strs])


ISO8601_FORMATS_DATETIME = []

for T in ["T", " "]:
    for hms in (
        [
            f"{T}%H:%M:%S",
            f"{T}%H%M%S",
            f"{T}%H:%M",
            f"{T}%H%M",
        ]
        + [f"{T}%H:%M:%S.{fraction}" for fraction in ["%9f", "%6f", "%3f"]]
        + [f"{T}%H%M%S.{fraction}" for fraction in ["%9f", "%6f", "%3f"]]
        + [""]
    ):
        for date_sep in ("/", "-"):
            fmt = f"%Y{date_sep}%m{date_sep}%d{hms}"
            ISO8601_FORMATS_DATETIME.append(fmt)


@pytest.fixture(params=ISO8601_FORMATS_DATETIME)
def iso8601_format_datetime(request: pytest.FixtureRequest) -> list[str]:
    return cast(List[str], request.param)


ISO8601_TZ_AWARE_FORMATS_DATETIME = []

for T in ["T", " "]:
    for hms in (
        [
            f"{T}%H:%M:%S",
            f"{T}%H%M%S",
            f"{T}%H:%M",
            f"{T}%H%M",
        ]
        + [f"{T}%H:%M:%S.{fraction}" for fraction in ["%9f", "%6f", "%3f"]]
        + [f"{T}%H%M%S.{fraction}" for fraction in ["%9f", "%6f", "%3f"]]
    ):
        for date_sep in ("/", "-"):
            fmt = f"%Y{date_sep}%m{date_sep}%d{hms}%#z"
            ISO8601_TZ_AWARE_FORMATS_DATETIME.append(fmt)


@pytest.fixture(params=ISO8601_TZ_AWARE_FORMATS_DATETIME)
def iso8601_tz_aware_format_datetime(request: pytest.FixtureRequest) -> list[str]:
    return cast(List[str], request.param)


ISO8601_FORMATS_DATE = []

for date_sep in ("/", "-"):
    fmt = f"%Y{date_sep}%m{date_sep}%d"
    ISO8601_FORMATS_DATE.append(fmt)


@pytest.fixture(params=ISO8601_FORMATS_DATE)
def iso8601_format_date(request: pytest.FixtureRequest) -> list[str]:
    return cast(List[str], request.param)


class MemoryUsage:
    """
    Provide an API for measuring peak memory usage.

    Gets info from PyArrow, if possible, by overriding the memory pool.
    """

    def __init__(self):
        self.reset_tracking()

    def reset_tracking(self) -> None:
        """Reset tracking to zero."""
        # Clear existing pool's memory:
        pa.default_memory_pool().release_unused()
        # Setup a new pool so that we can track peak allocations from this
        # point onwards.
        pool = pa.system_memory_pool()
        pa.set_memory_pool(pool)
        self._pool = pool
        tracemalloc.reset_peak()

    def get_peak(self) -> int:
        """Return peak allocated memory, in bytes.

        This returns peak allocations since this object was created or
        ``reset_tracking()`` was called, whichever is later.
        """
        result = tracemalloc.get_traced_memory()[1]
        result += self._pool.max_memory()
        return result


@pytest.fixture
def memory_usage() -> Generator[MemoryUsage, Any, Any]:
    """
    Provide an API for measuring peak memory usage.

    Not thread-safe: there should only be one instance of MemoryUsage at any
    given time.
    """
    if not pl.build_info()["build"]["debug"]:
        pytest.skip("Memory usage only available in debug/dev builds.")

    gc.collect()
    tracemalloc.start()
    try:
        yield MemoryUsage()
    finally:
        tracemalloc.stop()
