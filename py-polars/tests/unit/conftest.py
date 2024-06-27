from __future__ import annotations

import gc
import os
import random
import string
import sys
import tracemalloc
from typing import Any, Generator, List, cast

import numpy as np
import pytest

import polars as pl
from polars.testing.parametric import load_profile

load_profile(
    profile=os.environ.get("POLARS_HYPOTHESIS_PROFILE", "fast"),  # type: ignore[arg-type]
)

# Data type groups
SIGNED_INTEGER_DTYPES = [pl.Int8(), pl.Int16(), pl.Int32(), pl.Int64()]
UNSIGNED_INTEGER_DTYPES = [pl.UInt8(), pl.UInt16(), pl.UInt32(), pl.UInt64()]
INTEGER_DTYPES = SIGNED_INTEGER_DTYPES + UNSIGNED_INTEGER_DTYPES
FLOAT_DTYPES = [pl.Float32(), pl.Float64()]
NUMERIC_DTYPES = INTEGER_DTYPES + FLOAT_DTYPES

DATETIME_DTYPES = [pl.Datetime("ms"), pl.Datetime("us"), pl.Datetime("ns")]
DURATION_DTYPES = [pl.Duration("ms"), pl.Duration("us"), pl.Duration("ns")]
TEMPORAL_DTYPES = [*DATETIME_DTYPES, *DURATION_DTYPES, pl.Date(), pl.Time()]

NESTED_DTYPES = [pl.List, pl.Struct, pl.Array]


@pytest.fixture()
def partition_limit() -> int:
    """The limit at which Polars will start partitioning in debug builds."""
    return 15


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
        pl.col("date").cast(pl.Date),
        pl.col("datetime").cast(pl.Datetime),
        pl.col("strings").cast(pl.Categorical).alias("cat"),
        pl.col("strings").cast(pl.Enum(["foo", "ham", "bar"])).alias("enum"),
        pl.col("time").cast(pl.Time),
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

    Memory from PyArrow is not tracked at the moment.
    """

    def reset_tracking(self) -> None:
        """Reset tracking to zero."""
        gc.collect()
        tracemalloc.stop()
        tracemalloc.start()
        assert self.get_peak() < 100_000

    def get_current(self) -> int:
        """
        Return currently allocated memory, in bytes.

        This only tracks allocations since this object was created or
        ``reset_tracking()`` was called, whichever is later.
        """
        return tracemalloc.get_traced_memory()[0]

    def get_peak(self) -> int:
        """
        Return peak allocated memory, in bytes.

        This returns peak allocations since this object was created or
        ``reset_tracking()`` was called, whichever is later.
        """
        return tracemalloc.get_traced_memory()[1]


@pytest.fixture()
def memory_usage_without_pyarrow() -> Generator[MemoryUsage, Any, Any]:
    """
    Provide an API for measuring peak memory usage.

    Not thread-safe: there should only be one instance of MemoryUsage at any
    given time.

    Memory usage from PyArrow is not tracked.
    """
    if not pl.build_info()["compiler"]["debug"]:
        pytest.skip("Memory usage only available in debug/dev builds.")

    if os.getenv("POLARS_FORCE_ASYNC", "0") == "1":
        pytest.skip("Hangs when combined with async glob")

    if sys.platform == "win32":
        # abi3 wheels don't have the tracemalloc C APIs, which breaks linking
        # on Windows.
        pytest.skip("Windows not supported at the moment.")

    gc.collect()
    tracemalloc.start()
    try:
        yield MemoryUsage()
    finally:
        tracemalloc.stop()
