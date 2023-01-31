from __future__ import annotations

from typing import List, cast

import pytest

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
        }
    )


ISO8601_FORMATS = []
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
        for date_sep in ("/", "-", ""):
            fmt = f"%Y{date_sep}%m{date_sep}%d{hms}"
            ISO8601_FORMATS.append(fmt)


@pytest.fixture(params=ISO8601_FORMATS)
def iso8601_format(request: pytest.FixtureRequest) -> list[str]:
    return cast(List[str], request.param)
