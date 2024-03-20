from __future__ import annotations

import random
import sqlite3
import string
from datetime import date
from typing import TYPE_CHECKING, List, cast

import numpy as np
import pytest

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


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


@pytest.fixture()
def tmp_sqlite_db(tmp_path: Path) -> Path:
    test_db = tmp_path / "test.db"
    test_db.unlink(missing_ok=True)

    def convert_date(val: bytes) -> date:
        """Convert ISO 8601 date to datetime.date object."""
        return date.fromisoformat(val.decode())

    # NOTE: at the time of writing adcb/connectorx have weak SQLite support (poor or
    # no bool/date/datetime dtypes, for example) and there is a bug in connectorx that
    # causes float rounding < py 3.11, hence we are only testing/storing simple values
    # in this test db for now. as support improves, we can add/test additional dtypes).
    sqlite3.register_converter("date", convert_date)
    conn = sqlite3.connect(test_db)

    # ┌─────┬───────┬───────┬────────────┐
    # │ id  ┆ name  ┆ value ┆ date       │
    # │ --- ┆ ---   ┆ ---   ┆ ---        │
    # │ i64 ┆ str   ┆ f64   ┆ date       │
    # ╞═════╪═══════╪═══════╪════════════╡
    # │ 1   ┆ misc  ┆ 100.0 ┆ 2020-01-01 │
    # │ 2   ┆ other ┆ -99.0 ┆ 2021-12-31 │
    # └─────┴───────┴───────┴────────────┘
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS test_data (
            id    INTEGER PRIMARY KEY,
            name  TEXT NOT NULL,
            value FLOAT,
            date  DATE
        );
        REPLACE INTO test_data(name,value,date)
          VALUES ('misc',100.0,'2020-01-01'),
                 ('other',-99.5,'2021-12-31');
        """
    )
    conn.close()
    return test_db


@pytest.fixture()
def tmp_sqlite_inference_db(tmp_path: Path) -> Path:
    test_db = tmp_path / "test_inference.db"
    test_db.unlink(missing_ok=True)
    conn = sqlite3.connect(test_db)
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS test_data (name TEXT, value FLOAT);
        REPLACE INTO test_data(name,value) VALUES (NULL,NULL), ('foo',0);
        """
    )
    conn.close()
    return test_db
