from __future__ import annotations

import sqlite3
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars.type_aliases import (
        DbReadEngine,
        DbWriteEngine,
        DbWriteMode,
    )


@pytest.fixture()
def sample_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2],
            "name": ["misc", "other"],
            "value": [100.0, -99.0],
            "date": ["2020-01-01", "2021-12-31"],
        }
    )


def create_temp_sqlite_db(test_db: str) -> None:
    Path(test_db).unlink(missing_ok=True)

    # NOTE: at the time of writing adcb/connectorx have weak SQLite support (poor or
    # no bool/date/datetime dtypes, for example) and there is a bug in connectorx that
    # causes float rounding < py 3.11, hence we are only testing/storing simple values
    # in this test db for now. as support improves, we can add/test additional dtypes).

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
        CREATE TABLE test_data (
            id    INTEGER PRIMARY KEY,
            name  TEXT NOT NULL,
            value FLOAT,
            date  DATE
        );
        INSERT INTO test_data(name,value,date)
        VALUES ('misc',100.0,'2020-01-01'), ('other',-99.5,'2021-12-31');
        """
    )
    conn.close()


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("engine", "expected_dtypes", "expected_dates"),
    [
        pytest.param(
            "connectorx",
            {
                "id": pl.Int64,
                "name": pl.Utf8,
                "value": pl.Float64,
                "date": pl.Date,
            },
            [date(2020, 1, 1), date(2021, 12, 31)],
        ),
        pytest.param(
            "adbc",
            {
                "id": pl.Int64,
                "name": pl.Utf8,
                "value": pl.Float64,
                "date": pl.Utf8,
            },
            ["2020-01-01", "2021-12-31"],
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
        ),
    ],
)
def test_read_database(
    engine: DbReadEngine,
    expected_dtypes: dict[str, pl.DataType],
    expected_dates: list[date | str],
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)

    test_db = str(tmp_path / "test.db")
    create_temp_sqlite_db(test_db)

    df = pl.read_database(
        connection=f"sqlite:///{test_db}",
        query="SELECT * FROM test_data",
        engine=engine,
    )
    assert df.schema == expected_dtypes
    assert df.shape == (2, 4)
    assert df["date"].to_list() == expected_dates


@pytest.mark.parametrize(
    ("engine", "query", "database", "errclass", "err"),
    [
        pytest.param(
            "not_engine",
            "SELECT * FROM test_data",
            "sqlite",
            ValueError,
            "Engine 'not_engine' not implemented; use connectorx or adbc.",
            id="Not an available sql engine",
        ),
        pytest.param(
            "adbc",
            ["SELECT * FROM test_data", "SELECT * FROM test_data"],
            "sqlite",
            ValueError,
            "Only a single SQL query string is accepted for adbc.",
            id="Unavailable list of queries for adbc",
        ),
        pytest.param(
            "adbc",
            "SELECT * FROM test_data",
            "mysql",
            ImportError,
            "ADBC mysql driver not detected",
            id="Unavailable adbc driver",
        ),
        pytest.param(
            "adbc",
            "SELECT * FROM test_data",
            sqlite3.connect(":memory:"),
            TypeError,
            "Expect connection to be a URI string",
            id="Invalid connection URI",
        ),
    ],
)
def test_read_database_exceptions(
    engine: DbReadEngine,
    query: str,
    database: Any,
    errclass: type,
    err: str,
    tmp_path: Path,
) -> None:
    conn = f"{database}://test" if isinstance(database, str) else database
    with pytest.raises(errclass, match=err):
        pl.read_database(
            connection=conn,
            query=query,
            engine=engine,
        )


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("engine", "mode"),
    [
        pytest.param(
            "adbc",
            "create",
            id="adbc_create",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
        ),
        pytest.param(
            "adbc",
            "append",
            id="adbc_append",
            marks=pytest.mark.skipif(
                sys.version_info < (3, 9) or sys.platform == "win32",
                reason="adbc_driver_sqlite not available below Python 3.9 / on Windows",
            ),
        ),
        pytest.param(
            "sqlalchemy",
            "create",
            id="sa_create",
        ),
        pytest.param(
            "sqlalchemy",
            "append",
            id="sa_append",
        ),
    ],
)
def test_write_database(
    engine: DbWriteEngine, mode: DbWriteMode, sample_df: pl.DataFrame, tmp_path: Path
) -> None:
    tmp_path.mkdir(exist_ok=True)
    tmp_db = f"test_{engine}.db"
    test_db = str(tmp_path / tmp_db)

    # note: test a table name that requires quotes to ensure that we handle
    # it correctly (also supply an explicit db schema with/without quotes)
    tbl_name = '"test-data"'

    sample_df.write_database(
        table_name=f"main.{tbl_name}",
        connection=f"sqlite:///{test_db}",
        if_exists="replace",
        engine=engine,
    )
    if mode == "append":
        sample_df.write_database(
            table_name=f'"main".{tbl_name}',
            connection=f"sqlite:///{test_db}",
            if_exists="append",
            engine=engine,
        )
        sample_df = pl.concat([sample_df, sample_df])

    result = pl.read_database(f"SELECT * FROM {tbl_name}", f"sqlite:///{test_db}")
    sample_df = sample_df.with_columns(pl.col("date").cast(pl.Utf8))
    assert_frame_equal(sample_df, result)

    # check that some invalid parameters raise errors
    for invalid_params in (
        {"table_name": "w.x.y.z"},
        {"if_exists": "crunk", "table_name": f"main.{tbl_name}"},
    ):
        with pytest.raises(ValueError):
            sample_df.write_database(
                connection=f"sqlite:///{test_db}",
                engine=engine,
                **invalid_params,  # type: ignore[arg-type]
            )
