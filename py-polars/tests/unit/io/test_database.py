from __future__ import annotations

import os
from contextlib import suppress
from datetime import date
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from polars.internals.type_aliases import (
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
            "value": [100.0, -99.5],
            "date": ["2020-01-01", "2021-12-31"],
        }
    )


def create_temp_sqlite_db(test_db: str) -> None:
    import sqlite3

    conn = sqlite3.connect(test_db)
    # ┌─────┬───────┬───────┬────────────┐
    # │ id  ┆ name  ┆ value ┆ date       │
    # │ --- ┆ ---   ┆ ---   ┆ ---        │
    # │ i64 ┆ str   ┆ f64   ┆ date       │
    # ╞═════╪═══════╪═══════╪════════════╡
    # │ 1   ┆ misc  ┆ 100.0 ┆ 2020-01-01 │
    # │ 2   ┆ other ┆ -99.5 ┆ 2021-12-31 │
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
        ),
    ],
)
def test_read_database(
    engine: DbReadEngine,
    expected_dtypes: dict[str, pl.DataType],
    expected_dates: list[date | str],
) -> None:
    import tempfile

    with suppress(ImportError):
        import connectorx  # noqa: F401

        with tempfile.TemporaryDirectory() as tmpdir_name:
            test_db = os.path.join(tmpdir_name, "test.db")
            create_temp_sqlite_db(test_db)

            df = pl.read_database(
                connection_uri=f"sqlite:///{test_db}",
                sql="SELECT * FROM test_data",
                engine=engine,
            )

            assert df.schema == expected_dtypes
            assert df.shape == (2, 4)
            assert df["date"].to_list() == expected_dates


@pytest.mark.parametrize(
    ("engine", "sql", "database", "err"),
    [
        pytest.param(
            "not_engine",
            "SELECT * FROM test_data",
            "sqlite",
            "Engine is not implemented, try either connectorx or adbc.",
            id="Not an available sql engine",
        ),
        pytest.param(
            "adbc",
            ["SELECT * FROM test_data", "SELECT * FROM test_data"],
            "sqlite",
            "Only a single SQL query string is accepted for adbc.",
            id="Unavailable list of queries for adbc.",
        ),
        pytest.param(
            "adbc",
            "SELECT * FROM test_data",
            "mysql",
            "ADBC does not currently support this database.",
            id="Unavailable database for adbc.",
        ),
    ],
)
def test_read_database_exceptions(
    engine: DbReadEngine, sql: str, database: str, err: str
) -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir_name:
        test_db = os.path.join(tmpdir_name, "test.db")
        create_temp_sqlite_db(test_db)

        with pytest.raises(ValueError, match=err):
            pl.read_database(
                connection_uri=f"{database}:///{test_db}",
                sql=sql,
                engine=engine,
            )


@pytest.mark.write_disk()
@pytest.mark.parametrize(
    ("engine", "mode"),
    [
        pytest.param("adbc", "create", id="create"),
        pytest.param("adbc", "append", id="append"),
    ],
)
def test_write_database(
    engine: DbWriteEngine, mode: DbWriteMode, sample_df: pl.DataFrame
) -> None:
    import tempfile

    with suppress(ImportError), tempfile.TemporaryDirectory() as tmpdir_name:
        test_db = os.path.join(tmpdir_name, "test.db")

        sample_df.write_database(
            table_name="test_data",
            connection_uri=f"sqlite:///{test_db}",
            mode="create",
            engine=engine,
        )

        if mode == "append":
            sample_df.write_database(
                table_name="test_data",
                connection_uri=f"sqlite:///{test_db}",
                mode="append",
                engine=engine,
            )
            sample_df = pl.concat([sample_df, sample_df])

        assert_frame_equal(
            sample_df,
            pl.read_database("SELECT * FROM test_data", f"sqlite:///{test_db}"),
        )
