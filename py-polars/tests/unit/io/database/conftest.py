from __future__ import annotations

import sqlite3
from datetime import date
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path


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
