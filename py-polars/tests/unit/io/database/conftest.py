from __future__ import annotations

import sqlite3
from datetime import date
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from sqlalchemy.engine import Engine





def close_connections(*connections: Any) -> None:
    """Fully release the underlying DB-API connection for each given object."""
    from sqlalchemy.engine import Connection, Engine

    for conn in connections:
        if isinstance(conn, str):
            continue  # eg: connection URI, nothing to close
        elif isinstance(conn, Engine):
            conn.dispose()
        elif isinstance(conn, Connection):
            engine = conn.engine
            conn.close()
            engine.dispose()
        elif hasattr(conn, "close"):
            conn.close()

def create_sqlite_engine(target: str | Path, **kwargs: Any) -> Engine:
    """Create a sqlite SQLAlchemy ``Engine`` that uses NullPool connections."""
    from sqlalchemy import create_engine
    from sqlalchemy.pool import NullPool

    if not (uri := str(target)).startswith("sqlite:"):
        uri = f"sqlite:///{uri}"
    return create_engine(uri, poolclass=NullPool, **kwargs)
@pytest.fixture
def sqlite_engine(tmp_sqlite_db: Path) -> Iterator[Engine]:
    """A NullPool SQLAlchemy engine bound to the standard test database."""
    engine = create_sqlite_engine(tmp_sqlite_db)
    try:
        yield engine
    finally:
        engine.dispose()


@pytest.fixture
def tmp_sqlite_db(tmp_path: Path) -> Path:
    test_db = tmp_path / "test.db"
    test_db.unlink(missing_ok=True)

    def convert_date(val: bytes) -> date:
        """Convert ISO 8601 date to datetime.date object."""
        return date.fromisoformat(val.decode())

    # NOTE: at the time of writing adbc/connectorx have weak SQLite support (poor or
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


@pytest.fixture
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
