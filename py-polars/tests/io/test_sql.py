from __future__ import annotations

import os
from datetime import date

import polars as pl


def test_read_sql() -> None:
    import sqlite3
    import tempfile

    try:
        import connectorx  # noqa: F401

        with tempfile.TemporaryDirectory() as tmpdir_name:
            test_db = os.path.join(tmpdir_name, "test.db")
            conn = sqlite3.connect(test_db)
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

            df = pl.read_sql(
                connection_uri=f"sqlite:///{test_db}", sql="SELECT * FROM test_data"
            )
            # ┌─────┬───────┬───────┬────────────┐
            # │ id  ┆ name  ┆ value ┆ date       │
            # │ --- ┆ ---   ┆ ---   ┆ ---        │
            # │ i64 ┆ str   ┆ f64   ┆ date       │
            # ╞═════╪═══════╪═══════╪════════════╡
            # │ 1   ┆ misc  ┆ 100.0 ┆ 2020-01-01 │
            # ├╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌┤
            # │ 2   ┆ other ┆ -99.5 ┆ 2021-12-31 │
            # └─────┴───────┴───────┴────────────┘

            expected = {
                "id": pl.Int64,
                "name": pl.Utf8,
                "value": pl.Float64,
                "date": pl.Date,
            }
            assert df.schema == expected
            assert df.shape == (2, 4)
            assert df["date"].to_list() == [date(2020, 1, 1), date(2021, 12, 31)]
            # assert df.rows() == ...

    except ImportError:
        pass  # if connectorx not installed on test machine
