# --8<-- [start:setup]
import sqlite3
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path

import polars as pl

path = Path(tempfile.mkdtemp()) / "shop.db"
with closing(sqlite3.connect(path)) as conn, conn:
    conn.execute(
        "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, "
        "status TEXT, amount REAL, created_at TEXT)"
    )
    conn.execute(
        "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, region TEXT)"
    )
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?)",
        [
            (
                i,
                i % 10,
                "paid" if i % 3 else "open",
                i * 1.5,
                f"2026-01-{1 + i % 28:02d}",
            )
            for i in range(1, 1001)
        ],
    )
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?)",
        [(i, "EU" if i % 2 else "US") for i in range(10)],
    )
uri = f"sqlite:///{path}"
# --8<-- [end:setup]

# --8<-- [start:scan]
lf = pl.scan_database("orders", uri, schema_overrides={"created_at": pl.Date})

(
    lf.filter(pl.col("created_at") >= date(2026, 1, 15), pl.col("status") == "paid")
    .group_by("customer_id")
    .agg(pl.col("amount").sum())
    .collect()
)
# --8<-- [end:scan]

# --8<-- [start:scan_explain]
print(lf.filter(pl.col("status") == "paid").select("id", "amount").explain())
# --8<-- [end:scan_explain]

# --8<-- [start:show_drivers]
pl.io.database.show_drivers("sqlite")
# --8<-- [end:show_drivers]

# --8<-- [start:scan_driver]
pl.scan_database("orders", uri, driver="sqlite3")
pl.scan_database("orders", f"sqlite+pysqlite:///{path}")  # the SQLAlchemy form
# --8<-- [end:scan_driver]

# --8<-- [start:scan_partitions]
pl.scan_database(
    "orders",
    uri,
    partition_on="id",
    partition_num=16,
    max_connections=4,
).collect()
# --8<-- [end:scan_partitions]

# --8<-- [start:scan_query]
pl.scan_database(
    "SELECT o.*, c.region FROM orders o JOIN customers c USING (customer_id)",
    uri,
).filter(pl.col("region") == "EU").collect()
# --8<-- [end:scan_query]

# --8<-- [start:scan_params]
pl.scan_database(
    "SELECT * FROM orders WHERE customer_id = ?",
    uri,
    params=[3],
    driver="sqlite3",  # params are bound by the driver
).collect()
# --8<-- [end:scan_params]

# --8<-- [start:scan_factory]
pl.scan_database(
    "orders",
    lambda: sqlite3.connect(path, check_same_thread=False),
).collect()
# --8<-- [end:scan_factory]
