---
render_macros: false
---

# Redis through ADBC

Read SQL results from Redis and write Polars DataFrames through the
[community Redis ADBC driver](https://github.com/redis/redis-adbc-driver). The driver executes
supported SQL over Redis data; Redis must run separately with the Query Engine (Search). Existing
arbitrary Redis keys do not automatically become SQL tables. Create, ingest, or adopt tables through
the driver first.

## Install and connect

```sh
python -m pip install polars 'adbc-driver-redis[dbapi]'
```

The driver bundles its native library in supported Linux amd64/arm64 and macOS arm64 wheels. See its
[installation guide](https://github.com/redis/redis-adbc-driver/blob/main/docs/python-integrations.md#install-and-discover-the-driver)
for platform requirements, source builds and TLS configuration. Installing the package does not
start Redis.

```python
import os
import polars as pl

result = pl.read_database_uri(
    query="SELECT 42 AS answer",
    uri=os.environ["REDIS_URI"],
    engine="adbc",
)
assert result.to_dicts() == [{"answer": 42}]
```

Set `REDIS_URI` in your environment, for example `redis://localhost:6379/0`. Use `rediss://` for
verified TLS. The URI's scheme selects `adbc_driver_redis` or its TLS alias; do not use the default
ConnectorX engine for this path.

## Write and read a DataFrame

Use a dedicated test database for this example. It creates a unique table and removes it afterward:

```python
import os
from uuid import uuid4

import polars as pl
from adbc_driver_redis import dbapi
from polars.testing import assert_frame_equal

name = "polars_example_" + uuid4().hex
source = pl.DataFrame({"id": [1, 2], "amount": [10.5, 20.0]})
with dbapi.connect(os.environ["REDIS_URI"]) as connection:
    try:
        source.write_database(name, connection, engine="adbc")
        result = pl.read_database(f'SELECT * FROM "{name}" ORDER BY id', connection)
        assert_frame_equal(result, source)
    finally:
        with connection.cursor() as cursor:
            cursor.execute(f'DROP TABLE IF EXISTS "{name}"')
```

Use `if_table_exists="append"` for later batches. Writes autocommit; a failed write can partially
apply and rollback cannot undo it. Replacement can remove existing data. Do not assume transaction
isolation or exactly-once retries.

The driver's
[regression record](https://github.com/redis/redis-adbc-driver/blob/main/docs/sdlc/changes/polars-regression-rerun/validation.md)
reports connection/URI reads, typed write/append round trips, filtering, aggregation, NULLs and
empty schemas with Polars 2.1.0. Those local source-build results do not establish every platform,
wheel, deployment or Flight SQL path.
