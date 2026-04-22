# Persistence and query history

The observatory database is used to store more than metrics. Restarting a Polars on-premises cluster
with the same database path will allow the observatory to read back historical query data. Keep in
mind that the engine (SQLite) does not allow concurrent access, and a database can only be
associated with a single cluster at a time.

The location of the observatory database can be configured through the `observatory.database_path`
configuration option. If this points to a directory, a file in that directory will be created.

```toml
[observatory]
enabled = true
database_path = "/mnt/data/observatory.db"
```
