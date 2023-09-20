"""
# --8<-- [start:read]
import polars as pl

connection_uri = "postgres://username:password@server:port/database"
query = "SELECT * FROM foo"

pl.read_database(query=query, connection_uri=connection_uri)
# --8<-- [end:read]

# --8<-- [start:adbc]
connection_uri = "postgres://username:password@server:port/database"
query = "SELECT * FROM foo"

pl.read_database(query=query, connection_uri=connection_uri, engine="adbc")
# --8<-- [end:adbc]

# --8<-- [start:write]
connection_uri = "postgres://username:password@server:port/database"
df = pl.DataFrame({"foo": [1, 2, 3]})

df.write_database(table_name="records",  connection_uri=connection_uri)
# --8<-- [end:write]

# --8<-- [start:write_adbc]
connection_uri = "postgres://username:password@server:port/database"
df = pl.DataFrame({"foo": [1, 2, 3]})

df.write_database(table_name="records", connection_uri=connection_uri, engine="adbc")
# --8<-- [end:write_adbc]

"""
