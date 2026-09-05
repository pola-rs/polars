"""
# --8<-- [start:read_uri]
import polars as pl

uri = "postgresql://username:password@server:port/database"
query = "SELECT * FROM foo"

pl.read_database_uri(query=query, uri=uri)
# --8<-- [end:read_uri]

# --8<-- [start:read_cursor]
import polars as pl
from sqlalchemy import create_engine

conn = create_engine(f"sqlite:///test.db")

query = "SELECT * FROM foo"

pl.read_database(query=query, connection=conn.connect())
# --8<-- [end:read_cursor]


# --8<-- [start:adbc]
uri = "postgresql://username:password@server:port/database"
query = "SELECT * FROM foo"

pl.read_database_uri(query=query, uri=uri, engine="adbc")
# --8<-- [end:adbc]

# --8<-- [start:write]
uri = "postgresql://username:password@server:port/database"
df = pl.DataFrame({"foo": [1, 2, 3]})

df.write_database(table_name="records",  connection=uri)
# --8<-- [end:write]

# --8<-- [start:write_adbc]
uri = "postgresql://username:password@server:port/database"
df = pl.DataFrame({"foo": [1, 2, 3]})

df.write_database(table_name="records", connection=uri, engine="adbc")
# --8<-- [end:write_adbc]
"""
