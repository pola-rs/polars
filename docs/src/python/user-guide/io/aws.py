"""
# --8<-- [start:read_parquet]
import polars as pl

bucket = "<YOUR_BUCKET>"
path = "<YOUR_PATH>"

df = pl.read_parquet(f"s3://{bucket}/{path}")
# --8<-- [end:bucket]

# --8<-- [start:scan_parquet]
import polars as pl

bucket = "<YOUR_BUCKET>"
path = "<YOUR_PATH>"

df = pl.scan_parquet(f"s3://{bucket}/{path}")
# --8<-- [end:scan_parquet]

# --8<-- [start:scan_parquet_query]
import polars as pl

bucket = "<YOUR_BUCKET>"
path = "<YOUR_PATH>"

df = pl.scan_parquet(f"s3://{bucket}/{path}").filter(pl.col("id") < 100).select("id","value")
# --8<-- [end:scan_parquet_query]


"""
