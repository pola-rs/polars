"""
# --8<-- [start:read_parquet]
import polars as pl

source = "s3://bucket/*.parquet"

df = pl.read_parquet(source)
# --8<-- [end:read_parquet]

# --8<-- [start:scan_parquet]
import polars as pl

source = "s3://bucket/*.parquet"

storage_options = {
    "aws_access_key_id": "<secret>",
    "aws_secret_access_key": "<secret>",
    "aws_region": "us-east-1",
}
df = pl.scan_parquet(source, storage_options=storage_options)
# --8<-- [end:scan_parquet]

# --8<-- [start:scan_parquet_query]
import polars as pl

source = "s3://bucket/*.parquet"


df = pl.scan_parquet(source).filter(pl.col("id") < 100).select("id","value").collect()
# --8<-- [end:scan_parquet_query]

# --8<-- [start:scan_pyarrow_dataset]
import polars as pl
import pyarrow.dataset as ds

dset = ds.dataset("s3://my-partitioned-folder/", format="parquet")
(
    pl.scan_pyarrow_dataset(dset)
    .filter(pl.col("foo") == "a")
    .select(["foo", "bar"])
    .collect()
)
# --8<-- [end:scan_pyarrow_dataset]

# --8<-- [start:write_parquet]

import polars as pl
import s3fs

df = pl.DataFrame({
    "foo": ["a", "b", "c", "d", "d"],
    "bar": [1, 2, 3, 4, 5],
})

fs = s3fs.S3FileSystem()
destination = "s3://bucket/my_file.parquet"

# write parquet
with fs.open(destination, mode='wb') as f:
    df.write_parquet(f)
# --8<-- [end:write_parquet]
"""
