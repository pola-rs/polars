"""
# --8<-- [start:read_parquet]
import polars as pl

source = "s3://bucket/*.parquet"

df = pl.read_parquet(source)
# --8<-- [end:read_parquet]

# --8<-- [start:scan_parquet_query]
import polars as pl

source = "s3://bucket/*.parquet"

df = pl.scan_parquet(source).filter(pl.col("id") < 100).select("id","value").collect()
# --8<-- [end:scan_parquet_query]


# --8<-- [start:scan_parquet_storage_options_aws]
import polars as pl

source = "s3://bucket/*.parquet"

storage_options = {
    "aws_access_key_id": "<secret>",
    "aws_secret_access_key": "<secret>",
    "aws_region": "us-east-1",
}
df = pl.scan_parquet(source, storage_options=storage_options).collect()
# --8<-- [end:scan_parquet_storage_options_aws]

# --8<-- [start:credential_provider_class]
lf = pl.scan_parquet(
    "s3://.../...",
    credential_provider=pl.CredentialProviderAWS(
        profile_name="..."
        assume_role={
            "RoleArn": f"...",
            "RoleSessionName": "...",
        }
    ),
)

df = lf.collect()
# --8<-- [end:credential_provider_class]

# --8<-- [start:credential_provider_custom_func]
def get_credentials() -> pl.CredentialProviderFunctionReturn:
    expiry = None

    return {
        "aws_access_key_id": "...",
        "aws_secret_access_key": "...",
        "aws_session_token": "...",
    }, expiry


lf = pl.scan_parquet(
    "s3://.../...",
    credential_provider=get_credentials,
)

df = lf.collect()
# --8<-- [end:credential_provider_custom_func]

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
