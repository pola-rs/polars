"""
# --8<-- [start:read]
import polars as pl
from google.cloud import bigquery

client = bigquery.Client()

# Perform a query.
QUERY = (
    'SELECT name FROM `bigquery-public-data.usa_names.usa_1910_2013` '
    'WHERE state = "TX" '
    'LIMIT 100')
query_job = client.query(QUERY)  # API request
rows = query_job.result()  # Waits for query to finish

df = pl.from_arrow(rows.to_arrow())
# --8<-- [end:read]

# --8<-- [start:write]
from google.cloud import bigquery

client = bigquery.Client()

# Write DataFrame to stream as parquet file; does not hit disk
with io.BytesIO() as stream:
    df.write_parquet(stream)
    stream.seek(0)
    parquet_options = bigquery.ParquetOptions()
    parquet_options.enable_list_inference = True
    job = client.load_table_from_file(
        stream,
        destination='tablename',
        project='projectname',
        job_config=bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            parquet_options=parquet_options,
        ),
    )
job.result()  # Waits for the job to complete
# --8<-- [end:write]
"""


"""
# --8<-- [start:scan]
import polars as pl

df = pl.scan_bigquery(
    'bigquery-public-data.usa_names.usa_1910_2013', 
    billing_project_id="swast-scratch",
)
df.filter(
    (pl.col("state") == "TX")
    & (pl.col('year') == 2000)
).select(
    pl.col("name"),
    pl.col("number"),
).collect()
# --8<-- [end:scan]
"""
