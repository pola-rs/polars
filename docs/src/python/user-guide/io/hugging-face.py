# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:scan_iris_csv]
print(pl.scan_csv("hf://datasets/nameexhaustion/polars-docs/iris.csv").collect())
# --8<-- [end:scan_iris_csv]

# --8<-- [start:scan_iris_ndjson]
print(pl.scan_ndjson("hf://datasets/nameexhaustion/polars-docs/iris.jsonl").collect())
# --8<-- [end:scan_iris_ndjson]

# --8<-- [start:scan_parquet_hive]
print(pl.scan_parquet("hf://datasets/nameexhaustion/polars-docs/hive_dates/").collect())
# --8<-- [end:scan_parquet_hive]

# --8<-- [start:scan_ipc]
print(pl.scan_ipc("hf://spaces/nameexhaustion/polars-docs/orders.feather").collect())
# --8<-- [end:scan_ipc]
