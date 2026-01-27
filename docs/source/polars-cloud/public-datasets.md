# Public datasets

# Public datasets

Start experimenting with Polars Cloud immediately using our curated public datasets. These datasets
span different scale factors, letting you test performance across various data sizes—from small
exploratory queries to large-scale processing workloads.

## Available datasets

**PDSH** - derived from TPC-H benchmark Standard analytical queries for testing joins, aggregations,
and filtering operations. Queries available in the
[Polars benchmark repository](https://github.com/pola-rs/polars-benchmark/tree/main/queries/polars).

**PDSDS** - derived from TPC-DS benchmark Decision support dataset designed for complex analytical
workloads.

**NYC Taxi** - [source: NYC.gov](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
Real-world transportation data with temporal patterns and geospatial dimensions.

## Usage

Access any dataset directly from your Polars code and execute in Polars Cloud:

```python
data = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/{dataset}/{scale_factor/year}/",
    storage_options={"request_payer": "true"}
)
query = data.select().remote(ctx).execute()
```

**Note:** These buckets use
[AWS Requester Pays](https://docs.aws.amazon.com/AmazonS3/latest/userguide/RequesterPaysBuckets.html),
meaning you pay only for pays the cost of the request and the data download from the bucket. The
storage costs are covered.

## Dataset URLs

All datasets are hosted in AWS region `us-east-2` and use Requester Pays buckets.

### PDSH (TPC-H derived)

| Scale Factor | Size   | URL Pattern                                                             | Format       |
| ------------ | ------ | ----------------------------------------------------------------------- | ------------ |
| SF10         | ~10GB  | `s3://polars-cloud-samples-us-east-2-prd/pdsh/sf10/{filename}.parquet`  | Single files |
| SF100        | ~100GB | `s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/{table}/_.parquet`  | Partitioned  |
| SF1000       | ~1TB   | `s3://polars-cloud-samples-us-east-2-prd/pdsh/sf1000/{table}/_.parquet` | Partitioned  |

#### Example

```python
data = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf10/lineitem.parquet",
    storage_options={"request_payer": "true"}
)

partitioned_data = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf100/lineitem/*.parquet",
    storage_options={"request_payer": "true"}
)
```

### PDSDS (TPC-DS derived)

| Scale Factor | Size   | URL Pattern                                                              |
| ------------ | ------ | ------------------------------------------------------------------------ |
| SF1          | ~1GB   | `s3://polars-cloud-samples-us-east-2-prd/pdsds/sf1/{filename}.parquet`   |
| SF10         | ~10GB  | `s3://polars-cloud-samples-us-east-2-prd/pdsds/sf10/{filename}.parquet`  |
| SF100        | ~100GB | `s3://polars-cloud-samples-us-east-2-prd/pdsds/sf100/{filename}.parquet` |
| SF300        | ~300GB | `s3://polars-cloud-samples-us-east-2-prd/pdsds/sf300/{filename}.parquet` |

#### Example

```python
data = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/pdsh/sf10/store_sales.parquet",
    storage_options={"request_payer": "true"}
)
```

### NYC Taxi

| Year | URL Pattern                                                            |
| ---- | ---------------------------------------------------------------------- |
| 2023 | `s3://polars-cloud-samples-us-east-2-prd/taxi/2023/{filename}.parquet` |
| 2024 | `s3://polars-cloud-samples-us-east-2-prd/taxi/2024/{filename}.parquet` |

#### Example

```python
data = pl.scan_parquet(
    "s3://polars-cloud-samples-us-east-2-prd/taxi/2024/yellow_tripdata_2024-01.parquet",
    storage_options={"request_payer": "true"}
)
```
