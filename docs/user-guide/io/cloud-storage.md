# Cloud storage

Polars can read and write to AWS S3, Azure Blob Storage and Google Cloud Storage. The API is the same for all three storage providers.

To read from cloud storage, additional dependencies may be needed depending on the use case and cloud storage provider:

=== ":fontawesome-brands-python: Python"

    ```shell
    $ pip install fsspec s3fs adlfs gcsfs
    ```

=== ":fontawesome-brands-rust: Rust"

    ```shell
    $ cargo add aws_sdk_s3 aws_config tokio --features tokio/full
    ```

## Reading from cloud storage

Polars can read a CSV, IPC or Parquet file in eager mode from cloud storage.

{{code_block('user-guide/io/cloud-storage','read_parquet',['read_parquet','read_csv','read_ipc'])}}

This eager query downloads the file to a buffer in memory and creates a `DataFrame` from there. Polars uses `fsspec` to manage this download internally for all cloud storage providers.

## Scanning from cloud storage with query optimisation

Polars can scan a Parquet file in lazy mode from cloud storage. We may need to provide further details beyond the source url such as authentication details or storage region. Polars looks for these as environment variables but we can also do this manually by passing a `dict` as the `storage_options` argument.

{{code_block('user-guide/io/cloud-storage','scan_parquet',['scan_parquet'])}}

This query creates a `LazyFrame` without downloading the file. In the `LazyFrame` we have access to file metadata such as the schema. Polars uses the `object_store.rs` library internally to manage the interface with the cloud storage providers and so no extra dependencies are required in Python to scan a cloud Parquet file.

If we create a lazy query with [predicate and projection pushdowns](../lazy/optimizations.md), the query optimizer will apply them before the file is downloaded. This can significantly reduce the amount of data that needs to be downloaded. The query evaluation is triggered by calling `collect`.

{{code_block('user-guide/io/cloud-storage','scan_parquet_query',[])}}

## Scanning with PyArrow

We can also scan from cloud storage using PyArrow. This is particularly useful for partitioned datasets such as Hive partitioning.

We first create a PyArrow dataset and then create a `LazyFrame` from the dataset.

{{code_block('user-guide/io/cloud-storage','scan_pyarrow_dataset',['scan_pyarrow_dataset'])}}

## Writing to cloud storage

We can write a `DataFrame` to cloud storage in Python using s3fs for S3, adlfs for Azure Blob Storage and gcsfs for Google Cloud Storage. In this example, we write a Parquet file to S3.

{{code_block('user-guide/io/cloud-storage','write_parquet',['write_parquet'])}}
