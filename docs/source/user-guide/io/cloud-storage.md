# Cloud storage

Polars can read and write to AWS S3, Azure Blob Storage and Google Cloud Storage. The API is the
same for all three storage providers.

To read from cloud storage, additional dependencies may be needed depending on the use case and
cloud storage provider:

=== ":fontawesome-brands-python: Python"

    ```shell
    $ pip install fsspec s3fs adlfs gcsfs
    ```

=== ":fontawesome-brands-rust: Rust"

    ```shell
    $ cargo add aws_sdk_s3 aws_config tokio --features tokio/full
    ```

## Reading from cloud storage

Polars supports reading Parquet, CSV, IPC and NDJSON files from cloud storage:

{{code_block('user-guide/io/cloud-storage','read_parquet',['read_parquet','read_csv','read_ipc'])}}

## Scanning from cloud storage with query optimisation

Using `pl.scan_*` functions to read from cloud storage can benefit from
[predicate and projection pushdowns](../lazy/optimizations.md), where the query optimizer will apply
them before the file is downloaded. This can significantly reduce the amount of data that needs to
be downloaded. The query evaluation is triggered by calling `collect`.

{{code_block('user-guide/io/cloud-storage','scan_parquet_query',[])}}

## Cloud authentication

Polars is able to automatically load default credential configurations for some cloud providers. For
cases when this does not happen, it is possible to manually configure the credentials for Polars to
use for authentication. This can be done in a few ways:

### Using `storage_options`:

- Credentials can be passed as configuration keys in a dict with the `storage_options` parameter:

{{code_block('user-guide/io/cloud-storage','scan_parquet_storage_options_aws',['scan_parquet'])}}

### Using one of the available `CredentialProvider*` utility classes

- There may be a utility class `pl.CredentialProvider*` that provides the required authentication
  functionality. For example, `pl.CredentialProviderAWS` supports selecting AWS profiles, as well as
  assuming an IAM role:

{{code_block('user-guide/io/cloud-storage','credential_provider_class',['scan_parquet',
'CredentialProviderAWS'])}}

### Using a custom `credential_provider` function

- Some environments may require custom authentication logic (e.g. AWS IAM role-chaining). For these
  cases a Python function can be provided for Polars to use to retrieve credentials:

{{code_block('user-guide/io/cloud-storage','credential_provider_custom_func',['scan_parquet'])}}

- Example for Azure:

{{code_block('user-guide/io/cloud-storage','credential_provider_custom_func_azure',['scan_parquet',
'CredentialProviderAzure'])}}

## Scanning with PyArrow

We can also scan from cloud storage using PyArrow. This is particularly useful for partitioned
datasets such as Hive partitioning.

We first create a PyArrow dataset and then create a `LazyFrame` from the dataset.

{{code_block('user-guide/io/cloud-storage','scan_pyarrow_dataset',['scan_pyarrow_dataset'])}}

## Writing to cloud storage

We can write a `DataFrame` to cloud storage in Python using s3fs for S3, adlfs for Azure Blob
Storage and gcsfs for Google Cloud Storage. In this example, we write a Parquet file to S3.

{{code_block('user-guide/io/cloud-storage','write_parquet',['write_parquet'])}}
