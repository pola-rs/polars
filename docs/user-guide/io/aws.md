# AWS

--8<-- "docs/_build/snippets/under_construction.md"

To read from or write to an AWS bucket, additional dependencies may be needed:
=== ":fontawesome-brands-python: Python"

```shell
$ pip install fsspec
```

=== ":fontawesome-brands-rust: Rust"

```shell
$ cargo add aws_sdk_s3 aws_config tokio --features tokio/full
```

## Read

We can read a `.parquet` file in eager mode from an AWS bucket:

{{code_block('user-guide/io/aws','read_parquet',[])}}

This downloads the file to a temporary location and reads it from there.

## Scan

We can scan a `.parquet` file in lazy mode from an AWS bucket:

{{code_block('user-guide/io/aws','scan_parquet',[])}}

This creates a `LazyFrame` without downloading the file. We have access to file metadata such as the schema.

If we create a lazy query with predicate and projection pushdowns the query optimiser will apply them before the file is downloaded

{{code_block('user-guide/io/aws','scan_parquet_query',[])}}

