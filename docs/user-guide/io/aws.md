# AWS

--8<-- "docs/_build/snippets/under_construction.md"

To read from or write to an AWS bucket, additional dependencies are needed in Rust:

=== ":fontawesome-brands-rust: Rust"

```shell
$ cargo add aws_sdk_s3 aws_config tokio --features tokio/full
```

In the next few snippets we'll demonstrate interacting with a `Parquet` file
located on an AWS bucket.

## Read

Load a `.parquet` file using:

{{code_block('user-guide/io/aws','bucket',[])}}
