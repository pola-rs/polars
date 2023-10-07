# Parquet

Loading or writing [`Parquet` files](https://parquet.apache.org/) is lightning fast.
`Pandas` uses [`PyArrow`](https://arrow.apache.org/docs/python/) -`Python` bindings
exposed by `Arrow`- to load `Parquet` files into memory, but it has to copy that data into
`Pandas` memory. With `Polars` there is no extra cost due to
copying as we read `Parquet` directly into `Arrow` memory and _keep it there_.

## Read

{{code_block('user-guide/io/parquet','read',['read_parquet'])}}

## Write

{{code_block('user-guide/io/parquet','write',['write_parquet'])}}

## Scan

`Polars` allows you to _scan_ a `Parquet` input. Scanning delays the actual parsing of the
file and instead returns a lazy computation holder called a `LazyFrame`.

{{code_block('user-guide/io/parquet','scan',['scan_parquet'])}}

If you want to know why this is desirable, you can read more about those `Polars` optimizations [here](../concepts/lazy-vs-eager.md).
