# Polars IO

`polars-io` is a submodule of polars-lazy that provides IO functionality.

## Repository

The source code for this crate can be found on GitHub: [https://github.com/pola-rs/polars](https://github.com/pola-rs/polars)

## Features

| Feature           | Description                                          |
| ----------------- | ---------------------------------------------------- |
| json              | reading & writing for JSON files.                    |
| parquet           | reading & writing for Parquet files.                 |
| ipc               | reading & writing for IPC/Arrow files.               |
| ipc_streaming     | Arrow's streaming IPC file parsing support           |
| avro              | Arrow's Avro parsing support                         |
| csv               | reading & writing for CSV files                      |
| decompress        | decompression support                                |
| decompress-fast   | faster decompression                                 |
| dtype-categorical | `Categorical` data type                              |
| dtype-date        | `Date` data type                                     |
| dtype-datetime    | `Datetime` data type                                 |
| dtype-time        | `Time` data type.                                    |
| dtype-struct      | `Struct` data type                                   |
| timezones         | timezone parsing via chrono-tz                       |
| fmt               | pretty formatting of dataframes                      |
| lazy              | Reserved for future use.                             |
| async             | async support.                                       |
| aws               | reading from AWS S3 object store.                    |
| azure             | reading from Azure Blob Storage.                     |
| gcp               | reading from Google Cloud Storage.                   |
| partition         | partitioned writing                                  |
| temporal          | temporal data types: `Datetime`, `Date`, and `Time`. |
| simd              | Reserved for future use.                             |
| private           | private APIs. <sup>[1](#footnote1)</sup>             |

<sup><a name="footnote1">1</a></sup> Private APIs in `polars` are not intended for public use and may change without notice. Use at your own risk.
