# Polars SQL

`polars-sql` is a Rust crate that provides an SQL transpiler for Polars. It can convert SQL queries to Polars logical plans.

## Usage

To use `polars-sql`, add it as a dependency to your Rust project's `Cargo.toml` file:

```toml
[dependencies]
polars-sql = "0.29.0"
```

You can then import the crate in your Rust code using:

```rust
use polars_sql::*;
```

## Features

`polars-sql` has the following features:

| Feature | Description                                       |
| ------- | ------------------------------------------------- |
| csv     | Enables support for CSV files.                    |
| json    | Enables support for JSON files.                   |
| default | The default feature set for Polars.               |
| ipc     | Enables support for IPC/Arrow files.              |
| parquet | Enables support for Parquet files.                |
| private | Enables private APIs. <sup> [1](#footnote1)</sup> |

<sup><a name="footnote1">1</a></sup> Private APIs in `polars` are not intended for public use and may change without notice. Use at your own risk.
