# polars-sql

`polars-sql` is a sub-crate of the [Polars](https://crates.io/crates/polars) library, offering a SQL transpiler. It allows for SQL query conversion to Polars logical plans.

## Usage

To use `polars-sql`, add it as a dependency to your Rust project's `Cargo.toml` file:

```toml
[dependencies]
polars-sql = "0.30.0"
```

You can then import the crate in your Rust code using:

```rust
use polars_sql::*;
```

**Important Note**: This crate is **not intended for external usage**. Please refer to the main [Polars crate](https://crates.io/crates/polars) for intended usage.
