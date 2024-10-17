# parquet-format-safe

This crate contains an implementation of Thrift and generated Rust code
associated to Parquet's thrift definition.

- supports `sync` and `async` read API
- supports `sync` and `async` write API
- the write API returns the number of written bytes
- the read API is panic free
- the read API has a bound on the maximum number of possible bytes read, to avoid OOM.

The Rust generated code is done by a fork of thrift's compiler, available
at <https://github.com/coastalwhite/thrift/tree/safe>.

## Usage

To regenerate the thrift format implementation in Rust.

```bash
nix-shell
generate_parquet_format
mv parquet.rs src/parquet_format.rs
```
