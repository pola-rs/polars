# Standalone expression plugin

This example demonstrates that the Polars expression-plugin runtime can be compiled as a `cdylib`
without PyO3 or a Python extension build.

For the documented and supported Python plugin workflow, use `pyo3-polars` and see the Polars
expression-plugin user guide.

Build it with:

```bash
cargo build --manifest-path Cargo.toml
```
