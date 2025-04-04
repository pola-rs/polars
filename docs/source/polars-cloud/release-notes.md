# Release Notes

## Polars Cloud 0.0.6 (2025-03-31)

This version of Polars Cloud is only compatible with the 1.26.x releases of Polars Open Source.

### Enhancements

- Added support for [Polars IO plugins](https://docs.pola.rs/user-guide/plugins/io_plugins/).
  - These allow you to register different file formats as sources to the Polars engines which allows
    you to benefit from optimizations like projection pushdown, predicate pushdown, early stopping
    and support of our streaming engine.
- Added `.show()` for remote queries
  - You can now call `.remote().show()` instead of `remote().limit().collect().collect()`
- All Polars features that rely on external Python dependencies are now available in Polars Cloud
  such as:
  - `scan_iceberg()`
  - `scan_delta()`
  - `read_excel()`
  - `read_database()`

### Bug fixes

- Fixed an issue where specifying CPU/memory minimums could select an incompatible legacy EC2
  instance.
- The API now returns clear errors when trying to start a query on an uninitialized workspace.
- Fixed a 404 error when opening a query overview page in a workspace different from the currently
  selected workspace.
- Viewing another workspace members compute detail page no longer shows a blank screen.
