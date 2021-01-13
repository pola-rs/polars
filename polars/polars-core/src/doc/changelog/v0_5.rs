//! # Changelog v0.5
//!
//! * `DataFrame.column` returns `Result<_>` **breaking change**.
//! * Define idiomatic way to do inplace operations on a `DataFrame` with `apply`, `may_apply` and `ChunkSet`
//! * `ChunkSet` Trait.
//! * `Groupby` aggregations can be done on a selection of multiple columns.
//! * `Groupby` operation can be done on multiple keys.
//! * `Groupby` `first` operation.
//! * `Pivot` operation.
//! * Random access to `ChunkedArray` types via `.get` and `.get_unchecked`.
//!
