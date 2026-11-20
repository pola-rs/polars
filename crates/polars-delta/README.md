# polars-delta

Native Delta Lake scan planning for Polars. This crate builds the `polars_delta` Python extension
and exports the `polars.io_plugin.delta.v1` planning contract. The same Rust crate exposes
`polars_delta::codec` for native deletion-vector decoding without planner features.

The planner reads Delta logs and checkpoints through host-provided storage, and returns snapshot
schema, active files, partition values, statistics and deletion-vector descriptors. Polars reads the
data files and applies predicates and deletion vectors. Unsupported reader features produce an
explicit unsupported-feature result; malformed metadata is an error.

## Workspace setup

This crate lives at `crates/polars-delta` in the Polars repository and is a workspace member; it
inherits workspace metadata, lints and dependency versions. Its package version is independent of
Polars.

The Rust package and library are `polars-delta` and `polars_delta` (an `rlib`). The Python
distribution and import module are `polars-delta` and `polars_delta`, built from
`py-polars/polars-delta`: its `polars-delta-python` crate enables the `python` Cargo feature of this
crate to build the extension module. `make build-polars-delta` installs it with maturin.

## Integration

A compatible Polars build selects the installed plugin automatically. Set
`POLARS_DELTA_PLANNER=plugin` to require it, or `POLARS_DELTA_PLANNER=deltalake` to use the
compatibility planner. Automatic mode may fall back for unsupported features; requiring the plugin
reports them directly. `use_pyarrow=True` and writes still use deltalake.

The module advertises supported contracts through `_polars_io_plugin_ids`; `_capsule` returns the
selected contract. Contract compatibility, rather than the package version alone, determines whether
a Polars build can load the plugin.

This development implementation requires a matching Polars build. The host and plugin must be
released together with compatible contracts; installing this development source alone does not add
plugin support to older Polars builds.

## Feature separation

Default features are empty. The always-available `polars_delta::codec` module validates Delta
deletion-vector framing and decodes portable Roaring data into packed keep bits. Its only direct
non-standard-library dependency is `crc32fast`. The host depends on `polars-delta` with default
features disabled; it keeps storage credentials, lazy range reads, pruning and reader scheduling.
The Python planning plugin remains optional at runtime.

The `planner` feature enables snapshot planning and its Arrow/Parquet, storage-interface and other
planning dependencies. The `python` feature includes `planner` and the Python extension; maturin
enables it through `pyproject.toml`. A workspace build selecting the extension can unify features
with the host dependency, so check the minimal build independently:

```bash
cargo test -p polars-delta --no-default-features
cargo test -p polars-delta --no-default-features --features planner
```

The codec caller supplies the maximum row bound and performs storage reads. Validate the file header
before fetching a stored frame; inline/frame decoding validates sizes, CRC where present,
cardinality and portable Roaring contents. Results own packed least-significant-bit-first keep bits
and a logical length; rows beyond that length are kept. The serialized input bound is 512 MiB.
Per-mask bounds do not bound aggregate memory across readers.

This package retains `publish = false`. Disabling planner features does not remove optional planner
dependencies from Cargo registry resolution, including the unpublished host FFI package. Rust
registry publication needs a separate release design; Python wheels and source distributions are
built from the matching workspace sources.
