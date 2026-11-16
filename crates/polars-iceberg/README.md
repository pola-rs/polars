# polars-iceberg

Native Iceberg scan planning plugin for [Polars](https://github.com/pola-rs/polars), released to
PyPI as `polars-iceberg` (Python module `polars_iceberg`).

Polars loads the plugin through the versioned FFI contracts in
[`polars-io-ext-ffi`](https://github.com/pola-rs/polars/tree/main/crates/polars-io-ext-ffi);
compatibility is decided by the plugin IDs the module exports, not by the package version.

## Development

This crate lives at `crates/polars-iceberg` in the Polars repository and is a workspace member;
it inherits workspace metadata, lints and dependency versions. Its package version is independent
of Polars, and Polars' default workspace builds exclude it.

The Rust package and library are `polars-iceberg` and `polars_iceberg` (an `rlib`). The Python
distribution and import module are `polars-iceberg` and `polars_iceberg`, built from
`py-polars/polars-iceberg`: its `polars-iceberg-python` crate enables the `python`
Cargo feature of this crate to build the extension module. From the Polars root:

```bash
make -C py-polars build-polars-iceberg  # build and install into the Polars venv
make -C py-polars test                  # also builds the plugin
```

## Releasing

The released version is the `version` of `py-polars/polars-iceberg/Cargo.toml` (the PyPI
package's version), which must equal the `version` of this crate's `Cargo.toml` (the module's
`__version__`).

1. Bump both versions and merge, then push the tag `polars-iceberg-<version>` for that commit.
2. Run the `release-polars-iceberg` workflow with that tag. It builds the sdist and abi3 wheels
   (one per target) and publishes them to PyPI.
3. To make Polars require the release, update the `polars-iceberg >= <version>, < <next major
   version>` requirement in the `iceberg` extra of `py-polars/pyproject.toml`. Polars'
   Python release workflow runs its test suite against that lower bound installed from PyPI.

The PyPI project page is `py-polars/polars-iceberg/README.pypi.md`.
