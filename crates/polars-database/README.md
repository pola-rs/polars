# polars-database

Native database scan plugin for [Polars](https://github.com/pola-rs/polars), released to PyPI as
`polars-database` (Python module `polars_database`). It implements `polars.scan_database`: dialects,
SQL pushdown, partitioning, driver resolution and the ADBC and Python driver backends.

Polars calls the plugin directly from its engine through the versioned FFI contracts in
[`polars-io-ext-ffi`](https://github.com/pola-rs/polars/tree/main/crates/polars-io-ext-ffi)
(`lazyframe_resolver.v1`, `file_reader.v1`); compatibility is decided by the plugin IDs the module
exports, not by the package version. Polars also calls a few Python functions of the module
(`scan_resolver`, `find_adbc_driver`, `driver_statuses`, `dialect_from_scheme`); they stay
compatible within a major version, which is what Polars' `polars-database >= x, < next major`
requirement relies on.

## Layout

This crate is the Rust side of the plugin, in the
[Polars repository](https://github.com/pola-rs/polars): an `rlib` whose pyo3 module
(`polars_database._lib`) is behind the `python` feature. The PyPI package `polars-database` is the
maturin project in [`py-polars/polars-database`](../../py-polars/polars-database): the Python
sources of the module and a one-line `cdylib` crate that exports the module from this crate. The
plugin's Python tests are in `py-polars/tests/unit/io/database`.

## Development

From the Polars root:

```bash
make -C py-polars build-polars-database  # build and install into the Polars venv
make -C py-polars test                   # also builds the plugin
```

The plugin's tests run with the rest of Polars' Python tests, from `py-polars`:

```bash
../.venv/bin/pytest tests/unit/io/database -n auto
```

Tests against database servers skip unless their `POLARS_TEST_*_URI` variables are set (for example
`POLARS_TEST_POSTGRES_URI=postgresql://user:password@host:5432/db`; MySQL, MariaDB, Oracle and
ClickHouse have their own). The load tests also need `POLARS_TEST_DATABASE_LOAD=1` and run with
`-m slow`.

## Releasing

The version of the package is the same in `project.version` of
`py-polars/polars-database/pyproject.toml`, in the `version` of
`py-polars/polars-database/Cargo.toml`, and in the `version` of this crate's `Cargo.toml` (the
module's `__version__`). Releases are tagged `polars-database-<version>` in the Polars repository.

1. Bump the version in those three files and push a tag `polars-database-<version>` for the commit.
2. Run Polars' `release-polars-database` workflow with that tag. It builds the sdist and abi3 wheels
   (one per target) from that commit and publishes them to PyPI.
3. To make Polars require the release, update the
   `polars-database >= <version>, < <next major version>` requirement in the `database` extra of
   Polars' `py-polars/pyproject.toml`.

The PyPI project page is `py-polars/polars-database/README.pypi.md`.
