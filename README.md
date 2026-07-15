<h1 align="center">
  <a href="https://pola.rs">
    <img src="https://raw.githubusercontent.com/pola-rs/polars-static/master/banner/polars_github_banner.svg" alt="Polars logo">
  </a>
</h1>

<div align="center">
  <a href="https://crates.io/crates/polars">
    <img src="https://img.shields.io/crates/v/polars.svg" alt="crates.io Latest Release"/>
  </a>
  <a href="https://pypi.org/project/polars/">
    <img src="https://img.shields.io/pypi/v/polars.svg" alt="PyPi Latest Release"/>
  </a>
  <a href="https://www.npmjs.com/package/nodejs-polars">
    <img src="https://img.shields.io/npm/v/nodejs-polars.svg" alt="NPM Latest Release"/>
  </a>
  <a href="https://community.r-multiverse.org/polars">
    <img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fcommunity.r-multiverse.org%2Fapi%2Fpackages%2Fpolars&query=%24.Version&label=r-multiverse" alt="R-multiverse Latest Release"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.7697217">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7697217.svg" alt="DOI Latest Release"/>
  </a>
</div>

<p align="center">
  <b>Documentation</b>:
  <a href="https://docs.pola.rs/api/python/stable/reference/index.html">Python</a>
  -
  <a href="https://docs.rs/polars/latest/polars/">Rust</a>
  -
  <a href="https://pola-rs.github.io/nodejs-polars/index.html">Node.js</a>
  -
  <a href="https://pola-rs.github.io/r-polars/index.html">R</a>
  <b>Agents</b>:
  <a href="https://github.com/polars-inc/skills/tree/main/polars">Skill</a>
  -
  <a href="https://docs.pola.rs/user-guide/misc/polars_llms/">MCP</a>
  |
  <a href="https://docs.pola.rs/">User guide</a>
  |
  <a href="https://discord.gg/4UfP5cfBE7">Discord</a>
</p>

## Polars: Extremely fast Query Engine for DataFrames

Polars is an analytical query engine for DataFrames, written in Rust. It is designed to be fast,
easy to use and expressive. Key features are:

- **Fast** — written from the ground up in Rust with multi-threaded, vectorized (SIMD) execution
- **Lazy & eager execution** — with query optimization out of the box
- **Larger-than-RAM** — the streaming engine processes datasets that don't fit in memory
- **Expressive API** — compose complex queries with powerful expressions
- **Extensible** — write your own
  [I/O plugins and expression plugins](https://docs.pola.rs/user-guide/plugins/)
- **Multi-language** — front ends for Python, Rust, Node.js, R, and SQL
- **GPU support** — optionally accelerate queries on NVIDIA GPUs
- **Interoperable** — uses the
  [Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html) for zero-copy
  data sharing

To learn more, read the [user guide](https://docs.pola.rs/).

## Polars in action

Queries are composed from expressions. This lazy query is optimized and runs in parallel across all
your cores:

```python
import polars as pl

df = (
    pl.scan_parquet("orders.parquet")
    .filter(pl.col("status") == "shipped")
    .group_by("customer_id")
    .agg(
        pl.col("amount").sum().alias("total"),
        pl.len().alias("n_orders"),
    )
    .sort("total", descending=True)
    .collect()
)
```

The same query handles larger-than-RAM data with `.collect(engine="streaming")`.

## Performance 🚀🚀

### Blazingly fast

Polars is very fast. In fact, it is one of the best performing solutions available. See the
[PDS-H benchmarks](https://www.pola.rs/benchmarks.html) results.

### Handles larger-than-RAM data

If you have data that does not fit into memory, Polars' query engine is able to process your query
(or parts of your query) in a streaming fashion. This drastically reduces memory requirements, so
you might be able to process your 250GB dataset on your laptop. Collect with
`collect(engine='streaming')` to run the query streaming.

## Setup

### Python

Install the latest Polars version with:

```sh
pip install polars
```

See the [User Guide](https://docs.pola.rs/user-guide/installation/#feature-flags) for more details
on optional dependencies

<details>
<summary><b>Compile Polars from source</b></summary>

If you want a bleeding edge release or maximal performance you should compile Polars from source.

This can be done by going through the following steps in sequence:

1. Install the latest [Rust compiler](https://www.rust-lang.org/tools/install)
2. Install [maturin](https://maturin.rs/): `pip install maturin`
3. `cd py-polars` and choose one of the following:
   - `make build`, slow binary with debug assertions and symbols, fast compile times
   - `make build-release`, fast binary without debug assertions, minimal debug symbols, long compile
     times
   - `make build-nodebug-release`, same as build-release but without any debug symbols, slightly
     faster to compile
   - `make build-debug-release`, same as build-release but with full debug symbols, slightly slower
     to compile
   - `make build-dist-release`, fastest binary, extreme compile times

By default the binary is compiled with optimizations turned on for a modern CPU. Specify `LTS_CPU=1`
with the command if your CPU is older and does not support e.g. AVX2.

Note that the Rust crate implementing the Python bindings is called `py-polars` to distinguish from
the wrapped Rust crate `polars` itself. However, both the Python package and the Python module are
named `polars`, so you can `pip install polars` and `import polars`.

</details>

Check the [Installation chapter in the user guide](https://docs.pola.rs/user-guide/installation/)
for more advanced installations. For example when you expect more than 2^32 (~4.2 billion) rows, run
on an old CPU (e.g. dating from before 2011), or on an `x86-64` build of Python on Apple Silicon
under Rosetta

## Contributing

Want to contribute? Read our [contributing guide](https://docs.pola.rs/development/contributing/).
You can [join the Polars Discord server](https://discord.gg/4UfP5cfBE7) for any help along the way.

## Distributed Polars

Running into hardware limitations executing your queries? Read how you can
[horizontally scale your Polars query on a cluster](https://docs.pola.rs/polars-cloud/).

## License

Polars is licensed under the [MIT License](LICENSE) (SPDX: `MIT`).
