<h1 align="center">
  <img src="https://raw.githubusercontent.com/pola-rs/polars-static/master/banner/polars_github_banner.svg" alt="Polars logo">
  <br>
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
  <a href="https://rpolars.r-universe.dev">
    <img src="https://rpolars.r-universe.dev/badges/polars" alt="R-universe Latest Release"/>
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
  |
  <b>StackOverflow</b>:
  <a href="https://stackoverflow.com/questions/tagged/python-polars">Python</a>
  -
  <a href="https://stackoverflow.com/questions/tagged/rust-polars">Rust</a>
  -
  <a href="https://stackoverflow.com/questions/tagged/nodejs-polars">Node.js</a>
  -
  <a href="https://stackoverflow.com/questions/tagged/r-polars">R</a>
  |
  <a href="https://docs.pola.rs/">User guide</a>
  |
  <a href="https://discord.gg/4UfP5cfBE7">Discord</a>
</p>

## Polars: Blazingly fast DataFrames in Rust, Python, Node.js, R, and SQL

Polars is a DataFrame interface on top of an OLAP Query Engine implemented in Rust using
[Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html) as the memory model.

- Lazy | eager execution
- Multi-threaded
- SIMD
- Query optimization
- Powerful expression API
- Hybrid Streaming (larger-than-RAM datasets)
- Rust | Python | NodeJS | R | ...

To learn more, read the [user guide](https://docs.pola.rs/).

## Python

```python
>>> import polars as pl
>>> df = pl.DataFrame(
...     {
...         "A": [1, 2, 3, 4, 5],
...         "fruits": ["banana", "banana", "apple", "apple", "banana"],
...         "B": [5, 4, 3, 2, 1],
...         "cars": ["beetle", "audi", "beetle", "beetle", "beetle"],
...     }
... )

# embarrassingly parallel execution & very expressive query language
>>> df.sort("fruits").select(
...     "fruits",
...     "cars",
...     pl.lit("fruits").alias("literal_string_fruits"),
...     pl.col("B").filter(pl.col("cars") == "beetle").sum(),
...     pl.col("A").filter(pl.col("B") > 2).sum().over("cars").alias("sum_A_by_cars"),
...     pl.col("A").sum().over("fruits").alias("sum_A_by_fruits"),
...     pl.col("A").reverse().over("fruits").alias("rev_A_by_fruits"),
...     pl.col("A").sort_by("B").over("fruits").alias("sort_A_by_B_by_fruits"),
... )
shape: (5, 8)
┌──────────┬──────────┬──────────────┬─────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ fruits   ┆ cars     ┆ literal_stri ┆ B   ┆ sum_A_by_ca ┆ sum_A_by_fr ┆ rev_A_by_fr ┆ sort_A_by_B │
│ ---      ┆ ---      ┆ ng_fruits    ┆ --- ┆ rs          ┆ uits        ┆ uits        ┆ _by_fruits  │
│ str      ┆ str      ┆ ---          ┆ i64 ┆ ---         ┆ ---         ┆ ---         ┆ ---         │
│          ┆          ┆ str          ┆     ┆ i64         ┆ i64         ┆ i64         ┆ i64         │
╞══════════╪══════════╪══════════════╪═════╪═════════════╪═════════════╪═════════════╪═════════════╡
│ "apple"  ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 7           ┆ 4           ┆ 4           │
│ "apple"  ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 7           ┆ 3           ┆ 3           │
│ "banana" ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 8           ┆ 5           ┆ 5           │
│ "banana" ┆ "audi"   ┆ "fruits"     ┆ 11  ┆ 2           ┆ 8           ┆ 2           ┆ 2           │
│ "banana" ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 8           ┆ 1           ┆ 1           │
└──────────┴──────────┴──────────────┴─────┴─────────────┴─────────────┴─────────────┴─────────────┘
```

## SQL

```python
>>> df = pl.scan_csv("docs/data/iris.csv")
>>> ## OPTION 1
>>> # run SQL queries on frame-level
>>> df.sql("""
...	SELECT species,
...	  AVG(sepal_length) AS avg_sepal_length
...	FROM self
...	GROUP BY species
...	""").collect()
shape: (3, 2)
┌────────────┬──────────────────┐
│ species    ┆ avg_sepal_length │
│ ---        ┆ ---              │
│ str        ┆ f64              │
╞════════════╪══════════════════╡
│ Virginica  ┆ 6.588            │
│ Versicolor ┆ 5.936            │
│ Setosa     ┆ 5.006            │
└────────────┴──────────────────┘
>>> ## OPTION 2
>>> # use pl.sql() to operate on the global context
>>> df2 = pl.LazyFrame({
...    "species": ["Setosa", "Versicolor", "Virginica"],
...    "blooming_season": ["Spring", "Summer", "Fall"]
...})
>>> pl.sql("""
... SELECT df.species,
...     AVG(df.sepal_length) AS avg_sepal_length,
...     df2.blooming_season
... FROM df
... LEFT JOIN df2 ON df.species = df2.species
... GROUP BY df.species, df2.blooming_season
... """).collect()
```

SQL commands can also be run directly from your terminal using the Polars CLI:

```bash
# run an inline SQL query
> polars -c "SELECT species, AVG(sepal_length) AS avg_sepal_length, AVG(sepal_width) AS avg_sepal_width FROM read_csv('docs/data/iris.csv') GROUP BY species;"

# run interactively
> polars
Polars CLI v0.3.0
Type .help for help.

> SELECT species, AVG(sepal_length) AS avg_sepal_length, AVG(sepal_width) AS avg_sepal_width FROM read_csv('docs/data/iris.csv') GROUP BY species;
```

Refer to the [Polars CLI repository](https://github.com/pola-rs/polars-cli) for more information.

## Performance 🚀🚀

### Blazingly fast

Polars is very fast. In fact, it is one of the best performing solutions available. See the [TPC-H benchmarks](https://www.pola.rs/benchmarks.html) results.

### Lightweight

Polars is also very lightweight. It comes with zero required dependencies, and this shows in the import times:

- polars: 70ms
- numpy: 104ms
- pandas: 520ms

### Handles larger-than-RAM data

If you have data that does not fit into memory, Polars' query engine is able to process your query (or parts of your query) in a streaming fashion.
This drastically reduces memory requirements, so you might be able to process your 250GB dataset on your laptop.
Collect with `collect(streaming=True)` to run the query streaming.
(This might be a little slower, but it is still very fast!)

## Setup

### Python

Install the latest Polars version with:

```sh
pip install polars
```

We also have a conda package (`conda install -c conda-forge polars`), however pip is the preferred way to install Polars.

Install Polars with all optional dependencies.

```sh
pip install 'polars[all]'
```

You can also install a subset of all optional dependencies.

```sh
pip install 'polars[numpy,pandas,pyarrow]'
```

See the [User Guide](https://docs.pola.rs/user-guide/installation/#feature-flags) for more details on optional dependencies

To see the current Polars version and a full list of its optional dependencies, run:

```python
pl.show_versions()
```

Releases happen quite often (weekly / every few days) at the moment, so updating Polars regularly to get the latest bugfixes / features might not be a bad idea.

### Rust

You can take latest release from `crates.io`, or if you want to use the latest features / performance
improvements point to the `main` branch of this repo.

```toml
polars = { git = "https://github.com/pola-rs/polars", rev = "<optional git tag>" }
```

Requires Rust version `>=1.80`.

## Contributing

Want to contribute? Read our [contributing guide](https://docs.pola.rs/development/contributing/).

## Python: compile Polars from source

If you want a bleeding edge release or maximal performance you should compile Polars from source.

This can be done by going through the following steps in sequence:

1. Install the latest [Rust compiler](https://www.rust-lang.org/tools/install)
2. Install [maturin](https://maturin.rs/): `pip install maturin`
3. `cd py-polars` and choose one of the following:
   - `make build-release`, fastest binary, very long compile times
   - `make build-opt`, fast binary with debug symbols, long compile times
   - `make build-debug-opt`, medium-speed binary with debug assertions and symbols, medium compile times
   - `make build`, slow binary with debug assertions and symbols, fast compile times

   Append `-native` (e.g. `make build-release-native`) to enable further optimizations specific to
   your CPU. This produces a non-portable binary/wheel however.

Note that the Rust crate implementing the Python bindings is called `py-polars` to distinguish from the wrapped
Rust crate `polars` itself. However, both the Python package and the Python module are named `polars`, so you
can `pip install polars` and `import polars`.

## Using custom Rust functions in Python

Extending Polars with UDFs compiled in Rust is easy. We expose PyO3 extensions for `DataFrame` and `Series`
data structures. See more in https://github.com/pola-rs/pyo3-polars.

## Going big...

Do you expect more than 2^32 (~4.2 billion) rows? Compile Polars with the `bigidx` feature
flag or, for Python users, install `pip install polars-u64-idx`.

Don't use this unless you hit the row boundary as the default build of Polars is faster and consumes less memory.

## Legacy

Do you want Polars to run on an old CPU (e.g. dating from before 2011), or on an `x86-64` build
of Python on Apple Silicon under Rosetta? Install `pip install polars-lts-cpu`. This version of
Polars is compiled without [AVX](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) target
features.

## Sponsors

[<img src="https://www.jetbrains.com/company/brand/img/jetbrains_logo.png" height="50" alt="JetBrains logo" />](https://www.jetbrains.com)
