<h1 align="center">
  <img src="https://raw.githubusercontent.com/pola-rs/polars-static/master/logos/polars_github_logo_rect_dark_name.svg">
  <br>
</h1>

<div align="center">
  <a href="https://docs.rs/polars/latest/polars/">
    <img src="https://docs.rs/polars/badge.svg" alt="rust docs"/>
  </a>
  <a href="https://github.com/pola-rs/polars/actions">
    <img src="https://github.com/pola-rs/polars/workflows/Build%20and%20test/badge.svg" alt="Build and test"/>
  </a>
  <a href="https://crates.io/crates/polars">
    <img src="https://img.shields.io/crates/v/polars.svg"/>
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
  <a href="https://pola-rs.github.io/polars/py-polars/html/reference/index.html">Python</a>
  -
  <a href="https://pola-rs.github.io/polars/polars/index.html">Rust</a>
  -
  <a href="https://pola-rs.github.io/nodejs-polars/index.html">Node.js</a>
  -
  <a href="https://rpolars.github.io/index.html">R</a>
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
  <a href="https://pola-rs.github.io/polars-book/">User Guide</a>
  |
  <a href="https://discord.gg/4UfP5cfBE7">Discord</a>
</p>

## Polars: Blazingly fast DataFrames in Rust, Python, Node.js, R and SQL

Polars is a DataFrame interface on top of an OLAP Query Engine implemented in Rust using
[Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html) as the memory model.

- Lazy | eager execution
- Multi-threaded
- SIMD
- Query optimization
- Powerful expression API
- Hybrid Streaming (larger than RAM datasets)
- Rust | Python | NodeJS | R | ...

To learn more, read the [User Guide](https://pola-rs.github.io/polars-book/).

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
>>> # create a sql context
>>> context = pl.SQLContext()
>>> # register a table
>>> table = pl.scan_ipc("file.arrow")
>>> context.register("my_table", table)
>>> # the query we want to run
>>> query = """
... SELECT sum(v1) as sum_v1, min(v2) as min_v2 FROM my_table
... WHERE id1 = 'id016'
... LIMIT 10
... """
>>> ## OPTION 1
>>> # run query to materialization
>>> context.query(query)
 shape: (1, 2)
 ┌────────┬────────┐
 │ sum_v1 ┆ min_v2 │
 │ ---    ┆ ---    │
 │ i64    ┆ i64    │
 ╞════════╪════════╡
 │ 298268 ┆ 1      │
 └────────┴────────┘
>>> ## OPTION 2
>>> # Don't materialize the query, but return as LazyFrame
>>> # and continue in python
>>> lf = context.execute(query)
>>> (lf.join(other_table)
...      .groupby("foo")
...      .agg(
...     pl.col("sum_v1").count()
... ).collect())
```

SQL commands can also be ran directly from your terminal.

```bash
> cargo install polars-cli --locked
# run an inline sql query
> polars -c "SELECT sum(v1) as sum_v1, min(v2) as min_v2 FROM read_ipc('file.arrow') WHERE id1 = 'id016' LIMIT 10"

# run interactively
> polars
Polars CLI v0.1.0
Type .help for help.

> SELECT sum(v1) as sum_v1, min(v2) as min_v2 FROM read_ipc('file.arrow') WHERE id1 = 'id016' LIMIT 10;
```

Refer to [polars-cli](./polars-cli/README.md) for more information.

## Performance 🚀🚀

### Blazingly fast

Polars is very fast. In fact, it is one of the best performing solutions available.
See the results in [DuckDB's db-benchmark](https://duckdblabs.github.io/db-benchmark/).

In the [TPCH benchmarks](https://www.pola.rs/benchmarks.html) polars is orders of magnitudes faster than pandas, dask, modin and vaex
on full queries (including IO).

### Lightweight

Polars is also very lightweight. It comes with zero required dependencies, and this shows in the import times:

- polars: 70ms
- numpy: 104ms
- pandas: 520ms

### Handles larger than RAM data

If you have data that does not fit into memory, polars lazy is able to process your query (or parts of your query) in a
streaming fashion, this drastically reduces memory requirements so you might be able to process your 250GB dataset on your
laptop. Collect with `collect(streaming=True)` to run the query streaming. (This might be a little slower, but
it is still very fast!)

## Setup

### Python

Install the latest polars version with:

```sh
pip install polars
```

We also have a conda package (`conda install -c conda-forge polars`), however pip is the preferred way to install Polars.

Install Polars with all optional dependencies.

```sh
pip install 'polars[all]'
pip install 'polars[numpy,pandas,pyarrow]'  # install a subset of all optional dependencies
```

You can also install the dependencies directly.

| Tag        | Description                                                                  |
| ---------- | ---------------------------------------------------------------------------- |
| **all**    | Install all optional dependencies (all of the following)                     |
| pandas     | Install with Pandas for converting data to and from Pandas Dataframes/Series |
| numpy      | Install with numpy for converting data to and from numpy arrays              |
| pyarrow    | Reading data formats using PyArrow                                           |
| fsspec     | Support for reading from remote file systems                                 |
| connectorx | Support for reading from SQL databases                                       |
| xlsx2csv   | Support for reading from Excel files                                         |
| deltalake  | Support for reading from Delta Lake Tables                                   |
| timezone   | Timezone support, only needed if are on Python<3.9 or you are on Windows     |

Releases happen quite often (weekly / every few days) at the moment, so updating polars regularly to get the latest bugfixes / features might not be a bad idea.

### Rust

You can take latest release from `crates.io`, or if you want to use the latest features / performance improvements
point to the `main` branch of this repo.

```toml
polars = { git = "https://github.com/pola-rs/polars", rev = "<optional git tag>" }
```

Required Rust version `>=1.62`

## Contributing

Want to contribute? Read our [contribution guideline](./CONTRIBUTING.md).

## Python: compile polars from source

If you want a bleeding edge release or maximal performance you should compile **polars** from source.

This can be done by going through the following steps in sequence:

1. Install the latest [Rust compiler](https://www.rust-lang.org/tools/install)
2. Install [maturin](https://maturin.rs/): `pip install maturin`
3. Choose any of:
   - Fastest binary, very long compile times:
     ```sh
     $ cd py-polars && maturin develop --release -- -C target-cpu=native
     ```
   - Fast binary, Shorter compile times:
     ```sh
     $ cd py-polars && maturin develop --release -- -C codegen-units=16 -C lto=thin -C target-cpu=native
     ```

Note that the Rust crate implementing the Python bindings is called `py-polars` to distinguish from the wrapped
Rust crate `polars` itself. However, both the Python package and the Python module are named `polars`, so you
can `pip install polars` and `import polars`.

## Use custom Rust function in python?

Extending polars with UDFs compiled in Rust is easy. We expose pyo3 extensions for `DataFrame` and `Series`
data structures. See more in https://github.com/pola-rs/pyo3-polars.

## Going big...

Do you expect more than `2^32` ~4,2 billion rows? Compile polars with the `bigidx` feature flag.

Or for python users install `pip install polars-u64-idx`.

Don't use this unless you hit the row boundary as the default polars is faster and consumes less memory.

## Legacy

Do you want polars to run on an old CPU (e.g. dating from before 2011)? Install `pip install polars-lts-cpu`. This polars project is
compiled without [avx](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) target features.

## Acknowledgements

Development of Polars is proudly powered by

[![Xomnia](https://raw.githubusercontent.com/pola-rs/polars-static/master/sponsors/xomnia.png)](https://www.xomnia.com/)

## Sponsors

[<img src="https://raw.githubusercontent.com/pola-rs/polars-static/master/sponsors/xomnia.png" height="40" />](https://www.xomnia.com/) &emsp; [<img src="https://www.jetbrains.com/company/brand/img/jetbrains_logo.png" height="50" />](https://www.jetbrains.com)
