# Polars

[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
[![Build and test](https://github.com/pola-rs/polars/workflows/Build%20and%20test/badge.svg)](https://github.com/pola-rs/polars/actions)
[![](https://img.shields.io/crates/v/polars.svg)](https://crates.io/crates/polars)
[![PyPI Latest Release](https://img.shields.io/pypi/v/polars.svg)](https://pypi.org/project/polars/)
[![NPM Latest Release](https://img.shields.io/npm/v/nodejs-polars.svg)](https://www.npmjs.com/package/nodejs-polars)

<p align="center">
  <b>Documentation</b>:
  <a href="https://pola-rs.github.io/polars/py-polars/html/reference/index.html">Python</a>
  -
  <a href="https://pola-rs.github.io/polars/polars/index.html">Rust</a>
  -
  <a href="https://pola-rs.github.io/nodejs-polars/index.html">Node.js</a>
  |
  <b>StackOverflow</b>:
  <a href="https://stackoverflow.com/questions/tagged/python-polars">Python</a>
  -
  <a href="https://stackoverflow.com/questions/tagged/rust-polars">Rust</a>
  -
  <a href="https://stackoverflow.com/questions/tagged/nodejs-polars">Node.js</a>
  |
  <a href="https://pola-rs.github.io/polars-book/">User Guide</a>
  |
  <a href="https://discord.gg/4UfP5cfBE7">Discord</a>
</p>


## Blazingly fast DataFrames in Rust, Python & Node.js

Polars is a blazingly fast DataFrames library implemented in Rust using
[Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html) as the memory model.

  * Lazy | eager execution
  * Multi-threaded
  * SIMD
  * Query optimization
  * Powerful expression API
  * Rust | Python | ...

To learn more, read the [User Guide](https://pola-rs.github.io/polars-book/).

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

# embarrassingly parallel execution
# very expressive query language
>>> (
...     df
...     .sort("fruits")
...     .select(
...         [
...             "fruits",
...             "cars",
...             pl.lit("fruits").alias("literal_string_fruits"),
...             pl.col("B").filter(pl.col("cars") == "beetle").sum(),
...             pl.col("A").filter(pl.col("B") > 2).sum().over("cars").alias("sum_A_by_cars"),     # groups by "cars"
...             pl.col("A").sum().over("fruits").alias("sum_A_by_fruits"),                         # groups by "fruits"
...             pl.col("A").reverse().over("fruits").alias("rev_A_by_fruits"),                     # groups by "fruits
...             pl.col("A").sort_by("B").over("fruits").alias("sort_A_by_B_by_fruits"),            # groups by "fruits"
...         ]
...     )
... )
shape: (5, 8)
┌──────────┬──────────┬──────────────┬─────┬─────────────┬─────────────┬─────────────┬─────────────┐
│ fruits   ┆ cars     ┆ literal_stri ┆ B   ┆ sum_A_by_ca ┆ sum_A_by_fr ┆ rev_A_by_fr ┆ sort_A_by_B │
│ ---      ┆ ---      ┆ ng_fruits    ┆ --- ┆ rs          ┆ uits        ┆ uits        ┆ _by_fruits  │
│ str      ┆ str      ┆ ---          ┆ i64 ┆ ---         ┆ ---         ┆ ---         ┆ ---         │
│          ┆          ┆ str          ┆     ┆ i64         ┆ i64         ┆ i64         ┆ i64         │
╞══════════╪══════════╪══════════════╪═════╪═════════════╪═════════════╪═════════════╪═════════════╡
│ "apple"  ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 7           ┆ 4           ┆ 4           │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ "apple"  ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 7           ┆ 3           ┆ 3           │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ "banana" ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 8           ┆ 5           ┆ 5           │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ "banana" ┆ "audi"   ┆ "fruits"     ┆ 11  ┆ 2           ┆ 8           ┆ 2           ┆ 2           │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ "banana" ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 8           ┆ 1           ┆ 1           │
└──────────┴──────────┴──────────────┴─────┴─────────────┴─────────────┴─────────────┴─────────────┘

```


## Performance 🚀🚀

Polars is very fast. In fact, it is one of the best performing solutions available.
See the results in [h2oai's db-benchmark](https://h2oai.github.io/db-benchmark/).


## Python setup

Install the latest polars version with:

```
# Install Polars only.
$ pip3 install -U 'polars'

# Install Polars with all optional dependencies.
$ pip3 install -U 'polars[all]'

# Install Polars and numpy.
$ pip3 install -U 'polars[numpy]'

# Install Polars and pyarrow/pandas/numpy to be able to convert to/from pandas and/or read data with pyarrow.
$ pip3 install -U 'polars[pyarrow]'

# Install Polars and pyarrow/pandas/numpy and fsspec (read from e.g. remote filesystems, compressed files).
$ pip3 install -U 'polars[pyarrow,fsspec]'

# Install Polars and connectorx (read data from SQL databases).
$ pip3 install -U 'polars[connectorx]'

# Install Polars and xlsx2csv (read data from Excel).
$ pip3 install -U 'polars[xlsx2csv]'

# Install Polars with timezone support, only needed if
#   1. you are on Python < 3.9, Python 3.9+ has this in stdlib
#   2. you are on Windows
$ pip3 install -U 'polars[timezone]'
```

Releases happen quite often (weekly / every few days) at the moment, so updating polars regularly to get the latest bugfixes / features might not be a bad idea.


## Rust setup

You can take latest release from `crates.io`, or if you want to use the latest features / performance improvements
point to the `master` branch of this repo.

```toml
polars = { git = "https://github.com/pola-rs/polars", rev = "<optional git tag>" }
```


#### Rust version

Required Rust version `>=1.58`


## Documentation

Want to know about all the features Polars supports? Read the docs!

#### Python

  * Installation guide: `$ pip3 install polars`
  * [Python documentation](https://pola-rs.github.io/polars/py-polars/html/reference/index.html)
  * [User guide](https://pola-rs.github.io/polars-book/)

#### Rust

  * [Rust documentation (master branch)](https://pola-rs.github.io/polars/polars/index.html)
  * [User guide](https://pola-rs.github.io/polars-book/)

#### Node

  * Installation guide: `$ yarn add nodejs-polars`
  * [Node documentation](https://pola-rs.github.io/nodejs-polars/index.html)
  * [User guide](https://pola-rs.github.io/polars-book/)
  * [Github](https://github.com/pola-rs/nodejs-polars)


## Contribution

Want to contribute? Read our [contribution guideline](https://github.com/pola-rs/polars/blob/master/CONTRIBUTING.md).


## \[Python\]: compile polars from source

If you want a bleeding edge release or maximal performance you should compile **polars** from source.

This can be done by going through the following steps in sequence:

  1. Install the latest [Rust compiler](https://www.rust-lang.org/tools/install)
  2. Install [maturin](https://maturin.rs/): `$ pip3 install maturin`
  3. Choose any of:
      * Fastest binary, very long compile times:
        ```bash
        $ cd py-polars && maturin develop --release -- -C target-cpu=native
        ```
      * Fast binary, Shorter compile times:
        ```bash
        $ cd py-polars && maturin develop --release -- -C codegen-units=16 -C lto=thin -C target-cpu=native
        ```

Note that the Rust crate implementing the Python bindings is called `py-polars` to distinguish from the wrapped
Rust crate `polars` itself. However, both the Python package and the Python module are named `polars`, so you
can `pip install polars` and `import polars`.


## Arrow2

Polars has transitioned to [arrow2](https://crates.io/crates/arrow2).
Arrow2 is a faster and safer implementation of the [Apache Arrow Columnar Format](https://arrow.apache.org/docs/format/Columnar.html).
Arrow2 also has a more granular code base, helping to reduce the compiler bloat.

## Use custom Rust function in python?
See [this example](./examples/python_rust_compiled_function).

# Going big...
Do you expect more than `2^32` ~4,2 billion rows? Compile polars with the `bigidx` feature flag.

Or for python users install `$ pip install -U polars-u64-idx`.

Don't use this unless you hit the row boundary as the default polars is faster and consumes less memory.

# Legacy
Do you want polars to run on an old CPU (e.g. dating from before 2011)? Install `$pip -U polars-lts-cpu`. This polars project is 
compiled without [avx](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) target features.

## Acknowledgements

Development of Polars is proudly powered by


[![Xomnia](https://raw.githubusercontent.com/pola-rs/polars-static/master/sponsors/xomnia.png)](https://www.xomnia.com/)


## Sponsors

[<img src="https://raw.githubusercontent.com/pola-rs/polars-static/master/sponsors/xomnia.png" height="40" />](https://www.xomnia.com/) &emsp; [<img src="https://www.jetbrains.com/company/brand/img/jetbrains_logo.png" height="50" />](https://www.jetbrains.com)
