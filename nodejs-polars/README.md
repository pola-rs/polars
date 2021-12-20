# Polars

[![rust docs](https://docs.rs/polars/badge.svg)](https://docs.rs/polars/latest/polars/)
[![Build and test](https://github.com/pola-rs/polars/workflows/Build%20and%20test/badge.svg)](https://github.com/pola-rs/polars/actions)
[![](https://img.shields.io/crates/v/polars.svg)](https://crates.io/crates/polars)
[![PyPI Latest Release](https://img.shields.io/pypi/v/polars.svg)](https://pypi.org/project/polars/)


```js

import pl from "polars"

const df = pl.DataFrame(
  {
    A: [1, 2, 3, 4, 5],
    fruits: ["banana", "banana", "apple", "apple", "banana"],
    B: [5, 4, 3, 2, 1],
    cars: ["beetle", "audi", "beetle", "beetle", "beetle"],
  }
)
const df2 = df
  .sort("fruits")
  .select(
    "fruits",
    "cars",
    pl.lit("fruits").alias("literal_string_fruits"),
    pl.col("B").filter(pl.col("cars").eq(lit("beetle"))).sum(),
    pl.col("A").filter(pl.col("B").gt(2)).sum().over("cars").alias("sum_A_by_cars"),
    pl.col("A").sum().over("fruits").alias("sum_A_by_fruits"),
    pl.col("A").reverse().over("fruits").flatten().alias("rev_A_by_fruits")
  )
  
console.log(df2)
shape: (5, 8)
┌──────────┬──────────┬──────────────┬─────┬─────────────┬─────────────┬─────────────┐
│ fruits   ┆ cars     ┆ literal_stri ┆ B   ┆ sum_A_by_ca ┆ sum_A_by_fr ┆ rev_A_by_fr │
│ ---      ┆ ---      ┆ ng_fruits    ┆ --- ┆ rs          ┆ uits        ┆ uits        │
│ str      ┆ str      ┆ ---          ┆ i64 ┆ ---         ┆ ---         ┆ ---         │
│          ┆          ┆ str          ┆     ┆ i64         ┆ i64         ┆ i64         │
╞══════════╪══════════╪══════════════╪═════╪═════════════╪═════════════╪═════════════╡
│ "apple"  ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 7           ┆ 4           │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ "apple"  ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 7           ┆ 3           │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ "banana" ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 8           ┆ 5           │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ "banana" ┆ "audi"   ┆ "fruits"     ┆ 11  ┆ 2           ┆ 8           ┆ 2           │
├╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌╌╌┤
│ "banana" ┆ "beetle" ┆ "fruits"     ┆ 11  ┆ 4           ┆ 8           ┆ 1           │
└──────────┴──────────┴──────────────┴─────┴─────────────┴─────────────┴─────────────┘

```

## Node setup


Install the latest polars version with:

```sh
$ yarn add nodejs-polars # yarn
$ npm i -s nodejs-polars # npm
```

Releases happen quite often (weekly / every few days) at the moment, so updating polars regularly to get the latest bugfixes / features might not be a bad idea.

#### Rust version

Required Rust version `>=1.52`


## Documentation

Want to know about all the features Polars supports? Read the docs!

#### Python

  * Installation guide: `$ pip3 install polars`
  * [Python documentation](https://pola-rs.github.io/polars/py-polars/html/reference/index.html)
  * [User guide](https://pola-rs.github.io/polars-book/)

#### Rust

  * [Rust documentation (master branch)](https://pola-rs.github.io/polars/polars/index.html)
  * [User guide](https://pola-rs.github.io/polars-book/)

### Node
  * COMING SOON!

## Contribution

Want to contribute? Read our [contribution guideline](https://github.com/pola-rs/polars/blob/master/CONTRIBUTING.md).

## \[Node\]: compile polars from source

If you want a bleeding edge release or maximal performance you should compile **polars** from source.
  1. Install the latest [Rust compiler](https://www.rust-lang.org/tools/install)
  2. Run `npm|yarn install`
  3. Choose any of:
     * Fastest binary, very long compile times:
        ```bash
        $ cd nodejs-polars && yarn build:all # this will generate a /bin directory with the compiles TS code, as well as the rust binary
        ```
     * Fast binary, Shorter compile times:
        ```bash
        $ cd nodejs-polars && yarn build:all:slim # this will generate a /bin directory with the compiles TS code, as well as the rust binary
        ```

### Debugging
for an unoptimized build, you can run `yarn build:debug` 


## Acknowledgements

Development of Polars is proudly powered by


[![Xomnia](https://raw.githubusercontent.com/pola-rs/polars-static/master/sponsors/xomnia.png)](https://www.xomnia.com/)


## Sponsors

[<img src="https://raw.githubusercontent.com/pola-rs/polars-static/master/sponsors/xomnia.png" height="40" />](https://www.xomnia.com/) &emsp; [<img src="https://www.jetbrains.com/company/brand/img/jetbrains_logo.png" height="50" />](https://www.jetbrains.com)