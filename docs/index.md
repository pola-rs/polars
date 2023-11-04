---
hide:
  - navigation
---

# Polars

![logo](https://raw.githubusercontent.com/pola-rs/polars-static/master/logos/polars_github_logo_rect_dark_name.svg)

<h1 style="text-align:center">Blazingly Fast DataFrame Library </h1>
<div align="center">
  <a href="https://docs.rs/polars/latest/polars/">
    <img src="https://docs.rs/polars/badge.svg" alt="rust docs"/>
  </a>
  <a href="https://crates.io/crates/polars">
    <img src="https://img.shields.io/crates/v/polars.svg"/>
  </a>
  <a href="https://pypi.org/project/polars/">
    <img src="https://img.shields.io/pypi/v/polars.svg" alt="PyPI Latest Release"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.7697217">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7697217.svg" alt="DOI Latest Release"/>
  </a>
</div>

Polars is a highly performant DataFrame library for manipulating structured data. The core is written in Rust, but the library is also available in Python. Its key features are:

- **Fast**: Polars is written from the ground up, designed close to the machine and without external dependencies.
- **I/O**: First class support for all common data storage layers: local, cloud storage & databases.
- **Easy to use**: Write your queries the way they were intended. Polars, internally, will determine the most efficient way to execute using its query optimizer.
- **Out of Core**: Polars supports out of core data transformation with its streaming API. Allowing you to process your results without requiring all your data to be in memory at the same time
- **Parallel**: Polars fully utilises the power of your machine by dividing the workload among the available CPU cores without any additional configuration.
- **Vectorized Query Engine**: Polars uses [Apache Arrow](https://arrow.apache.org/), a columnar data format, to process your queries in a vectorized manner. It uses [SIMD](https://en.wikipedia.org/wiki/Single_instruction,_multiple_data) to optimize CPU usage.

## About this guide

The Polars user guide is intended to live alongside the API documentation. Its purpose is to explain (new) users how to use Polars and to provide meaningful examples. The guide is split into two parts:

- [Getting started](user-guide/basics/index.md): A 10 minute helicopter view of the library and its primary function.
- [User guide](user-guide/index.md): A detailed explanation of how the library is setup and how to use it most effectively.

If you are looking for details on a specific level / object, it is probably best to go the API documentation: [Python](https://pola-rs.github.io/polars/py-polars/html/reference/index.html) | [Rust](https://docs.rs/polars/latest/polars/).

## Performance :rocket: :rocket:

Polars is very fast, and in fact is one of the best performing solutions available.
See the results in h2oai's [db-benchmark](https://duckdblabs.github.io/db-benchmark/), revived by the DuckDB project.

Polars [TPC-H Benchmark results](https://www.pola.rs/benchmarks.html) are now available on the official website.

## Example

{{code_block('home/example','example',['scan_csv','filter','group_by','collect'])}}

## Community

Polars has a very active community with frequent releases (approximately weekly). Below are some of the top contributors to the project:

--8<-- "docs/people.md"

## Contributing

We appreciate all contributions, from reporting bugs to implementing new features. Read our [contributing guide](development/contributing/index.md) to learn more.

## License

This project is licensed under the terms of the [MIT license](https://github.com/pola-rs/polars/blob/main/LICENSE).
