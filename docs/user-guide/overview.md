# Overview

![logo](https://raw.githubusercontent.com/pola-rs/polars-static/master/logos/polars_github_logo_rect_dark_name.svg)

<h1 style="text-align:center">Blazingly Fast DataFrame Library </h1>
<div align="center">
  <a href="https://docs.rs/polars/latest/polars/">
    <img src="https://docs.rs/polars/badge.svg" alt="Rust docs latest"/>
  </a>
  <a href="https://crates.io/crates/polars">
    <img src="https://img.shields.io/crates/v/polars.svg" alt="Rust crates Latest Release"/>
  </a>
  <a href="https://pypi.org/project/polars/">
    <img src="https://img.shields.io/pypi/v/polars.svg" alt="PyPI Latest Release"/>
  </a>
  <a href="https://doi.org/10.5281/zenodo.7697217">
    <img src="https://zenodo.org/badge/DOI/10.5281/zenodo.7697217.svg" alt="DOI Latest Release"/>
  </a>
</div>

Polars is a blazingly fast DataFrame library for manipulating structured data. The core is written in Rust, and available for Python, R and NodeJS.

## Key features
- **Fast**: Written from scratch in Rust, designed close to the machine and without external dependencies.
- **I/O**: First class support for all common data storage layers: local, cloud storage & databases.
- **Intuitive API**: Write your queries the way they were intended. Polars, internally, will determine the most efficient way to execute using its query optimizer.
- **Out of Core**: The streaming API allows you to process your results without requiring all your data to be in memory at the same time
- **Parallel**: Utilises the power of your machine by dividing the workload among the available CPU cores without any additional configuration.
- **Vectorized Query Engine**: Using [Apache Arrow](https://arrow.apache.org/), a columnar data format, to process your queries in a vectorized manner and SIMD to optimize CPU usage.


!!! info "Users new to Dataframes"
    A DataFrame is a 2-dimensional data structure that is useful for data manipulation and analysis. With labeled axes for rows and columns, each column can contain different data types, making complex data operations such as merging and aggregation much easier. Due to their flexibility and intuitive way of storing and working with data, DataFrames have become increasingly popular in modern data analytics and engineering.


## Philosophy

The goal of Polars is to provide a lightning fast DataFrame library that:

- Utilizes all available cores on your machine.
- Optimizes queries to reduce unneeded work/memory allocations.
- Handles datasets much larger than your available RAM.
- A consistent and predictable API.
- Adheres to a strict schema (data-types should be known before running the query).

Polars is written in Rust which gives it C/C++ performance and allows it to fully control performance critical parts in a query engine.

## Example

{{code_block('home/example','example',['scan_csv','filter','group_by','collect'])}}

A more extensive introduction can be found in the [next chapter](/user-guide/getting-started).

## Community

Polars has a very active community with frequent releases (approximately weekly). Below are some of the top contributors to the project:

--8<-- "docs/people.md"

## Contributing

We appreciate all contributions, from reporting bugs to implementing new features. Read our [contributing guide](../development/contributing/index.md) to learn more.

## License

This project is licensed under the terms of the [MIT license](https://github.com/pola-rs/polars/blob/main/LICENSE).
