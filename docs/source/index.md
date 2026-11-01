![logo](https://raw.githubusercontent.com/pola-rs/polars-static/master/banner/polars_github_banner.svg)

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

Polars is a blazingly fast DataFrame library for manipulating structured data. The core is written
in Rust, and available for Python, R and NodeJS.

## Key features

- **Fast**: Leading performance on analytical workloads. Polars runs queries on all available CPU
  cores with vectorized (SIMD) execution, without any configuration.
- **Optimizations out of the box**: Use the lazy API so Polars can optimize your queries before
  running it, for example by reading only the rows and columns it needs.
- **Larger-than-RAM**: The streaming engine processes data in batches, so you can query datasets
  that don't fit in memory.
- **Query profiling**: Profile lazy queries as they run on your machine, and see the time spent,
  rows processed and I/O for each operation.
- **Expressive API**: Compose complex queries from expressions that Polars runs in parallel.
- **Extensible**: Add your own functions with expression plugins written in Rust, and your own data
  sources with I/O plugins.
- **Multi-language**: Use Polars from Python, Rust, Node.js and R, or query your data with SQL.
- **GPU support**: Run queries on NVIDIA GPUs for maximum performance for in-memory or streaming
  workloads.
- **Interoperable**: Polars supports the Apache Arrow columnar format, so it can share data with
  other Arrow-based tools without copying.

<!-- dprint-ignore-start -->

!!! info "Users new to DataFrames"
    A DataFrame is a 2-dimensional data structure that is useful for data manipulation and analysis. With labeled axes for rows and columns, each column can contain different data types, making complex data operations such as merging and aggregation much easier. Due to their flexibility and intuitive way of storing and working with data, DataFrames have become increasingly popular in modern data analytics and engineering.

<!-- dprint-ignore-end -->

## Philosophy

The goal of Polars is to provide a lightning fast DataFrame library that:

- Utilizes all available cores on your machine.
- Optimizes queries to reduce unneeded work/memory allocations.
- Handles datasets much larger than your available RAM.
- A consistent and predictable API.
- Adheres to a strict schema (data-types should be known before running the query).

Polars is written in Rust which gives it C/C++ performance and allows it to fully control
performance-critical parts in a query engine.

## Example

{{code_block('home/example','example',['scan_csv','filter','group_by','collect'])}}

To see how each operation of this query performs on your machine, install the `polars-cloud` package
and enable query profiling before you collect it:

```python
pl.Config.enable_monitoring()

df = q.collect()
```

While the query runs, you can follow the time spent, rows processed and I/O of each operation in
Polars Cloud. To set it up, see [Query profiling](user-guide/query-profiling.md).

A more extensive introduction to Polars and how to get started can be found in the
[next chapter](user-guide/getting-started.md).

## Community

Polars has a very active community with frequent releases (approximately weekly). Below are some of
the top contributors to the project:

--8<-- "docs/assets/people.md"

## Contributing

We appreciate all contributions, from reporting bugs to implementing new features. Read our
[contributing guide](development/contributing/index.md) to learn more.

## License

This project is licensed under the terms of the
[MIT license](https://github.com/pola-rs/polars/blob/main/LICENSE).
