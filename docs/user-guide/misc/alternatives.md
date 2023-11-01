# Alternatives

These are some tools that share similar functionality to what polars does.

- Pandas

  A very versatile tool for small data. Read [10 things I hate about pandas](https://wesmckinney.com/blog/apache-arrow-pandas-internals/)
  written by the author himself. Polars has solved all those 10 things.
  Polars is a versatile tool for small and large data with a more predictable, less ambiguous, and stricter API.

- Pandas the API

  The API of pandas was designed for in memory data. This makes it a poor fit for performant analysis on large data
  (read anything that does not fit into RAM). Any tool that tries to distribute that API will likely have a
  suboptimal query plan compared to plans that follow from a declarative API like SQL or Polars' API.

- Dask

  Parallelizes existing single-threaded libraries like NumPy and pandas. As a consumer of those libraries Dask
  therefore has less control over low level performance and semantics.
  Those libraries are treated like a black box.
  On a single machine the parallelization effort can also be seriously stalled by pandas strings.
  Pandas strings, by default, are stored as Python objects in
  numpy arrays meaning that any operation on them is GIL bound and therefore single threaded. This can be circumvented
  by multi-processing but has a non-trivial cost.

- Modin

  Similar to Dask

- Vaex

  Vaexs method of out-of-core analysis is memory mapping files. This works until it doesn't. For instance parquet
  or csv files first need to be read and converted to a file format that can be memory mapped. Another downside is
  that the OS determines when pages will be swapped. Operations that need a full data shuffle, such as
  sorts, have terrible performance on memory mapped data.
  Polars' out of core processing is not based on memory mapping, but on streaming data in batches (and spilling to disk
  if needed), we control which data must be hold in memory, not the OS, meaning that we don't have unexpected IO stalls.

- DuckDB

  Polars and DuckDB have many similarities. DuckDB is focused on providing an in-process OLAP Sqlite alternative,
  Polars is focused on providing a scalable `DataFrame` interface to many languages. Those different front-ends lead to
  different optimization strategies and different algorithm prioritization. The interoperability between both is zero-copy.
  See more: https://duckdb.org/docs/guides/python/polars

- Spark

  Spark is designed for distributed workloads and uses the JVM. The setup for spark is complicated and the startup-time
  is slow. On a single machine Polars has much better performance characteristics. If you need to process TB's of data
  Spark is a better choice.

- CuDF

  GPU's and CuDF are fast!
  However, GPU's are not readily available and expensive in production. The amount of memory available on a GPU
  is often a fraction of the available RAM.
  This (and out-of-core) processing means that Polars can handle much larger data-sets.
  Next to that Polars can be close in [performance to CuDF](https://zakopilo.hatenablog.jp/entry/2023/02/04/220552).
  CuDF doesn't optimize your query, so is not uncommon that on ETL jobs Polars will be faster because it can elide
  unneeded work and materializations.

- Any

  Polars is written in Rust. This gives it strong safety, performance and concurrency guarantees.
  Polars is written in a modular manner. Parts of Polars can be used in other query programs and can be added as a library.
