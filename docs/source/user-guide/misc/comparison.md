# Comparison with other tools

There are several libraries and tools that share similar functionalities with Polars. This often
leads to questions from data experts about what the differences are. Below is a short comparison
between some of the more popular data processing tools and Polars, to help make a deliberate
decision on which tool to use.

You can find performance benchmarks of these tools in this
[Polars blog post](https://pola.rs/posts/benchmarks/) or in the
[db-benchmark maintained by DuckDB Labs](https://duckdblabs.github.io/db-benchmark/). You can also
run the benchmarks yourself using the
[polars-benchmark](https://github.com/pola-rs/polars-benchmark) repository.

## pandas

pandas stands as a widely-adopted and comprehensive tool in Python data analysis, renowned for its
rich feature set and strong community support. However, due to its single threaded nature, it can
struggle with performance and memory usage on medium and large datasets.

In contrast, Polars is optimised for high-performance multithreaded computing on a single machine,
providing significant improvements in speed and memory efficiency, particularly for medium to large
data operations. Its more composable and stricter API results in greater expressiveness and fewer
schema-related bugs. For a hands-on guide, see [Coming from pandas](../migration/pandas.md).

!!! note

    The name pandas is written in lowercase, even at the start of a sentence, as
    [requested by the pandas project](https://pandas.pydata.org/about/citing.html).

## Dask

Dask extends pandas' capabilities to large, distributed datasets. Dask mimics pandas' API, offering
a familiar environment for pandas users, but with the added benefit of parallel and distributed
computing.

While Dask excels at scaling pandas workflows across clusters, it only supports a subset of the
pandas API and therefore cannot be used for all use cases. Polars offers a more versatile API that
delivers strong performance on a single machine, and scales across a cluster with the distributed
engine.

Dask also supports workloads beyond DataFrames, such as distributed arrays and custom task graphs,
which are outside the scope of Polars.

## Modin

Modin provides a drop-in replacement for the pandas API that parallelises work across multiple cores
or a cluster, using Ray or Dask as the execution backend. The same considerations as for Dask apply.
In 2023, Snowflake acquired Ponder, the organisation that maintains Modin.

## Spark

Spark (specifically PySpark) represents a different approach to large-scale data processing. Spark
was designed for distributed data processing across clusters, making it suitable for extremely large
datasets, but its architecture carries the overhead of distributed execution into every workload,
including those that fit on a single machine. Polars is a suite of engines behind a single API: the
in-memory and streaming engines are built for optimal performance on a single machine, while the
distributed engine scales the same syntax across a cluster, available as
[Polars Cloud](https://cloud.pola.rs/) on AWS and as
[Polars On-Prem](https://docs.cloud.pola.rs/polars-on-premises/index.md) on Kubernetes.

With Spark, data scientists and engineers typically work with different tools (pandas and PySpark),
so deploying a pipeline often requires refactoring by engineers. With Polars, the same code runs in
local environments, on a single machine in the cloud, and across a cluster, so scaling a pipeline
does not require rewriting it. For a hands-on guide, see
[Coming from Apache Spark](../migration/spark.md).

## DuckDB

Polars and DuckDB have many similarities. However, DuckDB is focused on providing an in-process SQL
OLAP database management system, while Polars is focused on providing a scalable `DataFrame`
interface to many languages. The different front-ends lead to different optimisation strategies and
different algorithm prioritisation. The interoperability between both is zero-copy. DuckDB offers a
guide on [how to integrate with Polars](https://duckdb.org/docs/stable/guides/python/polars).
