# Ecosystem

## Introduction

On this page you can find a non-exhaustive list of libraries and tools that support Polars. As the
data ecosystem is evolving fast, more libraries will likely support Polars in the future. One of the
main drivers is that Polars makes adheres its memory layout to the `Apache Arrow` spec.

### Table of contents:

- [Apache Arrow](#apache-arrow)
- [Data visualisation](#data-visualisation)
- [IO](#io)
- [Machine learning](#machine-learning)
- [Other](#other)

---

### Apache Arrow

[Apache Arrow](https://arrow.apache.org/) enables zero-copy reads of data within the same process,
meaning that data can be directly accessed in its in-memory format without the need for copying or
serialisation. This enhances performance when integrating with different tools using Apache Arrow.
Polars is compatible with a wide range of libraries that also make use of Apache Arrow, like Pandas
and DuckDB.

### Data visualisation

See the [dedicated visualization section](misc/visualization.md).

### IO

#### Delta Lake

The [Delta Lake](https://github.com/delta-io/delta-rs) project aims to unlock the power of the
Deltalake for as many users and projects as possible by providing native low-level APIs aimed at
developers and integrators, as well as a high-level operations API that lets you query, inspect, and
operate your Delta Lake with ease. Delta Lake builds on the native Polars Parquet reader allowing
you to write standard Polars queries against a DeltaTable.

Read how to use Delta Lake with Polars
[at Delta Lake](https://delta-io.github.io/delta-rs/integrations/delta-lake-polars/#reading-a-delta-lake-table-with-polars).

### Machine Learning

#### Scikit Learn

The [Scikit Learn](https://scikit-learn.org/stable/) machine learning package accepts a Polars
`DataFrame` as input/output to all transformers and as input to models.

#### XGBoost & LightGBM

XGBoost and LightGBM are gradient boosting packages for doing regression or classification on
tabular data.
[XGBoost accepts Polars `DataFrame` and `LazyFrame` as input](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
while LightGBM accepts Polars `DataFrame` as input.

#### Time series forecasting

The
[Nixtla time series forecasting packages](https://nixtlaverse.nixtla.io/statsforecast/docs/getting-started/getting_started_complete_polars.html)
accept a Polars `DataFrame` as input.

#### Hugging Face

Hugging Face is a platform for working with machine learning datasets and models.
[Polars can be used to work with datasets downloaded from Hugging Face](io/hugging-face.md).

#### Deep learning frameworks

A `DataFrame` can be transformed
[into a PyTorch format using `to_torch`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_torch.html)
or
[into a JAX format using `to_jax`](https://docs.pola.rs/api/python/stable/reference/dataframe/api/polars.DataFrame.to_jax.html).

### Other

#### DuckDB

[DuckDB](https://duckdb.org) is a high-performance analytical database system. It is designed to be
fast, reliable, portable, and easy to use. DuckDB provides a rich SQL dialect, with support far
beyond basic SQL. DuckDB supports arbitrary and nested correlated subqueries, window functions,
collations, complex types (arrays, structs), and more. Read about integration with Polars
[on the DuckDB website](https://duckdb.org/docs/guides/python/polars).

#### Great Tables

With [Great Tables](https://posit-dev.github.io/great-tables/articles/intro.html) anyone can make
wonderful-looking tables in Python. Here is a
[blog post](https://posit-dev.github.io/great-tables/blog/polars-styling/) on how to use Great
Tables with Polars.

#### LanceDB

[LanceDB](https://lancedb.com/) is a developer-friendly, serverless vector database for AI
applications. They have added a direct integration with Polars. LanceDB can ingest Polars
dataframes, return results as polars dataframes, and export the entire table as a polars lazyframe.
You can find a quick tutorial in their blog
[LanceDB + Polars](https://blog.lancedb.com/lancedb-polars-2d5eb32a8aa3)

#### Mage

[Mage](https://www.mage.ai) is an open-source data pipeline tool for transforming and integrating
data. Learn about integration between Polars and Mage at
[docs.mage.ai](https://docs.mage.ai/integrations/polars).

#### marimo

[marimo](https://marimo.io) is a reactive notebook for Python and SQL that models notebooks as
dataflow graphs. It offers built-in support for Polars, allowing seamless integration of Polars
dataframes in an interactive, reactive environment - such as displaying rich Polars tables, no-code
transformations of Polars dataframes, or selecting points on a Polars-backed reactive chart.
