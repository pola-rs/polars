# Ecosystem

## Introduction

On this page you can find a non-exhaustive list of libraries and tools that support Polars. As the data ecosystem is evolving fast, more libraries will likely support Polars in the future. One of the main drivers is that Polars makes adheres its memory layout to the `Apache Arrow` spec.

### Table of contents:

- [Database integration](#database-integration)
- [Data pipelines](#data-pipelines)
- [Interchange and open storage formats](#interchange-and-storage-formats)
- [Machine learning](#machine-learning)
- [Validation and quality](#validation-and-quality)
- [Visualization and presentation](#visualization-and-presentation)

---

### Database integration

#### DuckDB

[DuckDB](https://duckdb.org) is a high-performance analytical database system. It is designed to be fast, reliable, portable, and easy to use. DuckDB provides a rich SQL dialect, with support far beyond basic SQL. DuckDB supports arbitrary and nested correlated subqueries, window functions, collations, complex types (arrays, structs), and more. Read about integration with Polars [on the DuckDB website](https://duckdb.org/docs/guides/python/polars).

#### LanceDB

[LanceDB](https://lancedb.com/) is a developer-friendly, serverless vector database for AI applications. They have added a direct integration with Polars. LanceDB can ingest Polars dataframes, return results as Polars dataframes, and export the entire table as a Polars lazyframe. You can find a quick tutorial in their blog [LanceDB + Polars](https://blog.lancedb.com/lancedb-polars-2d5eb32a8aa3)

### Data pipelines

#### Mage

[Mage](https://www.mage.ai) is an open-source data pipeline tool for transforming and integrating data. Learn about integration between Polars and Mage at [docs.mage.ai](https://docs.mage.ai/integrations/polars).

### Interchange and storage formats

#### Apache Arrow

[Apache Arrow](https://arrow.apache.org/) enables zero-copy reads of data within the same process, meaning that data can be directly accessed in its in-memory format without the need for copying or serialisation. This enhances performance when integrating with different tools using Apache Arrow. Polars is compatible with a wide range of libraries that also make use of Apache Arrow, like pandas and DuckDB.

#### Delta Lake

The [Delta Lake](https://github.com/delta-io/delta-rs) project aims to unlock the power of the Deltalake for as many users and projects as possible by providing native low-level APIs aimed at developers and integrators, as well as a high-level operations API that lets you query, inspect, and operate your Delta Lake with ease.

Read how to use Delta Lake with Polars [at Delta Lake](https://delta-io.github.io/delta-rs/integrations/delta-lake-polars/#reading-a-delta-lake-table-with-polars).

#### Apache Iceberg

[Iceberg](https://iceberg.apache.org/) is a high-performance format for huge analytic tables. It ensures reliable and efficient data handling by abstracting complex table operations and maintaining consistent table states across multiple platforms and services.

Read more on [using Apache Iceberg with Polars](https://tabular.io/apache-iceberg-cookbook/pyiceberg-polars/).

### Machine learning

#### Scikit Learn

Since [Scikit Learn](https://scikit-learn.org/stable/) 1.4, all transformers support Polars output. See the change log for [more details](https://scikit-learn.org/dev/whats_new/v1.4.html#changes-impacting-all-modules).

### Validation and quality

#### Pandera

[pandera](https://pandera.readthedocs.io/en/stable/) is a Union.ai open source project that provides a flexible and expressive API for performing data validation on dataframe-like objects to make data processing pipelines more readable and robust. Find out how to do [data validation with pandera and Polars](https://pandera.readthedocs.io/en/stable/polars.html)

### Visualization and presentation

#### hvPlot

[hvPlot](https://hvplot.holoviz.org/) is available as the default plotting backend for Polars making it simple to create interactive and static visualisations. You can use hvPlot by using the feature flag `plot` during installing.

```python
pip install 'polars[plot]'
```

#### Matplotlib

[Matplotlib](https://matplotlib.org/) is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and hard things possible.

#### Plotly

[Plotly](https://plotly.com/python/) is an interactive, open-source, and browser-based graphing library for Python. Built on top of plotly.js, it ships with over 30 chart types, including scientific charts, 3D graphs, statistical charts, SVG maps, financial charts, and more.

#### Seaborn

[Seaborn](https://seaborn.pydata.org/) is a Python data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

#### Great Tables

With [Great Tables](https://posit-dev.github.io/great-tables/articles/intro.html) anyone can make wonderful-looking tables in Python. Here is a [blog post](https://posit-dev.github.io/great-tables/blog/polars-styling/) on how to use Great Tables with Polars.
