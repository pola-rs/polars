# Polars ecosystem

## Introduction

On this page you can find a non-exhaustive list of libraries and tools that support Polars. As the data ecosystem is evolving fast, more libraries will likely support Polars in the future. One of the main drivers is that Polars makes use of `Apache Arrow` in it's backend.

#### [Apache Arrow](https://arrow.apache.org/)

Apache Arrow enables zero-copy reads of data within the same process, meaning that data can be directly accessed in its in-memory format without the need for copying or serialisation. This enhances performance when integrating with different tools Using Apache Arrow, Polars is compatible with a wide range of libraries that also make use of Apache Arrow, like Pandas and DuckDB.

### Data visualisation

#### [hvPlot](https://hvplot.holoviz.org/)

hvPlot is available as the default plotting backend for Polars making it simple to create interactive and static visualisations. You can use hvPlot by using the feature flag `plot` during installing.

```python
pip install 'polars[plot]'
```

#### [Matplotlib](https://matplotlib.org/)

Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python. Matplotlib makes easy things easy and hard things possible.

#### [Plotly Dash](https://github.com/plotly/dash)

Dash is the original low-code framework for rapidly building data apps in Python. Learn more about how to build fast Dash apps at [Plotly.com](https://plotly.com/blog/polars-to-build-fast-dash-apps-for-large-datasets/).

#### [Seaborn](https://seaborn.pydata.org/)

Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

### Machine Learning

#### [Scikit Learn](https://scikit-learn.org/stable/)

Since Scikit Learn 1.4, all transformers support Polars output. See the change log for [more details](https://scikit-learn.org/dev/whats_new/v1.4.html#changes-impacting-all-modules).

### Geospatial

#### [GeoPolars](https://github.com/geopolars/geopolars)

GeoPolars extends the Polars DataFrame library for use with geospatial data.

### IO

#### [Delta Lake](https://github.com/delta-io/delta-rs)

The Delta Lake project aims to unlock the power of the Deltalake for as many users and projects as possible by providing native low-level APIs aimed at developers and integrators, as well as a high-level operations API that lets you query, inspect, and operate your Delta Lake with ease.

Read how to use Delta Lake with Polars [at Delta Lake](https://delta-io.github.io/delta-rs/integrations/delta-lake-polars/#reading-a-delta-lake-table-with-polars).

### Other

#### [Great Tables](https://posit-dev.github.io/great-tables/articles/intro.html)

With Great Tables anyone can make wonderful-looking tables in Python. Here is a [blog post](https://posit-dev.github.io/great-tables/blog/polars-styling/) on how to use Great Tables with Polars.

#### [LanceDB](https://lancedb.com/)

LanceDB is a developer-friendly, serverless vector database for AI applications. They have added a direct integration with Polars. LanceDB can ingest Polars dataframes, return results as polars dataframes, and export the entire table as a polars lazyframe. Read more about in their blog [LanceDB + Polars](https://blog.lancedb.com/lancedb-polars-2d5eb32a8aa3)

#### [Mage](https://www.mage.ai)

Open-source data pipeline tool for transforming and integrating data. Learn about integration between Polars and Mage at [docs.mage.ai](https://docs.mage.ai/integrations/polars).
