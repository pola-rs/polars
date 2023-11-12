# Series & DataFrames

The core base data structures provided by Polars are `Series` and `DataFrames`.

## Series

Series are a 1-dimensional data structure. Within a series all elements have the same data type (e.g. int, string).
The snippet below shows how to create a simple named `Series` object. In a later section of this getting started guide we will learn how to read data from external sources (e.g. files, database), for now lets keep it simple.

{{code_block('user-guide/basics/series-dataframes','series',['Series'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:series"
```

### Methods

Although it is more common to work directly on a `DataFrame` object, `Series` implement a number of base methods which make it easy to perform transformations. Below are some examples of common operations you might want to perform. Note that these are for illustration purposes and only show a small subset of what is available.

##### Aggregations

`Series` out of the box supports all basic aggregations (e.g. min, max, mean, mode, ...).

{{code_block('user-guide/basics/series-dataframes','minmax',['min','max'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:minmax"
```

##### String

There are a number of methods related to string operations in the `StringNamespace`. These only work on `Series` with the Datatype `Utf8`.

{{code_block('user-guide/basics/series-dataframes','string',['str.replace'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:string"
```

##### Datetime

Similar to strings, there is a separate namespace for datetime related operations in the `DateLikeNameSpace`. These only work on `Series`with DataTypes related to dates.

{{code_block('user-guide/basics/series-dataframes','dt',['Series.dt.day'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:dt"
```

## DataFrame

A `DataFrame` is a 2-dimensional data structure that is backed by a `Series`, and it could be seen as an abstraction of on collection (e.g. list) of `Series`. Operations that can be executed on `DataFrame` are very similar to what is done in a `SQL` like query. You can `GROUP BY`, `JOIN`, `PIVOT`, but also define custom functions. In the next pages we will cover how to perform these transformations.

{{code_block('user-guide/basics/series-dataframes','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:dataframe"
```

### Viewing data

This part focuses on viewing data in a `DataFrame`. We will use the `DataFrame` from the previous example as a starting point.

#### Head

The `head` function shows by default the first 5 rows of a `DataFrame`. You can specify the number of rows you want to see (e.g. `df.head(10)`).

{{code_block('user-guide/basics/series-dataframes','head',['head'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:head"
```

#### Tail

The `tail` function shows the last 5 rows of a `DataFrame`. You can also specify the number of rows you want to see, similar to `head`.

{{code_block('user-guide/basics/series-dataframes','tail',['tail'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:tail"
```

#### Sample

If you want to get an impression of the data of your `DataFrame`, you can also use `sample`. With `sample` you get an _n_ number of random rows from the `DataFrame`.

{{code_block('user-guide/basics/series-dataframes','sample',['sample'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:sample"
```

#### Describe

`Describe` returns summary statistics of your `DataFrame`. It will provide several quick statistics if possible.

{{code_block('user-guide/basics/series-dataframes','describe',['describe'])}}

```python exec="on" result="text" session="getting-started/series"
--8<-- "python/user-guide/basics/series-dataframes.py:describe"
```
