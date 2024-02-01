# Data structures

The core base data structures provided by Polars are `Series` and `DataFrame`.

## Series

Series are a 1-dimensional data structure. Within a series all elements have the same [Data Type](data-types/overview.md) .
The snippet below shows how to create a simple named `Series` object.

{{code_block('user-guide/basics/series-dataframes','series',['Series'])}}

```python exec="on" result="text" session="user-guide/data-structures"
--8<-- "python/user-guide/basics/series-dataframes.py:series"
```

## DataFrame

A `DataFrame` is a 2-dimensional data structure that is backed by a `Series`, and it can be seen as an abstraction of a collection (e.g. list) of `Series`. Operations that can be executed on a `DataFrame` are very similar to what is done in a `SQL` like query. You can `GROUP BY`, `JOIN`, `PIVOT`, but also define custom functions.

{{code_block('user-guide/basics/series-dataframes','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/data-structures"
--8<-- "python/user-guide/basics/series-dataframes.py:dataframe"
```

### Viewing data

This part focuses on viewing data in a `DataFrame`. We will use the `DataFrame` from the previous example as a starting point.

#### Head

The `head` function shows by default the first 5 rows of a `DataFrame`. You can specify the number of rows you want to see (e.g. `df.head(10)`).

{{code_block('user-guide/basics/series-dataframes','head',['head'])}}

```python exec="on" result="text" session="user-guide/data-structures"
--8<-- "python/user-guide/basics/series-dataframes.py:head"
```

#### Tail

The `tail` function shows the last 5 rows of a `DataFrame`. You can also specify the number of rows you want to see, similar to `head`.

{{code_block('user-guide/basics/series-dataframes','tail',['tail'])}}

```python exec="on" result="text" session="user-guide/data-structures"
--8<-- "python/user-guide/basics/series-dataframes.py:tail"
```

#### Sample

If you want to get an impression of the data of your `DataFrame`, you can also use `sample`. With `sample` you get an _n_ number of random rows from the `DataFrame`.

{{code_block('user-guide/basics/series-dataframes','sample',['sample'])}}

```python exec="on" result="text" session="user-guide/data-structures"
--8<-- "python/user-guide/basics/series-dataframes.py:sample"
```

#### Describe

`Describe` returns summary statistics of your `DataFrame`. It will provide several quick statistics if possible.

{{code_block('user-guide/basics/series-dataframes','describe',['describe'])}}

```python exec="on" result="text" session="user-guide/data-structures"
--8<-- "python/user-guide/basics/series-dataframes.py:describe"
```
