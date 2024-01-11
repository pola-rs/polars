# Getting started

This chapter is here to help you get started with Polars. It covers all the fundamental features and functionalities of the library, making it easy for new users to familiarise themselves with the basics from initial installation and setup to core functionalities. If you're already an advanced user or familiar with Dataframes, feel free to skip ahead to the [next chapter about installation options](installation.md).

## Installing Polars

=== ":fontawesome-brands-python: Python"

    ``` bash
    pip install polars
    ```

=== ":fontawesome-brands-rust: Rust"

    ``` shell
    cargo add polars -F lazy

    # Or Cargo.toml
    [dependencies]
    polars = { version = "x", features = ["lazy", ...]}
    ```

## Reading & writing

Polars supports reading and writing to all common files (e.g. csv, json, parquet), cloud storage (S3, Azure Blob, BigQuery) and databases (e.g. postgres, mysql). In the following examples we will show how to operate on most common file formats. For the following dataframe

{{code_block('user-guide/basics/reading-writing','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:dataframe"
```

### CSV

Polars has its own fast implementation for csv reading with many flexible configuration options.

{{code_block('user-guide/basics/reading-writing','csv',['read_csv','write_csv'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:csv"
```

As we can see above, Polars made the datetimes a `string`. We can tell Polars to parse dates, when reading the csv, to ensure the date becomes a datetime. The example can be found below:

{{code_block('user-guide/basics/reading-writing','csv2',['read_csv'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:csv2"
```

### JSON

{{code_block('user-guide/basics/reading-writing','json',['read_json','write_json'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:json"
```

### Parquet

{{code_block('user-guide/basics/reading-writing','parquet',['read_parquet','write_parquet'])}}

```python exec="on" result="text" session="getting-started/reading"
--8<-- "python/user-guide/basics/reading-writing.py:parquet"
```

To see more examples and other data formats go to the [User Guide](io/csv.md), section IO.

## Expressions

`Expressions` are the core strength of Polars. The `expressions` offer a versatile structure that both solves easy queries and is easily extended to complex ones. Below we cover the basic components that serve as building block (or in Polars terminology contexts) for all your queries:

- `select`
- `filter`
- `with_columns`
- `group_by`

To learn more about expressions and the context in which they operate, see the User Guide sections: [Contexts](concepts/contexts.md) and [Expressions](concepts/expressions.md).

### Select statement

To select a column we need to do two things. Define the `DataFrame` we want the data from. And second, select the data that we need. In the example below you see that we select `col('*')`. The asterisk stands for all columns.

{{code_block('user-guide/basics/expressions','select',['select'])}}

```python exec="on" result="text" session="getting-started/expressions"
--8<-- "python/user-guide/basics/expressions.py:setup"
print(
    --8<-- "python/user-guide/basics/expressions.py:select"
)
```

You can also specify the specific columns that you want to return. There are two ways to do this. The first option is to pass the column names, as seen below.

{{code_block('user-guide/basics/expressions','select2',['select'])}}

```python exec="on" result="text" session="getting-started/expressions"
print(
    --8<-- "python/user-guide/basics/expressions.py:select2"
)
```

The second option is to specify each column using `pl.col`. This option is shown below.

{{code_block('user-guide/basics/expressions','select3',['select'])}}

```python exec="on" result="text" session="getting-started/expressions"
print(
    --8<-- "python/user-guide/basics/expressions.py:select3"
)
```

If you want to exclude an entire column from your view, you can simply use `exclude` in your `select` statement.

{{code_block('user-guide/basics/expressions','exclude',['select'])}}

```python exec="on" result="text" session="getting-started/expressions"
print(
    --8<-- "python/user-guide/basics/expressions.py:exclude"
)
```

### Filter

The `filter` option allows us to create a subset of the `DataFrame`. We use the same `DataFrame` as earlier and we filter between two specified dates.

{{code_block('user-guide/basics/expressions','filter',['filter'])}}

```python exec="on" result="text" session="getting-started/expressions"
print(
    --8<-- "python/user-guide/basics/expressions.py:filter"
)
```

With `filter` you can also create more complex filters that include multiple columns.

{{code_block('user-guide/basics/expressions','filter2',['filter'])}}

```python exec="on" result="text" session="getting-started/expressions"
print(
    --8<-- "python/user-guide/basics/expressions.py:filter2"
)
```

### With_columns

`with_columns` allows you to create new columns for your analyses. We create two new columns `e` and `b+42`. First we sum all values from column `b` and store the results in column `e`. After that we add `42` to the values of `b`. Creating a new column `b+42` to store these results.

{{code_block('user-guide/basics/expressions','with_columns',['with_columns'])}}

```python exec="on" result="text" session="getting-started/expressions"
print(
    --8<-- "python/user-guide/basics/expressions.py:with_columns"
)
```

### Group by

We will create a new `DataFrame` for the Group by functionality. This new `DataFrame` will include several 'groups' that we want to group by.

{{code_block('user-guide/basics/expressions','dataframe2',['DataFrame'])}}

```python exec="on" result="text" session="getting-started/expressions"
--8<-- "python/user-guide/basics/expressions.py:dataframe2"
print(df2)
```

{{code_block('user-guide/basics/expressions','group_by',['group_by'])}}

```python exec="on" result="text" session="getting-started/expressions"
print(
    --8<-- "python/user-guide/basics/expressions.py:group_by"
)
```

{{code_block('user-guide/basics/expressions','group_by2',['group_by'])}}

```python exec="on" result="text" session="getting-started/expressions"
print(
    --8<-- "python/user-guide/basics/expressions.py:group_by2"
)
```

### Combining operations

Below are some examples on how to combine operations to create the `DataFrame` you require.

{{code_block('user-guide/basics/expressions','combine',['select','with_columns'])}}

```python exec="on" result="text" session="getting-started/expressions"
--8<-- "python/user-guide/basics/expressions.py:combine"
```

{{code_block('user-guide/basics/expressions','combine2',['select','with_columns'])}}

```python exec="on" result="text" session="getting-started/expressions"
--8<-- "python/user-guide/basics/expressions.py:combine2"
```

## Combining DataFrames

There are two ways `DataFrame`s can be combined depending on the use case: join and concat.

### Join

Polars supports all types of join (e.g. left, right, inner, outer). Let's have a closer look on how to `join` two `DataFrames` into a single `DataFrame`. Our two `DataFrames` both have an 'id'-like column: `a` and `x`. We can use those columns to `join` the `DataFrames` in this example.

{{code_block('user-guide/basics/joins','join',['join'])}}

```python exec="on" result="text" session="getting-started/joins"
--8<-- "python/user-guide/basics/joins.py:setup"
--8<-- "python/user-guide/basics/joins.py:join"
```

To see more examples with other types of joins, see the [Transformations section](transformations/joins.md) in the user guide.

### Concat

We can also `concatenate` two `DataFrames`. Vertical concatenation will make the `DataFrame` longer. Horizontal concatenation will make the `DataFrame` wider. Below you can see the result of an horizontal concatenation of our two `DataFrames`.

{{code_block('user-guide/basics/joins','hstack',['hstack'])}}

```python exec="on" result="text" session="getting-started/joins"
--8<-- "python/user-guide/basics/joins.py:hstack"
```
