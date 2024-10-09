# Getting started

This chapter is here to help you get started with Polars. It covers all the fundamental features and functionalities of the library, making it easy for new users to familiarise themselves with the basics from initial installation and setup to core functionalities. If you're already an advanced user or familiar with dataframes, feel free to skip ahead to the [next chapter about installation options](installation.md).

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

Polars supports reading and writing for common file formats (e.g., csv, json, parquet), cloud storage (S3, Azure Blob, BigQuery) and databases (e.g., postgres, mysql). Below, we create a small dataframe and show how to write it to disk and read it back.

{{code_block('user-guide/getting-started','df',['DataFrame'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:df"
```

In the example below we write the dataframe to a csv file called `output.csv`. After that, we read it back using `read_csv` and then print the result for inspection.

{{code_block('user-guide/getting-started','csv',['read_csv','write_csv'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:csv"
```

For more examples on the CSV file format and other data formats, see the [IO section](io/index.md) of the user guide.

## Expressions and contexts

_Expressions_ are one of the main strengths of Polars because they provide a modular and flexible way of expressing data transformations.

Here is an example of a Polars expression:

```py
pl.col("weight") / (pl.col("height") ** 2)
```

As you might be able to guess, this expression takes the column named “weight” and divides its values by the square of the values in the column “height”, computing a person's BMI.
Note that the code above expresses an abstract computation: it's only inside a Polars _context_ that the expression materalizes into a series with the results.

Below, we will show examples of Polars expressions inside different contexts:

- `select`
- `with_columns`
- `filter`
- `group_by`

For a more [detailed exploration of expressions and contexts see the respective user guide section](concepts/expressions-and-contexts.md).

### `select`

The context `select` allows you to select and manipulate columns from a dataframe.
In the simplest case, each expression you provide will map to a column in the result dataframe:

{{code_block('user-guide/getting-started','select',['select','alias','Expr.dt'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:select"
```

Polars also supports a feature called “expression expansion”, in which one expression acts as shorthand for multiple expressions.
In the example below, we use expression expansion to manipulate the columns “weight” and “height” with a single expression.
When using expression expansion you can use `.name.suffix` to add a suffix to the names of the original columns:

{{code_block('user-guide/getting-started','expression-expansion',['select','alias','Expr.name'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:expression-expansion"
```

You can check other sections of the user guide to learn more about [basic operations](expressions/operators.md) or [column selections](expressions/column-selections.md).

### `with_columns`

The context `with_columns` is very similar to the context `select` but `with_columns` adds columns to the dataframe instead of selecting them.
Notice how the resulting dataframe contains the four columns of the original dataframe plus the two new columns introduced by the expressions inside `with_columns`:

{{code_block('user-guide/getting-started','with_columns',['with_columns'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:with_columns"
```

In the example above we also decided to use named expressions instead of the method `alias` to specify the names of the new columns.
Other contexts like `select` and `group_by` also accept named expressions.

### `filter`

The context `filter` allows us to create a second dataframe with a subset of the rows of the original one:

{{code_block('user-guide/getting-started','filter',['filter','Expr.dt'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:filter"
```

You can also provide multiple predicate expressions as separate parameters, which is more convenient than putting them all together with `&`:

{{code_block('user-guide/getting-started','filter-multiple',['filter','is_between'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:filter-multiple"
```

### `group_by`

The context `group_by` can be used to group together the rows of the dataframe that share the same value across one or more expressions.
The example below counts how many people were born in each decade:

{{code_block('user-guide/getting-started','group_by',['group_by','alias','Expr.dt'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:group_by"
```

The keyword argument `maintain_order` forces Polars to present the resulting groups in the same order as they appear in the original dataframe.
This slows down the grouping operation but is used here to ensure reproducibility of the examples.

After using the context `group_by` we can use `agg` to compute aggregations over the resulting groups:

{{code_block('user-guide/getting-started','group_by-agg',['group_by','agg'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:group_by-agg"
```

### More complex queries

Contexts and the expressions within can be chained to create more complex queries according to your needs.
In the example below we combine some of the contexts we have seen so far to create a more complex query:

{{code_block('user-guide/getting-started','complex',['group_by','agg','select','with_columns','Expr.str','Expr.list'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:complex"
```

## Combining dataframes

Polars provides a number of tools to combine two dataframes.
In this section, we show an example of a join and an example of a concatenation.

### Joinining dataframes

Polars provides many different join algorithms.
The example below shows how to use a left outer join to combine two dataframes when a column can be used as a unique identifier to establish a correspondence between rows across the dataframes:

{{code_block('user-guide/getting-started','join',['join'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:join"
```

Polars provides many different join algorithms that you can learn about in the [joins section of the user guide](transformations/joins.md).

### Concatenating dataframes

Concatenating dataframes creates a taller or wider dataframe, depending on the method used.
Assuming we have a second dataframe with data from other people, we could use vertical concatenation to create a taller dataframe:

{{code_block('user-guide/getting-started','concat',['concat'])}}

```python exec="on" result="text" session="getting-started"
--8<-- "python/user-guide/getting-started.py:concat"
```

Polars provides vertical and horizontal concatenation, as well as diagonal concatenation.
You can learn more about these in the [concatenations section of the user guide](transformations/concatenation.md).
