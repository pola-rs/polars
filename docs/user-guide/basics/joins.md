# Combining DataFrames

There are two ways `DataFrame`s can be combined depending on the use case: join and concat.

## Join

Polars supports all types of join (e.g. left, right, inner, outer). Let's have a closer look on how to `join` two `DataFrames` into a single `DataFrame`. Our two `DataFrames` both have an 'id'-like column: `a` and `x`. We can use those columns to `join` the `DataFrames` in this example.

{{code_block('getting-started/joins','join',['join'])}}

```python exec="on" result="text" session="getting-started/joins"
--8<-- "python/getting-started/joins.py:setup"
--8<-- "python/getting-started/joins.py:join"
```

To see more examples with other types of joins, go the [User Guide](../transformations/joins.md).

## Concat

We can also `concatenate` two `DataFrames`. Vertical concatenation will make the `DataFrame` longer. Horizontal concatenation will make the `DataFrame` wider. Below you can see the result of an horizontal concatenation of our two `DataFrames`.

{{code_block('getting-started/joins','hstack',['hstack'])}}

```python exec="on" result="text" session="getting-started/joins"
--8<-- "python/getting-started/joins.py:hstack"
```
