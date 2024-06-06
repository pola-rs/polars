# Concatenation

There are a number of ways to concatenate data from separate DataFrames:

- two dataframes with **the same columns** can be **vertically** concatenated to make a **longer** dataframe
- two dataframes with **non-overlapping columns** can be **horizontally** concatenated to make a **wider** dataframe
- two dataframes with **different numbers of rows and columns** can be **diagonally** concatenated to make a dataframe which might be longer and/ or wider. Where column names overlap values will be vertically concatenated. Where column names do not overlap new rows and columns will be added. Missing values will be set as `null`

## Vertical concatenation - getting longer

In a vertical concatenation you combine all of the rows from a list of `DataFrames` into a single longer `DataFrame`.

{{code_block('user-guide/transformations/concatenation','vertical',['concat'])}}

```python exec="on" result="text" session="user-guide/transformations/concatenation"
--8<-- "python/user-guide/transformations/concatenation.py:setup"
--8<-- "python/user-guide/transformations/concatenation.py:vertical"
```

Vertical concatenation fails when the dataframes do not have the same column names and dtypes.

For certain differences in dtypes, Polars can do a relaxed vertical concatenation where the differences in dtype are resolved by casting all columns with the same name but different dtypes to a *supertype*. For example, if column `'a'` in the first `DataFrame` is `Float32` but column `'a'` in the second `DataFrame` is `Int64`, then both columns are cast to their supertype `Float64` before concatenation. If the set of dtypes for a column do not have a supertype, the concatenation fails. The supertype mappings are defined internally in Polars.

{{code_block('user-guide/transformations/concatenation','vertical_relaxed',['concat'])}}

```python exec="on" result="text" session="user-guide/transformations/concatenation"
--8<-- "python/user-guide/transformations/concatenation.py:vertical_relaxed"
```
## Horizontal concatenation - getting wider

In a horizontal concatenation you combine all of the columns from a list of `DataFrames` into a single wider `DataFrame`.

{{code_block('user-guide/transformations/concatenation','horizontal',['concat'])}}

```python exec="on" result="text" session="user-guide/transformations/concatenation"
--8<-- "python/user-guide/transformations/concatenation.py:horizontal"
```

Horizontal concatenation fails when dataframes have overlapping columns.

When dataframes have different numbers of rows,
columns will be padded with `null` values at the end up to the maximum length.

{{code_block('user-guide/transformations/concatenation','horizontal_different_lengths',['concat'])}}

```python exec="on" result="text" session="user-guide/transformations/concatenation"
--8<-- "python/user-guide/transformations/concatenation.py:horizontal_different_lengths"
```

## Diagonal concatenation - getting longer, wider and `null`ier

In a diagonal concatenation you combine all of the rows and columns from a list of `DataFrames` into a single longer and/or wider `DataFrame`.

{{code_block('user-guide/transformations/concatenation','cross',['concat'])}}

```python exec="on" result="text" session="user-guide/transformations/concatenation"
--8<-- "python/user-guide/transformations/concatenation.py:cross"
```

Diagonal concatenation generates nulls when the column names do not overlap but fails if the dtypes do not match for columns with the same name. As with vertical concatenation there is an alternative `diagonal_relaxed` method that tries to cast columns to a supertype if columns with the same name have different dtypes.

When the dataframe shapes do not match and we have an overlapping semantic key then [we can join the dataframes](joins.md) instead of concatenating them.

## Rechunking

We have a `list` of `DataFrames` and we want to concatenate them. Each column in each `DataFrame` is stored in one or more chunks in memory. When we concatenate the `DataFrames` then the data from each column in each `DataFrame` can be copied to a single location in memory - this is known as **rechunking**. Rechunking is an expensive process as it requires copying data from one location to another. However, rechunking can make subsequent operations faster as the data is in a single location in memory.

By default when we do a concatenation in eager mode rechunking does not happen. If we want Polars to rechunk the concatenated `DataFrame` then specify `rechunk = True` when doing the concatenation. In lazy mode the query optimizer assesses whether to do rechunking based on the query plan.
