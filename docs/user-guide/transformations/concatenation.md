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

Vertical concatenation fails when the dataframes do not have the same column names.

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

In a diagonal concatenation you combine all of the row and columns from a list of `DataFrames` into a single longer and/or wider `DataFrame`.

{{code_block('user-guide/transformations/concatenation','cross',['concat'])}}

```python exec="on" result="text" session="user-guide/transformations/concatenation"
--8<-- "python/user-guide/transformations/concatenation.py:cross"
```

Diagonal concatenation generates nulls when the column names do not overlap.

When the dataframe shapes do not match and we have an overlapping semantic key then [we can join the dataframes](joins.md) instead of concatenating them.

## Rechunking

Before a concatenation we have two dataframes `df1` and `df2`. Each column in `df1` and `df2` is in one or more chunks in memory. By default, during concatenation the chunks in each column are copied to a single new chunk - this is known as **rechunking**. Rechunking is an expensive operation, but is often worth it because future operations will be faster.
If you do not want Polars to rechunk the concatenated `DataFrame` you specify `rechunk = False` when doing the concatenation.
