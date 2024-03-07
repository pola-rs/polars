# Pivots

`pivot` transforms a "long-format" `DataFrame`, where each row represents an
observation, into a "wide-format" one, where each element represents an
observation.

To perform a pivot, specify one or more columns for each of `values`, `index`,
and `columns`, either by name or via selectors. Typically, the columns in
`values`, `index`, and `columns` are mutually exclusive; specifying overlapping
columns will not give an error, but is rarely useful.

In the simplest case where `values`, `index` and `columns` are each a single
column:

- Each unique value of the `index` column will become the name of a row in
  the pivoted `DataFrame`. The first column of the pivoted `DataFrame` will
  contain these row names.
- Each unique value of the `columns` column will become the name of a column
  in the pivoted `DataFrame`.
- Each value of the `values` column will become a value in the pivoted
  `DataFrame`. For instance, if the nth row of the input `DataFrame` is
  `("values_n", "index_n", "columns_n")`, then the value `"values_n"` will
  be placed at row `"index_n"` (i.e. the row where the `index` column has
  the value `index_n`) and column `"columns_n"`.

Thus, in this simple case where `values`, `index` and `columns` are each a
single column, if there are `N` unique values in the `columns` column, there
will be `N + 1` columns in the pivoted `DataFrame`: one for the row names, the
remaining `N` for the values.

If there are multiple `index` columns instead of one, each unique _combination_
of their values will become a row in the pivoted `DataFrame`, and there will be
`len(index)` columns of row names instead of one.

If there are multiple `columns` columns instead of one, the result will be the
same as if you had combined them into a single `struct` column beforehand. In
other words, `df.pivot(..., columns=['a', 'b', 'c'])` is equivalent to
`df.with_columns(foo=pl.struct(['a', 'b', 'c']).pivot(..., columns='foo')`,
assuming `foo` is not already a column in `df`.

If there are multiple `values` columns instead of one, the pivot will be done
independently for each of the columns in `values`, and the results will be
concatenated horizontally. To avoid having duplicate column names, the names
of the non-index columns will be prefixed with `f'{value}_{columns}_'`, where
`value` is the column name in `values` from which the column's values are
taken. The `'_'` can be changed to a different string using the `separator`
argument.

When multiple rows of the input `DataFrame` have the same `values` for all the
columns in `index` and `columns`, `pivot` will raise an error unless these
multiple values are aggregated into a single value before pivoting. This can be
done prior to pivoting with a :func:`group_by`, but `pivot` also provides a
convenient way to do this aggregation internally, by specifying the
`aggregate_function` argument. You can specify one of 8 predefined aggregation
functions as strings:

- `'first'`
- `'last'`
- `'sum'`
- `'max'`
- `'min'`
- `'mean'`
- `'median'`
- `'len'`

or provide an expression that performs a custom aggregation, where
`pl.element()` represents the multiple `values` in each "group" with the same
`index` and `columns`. For example, `aggregate_function='mean'` is short for
`aggregate_function=pl.element().mean()`.

## Dataset

{{code_block('user-guide/transformations/pivot','df',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/pivot"
--8<-- "python/user-guide/transformations/pivot.py:setup"
--8<-- "python/user-guide/transformations/pivot.py:df"
```

## Eager

{{code_block('user-guide/transformations/pivot','eager',['pivot'])}}

```python exec="on" result="text" session="user-guide/transformations/pivot"
--8<-- "python/user-guide/transformations/pivot.py:eager"
```

## Lazy

A Polars `LazyFrame` always need to know the schema of a computation statically
(before collecting the query). Since the schema of a pivoted DataFrame depends
on the data, it is impossible to determine the schema without running the
query. As a result, `pivot` is not available in lazy mode. To use `collect()`
in a `LazyFrame` pipe chain, you must include a `collect()` before pivoting and
a `lazy()` after pivoting:

{{code_block('user-guide/transformations/pivot','lazy',['pivot'])}}

```python exec="on" result="text" session="user-guide/transformations/pivot"
--8<-- "python/user-guide/transformations/pivot.py:lazy"
```
