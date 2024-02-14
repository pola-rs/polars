# Melts

`melt` is the opposite of `pivot`: it transforms a "wide-format" `DataFrame`, 
where each element represents an observation, into a "long-format" one, where 
each row represents an observation.

To perform a melt, specify one or more columns as identifier variables (via the 
`id_vars` argument) and other columns as value variables (via the `value_vars`
argument), either by name or via selectors. Typically, the columns in `id_vars`
and `value_vars` are mutually exclusive; specifying overlapping columns will
not give an error, but is rarely useful. If `value_vars` is `None`, all
remaining columns not in `id_vars` will be treated as `value_vars`.

Each element in each of the `value_vars` columns of the input `DataFrame`
(including `null` elements) will become its own row in the output `DataFrame`.
The row for that element will contain `len(id_vars) + 2` columns:

- One column for each of the `id_vars`, containing the values of the `id_vars` 
  columns that were on same row as that element in the input `DataFrame`. You
  can think of these as the element's row names.
- One column called `'variable'` containing the name of the column in which 
  that element appeared, i.e. the element's column name. You can change the
  name of this column by specifying the `variable_name` argument.
- One column called `'value'` containing the element itself. You can change the
  name of this column by specifying the `value_name` argument.

## Dataset

{{code_block('user-guide/transformations/melt','df',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/transformations/melt"
--8<-- "python/user-guide/transformations/melt.py:df"
```

## Eager + lazy

Unlike `pivot`, `melt` works in both eager and lazy mode, with the same API.
This is because all the column names in the output `DataFrame` are known in
advance, and do not depend on the data.

{{code_block('user-guide/transformations/melt','melt',['melt'])}}

```python exec="on" result="text" session="user-guide/transformations/melt"
--8<-- "python/user-guide/transformations/melt.py:melt"
```
