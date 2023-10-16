# Column selections

Let's create a dataset to use in this section:

{{code_block('user-guide/expressions/column-selections','selectors_df',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:setup"
--8<-- "python/user-guide/expressions/column-selections.py:selectors_df"
```

## Expression expansion

As we've seen in the previous section, we can select specific columns using the `pl.col` method. It can also select multiple columns - both as a means of convenience, and to _expand_ the expression.

This kind of convenience feature isn't just decorative or syntactic sugar. It allows for a very powerful application of [DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principles in your code: a single expression that specifies multiple columns expands into a list of expressions (depending on the DataFrame schema), resulting in being able to select multiple columns + run computation on them!

### Select all, or all but some

We can select all columns in the `DataFrame` object by providing the argument `*`:

{{code_block('user-guide/expressions/column-selections', 'all',['all'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:all"
```

Often, we don't just want to include all columns, but include all _while_ excluding a few. This can be done easily as well:

{{code_block('user-guide/expressions/column-selections','exclude',['exclude'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:exclude"
```

### By multiple strings

Specifying multiple strings allows expressions to _expand_ to all matching columns:

{{code_block('user-guide/expressions/column-selections','expansion_by_names',['dt.to_string'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:expansion_by_names"
```

### By regular expressions

Multiple column selection is possible by regular expressions also, by making sure to wrap the regex by `^` and `$` to let `pl.col` know that a regex selection is expected:

{{code_block('user-guide/expressions/column-selections','expansion_by_regex',[])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:expansion_by_regex"
```

### By data type

`pl.col` can select multiple columns using Polars data types:

{{code_block('user-guide/expressions/column-selections','expansion_by_dtype',['n_unique'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:expansion_by_dtype"
```

## Using `selectors`

Polars also allows for the use of intuitive selections for columns based on their name, `dtype` or other properties; and this is built on top of existing functionality outlined in `col` used above. It is recommended to use them by importing and aliasing `polars.selectors` as `cs`.

### By `dtype`

To select just the integer and string columns, we can do:

{{code_block('user-guide/expressions/column-selections','selectors_intro',['selectors'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:selectors_intro"
```

### Applying set operations

These _selectors_ also allow for set based selection operations. For instance, to select the **numeric** columns **except** the **first** column that indicates row numbers:

{{code_block('user-guide/expressions/column-selections','selectors_diff',['cs.first', 'cs.numeric'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:selectors_diff"
```

We can also select the row number by name **and** any **non**-numeric columns:

{{code_block('user-guide/expressions/column-selections','selectors_union',['cs.by_name', 'cs.numeric'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:selectors_union"
```

### By patterns and substrings

_Selectors_ can also be matched by substring and regex patterns:

{{code_block('user-guide/expressions/column-selections','selectors_by_name',['cs.contains', 'cs.matches'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:selectors_by_name"
```

### Converting to expressions

What if we want to apply a specific operation on the selected columns (i.e. get back to representing them as **expressions** to operate upon)? We can simply convert them using `as_expr` and then proceed as normal:

{{code_block('user-guide/expressions/column-selections','selectors_to_expr',['cs.temporal'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:selectors_to_expr"
```

### Debugging `selectors`

Polars also provides two helpful utility functions to aid with using selectors: `is_selector` and `expand_selector`:

{{code_block('user-guide/expressions/column-selections','selectors_is_selector_utility',['is_selector'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:selectors_is_selector_utility"
```

To predetermine the column names that are selected, which is especially useful for a LazyFrame object:

{{code_block('user-guide/expressions/column-selections','selectors_colnames_utility',['expand_selector'])}}

```python exec="on" result="text" session="user-guide/column-selections"
--8<-- "python/user-guide/expressions/column-selections.py:selectors_colnames_utility"
```
