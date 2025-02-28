# Expression expansion

As you've seen in
[the section about expressions and contexts](../concepts/expressions-and-contexts.md), expression
expansion is a feature that enables you to write a single expression that can expand to multiple
different expressions, possibly depending on the schema of the context in which the expression is
used.

This feature isn't just decorative or syntactic sugar. It allows for a very powerful application of
[DRY](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself) principles in your code: a single
expression that specifies multiple columns expands into a list of expressions, which means you can
write one single expression and reuse the computation that it represents.

In this section we will show several forms of expression expansion and we will be using the
dataframe that you can see below for that effect:

{{code_block('user-guide/expressions/expression-expansion', 'df', [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:df"
```

## Function `col`

The function `col` is the most common way of making use of expression expansion features in Polars.
Typically used to refer to one column of a dataframe, in this section we explore other ways in which
you can use `col` (or its variants, when in Rust).

### Explicit expansion by column name

The simplest form of expression expansion happens when you provide multiple column names to the
function `col`.

The example below uses a single function `col` with multiple column names to convert the values in
USD to EUR:

{{code_block('user-guide/expressions/expression-expansion', 'col-with-names', ['col'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:col-with-names"
```

When you list the column names you want the expression to expand to, you can predict what the
expression will expand to. In this case, the expression that does the currency conversion is
expanded to a list of five expressions:

{{code_block('user-guide/expressions/expression-expansion', 'expression-list', ['col'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:expression-list"
```

### Expansion by data type

We had to type five column names in the previous example but the function `col` can also
conveniently accept one or more data types. If you provide data types instead of column names, the
expression is expanded to all columns that match one of the data types provided.

The example below performs the exact same computation as before:

{{code_block('user-guide/expressions/expression-expansion', 'col-with-dtype', [], ['col'],
['dtype_col'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:col-with-dtype"
```

When we use a data type with expression expansion we cannot know, beforehand, how many columns a
single expression will expand to. We need the schema of the input dataframe if we want to determine
what is the final list of expressions that is to be applied.

If we weren't sure about whether the price columns where of the type `Float64` or `Float32`, we
could specify both data types:

{{code_block('user-guide/expressions/expression-expansion', 'col-with-dtypes', [], ['col'],
['dtype_cols'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:col-with-dtypes"
```

### Expansion by pattern matching

You can also use regular expressions to specify patterns that are used to match the column names. To
distinguish between a regular column name and expansion by pattern matching, regular expressions
start and end with `^` and `$`, respectively. This also means that the pattern must match against
the whole column name string.

Regular expressions can be mixed with regular column names:

{{code_block('user-guide/expressions/expression-expansion', 'col-with-regex', ['col'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:col-with-regex"
```

### Arguments cannot be of mixed types

In Python, the function `col` accepts an arbitrary number of strings (as
[column names](#explicit-expansion-by-column-name) or as
[regular expressions](#expansion-by-pattern-matching)) or an arbitrary number of data types, but you
cannot mix both in the same function call:

```python
--8<-- "python/user-guide/expressions/expression-expansion.py:col-error"
```

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:col-error"
```

## Selecting all columns

Polars provides the function `all` as shorthand notation to refer to all columns of a dataframe:

{{code_block('user-guide/expressions/expression-expansion', 'all', ['all'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:all"
```

!!! note

    The function `all` is syntactic sugar for `col("*")`, but since the argument `"*"` is a special case and `all` reads more like English, the usage of `all` is preferred.

## Excluding columns

Polars also provides a mechanism to exclude certain columns from expression expansion. For that, you
use the function `exclude`, which accepts exactly the same types of arguments as `col`:

{{code_block('user-guide/expressions/expression-expansion', 'all-exclude', ['exclude'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:all-exclude"
```

Naturally, the function `exclude` can also be used after the function `col`:

{{code_block('user-guide/expressions/expression-expansion', 'col-exclude', ['exclude'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:col-exclude"
```

## Column renaming

By default, when you apply an expression to a column, the result keeps the same name as the original
column.

Preserving the column name can be semantically wrong and in certain cases Polars may even raise an
error if duplicate names occur:

{{code_block('user-guide/expressions/expression-expansion', 'duplicate-error', [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:duplicate-error"
```

To prevent errors like this, and to allow users to rename their columns when appropriate, Polars
provides a series of functions that let you change the name of a column or a group of columns.

### Renaming a single column with `alias`

The function `alias` has been used thoroughly in the documentation already and it lets you rename a
single column:

{{code_block('user-guide/expressions/expression-expansion', 'alias', ['alias'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:alias"
```

### Prefixing and suffixing column names

When using expression expansion you cannot use the function `alias` because the function `alias` is
designed specifically to rename a single column.

When it suffices to add a static prefix or a static suffix to the existing names, we can use the
functions `prefix` and `suffix` from the namespace `name`:

{{code_block('user-guide/expressions/expression-expansion', 'prefix-suffix', ['Expr.name', 'prefix',
'suffix'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:prefix-suffix"
```

### Dynamic name replacement

If a static prefix/suffix is not enough, the namespace `name` also provides the function `map` that
accepts a callable that accepts the old column names and produces the new ones:

{{code_block('user-guide/expressions/expression-expansion', 'name-map', ['Expr.name', 'map'])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:name-map"
```

See the API reference for the full contents of the namespace `name`.

## Programmatically generating expressions

Expression expansion is a very useful feature but it does not solve all of your problems. For
example, if we want to compute the day and year amplitude of the prices of the stocks in our
dataframe, expression expansion won't help us.

At first, you may think about using a `for` loop:

{{code_block('user-guide/expressions/expression-expansion', 'for-with_columns', [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:for-with_columns"
```

Do not do this. Instead, generate all of the expressions you want to compute programmatically and
use them only once in a context. Loosely speaking, you want to swap the `for` loop with the context
`with_columns`. In practice, you could do something like the following:

{{code_block('user-guide/expressions/expression-expansion', 'yield-expressions', [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:yield-expressions"
```

This produces the same final result and by specifying all of the expressions in one go we give
Polars the opportunity to:

1. do a better job at optimising the query; and
2. parallelise the execution of the actual computations.

## More flexible column selections

Polars comes with the submodule `selectors` that provides a number of functions that allow you to
write more flexible column selections for expression expansion.

!!! warning

    This functionality is not available in Rust yet. Refer to [Polars issue #10594](https://github.com/pola-rs/polars/issues/10594).

As a first example, here is how we can use the functions `string` and `ends_with`, and the set
operations that the functions from `selectors` support, to select all string columns and the columns
whose names end with `"_high"`:

{{code_block('user-guide/expressions/expression-expansion', 'selectors', [], ['selectors'], [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:selectors"
```

The submodule `selectors` provides
[a number of selectors that match based on the data type of the columns](#selectors-for-data-types),
of which the most useful are the functions that match a whole category of types, like `cs.numeric`
for all numeric data types or `cs.temporal` for all temporal data types.

The submodule `selectors` also provides
[a number of selectors that match based on patterns in the column names](#selectors-for-column-name-patterns)
which make it more convenient to specify common patterns you may want to check for, like the
function `cs.ends_with` that was shown above.

### Combining selectors with set operations

We can combine multiple selectors using set operations and the usual Python operators:

<!-- dprint-ignore-start -->
<!-- dprint doesn't understand `A | B` and thinks the | is a column separator. -->
| Operator                | Operation            |
| ----------------------- | -------------------- |
| `A | B`                 | Union                |
| `A & B`                 | Intersection         |
| `A - B`                 | Difference           |
| `A ^ B`                 | Symmetric difference |
| `~A`                    | Complement           |
<!-- dprint-ignore-end -->

The next example matches all non-string columns that contain an underscore in the name:

{{code_block('user-guide/expressions/expression-expansion', 'selectors-set-operations', [],
['selectors'], [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:selectors-set-operations"
```

### Resolving operator ambiguity

Expression functions can be chained on top of selectors:

{{code_block('user-guide/expressions/expression-expansion', 'selectors-expressions', [],
['selectors'], [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:selectors-expressions"
```

However, some operators have been overloaded to operate both on Polars selectors and on expressions.
For example, the operator `~` on a selector represents
[the set operation “complement”](#combining-selectors-with-set-operations) and on an expression
represents the Boolean operation of negation.

When you use a selector and then want to use, in the context of an expression, one of the
[operators that act as set operators for selectors](#combining-selectors-with-set-operations), you
can use the function `as_expr`.

Below, we want to negate the Boolean values in the columns “has_partner”, “has_kids”, and
“has_tattoos”. If we are not careful, the combination of the operator `~` and the selector
`cs.starts_with("has_")` will actually select the columns that we do not care about:

{{code_block('user-guide/expressions/expression-expansion', 'selector-ambiguity', [], [], [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:selector-ambiguity"
```

The correct solution uses `as_expr`:

{{code_block('user-guide/expressions/expression-expansion', 'as_expr', [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:as_expr"
```

### Debugging selectors

When you are not sure whether you have a Polars selector at hand or not, you can use the function
`cs.is_selector` to check:

{{code_block('user-guide/expressions/expression-expansion', 'is_selector', [], ['is_selector'],
[])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:is_selector"
```

This should help you avoid any ambiguous situations where you think you are operating with
expressions but are in fact operating with selectors.

Another helpful debugging utility is the function `expand_selector`. Given a target frame or schema,
you can check what columns a given selector will expand to:

{{code_block('user-guide/expressions/expression-expansion', 'expand_selector', [],
['expand_selector'], [])}}

```python exec="on" result="text" session="expressions/expression-expansion"
--8<-- "python/user-guide/expressions/expression-expansion.py:expand_selector"
```

### Complete reference

The tables below group the functions available in the submodule `selectors` by their type of
behaviour.

#### Selectors for data types

Selectors that match based on the data type of the column:

| Selector function  | Data type(s) matched                                               |
| ------------------ | ------------------------------------------------------------------ |
| `binary`           | `Binary`                                                           |
| `boolean`          | `Boolean`                                                          |
| `by_dtype`         | Data types specified as arguments                                  |
| `categorical`      | `Categorical`                                                      |
| `date`             | `Date`                                                             |
| `datetime`         | `Datetime`, optionally filtering by time unit/zone                 |
| `decimal`          | `Decimal`                                                          |
| `duration`         | `Duration`, optionally filtering by time unit                      |
| `float`            | All float types, regardless of precision                           |
| `integer`          | All integer types, signed and unsigned, regardless of precision    |
| `numeric`          | All numeric types, namely integers, floats, and `Decimal`          |
| `signed_integer`   | All signed integer types, regardless of precision                  |
| `string`           | `String`                                                           |
| `temporal`         | All temporal data types, namely `Date`, `Datetime`, and `Duration` |
| `time`             | `Time`                                                             |
| `unsigned_integer` | All unsigned integer types, regardless of precision                |

#### Selectors for column name patterns

Selectors that match based on column name patterns:

| Selector function | Columns selected                                             |
| ----------------- | ------------------------------------------------------------ |
| `alpha`           | Columns with alphabetical names                              |
| `alphanumeric`    | Columns with alphanumeric names (letters and the digits 0-9) |
| `by_name`         | Columns with the names specified as arguments                |
| `contains`        | Columns whose names contain the substring specified          |
| `digit`           | Columns with numeric names (only the digits 0-9)             |
| `ends_with`       | Columns whose names end with the given substring             |
| `matches`         | Columns whose names match the given regex pattern            |
| `starts_with`     | Columns whose names start with the given substring           |

#### Positional selectors

Selectors that match based on the position of the columns:

| Selector function | Columns selected                     |
| ----------------- | ------------------------------------ |
| `all`             | All columns                          |
| `by_index`        | The columns at the specified indices |
| `first`           | The first column in the context      |
| `last`            | The last column in the context       |

#### Miscellaneous functions

The submodule `selectors` also provides the following functions:

| Function          | Behaviour                                                                             |
| ----------------- | ------------------------------------------------------------------------------------- |
| `as_expr`*        | Convert a selector to an expression                                                   |
| `exclude`         | Selects all columns except those matching the given names, data types, or selectors   |
| `expand_selector` | Expand selector to matching columns with respect to a specific frame or target schema |
| `is_selector`     | Check whether the given object/expression is a selector                               |

*`as_expr` isn't a function defined on the submodule `selectors`, but rather a method defined on
selectors.
