# Structs

The data type `Struct` is a composite data type that can store multiple fields in a single column.

!!! tip "Python analogy"

    For Python users, the data type `Struct` is kind of like a Python
    dictionary. Even better, if you are familiar with Python typing, you can think of the data type
    `Struct` as `typing.TypedDict`.

In this page of the user guide we will see situations in which the data type `Struct` arises, we
will understand why it does arise, and we will see how to work with `Struct` values.

Let's start with a dataframe that captures the average rating of a few movies across some states in
the US:

{{code_block('user-guide/expressions/structs','ratings_df',['DataFrame'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:ratings_df"
```

## Encountering the data type `Struct`

A common operation that will lead to a `Struct` column is the ever so popular `value_counts`
function that is commonly used in exploratory data analysis. Checking the number of times a state
appears in the data is done as so:

{{code_block('user-guide/expressions/structs','state_value_counts',['value_counts'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:state_value_counts"
```

Quite unexpected an output, especially if coming from tools that do not have such a data type. We're
not in peril, though. To get back to a more familiar output, all we need to do is use the function
`unnest` on the `Struct` column:

{{code_block('user-guide/expressions/structs','struct_unnest',['unnest'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:struct_unnest"
```

The function `unnest` will turn each field of the `Struct` into its own column.

!!! note "Why `value_counts` returns a `Struct`"

    Polars expressions always operate on a single series and return another series.
    `Struct` is the data type that allows us to provide multiple columns as input to an expression, or to output multiple columns from an expression.
    Thus, we can use the data type `Struct` to specify each value and its count when we use `value_counts`.

## Inferring the data type `Struct` from dictionaries

When building series or dataframes, Polars will convert dictionaries to the data type `Struct`:

{{code_block('user-guide/expressions/structs','series_struct',['Series'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:series_struct"
```

The number of fields, their names, and their types, are inferred from the first dictionary seen.
Subsequent incongruences can result in `null` values or in errors:

{{code_block('user-guide/expressions/structs','series_struct_error',['Series'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:series_struct_error"
```

## Extracting individual values of a `Struct`

Let's say that we needed to obtain just the field `"Movie"` from the `Struct` in the series that we
created above. We can use the function `field` to do so:

{{code_block('user-guide/expressions/structs','series_struct_extract',['struct.field'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:series_struct_extract"
```

## Renaming individual fields of a `Struct`

What if we need to rename individual fields of a `Struct` column? We use the function
`rename_fields`:

{{code_block('user-guide/expressions/structs','series_struct_rename',['struct.rename_fields'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:series_struct_rename"
```

To be able to actually see that the field names were changed, we will create a dataframe where the
only column is the result and then we use the function `unnest` so that each field becomes its own
column. The column names will reflect the renaming operation we just did:

{{code_block('user-guide/expressions/structs','struct-rename-check',['struct.rename_fields'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:struct-rename-check"
```

## Practical use-cases of `Struct` columns

### Identifying duplicate rows

Let's get back to the `ratings` data. We want to identify cases where there are duplicates at a
“Movie” and “Theatre” level.

This is where the data type `Struct` shines:

{{code_block('user-guide/expressions/structs','struct_duplicates',['is_duplicated', 'struct'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:struct_duplicates"
```

We can identify the unique cases at this level also with `is_unique`!

### Multi-column ranking

Suppose, given that we know there are duplicates, we want to choose which rating gets a higher
priority. We can say that the column “Count” is the most important, and if there is a tie in the
column “Count” then we consider the column “Avg_Rating”.

We can then do:

{{code_block('user-guide/expressions/structs','struct_ranking',['is_duplicated', 'struct'])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:struct_ranking"
```

That's a pretty complex set of requirements done very elegantly in Polars! To learn more about the
function `over`, used above, [see the user guide section on window functions](window-functions.md).

### Using multiple columns in a single expression

As mentioned earlier, the data type `Struct` is also useful if you need to pass multiple columns as
input to an expression. As an example, suppose we want to compute
[the Ackermann function](https://en.wikipedia.org/wiki/Ackermann_function) on two columns of a
dataframe. There is no way of composing Polars expressions to compute the Ackermann function[^1], so
we define a custom function:

{{code_block('user-guide/expressions/structs', 'ack', [])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:ack"
```

Now, to compute the values of the Ackermann function on those arguments, we start by creating a
`Struct` with fields `m` and `n` and then use the function `map_elements` to apply the function
`ack` to each value:

{{code_block('user-guide/expressions/structs','struct-ack',[], ['map_elements'], [])}}

```python exec="on" result="text" session="expressions/structs"
--8<-- "python/user-guide/expressions/structs.py:struct-ack"
```

Refer to
[this section of the user guide to learn more about applying user-defined Python functions to your data](user-defined-python-functions.md).

[^1]: To say that something cannot be done is quite a bold claim. If you prove us wrong, please let
us know!
