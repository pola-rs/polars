# The Struct datatype

Polars `Struct`s are the idiomatic way of working with multiple columns. It is also a free operation i.e. moving columns into `Struct`s does not copy any data!

For this section, let's start with a `DataFrame` that captures the average rating of a few movies across some states in the U.S.:

{{code_block('user-guide/expressions/structs','ratings_df',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/structs"
--8<-- "python/user-guide/expressions/structs.py:setup"
--8<-- "python/user-guide/expressions/structs.py:ratings_df"
```

## Encountering the `Struct` type

A common operation that will lead to a `Struct` column is the ever so popular `value_counts` function that is commonly used in exploratory data analysis. Checking the number of times a state appears the data will be done as so:

{{code_block('user-guide/expressions/structs','state_value_counts',['value_counts'])}}

```python exec="on" result="text" session="user-guide/structs"
--8<-- "python/user-guide/expressions/structs.py:state_value_counts"
```

Quite unexpected an output, especially if coming from tools that do not have such a data type. We're not in peril though, to get back to a more familiar output, all we need to do is `unnest` the `Struct` column into its constituent columns:

{{code_block('user-guide/expressions/structs','struct_unnest',['unnest'])}}

```python exec="on" result="text" session="user-guide/structs"
--8<-- "python/user-guide/expressions/structs.py:struct_unnest"
```

!!! note "Why `value_counts` returns a `Struct`"

    Polars expressions always have a `Fn(Series) -> Series` signature and `Struct` is thus the data type that allows us to provide multiple columns as input/ouput of an expression. In other words, all expressions have to return a `Series` object, and `Struct` allows us to stay consistent with that requirement.

## Structs as `dict`s

Polars will interpret a `dict` sent to the `Series` constructor as a `Struct`:

{{code_block('user-guide/expressions/structs','series_struct',['Series'])}}

```python exec="on" result="text" session="user-guide/structs"
--8<-- "python/user-guide/expressions/structs.py:series_struct"
```

!!! note "Constructing `Series` objects"

    Note that `Series` here was constructed with the `name` of the series in the beginning, followed by the `values`. Providing the latter first
    is considered an anti-pattern in Polars, and must be avoided.

### Extracting individual values of a `Struct`

Let's say that we needed to obtain just the `movie` value in the `Series` that we created above. We can use the `field` method to do so:

{{code_block('user-guide/expressions/structs','series_struct_extract',['struct.field'])}}

```python exec="on" result="text" session="user-guide/structs"
--8<-- "python/user-guide/expressions/structs.py:series_struct_extract"
```

### Renaming individual keys of a `Struct`

What if we need to rename individual `field`s of a `Struct` column? We first convert the `rating_series` object to a `DataFrame` so that we can view the changes easily, and then use the `rename_fields` method:

{{code_block('user-guide/expressions/structs','series_struct_rename',['struct.rename_fields'])}}

```python exec="on" result="text" session="user-guide/structs"
--8<-- "python/user-guide/expressions/structs.py:series_struct_rename"
```

## Practical use-cases of `Struct` columns

### Identifying duplicate rows

Let's get back to the `ratings` data. We want to identify cases where there are duplicates at a `Movie` and `Theatre` level. This is where the `Struct` datatype shines:

{{code_block('user-guide/expressions/structs','struct_duplicates',['is_duplicated', 'struct'])}}

```python exec="on" result="text" session="user-guide/structs"
--8<-- "python/user-guide/expressions/structs.py:struct_duplicates"
```

We can identify the unique cases at this level also with `is_unique`!

### Multi-column ranking

Suppose, given that we know there are duplicates, we want to choose which rank gets a higher priority. We define `Count` of ratings to be more important than the actual `Avg_Rating` themselves, and only use it to break a tie. We can then do:

{{code_block('user-guide/expressions/structs','struct_ranking',['is_duplicated', 'struct'])}}

```python exec="on" result="text" session="user-guide/structs"
--8<-- "python/user-guide/expressions/structs.py:struct_ranking"
```

That's a pretty complex set of requirements done very elegantly in Polars!

### Using multi-column apply

This was discussed in the previous section on _User Defined Functions_.
