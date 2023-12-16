# Lists and Arrays

Polars has first-class support for `List` columns: that is, columns where each row is a list of homogeneous elements, of varying lengths. Polars also has an `Array` datatype, which is analogous to NumPy's `ndarray` objects, where the length is identical across rows.

Note: this is different from Python's `list` object, where the elements can be of any type. Polars can store these within columns, but as a generic `Object` datatype that doesn't have the special list manipulation features that we're about to discuss.

## Powerful `List` manipulation

Let's say we had the following data from different weather stations across a state. When the weather station is unable to get a result, an error code is recorded instead of the actual temperature at that time.

{{code_block('user-guide/expressions/lists','weather_df',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:setup"
--8<-- "python/user-guide/expressions/lists.py:weather_df"
```

### Creating a `List` column

For the `weather` `DataFrame` created above, it's very likely we need to run some analysis on the temperatures that are captured by each station. To make this happen, we need to first be able to get individual temperature measurements. This is done by:

{{code_block('user-guide/expressions/lists','string_to_list',['str.split'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:string_to_list"
```

One way we could go post this would be to convert each temperature measurement into its own row:

{{code_block('user-guide/expressions/lists','explode_to_atomic',['DataFrame.explode'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:explode_to_atomic"
```

However, in Polars, we often do not need to do this to operate on the `List` elements.

### Operating on `List` columns

Polars provides several standard operations on `List` columns. If we want the first three measurements, we can do a `head(3)`. The last three can be obtained via a `tail(3)`, or alternately, via `slice` (negative indexing is supported). We can also identify the number of observations via `lengths`. Let's see them in action:

{{code_block('user-guide/expressions/lists','list_ops',['Expr.list'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:list_ops"
```

!!! warning "`arr` then, `list` now"

    If you find references to the `arr` API on Stackoverflow or other sources, just replace `arr` with `list`, this was the old accessor for the `List` datatype. `arr` now refers to the newly introduced `Array` datatype (see below).

### Element-wise computation within `List`s

If we need to identify the stations that are giving the most number of errors from the starting `DataFrame`, we need to:

1. Parse the string input as a `List` of string values (already done).
2. Identify those strings that can be converted to numbers.
3. Identify the number of non-numeric values (i.e. `null` values) in the list, by row.
4. Rename this output as `errors` so that we can easily identify the stations.

The third step requires a casting (or alternately, a regex pattern search) operation to be perform on each element of the list. We can do this using by applying the operation on each element by first referencing them in the `pl.element()` context, and then calling a suitable Polars expression on them. Let's see how:

{{code_block('user-guide/expressions/lists','count_errors',['Expr.list', 'element'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:count_errors"
```

What if we chose the regex route (i.e. recognizing the presence of _any_ alphabetical character?)

{{code_block('user-guide/expressions/lists','count_errors_regex',['str.contains'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:count_errors_regex"
```

If you're unfamiliar with the `(?i)`, it's a good time to look at the documentation for the `str.contains` function in Polars! The Rust regex crate provides a lot of additional regex flags that might come in handy.

## Row-wise computations

This context is ideal for computing in row orientation.

We can apply **any** Polars operations on the elements of the list with the `list.eval` (`list().eval` in Rust) expression! These expressions run entirely on Polars' query engine and can run in parallel, so will be well optimized. Let's say we have another set of weather data across three days, for different stations:

{{code_block('user-guide/expressions/lists','weather_by_day',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:weather_by_day"
```

Let's do something interesting, where we calculate the percentage rank of the temperatures by day, measured across stations. Pandas allows you to compute the percentages of the `rank` values. Polars doesn't provide a special function to do this directly, but because expressions are so versatile we can create our own percentage rank expression for highest temperature. Let's try that!

{{code_block('user-guide/expressions/lists','weather_by_day_rank',['list.eval'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:weather_by_day_rank"
```

## Polars `Array`s

`Array`s are a new data type that was recently introduced, and are still pretty nascent in features that it offers. The major difference between a `List` and an `Array` is that the latter is limited to having the same number of elements per row, while a `List` can have a variable number of elements. Both still require that each element's data type is the same.

We can define `Array` columns in this manner:

{{code_block('user-guide/expressions/lists','array_df',['Array'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:array_df"
```

Basic operations are available on it:

{{code_block('user-guide/expressions/lists','array_ops',['Series.arr'])}}

```python exec="on" result="text" session="user-guide/lists"
--8<-- "python/user-guide/expressions/lists.py:array_ops"
```

Polars `Array`s are still being actively developed, so this section will likely change in the future.
