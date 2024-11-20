# Lists and arrays

Polars has first-class support for two homogeneous container data types: `List` and `Array`. Polars
supports many operations with the two data types and their APIs overlap, so this section of the user
guide has the objective of clarifying when one data type should be chosen in favour of the other.

## Lists vs arrays

### The data type `List`

The data type list is suitable for columns whose values are homogeneous 1D containers of varying
lengths.

The dataframe below contains three examples of columns with the data type `List`:

{{code_block('user-guide/expressions/lists', 'list-example', ['List'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:list-example"
```

Note that the data type `List` is different from Python's type `list`, where elements can be of any
type. If you want to store true Python lists in a column, you can do so with the data type `Object`
and your column will not have the list manipulation features that we're about to discuss.

### The data type `Array`

The data type `Array` is suitable for columns whose values are homogeneous containers of an
arbitrary dimension with a known and fixed shape.

The dataframe below contains two examples of columns with the data type `Array`.

{{code_block('user-guide/expressions/lists', 'array-example', ['Array'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:array-example"
```

The example above shows how to specify that the columns “bit_flags” and “tic_tac_toe” have the data
type `Array`, parametrised by the data type of the elements contained within and by the shape of
each array.

In general, Polars does not infer that a column has the data type `Array` for performance reasons,
and defaults to the appropriate variant of the data type `List`. In Python, an exception to this
rule is when you provide a NumPy array to build a column. In that case, Polars has the guarantee
from NumPy that all subarrays have the same shape, so an array of $n + 1$ dimensions will generate a
column of $n$ dimensional arrays:

{{code_block('user-guide/expressions/lists', 'numpy-array-inference', ['Array'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:numpy-array-inference"
```

### When to use each

In short, prefer the data type `Array` over `List` because it is more memory efficient and more
performant. If you cannot use `Array`, then use `List`:

- when the values within a column do not have a fixed shape; or
- when you need functions that are only available in the list API.

## Working with lists

### The namespace `list`

Polars provides many functions to work with values of the data type `List` and these are grouped
inside the namespace `list`. We will explore this namespace a bit now.

!!! warning "`arr` then, `list` now"

    In previous versions of Polars, the namespace for list operations used to be `arr`.
    `arr` is now the namespace for the data type `Array`.
    If you find references to the namespace `arr` on StackOverflow or other sources, note that those sources _may_ be outdated.

The dataframe `weather` defined below contains data from different weather stations across a region.
When the weather station is unable to get a result, an error code is recorded instead of the actual
temperature at that time.

{{code_block('user-guide/expressions/lists', 'weather', [])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:weather"
```

### Programmatically creating lists

Given the dataframe `weather` defined previously, it is very likely we need to run some analysis on
the temperatures that are captured by each station. To make this happen, we need to first be able to
get individual temperature measurements. We
[can use the namespace `str`](strings.md#the-string-namespace) for this:

{{code_block('user-guide/expressions/lists', 'split', ['str.split'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:split"
```

A natural follow-up would be to explode the list of temperatures so that each measurement is in its
own row:

{{code_block('user-guide/expressions/lists', 'explode', ['explode'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:explode"
```

However, in Polars we often do not need to do this to operate on the list elements.

### Operating on lists

Polars provides several standard operations on columns with the `List` data type.
[Similar to what you can do with strings](strings.md#slicing), lists can be sliced with the
functions `head`, `tail`, and `slice`:

{{code_block('user-guide/expressions/lists', 'list-slicing', ['Expr.list'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:list-slicing"
```

### Element-wise computation within lists

If we need to identify the stations that are giving the most number of errors we need to

1. try to convert the measurements into numbers;
2. count the number of non-numeric values (i.e., `null` values) in the list, by row; and
3. rename this output column as “errors” so that we can easily identify the stations.

To perform these steps, we need to perform a casting operation on each measurement within the list
values. The function `eval` is used as the entry point to perform operations on the elements of the
list. Within it, you can use the context `element` to refer to each single element of the list
individually, and then you can use any Polars expression on the element:

{{code_block('user-guide/expressions/lists', 'element-wise-casting', ['element'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:element-wise-casting"
```

Another alternative would be to use a regular expression to check if a measurement starts with a
letter:

{{code_block('user-guide/expressions/lists', 'element-wise-regex', ['element'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:element-wise-regex"
```

If you are unfamiliar with the namespace `str` or the notation `(?i)` in the regex, now is a good
time to
[look at how to work with strings and regular expressions in Polars](strings.md#check-for-the-existence-of-a-pattern).

### Row-wise computations

The function `eval` gives us access to the list elements and `pl.element` refers to each individual
element, but we can also use `pl.all()` to refer to all of the elements of the list.

To show this in action, we will start by creating another dataframe with some more weather data:

{{code_block('user-guide/expressions/lists', 'weather_by_day', [])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:weather_by_day"
```

Now, we will calculate the percentage rank of the temperatures by day, measured across stations.
Polars does not provide a function to do this directly, but because expressions are so versatile we
can create our own percentage rank expression for highest temperature. Let's try that:

{{code_block('user-guide/expressions/lists', 'rank_pct', ['element', 'rank'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:rank_pct"
```

## Working with arrays

### Creating an array column

As [we have seen above](#the-data-type-array), Polars usually does not infer the data type `Array`
automatically. You have to specify the data type `Array` when creating a series/dataframe or
[cast a column](casting.md) explicitly unless you create the column out of a NumPy array.

### The namespace `arr`

The data type `Array` was recently introduced and is still pretty nascent in features that it
offers. Even so, the namespace `arr` aggregates several functions that you can use to work with
arrays.

!!! warning "`arr` then, `list` now"

    In previous versions of Polars, the namespace for list operations used to be `arr`.
    `arr` is now the namespace for the data type `Array`.
    If you find references to the namespace `arr` on StackOverflow or other sources, note that those sources _may_ be outdated.

The API documentation should give you a good overview of the functions in the namespace `arr`, of
which we present a couple:

{{code_block('user-guide/expressions/lists', 'array-overview', ['Expr.arr'])}}

```python exec="on" result="text" session="expressions/lists"
--8<-- "python/user-guide/expressions/lists.py:array-overview"
```
