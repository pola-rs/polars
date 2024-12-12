# Casting

Casting converts the [underlying data type of a column](../concepts/data-types-and-structures.md) to
a new one. Casting is available through the function `cast`.

The function `cast` includes a parameter `strict` that determines how Polars behaves when it
encounters a value that cannot be converted from the source data type to the target data type. The
default behaviour is `strict=True`, which means that Polars will thrown an error to notify the user
of the failed conversion while also providing details on the values that couldn't be cast. On the
other hand, if `strict=False`, any values that cannot be converted to the target data type will be
quietly converted to `null`.

## Basic example

Let's take a look at the following dataframe which contains both integers and floating point
numbers:

{{code_block('user-guide/expressions/casting', 'dfnum', [])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:dfnum"
```

To perform casting operations between floats and integers, or vice versa, we use the function
`cast`:

{{code_block('user-guide/expressions/casting','castnum',['cast'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:castnum"
```

Note that floating point numbers are truncated when casting to an integer data type.

## Downcasting numerical data types

You can reduce the memory footprint of a column by changing the precision associated with its
numeric data type. As an illustration, the code below demonstrates how casting from `Int64` to
`Int16` and from `Float64` to `Float32` can be used to lower memory usage:

{{code_block('user-guide/expressions/casting','downcast',['cast', 'estimated_size'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:downcast"
```

When performing downcasting it is crucial to ensure that the chosen number of bits (such as 64, 32,
or 16) is sufficient to accommodate the largest and smallest numbers in the column. For example, a
32-bit signed integer (`Int32`) represents integers between -2147483648 and 2147483647, inclusive,
while an 8-bit signed integer only represents integers between -128 and 127, inclusive. Attempting
to downcast to a data type with insufficient precision results in an error thrown by Polars:

{{code_block('user-guide/expressions/casting','overflow',['cast'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:overflow"
```

If you set the parameter `strict` to `False` the overflowing/underflowing values are converted to
`null`:

{{code_block('user-guide/expressions/casting','overflow2',['cast'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:overflow2"
```

## Converting strings to numeric data types

Strings that represent numbers can be converted to the appropriate data types via casting. The
opposite conversion is also possible:

{{code_block('user-guide/expressions/casting','strings',['cast'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:strings"
```

In case the column contains a non-numerical value, or a poorly formatted one, Polars will throw an
error with details on the conversion error. You can set `strict=False` to circumvent the error and
get a `null` value instead.

{{code_block('user-guide/expressions/casting','strings2',['cast'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:strings2"
```

## Booleans

Booleans can be expressed as either 1 (`True`) or 0 (`False`). It's possible to perform casting
operations between a numerical data type and a Boolean, and vice versa.

When converting numbers to Booleans, the number 0 is converted to `False` and all other numbers are
converted to `True`, in alignment with Python's Truthy and Falsy values for numbers:

{{code_block('user-guide/expressions/casting','bool',['cast'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:bool"
```

## Parsing / formatting temporal data types

All temporal data types are represented internally as the number of time units elapsed since a
reference moment, usually referred to as the epoch. For example, values of the data type `Date` are
stored as the number of days since the epoch. For the data type `Datetime` the time unit is the
microsecond (us) and for `Time` the time unit is the nanosecond (ns).

Casting between numerical types and temporal data types is allowed and exposes this relationship:

{{code_block('user-guide/expressions/casting','dates',['cast'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:dates"
```

To format temporal data types as strings we can use the function `dt.to_string` and to parse
temporal data types from strings we can use the function `str.to_datetime`. Both functions adopt the
[chrono format syntax](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) for
formatting.

{{code_block('user-guide/expressions/casting','dates2',['dt.to_string','str.to_date'])}}

```python exec="on" result="text" session="user-guide/casting"
--8<-- "python/user-guide/expressions/casting.py:dates2"
```

It's worth noting that `str.to_datetime` features additional options that support timezone
functionality. Refer to the API documentation for further information.
