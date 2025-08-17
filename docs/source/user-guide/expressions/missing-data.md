# Missing data

This section of the user guide teaches how to work with missing data in Polars.

## `null` and `NaN` values

In Polars, missing data is represented by the value `null`. This missing value `null` is used for
all data types, including numerical types.

Polars also supports the value `NaN` (“Not a Number”) for columns with floating point numbers. The
value `NaN` is considered to be a valid floating point value, which is different from missing data.
[We discuss the value `NaN` separately below](#not-a-number-or-nan-values).

When creating a series or a dataframe, you can set a value to `null` by using the appropriate
construct for your language:

{{code_block('user-guide/expressions/missing-data','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:dataframe"
```

!!! info "Difference from pandas"

    In pandas, the value used to represent missing data depends on the data type of the column.
    In Polars, missing data is always represented by the value `null`.

## Missing data metadata

Polars keeps track of some metadata regarding the missing data of each series. This metadata allows
Polars to answer some basic queries about missing values in a very efficient way, namely how many
values are missing and which ones are missing.

To determine how many values are missing from a column you can use the function `null_count`:

{{code_block('user-guide/expressions/missing-data','count',['null_count'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:count"
```

The function `null_count` can be called on a dataframe, a column from a dataframe, or on a series
directly. The function `null_count` is a cheap operation because the result is already known.

Polars uses something called a “validity bitmap” to know which values are missing in a series. The
validity bitmap is memory efficient as it is bit encoded. If a series has length $n$, then its
validity bitmap will cost $n / 8$ bytes. The function `is_null` uses the validity bitmap to
efficiently report which values are `null` and which are not:

{{code_block('user-guide/expressions/missing-data','isnull',['is_null'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:isnull"
```

The function `is_null` can be used on a column of a dataframe or on a series directly. Again, this
is a cheap operation because the result is already known by Polars.

??? info "Why does Polars waste memory on a validity bitmap?"

    It all comes down to a tradeoff.
    By using a bit more memory per column, Polars can be much more efficient when performing most operations on your columns.
    If the validity bitmap wasn't known, every time you wanted to compute something you would have to check each position of the series to see if a legal value was present or not.
    With the validity bitmap, Polars knows automatically the positions where your operations can be applied.

## Filling missing data

Missing data in a series can be filled with the function `fill_null`. You can specify how missing
data is effectively filled in a couple of different ways:

- a literal of the correct data type;
- a Polars expression, such as replacing with values computed from another column;
- a strategy based on neighbouring values, such as filling forwards or backwards; and
- interpolation.

To illustrate how each of these methods work we start by defining a simple dataframe with two
missing values in the second column:

{{code_block('user-guide/expressions/missing-data','dataframe2',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:dataframe2"
```

### Fill with a specified literal value

You can fill the missing data with a specified literal value. This literal value will replace all of
the occurrences of the value `null`:

{{code_block('user-guide/expressions/missing-data','fill',['fill_null'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:fill"
```

However, this is actually just a special case of the general case where
[the function `fill_null` replaces missing values with the corresponding values from the result of a Polars expression](#fill-with-a-strategy-based-on-neighbouring-values),
as seen next.

### Fill with an expression

In the general case, the missing data can be filled by extracting the corresponding values from the
result of a general Polars expression. For example, we can fill the second column with values taken
from the double of the first column:

{{code_block('user-guide/expressions/missing-data','fillexpr',['fill_null'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:fillexpr"
```

### Fill with a strategy based on neighbouring values

You can also fill the missing data by following a fill strategy based on the neighbouring values.
The two simpler strategies look for the first non-`null` value that comes immediately before or
immediately after the value `null` that is being filled:

{{code_block('user-guide/expressions/missing-data','fillstrategy',['fill_null'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:fillstrategy"
```

You can find other fill strategies in the API docs.

### Fill with interpolation

Additionally, you can fill intermediate missing data with interpolation by using the function
`interpolate` instead of the function `fill_null`:

{{code_block('user-guide/expressions/missing-data','fillinterpolate',['interpolate'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:fillinterpolate"
```

Note: With interpolate, nulls at the beginning and end of the series remain null.

## Not a Number, or `NaN` values

Missing data in a series is only ever represented by the value `null`, regardless of the data type
of the series. Columns with a floating point data type can sometimes have the value `NaN`, which
might be confused with `null`.

The special value `NaN` can be created directly:

{{code_block('user-guide/expressions/missing-data','nan',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:nan"
```

And it might also arise as the result of a computation:

{{code_block('user-guide/expressions/missing-data','nan-computed',[])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:nan-computed"
```

!!! info

    By default, a `NaN` value in an integer column causes the column to be cast to a float data type in pandas.
    This does not happen in Polars; instead, an exception is raised.

`NaN` values are considered to be a type of floating point data and are **not considered to be
missing data** in Polars. This means:

- `NaN` values are **not** counted with the function `null_count`; and
- `NaN` values are filled when you use the specialised function `fill_nan` method but are **not**
  filled with the function `fill_null`.

Polars has the functions `is_nan` and `fill_nan`, which work in a similar way to the functions
`is_null` and `fill_null`. Unlike with missing data, Polars does not hold any metadata regarding the
`NaN` values, so the function `is_nan` entails actual computation.

One further difference between the values `null` and `NaN` is that numerical aggregating functions,
like `mean` and `sum`, skip the missing values when computing the result, whereas the value `NaN` is
considered for the computation and typically propagates into the result. If desirable, this behavior
can be avoided by replacing the occurrences of the value `NaN` with the value `null`:

{{code_block('user-guide/expressions/missing-data','nanfill',['fill_nan'])}}

```python exec="on" result="text" session="user-guide/missing-data"
--8<-- "python/user-guide/expressions/missing-data.py:nanfill"
```

You can learn more about the value `NaN` in
[the section about floating point number data types](../concepts/data-types-and-structures.md#floating-point-numbers).
