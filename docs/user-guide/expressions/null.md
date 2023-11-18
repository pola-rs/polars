# Missing data

This page sets out how missing data is represented in Polars and how missing data can be filled.

## `null` and `NaN` values

Each column in a `DataFrame` (or equivalently a `Series`) is an Arrow array or a collection of Arrow arrays [based on the Apache Arrow format](https://arrow.apache.org/docs/format/Columnar.html#null-count). Missing data is represented in Arrow and Polars with a `null` value. This `null` missing value applies for all data types including numerical values.

Polars also allows `NotaNumber` or `NaN` values for float columns. These `NaN` values are considered to be a type of floating point data rather than missing data. We discuss `NaN` values separately below.

You can manually define a missing value with the python `None` value:

{{code_block('user-guide/expressions/null','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:setup"
--8<-- "python/user-guide/expressions/null.py:dataframe"
```

!!! info

    In pandas the value for missing data depends on the dtype of the column. In Polars missing data is always represented as a `null` value.

## Missing data metadata

Each Arrow array used by Polars stores two kinds of metadata related to missing data. This metadata allows Polars to quickly show how many missing values there are and which values are missing.

The first piece of metadata is the `null_count` - this is the number of rows with `null` values in the column:

{{code_block('user-guide/expressions/null','count',['null_count'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:count"
```

The `null_count` method can be called on a `DataFrame`, a column from a `DataFrame` or a `Series`. The `null_count` method is a cheap operation as `null_count` is already calculated for the underlying Arrow array.

The second piece of metadata is an array called a _validity bitmap_ that indicates whether each data value is valid or missing.
The validity bitmap is memory efficient as it is bit encoded - each value is either a 0 or a 1. This bit encoding means the memory overhead per array is only (array length / 8) bytes. The validity bitmap is used by the `is_null` method in Polars.

You can return a `Series` based on the validity bitmap for a column in a `DataFrame` or a `Series` with the `is_null` method:

{{code_block('user-guide/expressions/null','isnull',['is_null'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:isnull"
```

The `is_null` method is a cheap operation that does not require scanning the full column for `null` values. This is because the validity bitmap already exists and can be returned as a Boolean array.

## Filling missing data

Missing data in a `Series` can be filled with the `fill_null` method. You have to specify how you want the `fill_null` method to fill the missing data. The main ways to do this are filling with:

- a literal such as 0 or "0"
- a strategy such as filling forwards
- an expression such as replacing with values from another column
- interpolation

We illustrate each way to fill nulls by defining a simple `DataFrame` with a missing value in `col2`:

{{code_block('user-guide/expressions/null','dataframe2',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:dataframe2"
```

### Fill with specified literal value

We can fill the missing data with a specified literal value with `pl.lit`:

{{code_block('user-guide/expressions/null','fill',['fill_null'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:fill"
```

### Fill with a strategy

We can fill the missing data with a strategy such as filling forward:

{{code_block('user-guide/expressions/null','fillstrategy',['fill_null'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:fillstrategy"
```

You can find other fill strategies in the API docs.

### Fill with an expression

For more flexibility we can fill the missing data with an expression. For example,
to fill nulls with the median value from that column:

{{code_block('user-guide/expressions/null','fillexpr',['fill_null'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:fillexpr"
```

In this case the column is cast from integer to float because the median is a float statistic.

### Fill with interpolation

In addition, we can fill nulls with interpolation (without using the `fill_null` function):

{{code_block('user-guide/expressions/null','fillinterpolate',['interpolate'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:fillinterpolate"
```

## `NotaNumber` or `NaN` values

Missing data in a `Series` has a `null` value. However, you can use `NotaNumber` or `NaN` values in columns with float datatypes. These `NaN` values can be created from Numpy's `np.nan` or the native python `float('nan')`:

{{code_block('user-guide/expressions/null','nan',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:nan"
```

!!! info

    In pandas by default a `NaN` value in an integer column causes the column to be cast to float. This does not happen in Polars - instead an exception is raised.

`NaN` values are considered to be a type of floating point data and are **not considered to be missing data** in Polars. This means:

- `NaN` values are **not** counted with the `null_count` method
- `NaN` values are filled when you use `fill_nan` method but are **not** filled with the `fill_null` method

Polars has `is_nan` and `fill_nan` methods which work in a similar way to the `is_null` and `fill_null` methods. The underlying Arrow arrays do not have a pre-computed validity bitmask for `NaN` values so this has to be computed for the `is_nan` method.

One further difference between `null` and `NaN` values is that taking the `mean` of a column with `null` values excludes the `null` values from the calculation but with `NaN` values taking the mean results in a `NaN`. This behaviour can be avoided by replacing the `NaN` values with `null` values;

{{code_block('user-guide/expressions/null','nanfill',['fill_nan'])}}

```python exec="on" result="text" session="user-guide/null"
--8<-- "python/user-guide/expressions/null.py:nanfill"
```
