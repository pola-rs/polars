# Casting

Casting converts the underlying [`DataType`](../concepts/data-types.md) of a column to a new one. Polars uses Arrow to manage the data in memory and relies on the compute kernels in the [rust implementation](https://github.com/jorgecarleitao/arrow2) to do the conversion. Casting is available with the `cast()` method.

The `cast` method includes a `strict` parameter that determines how Polars behaves when it encounters a value that can't be converted from the source `DataType` to the target `DataType`. By default, `strict=True`, which means that Polars will throw an error to notify the user of the failed conversion and provide details on the values that couldn't be cast. On the other hand, if `strict=False`, any values that can't be converted to the target `DataType` will be quietly converted to `null`.

## Numerics

Let's take a look at the following `DataFrame` which contains both integers and floating point numbers.

{{code_block('user-guide/expressions/casting','dfnum',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:setup"
--8<-- "python/user-guide/expressions/casting.py:dfnum"
```

To perform casting operations between floats and integers, or vice versa, we can invoke the `cast()` function.

{{code_block('user-guide/expressions/casting','castnum',['cast'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:castnum"
```

Note that in the case of decimal values these are rounded downwards when casting to an integer.

##### Downcast

Reducing the memory footprint is also achievable by modifying the number of bits allocated to an element. As an illustration, the code below demonstrates how casting from `Int64` to `Int16` and from `Float64` to `Float32` can be used to lower memory usage.

{{code_block('user-guide/expressions/casting','downcast',['cast'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:downcast"
```

#### Overflow

When performing downcasting, it is crucial to ensure that the chosen number of bits (such as 64, 32, or 16) is sufficient to accommodate the largest and smallest numbers in the column. For example, using a 32-bit signed integer (`Int32`) allows handling integers within the range of -2147483648 to +2147483647, while using `Int8` covers integers between -128 to 127. Attempting to cast to a `DataType` that is too small will result in a `ComputeError` thrown by Polars, as the operation is not supported.

{{code_block('user-guide/expressions/casting','overflow',['cast'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:overflow"
```

You can set the `strict` parameter to `False`, this converts values that are overflowing to null values.

{{code_block('user-guide/expressions/casting','overflow2',['cast'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:overflow2"
```

## Strings

Strings can be casted to numerical data types and vice versa:

{{code_block('user-guide/expressions/casting','strings',['cast'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:strings"
```

In case the column contains a non-numerical value, Polars will throw a `ComputeError` detailing the conversion error. Setting `strict=False` will convert the non float value to `null`.

{{code_block('user-guide/expressions/casting','strings2',['cast'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:strings2"
```

## Booleans

Booleans can be expressed as either 1 (`True`) or 0 (`False`). It's possible to perform casting operations between a numerical `DataType` and a boolean, and vice versa. However, keep in mind that casting from a string (`Utf8`) to a boolean is not permitted.

{{code_block('user-guide/expressions/casting','bool',['cast'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:bool"
```

## Dates

Temporal data types such as `Date` or `Datetime` are represented as the number of days (`Date`) and microseconds (`Datetime`) since epoch. Therefore, casting between the numerical types and the temporal data types is allowed.

{{code_block('user-guide/expressions/casting','dates',['cast'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:dates"
```

To convert between strings and `Dates`/`Datetimes`, `dt.to_string` and `str.to_datetime` are utilized. Polars adopts the [chrono format syntax](https://docs.rs/chrono/latest/chrono/format/strftime/index.html) for formatting. It's worth noting that `str.to_datetime` features additional options that support timezone functionality. Refer to the API documentation for further information.

{{code_block('user-guide/expressions/casting','dates2',['dt.to_string','str.to_date'])}}

```python exec="on" result="text" session="user-guide/cast"
--8<-- "python/user-guide/expressions/casting.py:dates2"
```
