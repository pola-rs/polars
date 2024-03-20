# User-defined functions (Python)

Polars expressions are quite powerful and flexible, so there is much less need for custom Python functions compared to other libraries.
Still, you may need to pass an expression's state to a third party library or apply your black box function to data in Polars.

In this part of the documentation we'll be using one specific API that allows you to do this:

- [:material-api: `map_batches`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_batches.html): Always passes the full `Series` to the function.

## Example: A slow, custom function written in Python

For demonstration purposes, let's say we want to calculate the difference between the mean of a `Series` and each value.
Here is our data:

{{code_block('user-guide/expressions/user-defined-functions','dataframe',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:setup"
--8<-- "python/user-guide/expressions/user-defined-functions.py:dataframe"
```

We can use `map_batches()` to run this function on either the full `Series` or individual groups in a `group_by()`:

{{code_block('user-guide/expressions/user-defined-functions','diff_from_mean',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:diff_from_mean"
```

## Fast operations with user-defined functions

The problem with a pure-Python implementation is that it's slow.
In general, you want to minimize how much Python code you call if you want fast results.
Calling a Python function for every `Series` isn't usually a problem, unless the `group_by()` produces a very large number of groups.
However, running the `for` loop in Python, and then summing the values in Python, will be very slow.

To maximize speed, you'll want to make sure that you're using a function written in a compiled language.
For numeric calculations Polars supports a pair of interfaces defined by NumPy called ["ufuncs"](https://numpy.org/doc/stable/reference/ufuncs.html) and ["generalized ufuncs"](https://numpy.org/neps/nep-0005-generalized-ufuncs.html).
The former runs on each item individually, and the latter accepts a whole NumPy array, which allows for more flexible operations.

[NumPy](https://numpy.org/doc/stable/reference/ufuncs.html) and other libraries like [SciPy](https://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special) come with pre-written ufuncs you can use with Polars.
For example:

{{code_block('user-guide/expressions/user-defined-functions','np_log',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:np_log"
```

## Example: A fast custom function using Numba

The pre-written functions are helpful, but our goal is to write our own functions.
For example, let's say we want a fast version of our `diff_from_mean()` example above.
The easiest way to write this in Python is to use [Numba](https://numba.readthedocs.io/en/stable/), which allows you to write custom functions in (a subset) of Python while still getting the benefit of compiled code.

In particular, Numba provides a decorator called [`@guvectorize`](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-guvectorize-decorator) that compiles a Python function to fast machine code, in a way that allows it to be used by Polars.

In the following example the `diff_from_mean_numba()` will be compiled to fast machine code at import time, which will take a little time.
After that all calls to the function will run quickly.
The `Series` will be converted to a NumPy array before being passed to the function:

{{code_block('user-guide/expressions/user-defined-functions','diff_from_mean_numba',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:diff_from_mean_numba"
```

## Missing data can break your calculation

Before being passed to a user-defined function like `diff_from_mean_numba()`, a `Series` will be converted to a NumPy array.
Unfortunately, NumPy arrays don't have a concept of missing data.
If there is missing data in the original `Series`, this means the resulting array won't actually match the `Series`.

If you're calculating results item by item, this doesn't matter.
For example, `numpy.log()` gets called on each individual value separately, so those missing values don't change the calculation.
But if the result of a user-defined function depend on multiple values in the `Series`, the result may be wrong:

{{code_block('user-guide/expressions/user-defined-functions','dataframe2',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:dataframe2"
```

{{code_block('user-guide/expressions/user-defined-functions','missing_data',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:missing_data"
```

How do you deal with missing data?
Either [fill it in](missing-data.md) or [drop it](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.drop_nulls.html) before calling your custom function.

## Combining multiple column values

If you want to pass multiple columns to a user-defined function, you can use `Struct`s, which are [covered in detail in a different section](structs.md).
The basic idea is to combine multiple columns into a `Struct`, and then the function can extract the columns back out:

{{code_block('user-guide/expressions/user-defined-functions','combine',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:combine"
```

## Streaming calculations

Passing the full `Series` to the user-defined function has a cost: it may use a lot of memory, as its contents are copied into a NumPy array.
You can use a `is_elementwise=True` argument to [:material-api: `map_batches`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_batches.html) to stream results into the function, which means it might not get all values at once.

For a function like `numpy.log()` this works fine, because `numpy.log()` effectively calculates each individual value separately anyway.
However, for our example `diff_from_mean()` function above, this would result in incorrect results, since it would calculate the mean on only part of the `Series`.

## Return types

Custom Python functions are often black boxes; Polars doesn't know what your function is doing or what it will return.
The return data type is therefore automatically inferred. We do that by waiting for the first non-null value. That value will then be used
to determine the type of the resulting `Series`.

The mapping of Python types to Polars data types is as follows:

- `int` -> `Int64`
- `float` -> `Float64`
- `bool` -> `Boolean`
- `str` -> `String`
- `list[tp]` -> `List[tp]` (where the inner type is inferred with the same rules)
- `dict[str, [tp]]` -> `struct`
- `Any` -> `object` (Prevent this at all times)

Rust types map as follows:

- `i32` or `i64` -> `Int64`
- `f32` or `f64` -> `Float64`
- `bool` -> `Boolean`
- `String` or `str` -> `String`
- `Vec<tp>` -> `List[tp]` (where the inner type is inferred with the same rules)

You can pass a `return_dtype` argument to [:material-api: `map_batches`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_batches.html) if you want to override the inferred type.
