# User-defined functions (Python)

Polars expressions are quite powerful and flexible, so there is much less need for custom Python functions compared to other libraries.
Still, you may need to pass an expression's state to a third party library or apply your black box function to data in Polars.

In this part of the documentation we'll be using two APIs that allows you to do this:

- [:material-api: `map_elements`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_elements.html): Call a function separately on each value in the `Series`.
- [:material-api: `map_batches`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_batches.html): Always passes the full `Series` to the function.

## Processing individual values with `map_elements()`

Let's start with the simplest case: we want to process each value in a `Series` individually.
Here is our data:

{{code_block('user-guide/expressions/user-defined-functions','dataframe',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:setup"
--8<-- "python/user-guide/expressions/user-defined-functions.py:dataframe"
```

We'll call `math.log()` on each individual value:

{{code_block('user-guide/expressions/user-defined-functions','individual_log',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:individual_log"
```

While this works, `map_elements()` has two problems:

1. **Limited to individual items:** Often you'll want to have a calculation that needs to operate on the whole `Series`, rather than individual items one by one.
2. **Performance overhead:** Even if you do want to process each item individually, calling a function for each individual item is slow; all those extra function calls add a lot of overhead.

Let's start by solving the first problem, and then we'll see how to solve the second problem.

## Processing a whole `Series` with `map_batches()`

We want to run a custom function on the contents of a whole `Series`.
For demonstration purposes, let's say we want to calculate the difference between the mean of a `Series` and each value.

We can use the `map_batches()` API to run this function on either the full `Series` or individual groups in a `group_by()`:

{{code_block('user-guide/expressions/user-defined-functions','diff_from_mean',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:diff_from_mean"
```

## Fast operations with user-defined functions

The problem with a pure-Python implementation is that it's slow.
In general, you want to minimize how much Python code you call if you want fast results.

To maximize speed, you'll want to make sure that you're using a function written in a compiled language.
For numeric calculations Polars supports a pair of interfaces defined by NumPy called ["ufuncs"](https://numpy.org/doc/stable/reference/ufuncs.html) and ["generalized ufuncs"](https://numpy.org/neps/nep-0005-generalized-ufuncs.html).
The former runs on each item individually, and the latter accepts a whole NumPy array, which allows for more flexible operations.

[NumPy](https://numpy.org/doc/stable/reference/ufuncs.html) and other libraries like [SciPy](https://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special) come with pre-written ufuncs you can use with Polars.
For example:

{{code_block('user-guide/expressions/user-defined-functions','np_log',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:np_log"
```

Notice that we can use `map_batches()`, because `numpy.log()` is able to run on both individual items and on whole NumPy arrays.
This means it will run much faster than our original example, since we only have a single Python call and then all processing happens in a fast low-level language.

## Example: A fast custom function using Numba

The pre-written functions NumPy provides are helpful, but our goal is to write our own functions.
For example, let's say we want a fast version of our `diff_from_mean()` example above.
The easiest way to write this in Python is to use [Numba](https://numba.readthedocs.io/en/stable/), which allows you to write custom functions in (a subset) of Python while still getting the benefit of compiled code.

In particular, Numba provides a decorator called [`@guvectorize`](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-guvectorize-decorator).
This creates a generalized ufunc by compiling a Python function to fast machine code, in a way that allows it to be used by Polars.

In the following example the `diff_from_mean_numba()` will be compiled to fast machine code at import time, which will take a little time.
After that all calls to the function will run quickly.
The `Series` will be converted to a NumPy array before being passed to the function:

{{code_block('user-guide/expressions/user-defined-functions','diff_from_mean_numba',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:diff_from_mean_numba"
```

## Missing data is not allowed when calling generalized ufuncs

Before being passed to a user-defined function like `diff_from_mean_numba()`, a `Series` will be converted to a NumPy array.
Unfortunately, NumPy arrays don't have a concept of missing data.
If there is missing data in the original `Series`, this means the resulting array won't actually match the `Series`.

If you're calculating results item by item, this doesn't matter.
For example, `numpy.log()` gets called on each individual value separately, so those missing values don't change the calculation.
But if the result of a user-defined function depend on multiple values in the `Series`, it's not clear what exactly should happen with the missing values.

Therefore, when calling generalized ufuncs such as Numba functions decorated with `@guvectorize`, Polars will raise an error if you try to pass in a `Series` with missing data.
How do you get rid of missing data?
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
You can use the `is_elementwise=True` argument to [:material-api: `map_batches`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_batches.html) to stream results into the function, which means it might not get all values at once.

!!! note
The `is_elementwise` argument can lead to incorrect results if set incorrectly.
If you set `is_elementwise=True`, make sure that your function actually operates
element-by-element (e.g. "calculate the logarithm of each value") - our example function `diff_from_mean()`,
for instance, does not.

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
