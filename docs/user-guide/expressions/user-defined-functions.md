# User-defined functions (Python)

You should be convinced by now that Polars expressions are so powerful and flexible that there is much less need for custom Python functions
than in other libraries.

Still, you need to have the power to be able to pass an expression's state to a third party library or apply your black box function
over data in Polars.

In this part of the documentation we'll be using one specific API:

- [:material-api: `map_batches`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_batches.html): Always passes the full `Series` to the function.

A later section will explain other available APIs for applying user-defined functions.

## Example: A slow, custom sum function written in Python

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
The former runs on each item individually, and the latter accepts a whole NumPy array, so allows for more flexible operations.

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

So how do you deal with missing data?
Either [fill it in](missing-data.md) or [drop it](https://docs.pola.rs/py-polars/html/reference/dataframe/api/polars.DataFrame.drop_nulls.html) before calling the customer user function.

## Combining multiple column values

TODO

If we want to have access to values of different columns in a single `map_elements` function call, we can create `struct` data
type. This data type collects those columns as fields in the `struct`. So if we'd create a struct from the columns
`"keys"` and `"values"`, we would get the following struct elements:

```python
[
    {"keys": "a", "values": 10},
    {"keys": "a", "values": 7},
    {"keys": "b", "values": 1},
]
```

In Python, those would be passed as `dict` to the calling Python function and can thus be indexed by `field: str`. In Rust, you'll get a `Series` with the `Struct` type. The fields of the struct can then be indexed and downcast.

{{code_block('user-guide/expressions/user-defined-functions','combine',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:combine"
```

`Structs` are covered in detail in the next section.

## Streaming calculations

Passing the full `Series` to the user-defined function has a cost: it will use a lot of memory.

TODO

## Return types

TODO

Custom Python functions are black boxes for Polars. We really don't know what kind of black arts you are doing, so we have
to infer and try our best to understand what you meant.

As a user it helps to understand what we do to better utilize custom functions.

The data type is automatically inferred. We do that by waiting for the first non-null value. That value will then be used
to determine the type of the `Series`.

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
