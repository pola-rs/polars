# User-defined functions (Python)

You should be convinced by now that Polars expressions are so powerful and flexible that there is much less need for custom Python functions
than in other libraries.

Still, you need to have the power to be able to pass an expression's state to a third party library or apply your black box function
over data in Polars.

For this we provide the following expressions:

- [:material-api: `map_batches`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_batches.html): Always passes the full `Series` to the function.
- [:material-api: `map_elements`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.map_elements.html): Passes the smallest logical for an operation; in a `select()` context this will be individual items, in a `group_by()` context this will be group-specific `Series`.

## Example: A slow, custom sum function written in Python

For demonstration purposes, let's say we want to sum the values in a `Series` using a function we write in Python.
Here is our data:

{{code_block('user-guide/expressions/user-defined-functions','dataframe',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:setup"
--8<-- "python/user-guide/expressions/user-defined-functions.py:dataframe"
```

We can use `map_batches()` to run this function on either the full `Series` or individual groups in a `group_by()`.
Since the result of the latter is a `Series`, we can also extract it into a single value if we want.

{{code_block('user-guide/expressions/user-defined-functions','custom_sum',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:custom_sum"
```

The problem with this implementation is that it's slow.
In general, you want to minimize how much Python code you call if you want fast results.
Calling a Python function for every `Series` isn't usually a problem, unless the `group_by()` produces a very large number of groups.
However, running the `for` loop in Python, and then summing the values in Python, will be very slow.

## Fast operations with user-defined functions

In general, user-defined functions will run most quickly when two conditions are met:

1. **You're operating on a whole `Series`.**
   That means you'll want to use `map_batches()` in `select()` contexts, and `map_elements()` in `group_by()` contexts so your function gets called per group.
   See below for more details about the difference between the two APIs.
2. **You're using a function written in a compiled language.**
   For numeric calculations Polars supports a pair of interfaces defined by NumPy called ["ufuncs"](https://numpy.org/doc/stable/reference/ufuncs.html) and ["generalized ufuncs"](https://numpy.org/neps/nep-0005-generalized-ufuncs.html).
   The latter runs on each item individually, but the latter accepts a whole array.
   The easiest way to write these in Python is to use [Numba](https://numba.readthedocs.io/en/stable/), which allows you to write custom functions in (a subset) of Python while still getting the benefit of compiled code.

## Example: A fast custom sum function in Python using Numba

Numba provides a decorator called [`@guvectorize`](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-guvectorize-decorator) that takes a Python function and compiles it to fast machine code, in a way that allows it to be used by Polars.

In the following example the `custom_sum_numba()` will be compiled to fast machine code at import time, which will take a little time.
After that all calls to the function will run quickly.
The `Series` will be converted to a NumPy array before being passed to the function:

{{code_block('user-guide/expressions/user-defined-functions','custom_sum_numba',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:custom_sum_numba"
```

## Missing data can break your calculation

As mentioned above, before being passed to a generalized `ufunc` like our Numba function a `Series` will be converted to a NumPy array.
Unfortunately, NumPy arrays don't have a concept of missing data, which means the array won't actually match the `Series`.

If you're calculating results item by item, this doesn't matter.
But if the result depends on more than one value in the `Series`, the result will be wrong:

{{code_block('user-guide/expressions/user-defined-functions','dataframe2',[])}}
{{code_block('user-guide/expressions/user-defined-functions','custom_mean_numba',[])}}

```python exec="on" result="text" session="user-guide/udf"
--8<-- "python/user-guide/expressions/user-defined-functions.py:dataframe2"
--8<-- "python/user-guide/expressions/user-defined-functions.py:custom_mean_numba"
```

## Pre-written fast functions



## Combining multiple column values

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

## Return types

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
