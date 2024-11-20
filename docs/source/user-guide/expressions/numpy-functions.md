# Numpy functions

Polars expressions support NumPy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html). See
[the NumPy documentation for a list of all supported NumPy functions](https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs).

This means that if a function is not provided by Polars, we can use NumPy and we still have fast
columnar operations through the NumPy API.

## Example

{{code_block('user-guide/expressions/numpy-example',api_functions=['DataFrame','np.log'])}}

```python exec="on" result="text" session="user-guide/numpy"
--8<-- "python/user-guide/expressions/numpy-example.py"
```

## Interoperability

Polars' series have support for NumPy universal functions (ufuncs) and generalized ufuncs.
Element-wise functions such as `np.exp`, `np.cos`, `np.div`, etc, all work with almost zero
overhead.

However, bear in mind that
[Polars keeps track of missing values with a separate bitmask](missing-data.md) and NumPy does not
receive this information. This can lead to a window function or a `np.convolve` giving flawed or
incomplete results, so an error will be raised if you pass a series with missing data to a
generalized ufunc. Convert a Polars series to a NumPy array with the function `to_numpy`. Missing
values will be replaced by `np.nan` during the conversion.
