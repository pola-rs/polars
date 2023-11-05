# Numpy

Polars expressions support NumPy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html). See [here](https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs)
for a list on all supported numpy functions.

This means that if a function is not provided by Polars, we can use NumPy and we still have fast columnar operation through the NumPy API.

### Example

{{code_block('user-guide/expressions/numpy-example',api_functions=['DataFrame','np.log'])}}

```python exec="on" result="text" session="user-guide/numpy"
--8<-- "python/user-guide/expressions/numpy-example.py"
```

### Interoperability

Polars `Series` have support for NumPy universal functions (ufuncs). Element-wise functions such as `np.exp()`, `np.cos()`, `np.div()`, etc. all work with almost zero overhead.

However, as a Polars-specific remark: missing values are a separate bitmask and are not visible by NumPy. This can lead to a window function or a `np.convolve()` giving flawed or incomplete results.

Convert a Polars `Series` to a NumPy array with the `.to_numpy()` method. Missing values will be replaced by `np.nan` during the conversion. If the `Series` does not include missing values, or those values are not desired anymore, the `.view()` method can be used instead, providing a zero-copy NumPy array of the data.
