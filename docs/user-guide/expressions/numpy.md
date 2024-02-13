# Numpy ufuncs

Polars expressions support NumPy [ufuncs](https://numpy.org/doc/stable/reference/ufuncs.html). See [here](https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs)
for a list on all supported numpy functions. Additionally, SciPy offers a wide host of ufuncs. Specifically, the [scipy.special](https://docs.scipy.org/doc/scipy/reference/special.html#module-scipy.special) namespace has ufunc versions of many (possibly most) of what is available under stats. 

This means that if a function is not provided by Polars, we can use NumPy and we still have fast columnar operation through the NumPy API. ufuncs have a hook that diverts their own execution when one of its inputs is a class with the [`__array_ufunc__`](https://numpy.org/doc/stable/reference/arrays.classes.html#special-attributes-and-methods) method. Polars Expr class has this method which allows ufuncs to be input directly in a context (`select`, `with_columns`, `agg`) with relevant expressions as the input. This syntax extends even to multiple input functions. 

### Example

{{code_block('user-guide/expressions/numpy-example',api_functions=['DataFrame','np.log'])}}

```python exec="on" result="text" session="user-guide/numpy"
--8<-- "python/user-guide/expressions/numpy-example.py"
```

## Numba

[NumBa](https://numba.pydata.org/) is an open source JIT compiler that allows you to create your own ufuncs entirely within python. The key is to use the [@guvectorize](https://numba.readthedocs.io/en/stable/user/vectorize.html#the-guvectorize-decorator) decorator. One popular use case is conditional cumulative functions. For example, suppose you want to take a cumulative sum but have it reset whenever it gets to a threshold.

### Example

{{code_block('user-guide/expressions/numpy-example',api_functions=['DataFrame'])}}

```python exec="on" result="text" session="user-guide/numpy"
--8<-- "python/user-guide/expressions/numba-example.py"
```

### Interoperability

Polars `Series` have support for NumPy universal functions (ufuncs). Element-wise functions such as `np.exp()`, `np.cos()`, `np.div()`, etc. all work with almost zero overhead.

However, as a Polars-specific remark: missing values are a separate bitmask and are not visible by NumPy. This can lead to a window function or a `np.convolve()` giving flawed or incomplete results.

Convert a Polars `Series` to a NumPy array with the `.to_numpy()` method. Missing values will be replaced by `np.nan` during the conversion.

### Note on Performance

The speed of ufuncs comes from being vectorized, and compiled. That said, there's no inherent benefit in using ufuncs just to avoid the use of `map_batches`. As mentioned above, ufuncs use a hook which gives polars the opportunity to run its own code before the ufunc is executed. In that way polars is still executing the ufunc with `map_batches`.