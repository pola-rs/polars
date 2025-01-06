# Searching

Polars provides expression APIs for finding data in a `Series` or the result of
previous expressions.

## Searching for a value's specific index with `index_of()`

If you want to find the index of a particular value, you can use the `index_of()`
API, which is similar to Python lists' `index()` method.
Given a dataframe:

{{code_block('user-guide/expressions/casting', 'dfnum', [])}}

```python exec="on" result="text" session="user-guide/searching"
--8<-- "python/user-guide/expressions/searching.py:dfnum"
```

You can find the index of both values and nulls using `index_of()`:

{{code_block('user-guide/expressions/searching','index_of',[])}}

```python exec="on" result="text" session="user-guide/searching"
--8<-- "python/user-guide/expressions/searching.py:index_of"
```

Searching for a non-existent value will return `None`/`null`:

{{code_block('user-guide/expressions/searching','index_of_not_found',[])}}

```python exec="on" result="text" session="user-guide/searching"
--8<-- "python/user-guide/expressions/searching.py:index_of_not_found"
```
