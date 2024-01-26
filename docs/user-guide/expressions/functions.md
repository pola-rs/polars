# Functions

Polars expressions have a large number of built in functions. These allow you to create complex queries without the need for [user defined functions](user-defined-functions.md). There are too many to go through here, but we will cover some of the more popular use cases. If you want to view all the functions go to the API Reference for your programming language.

In the examples below we will use the following `DataFrame`:

{{code_block('user-guide/expressions/functions','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/functions"
--8<-- "python/user-guide/expressions/functions.py:setup"
--8<-- "python/user-guide/expressions/functions.py:dataframe"
```

## Column naming

By default if you perform an expression it will keep the same name as the original column. In the example below we perform an expression on the `nrs` column. Note that the output `DataFrame` still has the same name.

{{code_block('user-guide/expressions/functions','samename',[])}}

```python exec="on" result="text" session="user-guide/functions"
--8<-- "python/user-guide/expressions/functions.py:samename"
```

This might get problematic in the case you use the same column multiple times in your expression as the output columns will get duplicated. For example, the following query will fail.

{{code_block('user-guide/expressions/functions','samenametwice',[])}}

```python exec="on" result="text" session="user-guide/functions"
--8<-- "python/user-guide/expressions/functions.py:samenametwice"
```

You can change the output name of an expression by using the `alias` function

{{code_block('user-guide/expressions/functions','samenamealias',['alias'])}}

```python exec="on" result="text" session="user-guide/functions"
--8<-- "python/user-guide/expressions/functions.py:samenamealias"
```

In case of multiple columns for example when using `all()` or `col(*)` you can apply a mapping function `name.map` to change the original column name into something else. In case you want to add a suffix (`name.suffix()`) or prefix (`name.prefix()`) these are also built in.

=== ":fontawesome-brands-python: Python"
[:material-api: `name.prefix`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.name.prefix.html)
[:material-api: `name.suffix`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.name.suffix.html)
[:material-api: `name.map`](https://docs.pola.rs/py-polars/html/reference/expressions/api/polars.Expr.name.map.html)

## Count unique values

There are two ways to count unique values in Polars: an exact methodology and an approximation. The approximation uses the [HyperLogLog++](https://en.wikipedia.org/wiki/HyperLogLog) algorithm to approximate the cardinality and is especially useful for very large datasets where an approximation is good enough.

{{code_block('user-guide/expressions/functions','countunique',['n_unique','approx_n_unique'])}}

```python exec="on" result="text" session="user-guide/functions"
--8<-- "python/user-guide/expressions/functions.py:countunique"
```

## Conditionals

Polars supports if-else like conditions in expressions with the `when`, `then`, `otherwise` syntax. The predicate is placed in the `when` clause and when this evaluates to `true` the `then` expression is applied otherwise the `otherwise` expression is applied (row-wise).

{{code_block('user-guide/expressions/functions','conditional',['when'])}}

```python exec="on" result="text" session="user-guide/functions"
--8<-- "python/user-guide/expressions/functions.py:conditional"
```
