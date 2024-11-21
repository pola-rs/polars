# Basic operations

This section shows how to do basic operations on dataframe columns, like do basic arithmetic
calculations, perform comparisons, and other general-purpose operations. We will use the following
dataframe for the examples that follow:

{{code_block('user-guide/expressions/operations', 'dataframe', ['DataFrame'])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:dataframe"
```

## Basic arithmetic

Polars supports basic arithmetic between series of the same length, or between series and literals.
When literals are mixed with series, the literals are broadcast to match the length of the series
they are being used with.

{{code_block('user-guide/expressions/operations', 'arithmetic', ['operators'])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:arithmetic"
```

The example above shows that when an arithmetic operation takes `null` as one of its operands, the
result is `null`.

Polars uses operator overloading to allow you to use your language's native arithmetic operators
within your expressions. If you prefer, in Python you can use the corresponding named functions, as
the snippet below demonstrates:

```python
--8<-- "python/user-guide/expressions/operations.py:operator-overloading"
```

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:operator-overloading"
```

## Comparisons

Like with arithmetic operations, Polars supports comparisons via the overloaded operators or named
functions:

{{code_block('user-guide/expressions/operations','comparison',['operators'])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:comparison"
```

## Boolean and bitwise operations

Depending on the language, you may use the operators `&`, `|`, and `~`, for the Boolean operations
“and”, “or”, and “not”, respectively, or the functions of the same name:

{{code_block('user-guide/expressions/operations', 'boolean', ['operators'])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:boolean"
```

??? info "Python trivia"

    The Python functions are called `and_`, `or_`, and `not_`, because the words `and`, `or`, and `not` are reserved keywords in Python.
    Similarly, we cannot use the keywords `and`, `or`, and `not`, as the Boolean operators because these Python keywords will interpret their operands in the context of Truthy and Falsy through the dunder method `__bool__`.
    Thus, we overload the bitwise operators `&`, `|`, and `~`, as the Boolean operators because they are the second best choice.

These operators/functions can also be used for the respective bitwise operations, alongside the
bitwise operator `^` / function `xor`:

{{code_block('user-guide/expressions/operations', 'bitwise', [])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:bitwise"
```

## Counting (unique) values

Polars has two functions to count the number of unique values in a series. The function `n_unique`
can be used to count the exact number of unique values in a series. However, for very large data
sets, this operation can be quite slow. In those cases, if an approximation is good enough, you can
use the function `approx_n_unique` that uses the algorithm
[HyperLogLog++](https://en.wikipedia.org/wiki/HyperLogLog) to estimate the result.

The example below shows an example series where the `approx_n_unique` estimation is wrong by 0.9%:

{{code_block('user-guide/expressions/operations', 'count', ['n_unique', 'approx_n_unique'])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:count"
```

You can get more information about the unique values and their counts with the function
`value_counts`, that Polars also provides:

{{code_block('user-guide/expressions/operations', 'value_counts', ['value_counts'])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:value_counts"
```

The function `value_counts` returns the results in
[structs, a data type that we will explore in a later section](structs.md).

Alternatively, if you only need a series with the unique values or a series with the unique counts,
they are one function away:

{{code_block('user-guide/expressions/operations', 'unique_counts', ['unique', 'unique_counts'])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:unique_counts"
```

Note that we need to specify `maintain_order=True` in the function `unique` so that the order of the
results is consistent with the order of the results in `unique_counts`. See the API reference for
more information.

## Conditionals

Polars supports something akin to a ternary operator through the function `when`, which is followed
by one function `then` and an optional function `otherwise`.

The function `when` accepts a predicate expression. The values that evaluate to `True` are replaced
by the corresponding values of the expression inside the function `then`. The values that evaluate
to `False` are replaced by the corresponding values of the expression inside the function
`otherwise` or `null`, if `otherwise` is not provided.

The example below applies one step of the
[Collatz conjecture](https://en.wikipedia.org/wiki/Collatz_conjecture) to the numbers in the column
“nrs”:

{{code_block('user-guide/expressions/operations', 'collatz', ['when'])}}

```python exec="on" result="text" session="expressions/operations"
--8<-- "python/user-guide/expressions/operations.py:collatz"
```

You can also emulate a chain of an arbitrary number of conditionals, akin to Python's `elif`
statement, by chaining an arbitrary number of consecutive blocks of `.when(...).then(...)`. In those
cases, and for each given value, Polars will only consider a replacement expression that is deeper
within the chain if the previous predicates all failed for that value.
