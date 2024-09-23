# Folds

Polars provides expressions/methods for horizontal aggregations like `sum`,`min`, `mean`,
etc. However, when you need a more complex aggregation the default methods Polars supplies may not be sufficient. That's when `folds` come in handy.

The `fold` expression operates on columns for maximum speed. It utilizes the data layout very efficiently and often has vectorized execution.

### Manual sum

Let's start with an example by implementing the `sum` operation ourselves, with a `fold`.

{{code_block('user-guide/expressions/folds','mansum',['fold'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:setup"
--8<-- "python/user-guide/expressions/folds.py:mansum"
```

The snippet above recursively applies the function `f(acc, x) -> acc` to an accumulator `acc` and a new column `x`. The function operates on columns individually and can take advantage of cache efficiency and vectorization.

### Conditional

In the case where you'd want to apply a condition/predicate on all columns in a `DataFrame` a `fold` operation can be a very concise way to express this.

{{code_block('user-guide/expressions/folds','conditional',['fold'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:conditional"
```

In the snippet we filter all rows where **each** column value is `> 1`.

### Folds and string data

Folds could be used to concatenate string data. However, due to the materialization of intermediate columns, this operation will have squared complexity.

Therefore, we recommend using the `concat_str` expression for this.

{{code_block('user-guide/expressions/folds','string',['concat_str'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:string"
```
