# Folds

Polars provides many expressions to perform computations across columns, like `sum_horizontal`,
`mean_horizontal`, and `min_horizontal`. However, these are just special cases of a general
algorithm called a fold, and Polars provides a general mechanism for you to compute custom folds for
when the specialised versions of Polars are not enough.

Folds computed with the function `fold` operate on the full columns for maximum speed. They utilize
the data layout very efficiently and often have vectorized execution.

## Basic example

As a first example, we will reimplement `sum_horizontal` with the function `fold`:

{{code_block('user-guide/expressions/folds','mansum',['fold'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:mansum"
```

The function `fold` expects a function `f` as the parameter `function` and `f` should accept two
arguments. The first argument is the accumulated result, which we initialise as zero, and the second
argument takes the successive values of the expressions listed in the parameter `exprs`. In our
case, they're the two columns “a” and “b”.

The snippet below includes a third explicit expression that represents what the function `fold` is
doing above:

{{code_block('user-guide/expressions/folds','mansum-explicit',['fold'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:mansum-explicit"
```

??? tip "`fold` in Python"

    Most programming languages include a higher-order function that implements the algorithm that the function `fold` in Polars implements.
    The Polars `fold` is very similar to Python's `functools.reduce`.
    You can [learn more about the power of `functools.reduce` in this article](http://mathspp.com/blog/pydonts/the-power-of-reduce).

## The initial value `acc`

The initial value chosen for the accumulator `acc` is typically, but not always, the
[identity element](https://en.wikipedia.org/wiki/Identity_element) of the operation you want to
apply. For example, if we wanted to multiply across the columns, we would not get the correct result
if our accumulator was set to zero:

{{code_block('user-guide/expressions/folds','manprod',['fold'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:manprod"
```

To fix this, the accumulator `acc` should be set to `1`:

{{code_block('user-guide/expressions/folds','manprod-fixed',['fold'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:manprod-fixed"
```

## Conditional

In the case where you'd want to apply a condition/predicate across all columns in a dataframe, a
fold can be a very concise way to express this.

{{code_block('user-guide/expressions/folds','conditional',['fold'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:conditional"
```

The snippet above filters all rows where all columns are greater than 1.

## Folds and string data

Folds could be used to concatenate string data. However, due to the materialization of intermediate
columns, this operation will have squared complexity.

Therefore, we recommend using the function `concat_str` for this:

{{code_block('user-guide/expressions/folds','string',['concat_str'])}}

```python exec="on" result="text" session="user-guide/folds"
--8<-- "python/user-guide/expressions/folds.py:string"
```
