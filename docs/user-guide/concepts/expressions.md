# Expressions

Polars has a powerful concept called expressions that is central to its very fast performance.

Expressions are at the core of many data science operations:

- taking a sample of rows from a column
- multiplying values in a column
- extracting a column of years from dates
- convert a column of strings to lowercase
- and so on!

However, expressions are also used within other operations:

- taking the mean of a group in a `group_by` operation
- calculating the size of groups in a `group_by` operation
- taking the sum horizontally across columns

Polars performs these core data transformations very quickly by:

- automatic query optimization on each expression
- automatic parallelization of expressions on many columns

Polars expressions are a mapping from a series to a series (or mathematically `Fn(Series) -> Series`). As expressions have a `Series` as an input and a `Series` as an output then it is straightforward to do a sequence of expressions (similar to method chaining in pandas).

## Examples

The following is an expression:

{{code_block('user-guide/concepts/expressions','example1',['col','sort','head'])}}

The snippet above says:

1. Select column "foo"
1. Then sort the column (not in reversed order)
1. Then take the first two values of the sorted output

The power of expressions is that every expression produces a new expression, and that they
can be _piped_ together. You can run an expression by passing them to one of Polars execution contexts.

Here we run two expressions by running `df.select`:

{{code_block('user-guide/concepts/expressions','example2',['select'])}}

All expressions are run in parallel, meaning that separate Polars expressions are **embarrassingly parallel**. Note that within an expression there may be more parallelization going on.

## Conclusion

This is the tip of the iceberg in terms of possible expressions. There are a ton more, and they can be combined in a variety of ways. This page is intended to get you familiar with the concept of expressions, in the section on [expressions](../expressions/operators.md) we will dive deeper.
